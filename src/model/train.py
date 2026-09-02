"""Trains MuleGCN on the synthetic transaction graph and saves the
trained weights. Run from the repo root with:  python -m src.model.train
"""

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.data_gen.generate_graph import build_graph
from src.data_gen.node_features import compute_node_features
from src.data_gen.to_pyg import graph_to_pyg_data
from src.model.gnn import MuleGCN

# Same seed as generate_graph.py so the whole pipeline is reproducible.
SEED = 42
torch.manual_seed(SEED)

EPOCHS = 150
HIDDEN_CHANNELS = 16
DROPOUT = 0.3
LEARNING_RATE = 0.01
MODEL_SAVE_PATH = "src/model/mule_gcn.pt"
BEST_MODEL_SAVE_PATH = "src/model/mule_gcn_best.pt"
PATIENCE = 20


def make_split_masks(y, seed=SEED):
    """Stratified 60/20/20 train/val/test split as boolean masks over all
    nodes (not separate Data objects).

    Stratified because mules are only ~4.5% of nodes — a plain random split
    could easily land 0 mules in test/val by chance. Masks instead of
    separate graphs because GCNConv still needs val/test nodes' neighbors
    visible during the forward pass; only the loss is restricted to train_mask.
    """
    num_nodes = y.shape[0]
    indices = torch.arange(num_nodes)

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=seed
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def make_class_weights(y, mask):
    """Inverse-frequency class weights for CrossEntropyLoss, from the
    training split only (val/test shouldn't influence training at all).

    Without this, predicting "not mule" for everyone already gets ~95% of
    the loss right, so gradient descent barely bothers learning to predict
    "mule". Weighting by 1/count makes each class matter equally in the loss.
    """
    y_train = y[mask]
    counts = torch.bincount(y_train, minlength=2).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()  # just keeps the loss scale readable
    return weights


def accuracy(logits, y, mask):
    preds = logits[mask].argmax(dim=1)
    return (preds == y[mask]).float().mean().item()


def precision_recall(logits, y, mask):
    """Precision/recall on the mule class (label 1) only — accuracy alone
    hides whether the model is actually catching mules vs. just predicting
    the majority class.
    """
    preds = logits[mask].argmax(dim=1)
    targets = y[mask]

    true_positive = ((preds == 1) & (targets == 1)).sum().item()
    predicted_positive = (preds == 1).sum().item()
    actual_positive = (targets == 1).sum().item()

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0
    return precision, recall


def f1_score(precision, recall):
    """Harmonic mean of precision/recall — not arithmetic. Harmonic mean
    gets pulled toward whichever is smaller, so a "flag everything" model
    (precision 0.067, recall 1.0, like our epoch-1 checkpoint) still scores
    near 0 instead of the ~0.53 a plain average would give it.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    graph, mule_ids = build_graph()
    features = compute_node_features(graph)
    data = graph_to_pyg_data(graph, features)

    train_mask, val_mask, test_mask = make_split_masks(data.y)
    print(
        f"Split sizes -> train: {train_mask.sum().item()} "
        f"(mules: {data.y[train_mask].sum().item()}), "
        f"val: {val_mask.sum().item()} (mules: {data.y[val_mask].sum().item()}), "
        f"test: {test_mask.sum().item()} (mules: {data.y[test_mask].sum().item()})"
    )

    class_weights = make_class_weights(data.y, train_mask)

    model = MuleGCN(
        in_channels=data.x.shape[1], hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Track best val F1 + a patience counter for early stopping. Training
    # loss kept falling well past the point where val recall started
    # decaying (overfitting on the 8 training mules) — checkpointing on F1
    # instead of recall avoids locking in a degenerate "flag everything"
    # epoch like we saw at epoch 1 (recall 1.0, precision 0.067).
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        # Forward pass sees the whole graph (transductive setting) — loss
        # is just restricted to train_mask so val/test never leaks into it.
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(val_out[val_mask], data.y[val_mask], weight=class_weights)
            val_acc = accuracy(val_out, data.y, val_mask)
            val_precision, val_recall = precision_recall(val_out, data.y, val_mask)
            val_f1 = f1_score(val_precision, val_recall)

        if epoch % 10 == 0 or epoch == 1:
            train_acc = accuracy(out, data.y, train_mask)
            print(
                f"epoch {epoch:3d} | train_loss {loss.item():.4f} train_acc {train_acc:.3f} "
                f"| val_loss {val_loss.item():.4f} val_acc {val_acc:.3f} "
                f"val_precision {val_precision:.3f} val_recall {val_recall:.3f} "
                f"val_f1 {val_f1:.3f}"
            )

        if val_f1 > best_val_f1:
            # Save to disk right away, not just in memory, so we don't lose
            # the best model if training crashes or gets interrupted later.
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= PATIENCE:
            stopped_early = True
            print(
                f"\nEarly stopping at epoch {epoch}: val F1 hasn't improved "
                f"for {PATIENCE} epochs (best val F1 {best_val_f1:.3f} "
                f"at epoch {best_epoch})"
            )
            break

    if not stopped_early:
        print(
            f"\nReached max epochs ({EPOCHS}) without triggering early stopping "
            f"(best val F1 {best_val_f1:.3f} at epoch {best_epoch})"
        )

    # Reload the best checkpoint — whatever's in memory now is just the
    # last epoch, which can be a worse, overfit version of the model.
    model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, weights_only=True))

    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_acc = accuracy(test_out, data.y, test_mask)
        test_precision, test_recall = precision_recall(test_out, data.y, test_mask)
        test_f1 = f1_score(test_precision, test_recall)
    print(
        f"\nHeld-out test set (best checkpoint, epoch {best_epoch}, never used "
        f"for training/tuning) -> acc {test_acc:.3f} precision {test_precision:.3f} "
        f"recall {test_recall:.3f} f1 {test_f1:.3f}"
    )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved best model weights to {MODEL_SAVE_PATH} (and {BEST_MODEL_SAVE_PATH})")


if __name__ == "__main__":
    main()
