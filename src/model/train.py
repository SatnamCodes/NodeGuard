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

# Match the seed already used for data generation so the whole pipeline
# (graph, split, model init) is reproducible end to end.
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
    """Stratified 60/20/20 train/val/test split, returned as boolean
    masks over all nodes (not separate Data objects).

    Splitting is stratified because mules are only ~4.5% of nodes (14/314):
    a plain random split can, by chance, put very few or zero mule nodes
    into test/val, making that split's accuracy meaningless either way
    (undefined precision/recall, or a lucky/unlucky single example
    deciding the whole score). stratify= keeps each split's mule ratio
    close to the full dataset's ratio.

    Masks (not separate Data objects) because message passing needs every
    node's features/edges visible regardless of split — GCNConv aggregates
    over `edge_index`, so a val/test node's *neighbors* still need to be
    present in the graph during the forward pass. Only the *loss* is
    restricted to train_mask nodes; val/test nodes are simply not counted
    in that loss.
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
    """Inverse-frequency class weights for CrossEntropyLoss, computed from
    the TRAINING split only (val/test must stay untouched by anything that
    shapes training, including this).

    Without weighting, the loss is just an average over all training
    nodes. With ~95% of nodes being "not mule", a model that predicts
    "not mule" for everyone already gets ~95% of the loss "right" and the
    gradient barely pushes it to ever predict "mule" — high accuracy,
    ~0 recall on the class we actually care about. Weighting node class
    c's loss term by 1/count(c) makes each CLASS contribute equally to
    the loss regardless of how many nodes are in it, so mistakes on the
    14 mule nodes matter as much in aggregate as mistakes on the ~270
    normal ones.
    """
    y_train = y[mask]
    counts = torch.bincount(y_train, minlength=2).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()  # normalize, not required but keeps loss scale stable
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
    """Harmonic (not arithmetic) mean of precision and recall.

    The harmonic mean is what makes this fix the degenerate-checkpoint bug:
    it's pulled toward whichever of the two inputs is SMALLER, so a model
    that "flags everything" (precision near 0, recall 1.0 — exactly what we
    saw at epoch 1: precision 0.067, recall 1.000) still gets an F1 near 0,
    because a near-zero precision drags the harmonic mean down with it. A
    plain average of 0.067 and 1.000 would instead score that degenerate
    model at ~0.53, still looking "good" — which is why an arithmetic mean
    wouldn't fix this, only the harmonic mean does.
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

    # We track the BEST validation F1 seen so far, and how many epochs it's
    # been since it last improved (the "patience" counter). Training loss
    # falling every epoch does not mean the model is getting better at the
    # job we care about — we already saw it keep dropping past epoch 20
    # while val recall fell from 0.667 to 0.333, i.e. the model was
    # memorizing the 8 training-set mules rather than learning a pattern
    # that generalizes.
    #
    # We checkpoint on F1 rather than recall alone because recall alone is
    # gameable: a model that predicts "mule" for nearly every node gets
    # perfect recall for free. We hit this for real — epoch 1's near-random
    # weights scored val recall 1.000 (precision 0.067), and a recall-only
    # checkpoint locked that degenerate model in as "best" before training
    # had learned anything. F1 collapses toward 0 whenever precision OR
    # recall is near 0, so that failure mode can no longer look like a win.
    # (Plain accuracy is unusable here for the same underlying reason as the
    # class-weighted loss above: ~95% of nodes are "not mule", so accuracy
    # stays high even while recall on mules is 0.)
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        # Forward pass sees the WHOLE graph (transductive setting: unlike
        # image classification, there's one fixed graph and we're
        # predicting labels for a held-out subset of its nodes, not
        # generalizing to unseen graphs). Loss is computed only over
        # train_mask nodes so val/test labels never influence the gradient.
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
            # New best: checkpoint it immediately. We save to disk (not just
            # keep a copy in a variable) so the best weights survive even if
            # training crashes or we stop the process, and so BEST_MODEL_SAVE_PATH
            # always reflects the best model found so far, independent of
            # where training currently is.
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

    # Reload the BEST checkpoint before final evaluation. Without this, the
    # model in memory is whatever the LAST epoch produced — which, as we
    # saw, can already be a worse, overfit version of the model even though
    # a better checkpoint exists on disk. Loading it back guarantees the
    # test-set numbers we report describe the best model we actually found,
    # not whatever state training happened to end on.
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
