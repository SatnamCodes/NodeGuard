"""Loads the best trained checkpoint and prints the headline evaluation
report on the held-out test set: precision, recall, F1, confusion matrix,
and false-positive cost. Run from the repo root with:
    python -m src.eval.metrics

For a deeper per-node / per-pattern breakdown, see error_analysis.py.
"""

import torch

from src.data_gen.generate_graph import build_graph
from src.data_gen.node_features import compute_node_features
from src.data_gen.to_pyg import graph_to_pyg_data
from src.model.gnn import MuleGCN
from src.model.train import HIDDEN_CHANNELS, DROPOUT, f1_score, make_split_masks

BEST_MODEL_PATH = "src/model/mule_gcn_best.pt"


def confusion_counts(preds, y, mask):
    preds = preds[mask]
    targets = y[mask]

    true_positive = int(((preds == 1) & (targets == 1)).sum().item())
    false_positive = int(((preds == 1) & (targets == 0)).sum().item())
    true_negative = int(((preds == 0) & (targets == 0)).sum().item())
    false_negative = int(((preds == 0) & (targets == 1)).sum().item())
    return true_positive, false_positive, true_negative, false_negative


def main():
    graph, mule_ids = build_graph()
    features = compute_node_features(graph)
    data = graph_to_pyg_data(graph, features)

    # Same split logic as training, so this reproduces the identical test
    # set the checkpoint was actually evaluated against during training.
    _, _, test_mask = make_split_masks(data.y)

    model = MuleGCN(in_channels=data.x.shape[1], hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)

    tp, fp, tn, fn = confusion_counts(preds, data.y, test_mask)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(precision, recall)

    num_test_mules = tp + fn
    num_test_normal = tn + fp

    print("=" * 60)
    print("MULE-RING GCN — TEST SET METRICS REPORT")
    print("=" * 60)

    print(f"\nTest set: {num_test_mules} mule accounts, {num_test_normal} normal accounts")

    print("\nHeadline metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    print("\nConfusion matrix (raw counts):")
    print(f"  {'':>20}{'predicted mule':>18}{'predicted normal':>20}")
    print(f"  {'actual mule':>20}{tp:>18}{fn:>20}")
    print(f"  {'actual normal':>20}{fp:>18}{tn:>20}")
    print(f"\n  True positives  (mules correctly caught):        {tp}")
    print(f"  False negatives (mules missed):                   {fn}")
    print(f"  False positives (normal accounts wrongly flagged): {fp}")
    print(f"  True negatives  (normal accounts correctly cleared): {tn}")

    print("\nFalse-positive cost:")
    print(
        f"  {fp} normal account(s) would be sent for unnecessary manual review "
        f"if this model's test-set predictions were acted on directly."
    )

    print(
        "\nCaveat: Test set contains only 3 mule accounts; metrics at this "
        "sample size should be read as directional, not statistically precise."
    )


if __name__ == "__main__":
    main()
