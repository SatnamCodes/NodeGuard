"""Loads the best trained checkpoint, runs it on the held-out test set,
and prints a human-readable per-node error report. Run from the repo root
with:  python -m src.eval.error_analysis
"""

import torch

from src.data_gen.generate_graph import (
    NUM_FUNNEL_COLLECTORS,
    NUM_LAYERING_ACCOUNTS,
    NUM_NORMAL_ACCOUNTS,
    build_graph,
)
from src.data_gen.node_features import compute_node_features
from src.data_gen.to_pyg import graph_to_pyg_data
from src.model.gnn import MuleGCN
from src.model.train import HIDDEN_CHANNELS, DROPOUT, make_split_masks

BEST_MODEL_PATH = "src/model/mule_gcn_best.pt"

LAYERING_START = NUM_NORMAL_ACCOUNTS
LAYERING_END = LAYERING_START + NUM_LAYERING_ACCOUNTS  # exclusive
FUNNEL_START = LAYERING_END
FUNNEL_END = FUNNEL_START + NUM_FUNNEL_COLLECTORS  # exclusive


def pattern_type(node_id):
    """Which synthetic pattern this node belongs to, purely from its id
    range (ids are assigned contiguously per pattern in build_graph()).
    """
    if LAYERING_START <= node_id < LAYERING_END:
        return "layering"
    if FUNNEL_START <= node_id < FUNNEL_END:
        return "funnel"
    return "normal"


def main():
    graph, mule_ids = build_graph()
    features = compute_node_features(graph)
    data = graph_to_pyg_data(graph, features)

    # Same split logic as training, so this is the exact test set the
    # checkpoint was evaluated on.
    _, _, test_mask = make_split_masks(data.y)

    model = MuleGCN(in_channels=data.x.shape[1], hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)

    test_node_ids = test_mask.nonzero(as_tuple=True)[0].tolist()

    print("=" * 78)
    print("PER-NODE TEST SET REPORT")
    print("=" * 78)
    print(f"{'node_id':>8}  {'pattern':<10} {'true':<10} {'predicted':<10} {'result'}")
    print("-" * 78)

    correctly_caught = []   # true positive: real mule, flagged as mule
    missed = []              # false negative: real mule, flagged as normal
    false_positives = []     # real normal, flagged as mule
    correctly_cleared = []   # true negative: real normal, flagged as normal

    for node_id in sorted(test_node_ids):
        true_label = int(data.y[node_id].item())
        pred_label = int(preds[node_id].item())
        pattern = pattern_type(node_id)

        true_str = "MULE" if true_label == 1 else "normal"
        pred_str = "MULE" if pred_label == 1 else "normal"

        if true_label == 1 and pred_label == 1:
            result = "caught (TP)"
            correctly_caught.append((node_id, pattern))
        elif true_label == 1 and pred_label == 0:
            result = "MISSED (FN)"
            missed.append((node_id, pattern))
        elif true_label == 0 and pred_label == 1:
            result = "FALSE ALARM (FP)"
            false_positives.append((node_id, pattern))
        else:
            result = "correct (TN)"
            correctly_cleared.append((node_id, pattern))

        print(f"{node_id:>8}  {pattern:<10} {true_str:<10} {pred_str:<10} {result}")

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    total_mules_in_test = len(correctly_caught) + len(missed)
    print(f"\nMules in test set: {total_mules_in_test}")

    print(f"\nCorrectly caught ({len(correctly_caught)}):")
    if correctly_caught:
        for node_id, pattern in correctly_caught:
            print(f"  - node {node_id} ({pattern})")
    else:
        print("  (none)")

    print(f"\nMissed / false negatives ({len(missed)}):")
    if missed:
        for node_id, pattern in missed:
            print(f"  - node {node_id} ({pattern})")
    else:
        print("  (none)")

    print(f"\nFalse alarms / false positives ({len(false_positives)}):")
    if false_positives:
        for node_id, pattern in false_positives:
            print(f"  - node {node_id} (normal account incorrectly flagged as mule)")
    else:
        print("  (none)")

    # Split recall by pattern type — a blended number could hide that the
    # model nails one pattern and whiffs the other.
    print("\nRecall by pattern type (of mules present in test set):")
    for pattern in ("layering", "funnel"):
        caught = sum(1 for _, p in correctly_caught if p == pattern)
        total = caught + sum(1 for _, p in missed if p == pattern)
        if total == 0:
            print(f"  {pattern:<10}: no {pattern} mules in this test split")
        else:
            print(f"  {pattern:<10}: {caught}/{total} caught ({caught / total:.0%})")

    layering_caught = sum(1 for _, p in correctly_caught if p == "layering")
    layering_total = layering_caught + sum(1 for _, p in missed if p == "layering")
    funnel_caught = sum(1 for _, p in correctly_caught if p == "funnel")
    funnel_total = funnel_caught + sum(1 for _, p in missed if p == "funnel")

    print("\nDoes error clustering by pattern show up here?")
    if layering_total == 0 or funnel_total == 0:
        print(
            "  Can't compare — the test split doesn't contain mules from both "
            "patterns (small test set, only 3 mules total from a 14-mule pool)."
        )
    elif (layering_caught / layering_total) == (funnel_caught / funnel_total):
        print("  No difference in recall between layering and funnel mules in this split.")
    elif (layering_caught / layering_total) > (funnel_caught / funnel_total):
        print("  The model catches layering-chain mules more reliably than funnel mules here.")
    else:
        print("  The model catches funnel mules more reliably than layering-chain mules here.")

    print(
        f"\nNote: with only {total_mules_in_test} mules in the test split, each single "
        "node flips the per-pattern recall by a large amount — treat this "
        "breakdown as a qualitative signal, not a statistically solid claim."
    )

    print_false_positive_feature_comparison(false_positives, features, mule_ids, graph)
    print_neighbor_inspection([171, 219], graph)
    print_two_hop_trace([171, 219], graph)


FEATURE_NAMES = ["in_degree", "out_degree", "avg_in_amount", "avg_out_amount", "total_volume"]


def print_false_positive_feature_comparison(false_positives, features, mule_ids, graph):
    """Compares each false positive's raw features against normal/mule
    averages — are these outliers within the normal population, or does
    the model's mistake have no visible cause in the numbers?
    """
    print("\n" + "=" * 78)
    print("FALSE POSITIVE FEATURE COMPARISON")
    print("=" * 78)

    if not false_positives:
        print("\nNo false positives to analyze.")
        return

    normal_ids = [n for n in graph.nodes() if not graph.nodes[n]["is_mule"]]
    mule_id_set = set(mule_ids)

    def avg_features(node_ids):
        rows = [features[n] for n in node_ids]
        return [sum(col) / len(col) for col in zip(*rows)]

    normal_avg = avg_features(normal_ids)
    mule_avg = avg_features(mule_id_set)

    header = f"{'feature':<16}" + "".join(f"{name:>16}" for name in ("normal avg", "mule avg"))
    print(f"\n{header}")
    print("-" * len(header))
    for i, feat_name in enumerate(FEATURE_NAMES):
        print(f"{feat_name:<16}{normal_avg[i]:>16.2f}{mule_avg[i]:>16.2f}")

    for node_id, _pattern in false_positives:
        node_vals = features[node_id]
        print(f"\nnode {node_id} (true label: normal, predicted: MULE)")
        col_header = f"  {'feature':<16}{'this node':>14}{'normal avg':>14}{'mule avg':>14}{'closer to':>12}"
        print(col_header)
        print("  " + "-" * (len(col_header) - 2))

        closer_to_mule_count = 0
        for i, feat_name in enumerate(FEATURE_NAMES):
            value = node_vals[i]
            dist_to_normal = abs(value - normal_avg[i])
            dist_to_mule = abs(value - mule_avg[i])
            closer = "mule" if dist_to_mule < dist_to_normal else "normal"
            if closer == "mule":
                closer_to_mule_count += 1
            print(
                f"  {feat_name:<16}{value:>14.2f}{normal_avg[i]:>14.2f}"
                f"{mule_avg[i]:>14.2f}{closer:>12}"
            )

        majority = "closer to MULE profile overall" if closer_to_mule_count >= 3 else \
            "closer to NORMAL profile overall"
        print(
            f"  -> {closer_to_mule_count}/{len(FEATURE_NAMES)} features closer to the mule "
            f"average than the normal average: {majority}"
        )

    print(
        "\nInterpretation: a false positive whose features skew toward the mule "
        "profile is a reasonable model mistake — the raw numbers themselves are "
        "genuinely ambiguous, not just a model error with no visible cause. A "
        "false positive whose features look clearly normal instead points to the "
        "model relying on graph structure (its neighbors) rather than its own "
        "features to make that call, which the numbers alone can't explain."
    )


def print_neighbor_inspection(node_ids, graph):
    """Lists each node's direct neighbors and whether any are real mules —
    a mule neighbor would explain a false positive via message passing
    ("guilt by association"), separate from the node's own features.
    """
    print("\n" + "=" * 78)
    print("NEIGHBOR INSPECTION (guilt-by-association check)")
    print("=" * 78)

    for node_id in node_ids:
        print(f"\nnode {node_id} (true label: normal, predicted: MULE)")

        in_neighbors = [(u, graph.nodes[u]["is_mule"]) for u, _v in graph.in_edges(node_id)]
        out_neighbors = [(v, graph.nodes[v]["is_mule"]) for _u, v in graph.out_edges(node_id)]

        # Dedupe — multiple transactions to the same counterparty still
        # count as one neighbor for message passing.
        distinct_in = sorted(set(in_neighbors))
        distinct_out = sorted(set(out_neighbors))

        print(f"  Incoming (money received FROM) — {len(distinct_in)} distinct account(s):")
        if distinct_in:
            for neighbor_id, is_mule in distinct_in:
                flag = "  <-- MULE" if is_mule else ""
                print(f"    - node {neighbor_id} ({'mule' if is_mule else 'normal'}){flag}")
        else:
            print("    (none)")

        print(f"  Outgoing (money sent TO) — {len(distinct_out)} distinct account(s):")
        if distinct_out:
            for neighbor_id, is_mule in distinct_out:
                flag = "  <-- MULE" if is_mule else ""
                print(f"    - node {neighbor_id} ({'mule' if is_mule else 'normal'}){flag}")
        else:
            print("    (none)")

        all_neighbors = set(distinct_in) | set(distinct_out)
        mule_neighbor_count = sum(1 for _, is_mule in all_neighbors if is_mule)
        total_neighbors = len(all_neighbors)

        if mule_neighbor_count > 0:
            print(
                f"  -> Directly connected to {mule_neighbor_count}/{total_neighbors} real "
                "mule account(s): this supports a 'guilt by association' explanation — "
                "message passing would pull this node's representation toward its "
                "mule neighbor's features, even though its own features look normal."
            )
        else:
            print(
                f"  -> No direct neighbors are real mules (0/{total_neighbors}). The false "
                "positive isn't explained by a 1-hop mule neighbor — it would have to "
                "come from a 2-hop neighbor (a neighbor-of-a-neighbor, reachable through "
                "this model's 2nd GCNConv layer) or from the model itself being wrong."
            )


def _direct_neighbors(node_id, graph):
    """All distinct nodes directly connected to node_id, either direction."""
    in_ids = {u for u, _v in graph.in_edges(node_id)}
    out_ids = {v for _u, v in graph.out_edges(node_id)}
    return in_ids | out_ids


def print_two_hop_trace(node_ids, graph):
    """Checks every neighbor-of-a-neighbor (2-hop reach, i.e. what the 2nd
    GCNConv layer can actually see) for a real mule, and prints the path
    if found.
    """
    print("\n" + "=" * 78)
    print("2-HOP TRACE (does a 2nd-layer GCNConv reach explain the mistake?)")
    print("=" * 78)

    for node_id in node_ids:
        print(f"\nnode {node_id}:")
        direct = _direct_neighbors(node_id, graph)

        paths_to_mules = []
        for mid_node in sorted(direct):
            two_hop_neighbors = _direct_neighbors(mid_node, graph)
            for end_node in sorted(two_hop_neighbors):
                if end_node == node_id:
                    continue  # skip paths that lead straight back to the start
                if graph.nodes[end_node]["is_mule"]:
                    paths_to_mules.append((mid_node, end_node))

        if paths_to_mules:
            print(f"  Found {len(paths_to_mules)} 2-hop path(s) to a real mule:")
            for mid_node, mule_node in paths_to_mules:
                print(f"    {node_id} -> {mid_node} -> {mule_node} (MULE)")
            print(
                "  -> A 2-hop mule connection exists. This plausibly explains the false "
                "positive: the 2nd GCNConv layer gives the model exactly this reach, so "
                "this node's representation would have picked up signal from the mule "
                f"at the far end of the path via intermediate node {paths_to_mules[0][0]}."
            )
        else:
            print(f"  No 2-hop path to any real mule (checked all neighbors of all "
                  f"{len(direct)} direct neighbors).")
            print(
                "  -> No structural explanation at 1 or 2 hops. This false positive is "
                "most plausibly attributed to the model overgeneralizing from a small "
                "training set (only 14 total mule examples across the whole graph) "
                "rather than a specific graph-structural cause."
            )


if __name__ == "__main__":
    main()
