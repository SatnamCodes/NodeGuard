"""Runs the real pipeline once and dumps everything the static frontend
needs into web/data.json. No mocked data — reuses the same functions the
training/eval scripts use. Run from the repo root with:
    python -m src.app.export_data
"""

import json
import os

import torch

from src.data_gen.generate_graph import build_graph
from src.data_gen.node_features import compute_node_features
from src.data_gen.to_pyg import graph_to_pyg_data
from src.eval.error_analysis import FEATURE_NAMES, pattern_type
from src.eval.metrics import confusion_counts
from src.model.gnn import MuleGCN
from src.model.train import DROPOUT, HIDDEN_CHANNELS, f1_score, make_split_masks

BEST_MODEL_PATH = "src/model/mule_gcn_best.pt"
OUTPUT_PATH = "web/data.json"


def build_false_positive_verdicts(graph, features, mule_ids, false_positive_ids):
    normal_ids = [n for n in graph.nodes() if not graph.nodes[n]["is_mule"]]
    mule_id_set = set(mule_ids)

    def avg_features(node_ids):
        rows = [features[n] for n in node_ids]
        return [sum(col) / len(col) for col in zip(*rows)]

    normal_avg = avg_features(normal_ids)
    mule_avg = avg_features(mule_id_set)

    verdicts = []
    for node_id in false_positive_ids:
        vals = features[node_id]
        closer_to_mule = sum(
            1 for i in range(len(FEATURE_NAMES))
            if abs(vals[i] - mule_avg[i]) < abs(vals[i] - normal_avg[i])
        )
        explainable = closer_to_mule >= 3

        in_ids = {u for u, _v in graph.in_edges(node_id)}
        out_ids = {v for _u, v in graph.out_edges(node_id)}
        one_hop_mule = any(graph.nodes[n]["is_mule"] for n in in_ids | out_ids)

        two_hop_mule = False
        if not one_hop_mule:
            for mid in in_ids | out_ids:
                mid_neighbors = {u for u, _v in graph.in_edges(mid)} | {v for _u, v in graph.out_edges(mid)}
                if any(graph.nodes[n]["is_mule"] for n in mid_neighbors if n != node_id):
                    two_hop_mule = True
                    break

        if explainable:
            verdict = f"Feature-outlier explainable ({closer_to_mule}/{len(FEATURE_NAMES)} features closer to mule average)"
        elif one_hop_mule or two_hop_mule:
            hop = "1-hop" if one_hop_mule else "2-hop"
            verdict = f"Structurally explainable — {hop} connection to a real mule"
        else:
            verdict = "Unexplained at 1-2 hops — most plausibly the model overgeneralizing from only 14 total mule examples"

        verdicts.append({"node_id": node_id, "verdict": verdict})
    return verdicts


def main():
    graph, mule_ids = build_graph()
    features = compute_node_features(graph)
    data = graph_to_pyg_data(graph, features)

    train_mask, val_mask, test_mask = make_split_masks(data.y)

    model = MuleGCN(in_channels=data.x.shape[1], hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    tp, fp, tn, fn = confusion_counts(preds, data.y, test_mask)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(precision, recall)

    test_ids = sorted(test_mask.nonzero(as_tuple=True)[0].tolist())

    # Per-node record for every test-set account: prediction, confidence,
    # true label, pattern type, and its direct neighbors (for the frontend's
    # neighborhood graph — no need to ship the whole 314-node graph).
    nodes = {}
    for node_id in test_ids:
        pred_label = int(preds[node_id].item())
        true_label = int(data.y[node_id].item())
        confidence = float(probs[node_id, pred_label].item())

        in_ids = sorted({u for u, _v in graph.in_edges(node_id)})
        out_ids = sorted({v for _u, v in graph.out_edges(node_id)})

        neighbors = []
        for n in sorted(set(in_ids) | set(out_ids)):
            neighbors.append({
                "id": n,
                "direction": "in" if n in in_ids and n not in out_ids else
                             ("out" if n in out_ids and n not in in_ids else "both"),
                "predicted_mule": bool(int(preds[n].item()) == 1),
                "true_mule": bool(int(data.y[n].item()) == 1),
            })

        nodes[str(node_id)] = {
            "id": node_id,
            "predicted_mule": bool(pred_label == 1),
            "true_mule": bool(true_label == 1),
            "confidence": round(confidence, 4),
            "pattern": pattern_type(node_id),
            "neighbors": neighbors,
        }

    caught = {"layering": 0, "funnel": 0}
    total = {"layering": 0, "funnel": 0}
    false_positive_ids = []
    for node_id in test_ids:
        true_label = int(data.y[node_id].item())
        pred_label = int(preds[node_id].item())
        pattern = pattern_type(node_id)
        if true_label == 1:
            total[pattern] = total.get(pattern, 0) + 1
            if pred_label == 1:
                caught[pattern] = caught.get(pattern, 0) + 1
        elif true_label == 0 and pred_label == 1:
            false_positive_ids.append(node_id)

    false_positives = build_false_positive_verdicts(graph, features, mule_ids, false_positive_ids)

    output = {
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "test_mule_count": tp + fn,
        },
        "error_analysis": {
            "layering_caught": caught["layering"],
            "layering_total": total["layering"],
            "funnel_caught": caught["funnel"],
            "funnel_total": total["funnel"],
            "false_positives": false_positives,
        },
        "test_node_ids": test_ids,
        "nodes": nodes,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_PATH} — {len(test_ids)} test-set accounts, "
          f"{len(false_positives)} false positives.")


if __name__ == "__main__":
    main()
