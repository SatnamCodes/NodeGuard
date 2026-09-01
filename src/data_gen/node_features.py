
"""Computes per-node numeric features from the transaction graph, for
input into the GNN. These are the same category of signal real AML
systems use: degree, transaction velocity, and volume.
"""

import numpy as np


def compute_node_features(graph):
    """
    Returns a dict: {node_id: [in_degree, out_degree, avg_in_amount,
    avg_out_amount, total_volume]}
    """
    features = {}

    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        in_amounts = [d["amount"] for _, _, d in graph.in_edges(node, data=True)]
        out_amounts = [d["amount"] for _, _, d in graph.out_edges(node, data=True)]

        avg_in_amount = float(np.mean(in_amounts)) if in_amounts else 0.0
        avg_out_amount = float(np.mean(out_amounts)) if out_amounts else 0.0
        total_volume = float(sum(in_amounts) + sum(out_amounts))

        features[node] = [
            in_degree,
            out_degree,
            avg_in_amount,
            avg_out_amount,
            total_volume,
        ]

    return features