"""Converts the NetworkX transaction graph + hand-computed node features
into a single PyTorch Geometric `Data` object, ready to feed a GNN.
"""

import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def graph_to_pyg_data(graph, features):
    """
    graph: nx.DiGraph from build_graph(), with `is_mule` set on every node.
    features: dict {node_id: [in_degree, out_degree, avg_in_amount,
        avg_out_amount, total_volume]} from compute_node_features(graph).

    Returns a single torch_geometric.data.Data with x, edge_index, y set.
    """
    # Node ids are contiguous ints 0..N-1 already, but we sort explicitly
    # anyway: PyG has no concept of a node "id", only a row index into x.
    # If the row order here doesn't match the order used to build y, every
    # node gets silently paired with the wrong label — no error, just a
    # model that can't learn. Sorting pins that order down explicitly
    # instead of relying on dict/graph iteration order being stable.
    nodes = sorted(graph.nodes())

    raw_x = [features[n] for n in nodes]

    # The 5 features live on very different scales (degree is a small int,
    # total_volume can be in the thousands). A GCN layer is just a linear
    # layer applied per-node before aggregating neighbors, so an unscaled
    # feature with a huge range dominates the weighted sum and drowns out
    # the others. StandardScaler rescales every column to mean 0, std 1.
    x = StandardScaler().fit_transform(raw_x)
    x = torch.tensor(x, dtype=torch.float)

    y = torch.tensor([int(graph.nodes[n]["is_mule"]) for n in nodes], dtype=torch.long)

    # PyG expects edge_index as shape [2, num_edges]: row 0 = source node
    # indices, row 1 = target node indices for every edge, NOT a [num_edges, 2]
    # list of (source, target) pairs. Building it as a list of pairs and
    # forgetting the .t() (transpose) is the single most common PyG shape
    # bug for beginners.
    edge_list = list(graph.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)
