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
    # PyG has no concept of node "id", just row index — sort explicitly so
    # row i of x/y always lines up with account i. Get this wrong and nodes
    # get silently paired with the wrong label, no error, model just can't learn.
    nodes = sorted(graph.nodes())

    raw_x = [features[n] for n in nodes]

    # Features are on wildly different scales (degree ~10s, volume ~10,000s).
    # Unscaled, the big one dominates the GCN's linear layer and drowns out the rest.
    x = StandardScaler().fit_transform(raw_x)
    x = torch.tensor(x, dtype=torch.float)

    y = torch.tensor([int(graph.nodes[n]["is_mule"]) for n in nodes], dtype=torch.long)

    # PyG wants edge_index as [2, num_edges] (row 0 = sources, row 1 =
    # targets), not [num_edges, 2] — hence the transpose below. Forgetting
    # it is the classic PyG shape bug.
    edge_list = list(graph.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)
