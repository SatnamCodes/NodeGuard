"""A small 2-layer Graph Convolutional Network for binary node
classification (mule vs. not-mule) over the transaction graph.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MuleGCN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=16, out_channels=2, dropout=0.3):
        super().__init__()
        # 2 GCNConv layers = each node sees 2 hops out (neighbors of
        # neighbors) — needed to catch funnel feeders via their collector's
        # weird in-degree, not just direct neighbors.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # No-op at eval time (model.eval() flips self.training off).
        # Helps avoid overfitting with only 14 mule examples to learn from.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # Raw logits, not softmax — CrossEntropyLoss does log_softmax itself,
        # softmaxing here too would double up and break the gradient.
        return x
