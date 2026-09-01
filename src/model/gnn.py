"""A small 2-layer Graph Convolutional Network for binary node
classification (mule vs. not-mule) over the transaction graph.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MuleGCN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=16, out_channels=2, dropout=0.3):
        super().__init__()
        # A GCNConv layer replaces each node's features with a weighted
        # average of its own and its direct neighbors' features (then a
        # linear transform). Stacking 2 of these means a node's final
        # representation is influenced by neighbors up to 2 hops away —
        # e.g. a funnel collector's *feeder* accounts, not just the
        # collector itself. That 2-hop reach is exactly the "structure"
        # signal a flat per-account feature table can't see.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Dropout randomly zeroes activations during training only — it's
        # a no-op once you call model.eval() (F.dropout checks the module's
        # self.training flag internally). It forces the network not to
        # rely on any single feature/neighbor path too heavily, which
        # matters here because we have very few mule examples to learn
        # from and could otherwise overfit to quirks of those 14 nodes.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # Return raw logits, not softmax probabilities: nn.CrossEntropyLoss
        # applies log-softmax internally, so feeding it already-softmaxed
        # values would double-apply the transform and break the gradient.
        # (A 3rd GCNConv layer could be inserted here for a wider receptive
        # field, but 2 is kept as the default to stay easy to reason about.)
        return x
