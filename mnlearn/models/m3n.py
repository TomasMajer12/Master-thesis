"""
Maximum Margin Markov Network (M3N).

The M3N scores a labeling y given input x on a graph G = (V, E) as:

    F(y | x) = sum_{v in V}  unary(x_v)[y_v]
             + sum_{(u,v) in E}  pairwise[y_u, y_v]

where:
    - unary potentials come from a backbone network (per-node class scores)
    - pairwise potentials are a shared learnable matrix W of shape [K, K]
      (K = num_classes), encoding affinity between class pairs across edges

The graph structure (chain, grid, ...) is not baked into the model.
It is provided externally as an edge list, so the same M3N works for both
chain-structured HMC and grid-structured Sudoku.

Edge-list constructors (``chain_edges``, ``sudoku_edges``, ``build_graph``)
live in :mod:`mnlearn.models.graph`.
"""

import torch
import torch.nn as nn


class M3N(nn.Module):
    """Maximum Margin Markov Network.

    Args:
        backbone:             nn.Module mapping [batch, num_nodes, *input_shape]
                              -> [batch, num_nodes, num_classes]
        num_classes:          number of classes per node (K)
        pairwise_init_scale:  std of the initial pairwise weights. Kept small
                              so early training is dominated by unary
                              potentials, which are easier to learn first.
    """

    def __init__(self, backbone, num_classes, pairwise_init_scale: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        # W[i, j] = affinity between class i at node u and class j at node v.
        self.pairwise = nn.Parameter(
            torch.randn(num_classes, num_classes) * pairwise_init_scale
        )

    def unary(self, x):
        """Compute unary potentials from raw input.

        Args:
            x: [batch, num_nodes, input_dim]
        Returns:
            [batch, num_nodes, num_classes]
        """
        return self.backbone(x)

    def score(self, unary, y, edges):
        """Compute the score F(y | x) for a given labeling.

        This must be differentiable w.r.t. `unary` and `self.pairwise`
        (but not w.r.t. `y`, which is a fixed integer labeling).

        Args:
            unary: [batch, num_nodes, num_classes] — unary potentials (has grad)
            y:     [batch, num_nodes]              — integer class labels
            edges: [num_edges, 2]                  — edge list (LongTensor)

        Returns:
            [batch] — total score per sample
        """
        # --- Unary score ---
        # For each sample b and node v, pick unary[b, v, y[b,v]] and sum over v.
        # torch.gather picks elements along dim=2 at indices given by y.
        #   unary shape:        [batch, num_nodes, num_classes]
        #   y.unsqueeze(-1):    [batch, num_nodes, 1]
        #   after gather:       [batch, num_nodes, 1]  -> squeeze -> [batch, num_nodes]
        #   sum over nodes:     [batch]
        unary_score = torch.gather(unary, 2, y.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        # --- Pairwise score ---
        # For each edge (u, v), look up pairwise[y[b,u], y[b,v]] and sum over edges.
        if edges is not None and edges.shape[0] > 0:
            src, dst = edges[:, 0], edges[:, 1]     # [num_edges] each
            y_src = y[:, src]                        # [batch, num_edges]
            y_dst = y[:, dst]                        # [batch, num_edges]
            # Fancy indexing into pairwise: result shape [batch, num_edges]
            pw_score = self.pairwise[y_src, y_dst].sum(dim=1)  # [batch]
        else:
            pw_score = torch.zeros(y.shape[0], device=unary.device)

        return unary_score + pw_score

    def forward(self, x):
        """Convenience: returns unary potentials (used by inference)."""
        return self.unary(x)
