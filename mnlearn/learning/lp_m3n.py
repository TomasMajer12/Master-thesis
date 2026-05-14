"""
LP-M3N: LP relaxation loss for Maximum Margin Markov Networks.

Based on:
    Franc & Yermakov. "Learning Maximum Margin Markov Networks from
    examples with missing labels." ACML 2021.  Section 3.3, Eq. (24).

For fully supervised data (no missing labels), the LP-relaxed partial
margin-rescaling loss reduces to:

    L(w, phi) = U(x, y, phi, w) - score(x, y; w)

where U decomposes into cheap local argmax operations:

    U = sum_v  max_k  [ unary[v,k] + Delta(y_v, k)
                        - sum_{v' in N(v)} phi_{vv'}(k) ]
      + sum_e  max_{k,k'}  [ pairwise[k,k'] + phi_{vv'}(k) + phi_{v'v}(k') ]

Delta(y_v, k) = (1/T) * [[y_v != k]]   (normalised Hamming, per node).

Each training example i carries its own auxiliary variables phi_i of shape
[2, E, K].  These are jointly optimised with the model parameters w by
gradient descent, making training tractable on general (non-tree) graphs.
"""

import torch


def lp_m3n_loss(unary, pairwise, y_true, edges, phi):
    """LP-M3N loss for a mini-batch (fully supervised).

    Args:
        unary:    [B, T, K]    unary potentials (requires grad)
        pairwise: [K, K]       shared pairwise weights (requires grad)
        y_true:   [B, T]       ground-truth integer labels (LongTensor)
        edges:    [E, 2]       edge list (LongTensor) — shared across batch
        phi:      [B, 2, E, K] per-example auxiliary message variables,
                               stacked along the leading batch dim.
                               phi[b, 0, e, :] = messages for source node
                                                 of edge e in example b.
                               phi[b, 1, e, :] = messages for dest   node.

    Returns:
        loss: [B] tensor of per-example losses, differentiable w.r.t.
              ``unary``, ``pairwise``, and the leaves underlying ``phi``.
              The caller reduces over the batch (``.mean()`` in the
              trainer).
    """
    B, T, K = unary.shape
    E = edges.shape[0]
    src = edges[:, 0]  # [E]
    dst = edges[:, 1]  # [E]

    # --- Hamming loss augmentation (per batch element) ---
    # Δ_v(k) = 1/T if k != y_true[b, v] else 0. The /T normalisation
    # keeps the total Hamming task loss in [0, 1], 
    hamming_aug = torch.ones(B, T, K, device=unary.device, dtype=unary.dtype) / T
    hamming_aug.scatter_(2, y_true.unsqueeze(2), 0.0)
    aug_unary = unary + hamming_aug  # [B, T, K]

    # --- Accumulate phi messages into each node (per batch element) ---
    # For edge e = (src_e, dst_e):
    #   subtract phi[b, 0, e, :] from node src_e of example b
    #   subtract phi[b, 1, e, :] from node dst_e of example b
    # The non-in-place scatter_add lets gradients flow back to phi.
    # idx_src / idx_dst expand the shared edge endpoints into a
    # per-batch index tensor of shape [B, E, K] for scatter_add along
    # dim=1 (the T axis of node_phi).
    idx_src = src.unsqueeze(1).expand(E, K).unsqueeze(0).expand(B, E, K)
    idx_dst = dst.unsqueeze(1).expand(E, K).unsqueeze(0).expand(B, E, K)
    node_phi = torch.zeros(B, T, K, device=unary.device, dtype=unary.dtype)
    node_phi = node_phi.scatter_add(1, idx_src, phi[:, 0])  # [B, T, K]
    node_phi = node_phi.scatter_add(1, idx_dst, phi[:, 1])  # [B, T, K]
    node_scores = aug_unary - node_phi  # [B, T, K]

    # --- Node terms: sum_v max_k node_scores[b, v, k], per example ---
    u_nodes = node_scores.max(dim=2).values.sum(dim=1)  # [B]

    # --- Edge terms ---
    # edge_scores[b, e, k, k'] = pairwise[k, k'] + phi[b,0,e,k] + phi[b,1,e,k']
    edge_scores = (pairwise[None, None]               # [1, 1, K, K]
                   + phi[:, 0].unsqueeze(3)           # [B, E, K, 1]
                   + phi[:, 1].unsqueeze(2))          # [B, E, 1, K]  => [B, E, K, K]
    u_edges = edge_scores.reshape(B, E, K * K).max(dim=2).values.sum(dim=1)  # [B]

    # --- Score at ground truth, per example ---
    # unary_score[b] = sum_t unary[b, t, y_true[b, t]]
    unary_score = unary.gather(2, y_true.unsqueeze(2)).squeeze(2).sum(dim=1)  # [B]
    # pw_score[b]    = sum_e pairwise[y_true[b, src[e]], y_true[b, dst[e]]]
    y_src = y_true[:, src]  # [B, E]
    y_dst = y_true[:, dst]  # [B, E]
    pw_score = pairwise[y_src, y_dst].sum(dim=1)  # [B]
    score_true = unary_score + pw_score  # [B]

    return u_nodes + u_edges - score_true  # [B]
