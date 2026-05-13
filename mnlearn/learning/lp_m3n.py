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
    """LP-M3N loss for a single training example (fully supervised).

    Args:
        unary:    [T, K]    unary potentials (requires grad)
        pairwise: [K, K]    shared pairwise weights (requires grad)
        y_true:   [T]       ground-truth integer labels (LongTensor)
        edges:    [E, 2]    edge list (LongTensor)
        phi:      [2, E, K] per-example auxiliary message variables
                            phi[0, e, :] = messages for source node of edge e
                            phi[1, e, :] = messages for dest   node of edge e

    Returns:
        loss: scalar tensor (differentiable w.r.t. unary, pairwise, and phi)
    """
    T, K = unary.shape
    E = edges.shape[0]
    src = edges[:, 0]  # [E]
    dst = edges[:, 1]  # [E]

    # --- Hamming loss augmentation ---
    # Δ_v(k) = 1/T if k != y_true[v] else 0. The /T normalisation makes the total Hamming task
    # loss live in [0, 1]
    hamming_aug = torch.ones(T, K, device=unary.device, dtype=unary.dtype) / T
    hamming_aug.scatter_(1, y_true.unsqueeze(1), 0.0)
    aug_unary = unary + hamming_aug  # [T, K]

    # --- Accumulate phi messages into each node ---
    # For edge e = (src_e, dst_e):
    #   subtract phi[0, e, :] from node src_e
    #   subtract phi[1, e, :] from node dst_e
    # We use the non-in-place scatter_add so gradients flow to phi.
    idx_src = src.unsqueeze(1).expand(E, K)  # [E, K]
    idx_dst = dst.unsqueeze(1).expand(E, K)  # [E, K]
    node_phi = torch.zeros(T, K, device=unary.device, dtype=unary.dtype)
    node_phi = node_phi.scatter_add(0, idx_src, phi[0])
    node_phi = node_phi.scatter_add(0, idx_dst, phi[1])

    node_scores = aug_unary - node_phi  # [T, K]

    # --- Node terms: sum_v max_k node_scores[v, k] ---
    u_nodes = node_scores.max(dim=1).values.sum()

    # --- Edge terms: sum_e max_{k,k'} [pairwise[k,k'] + phi[0,e,k] + phi[1,e,k']] ---
    # edge_scores[e, k, k'] = pairwise[k, k'] + phi[0, e, k] + phi[1, e, k']
    edge_scores = (pairwise.unsqueeze(0)   # [1, K, K]
                   + phi[0].unsqueeze(2)   # [E, K, 1]
                   + phi[1].unsqueeze(1))  # [E, 1, K]  =>  [E, K, K]
    u_edges = edge_scores.reshape(E, K * K).max(dim=1).values.sum()

    # --- Score at ground truth: sum_v unary[v, y_v] + sum_e pairwise[y_src, y_dst] ---
    t_idx = torch.arange(T, device=unary.device)
    unary_score = unary[t_idx, y_true].sum()
    pw_score = pairwise[y_true[src], y_true[dst]].sum()
    score_true = unary_score + pw_score

    return u_nodes + u_edges - score_true