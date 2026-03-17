"""
LP relaxation for MAP inference via max-sum diffusion (Werner, 2007).

Tomáš Werner's approach solves the LP relaxation of the MAP-MRF problem
by iteratively reparameterizing the potentials until the max-marginals
become consistent across all edges.

Key idea:
    The MAP problem is:  max_y  Σ_i θ_i(y_i) + Σ_{(i,j)∈E} θ_ij(y_i, y_j)

    The LP relaxation replaces the integer constraint y_i ∈ {1..K} with
    marginal polytope constraints (local consistency).

    Max-sum diffusion solves this by reparameterizing potentials:
    we can shift values between node and edge potentials without
    changing the objective, aiming for "consistent max-marginals."

    At convergence (guaranteed for trees, including chains):
        max_l [θ_ij(k,l) + θ_j(l)]  =  θ_i(k) + const   for all k

    The decoded labeling is simply: y_i = argmax_k θ_i(k)

For tree-structured graphs (chains, trees), the LP relaxation is tight:
it gives the exact MAP solution, same as Viterbi. For loopy graphs, the
solution may be fractional, but rounding argmax_k θ_i(k) is a standard
heuristic that often works well.

References:
    Werner, T. (2007). "A Linear Programming Approach to Max-Sum Problem:
    A Review." IEEE TPAMI, 29(7), 1165-1179.
"""

import torch


def max_sum_diffusion(unary, pairwise, edges, max_iter=200, tol=1e-6):
    """Max-sum diffusion for MAP inference on any pairwise MRF.

    Iteratively reparameterizes node and edge potentials until
    max-marginals are consistent. For trees (including chains),
    this converges to the exact MAP solution.

    Args:
        unary:    [batch, N, K]    — unary potentials per node
        pairwise: [K, K]          — pairwise potentials (shared across edges)
        edges:    [num_edges, 2]   — edge list (LongTensor)
        max_iter: maximum number of iterations
        tol:      convergence threshold on max absolute update

    Returns:
        y:          [batch, N] (LongTensor) — decoded labeling
        info:       dict with convergence diagnostics
    """
    batch, N, K = unary.shape
    E = edges.shape[0]

    # Reparameterized potentials (work on copies)
    theta_node = unary.clone()                     # [batch, N, K]
    # Per-edge potentials: start from shared pairwise, expanded per edge
    theta_edge = pairwise.unsqueeze(0).unsqueeze(0).expand(batch, E, K, K).clone()
    # theta_edge[b, e, k, l] = potential for edge e, labels (k, l)

    converged_at = max_iter
    for iteration in range(max_iter):
        max_change = 0.0

        for e in range(E):
            i, j = edges[e, 0].item(), edges[e, 1].item()

            # --- Update node i from edge (i,j) ---
            # Max-marginal at i through edge e:
            #   gamma_i(k) = max_l [ theta_edge(k, l) + theta_node_j(l) ]
            # theta_edge[:, e]: [batch, K_i, K_j]
            # theta_node[:, j]: [batch, K_j]
            gamma_i = (theta_edge[:, e, :, :] + theta_node[:, j, :].unsqueeze(1)).max(dim=2).values
            # gamma_i: [batch, K]

            # Reparameterization: transfer half the difference
            delta_i = (gamma_i - theta_node[:, i, :]) / 2.0   # [batch, K]
            theta_node[:, i, :] += delta_i
            theta_edge[:, e, :, :] -= delta_i.unsqueeze(2)     # subtract from all l

            # --- Update node j from edge (i,j) ---
            # gamma_j(l) = max_k [ theta_edge(k, l) + theta_node_i(k) ]
            gamma_j = (theta_edge[:, e, :, :] + theta_node[:, i, :].unsqueeze(2)).max(dim=1).values
            # gamma_j: [batch, K]

            delta_j = (gamma_j - theta_node[:, j, :]) / 2.0   # [batch, K]
            theta_node[:, j, :] += delta_j
            theta_edge[:, e, :, :] -= delta_j.unsqueeze(1)     # subtract from all k

            max_change = max(max_change,
                             delta_i.abs().max().item(),
                             delta_j.abs().max().item())

        if max_change < tol:
            converged_at = iteration + 1
            break

    # Decode: take argmax of reparameterized node potentials
    y = theta_node.argmax(dim=2)

    info = {
        'converged_at': converged_at,
        'max_change': max_change,
        'theta_node': theta_node,   # useful for inspecting marginals
    }
    return y, info


def lp_decode(unary, pairwise, edges=None, max_iter=200, tol=1e-6):
    """LP relaxation decoding — drop-in replacement for viterbi_decode.

    If edges is None, assumes a chain graph (matching viterbi_decode interface).

    Args:
        unary:    [batch, T, K]  — unary potentials
        pairwise: [K, K]        — pairwise potentials
        edges:    [num_edges, 2] — edge list (optional, defaults to chain)
        max_iter: max diffusion iterations
        tol:      convergence tolerance

    Returns:
        y: [batch, T] (LongTensor) — decoded labeling
    """
    if edges is None:
        T = unary.shape[1]
        edges = _chain_edges(T, unary.device)

    y, _ = max_sum_diffusion(unary, pairwise, edges, max_iter, tol)
    return y


def loss_augmented_lp(unary, pairwise, y_true, edges=None, max_iter=200, tol=1e-6):
    """Loss-augmented LP relaxation for structured SVM training.

    Solves: y* = argmax_y [ F(y|x) + Delta(y, y_true) ]

    Same augmented-unary trick as loss_augmented_viterbi: add +1 to the
    unary potential of every class that differs from y_true at each position.

    Args:
        unary:    [batch, T, K]  — unary potentials
        pairwise: [K, K]        — pairwise potentials
        y_true:   [batch, T]    — ground truth labels
        edges:    [num_edges, 2] — edge list (optional, defaults to chain)
        max_iter: max diffusion iterations
        tol:      convergence tolerance

    Returns:
        y_star: [batch, T] (LongTensor) — most-violating labeling
    """
    batch, T, K = unary.shape

    if edges is None:
        edges = _chain_edges(T, unary.device)

    # Hamming loss augmentation: +1 for every class != y_true
    loss_term = torch.ones(batch, T, K, device=unary.device)
    loss_term.scatter_(2, y_true.unsqueeze(-1), 0.0)

    augmented_unary = unary.detach() + loss_term
    y_star, _ = max_sum_diffusion(augmented_unary, pairwise.detach(), edges, max_iter, tol)
    return y_star


def _chain_edges(seq_len, device=None):
    """Build chain edge list (utility, same as m3n.chain_edges)."""
    src = torch.arange(seq_len - 1, device=device)
    dst = torch.arange(1, seq_len, device=device)
    return torch.stack([src, dst], dim=1)