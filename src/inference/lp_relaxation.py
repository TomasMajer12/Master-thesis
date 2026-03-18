"""
LP relaxation for MAP inference using Schlesinger's ADAG algorithm.

Uses the manet library (Franc & Werner) for solving the LP relaxation of the
MAP-MRF problem via the ADAG (Adaptive Direction-Alternating primal-dual Gap
reduction) algorithm — an efficient solver for the max-sum LP relaxation on
arbitrary pairwise graphical models.

For chain-structured graphs, the LP relaxation is tight and gives the same
result as Viterbi. For general graphs (grids, loopy), ADAG solves the dual
of the local marginal polytope LP.

References:
    Werner, T. (2007). "A Linear Programming Approach to Max-Sum Problem:
    A Review." IEEE TPAMI, 29(7), 1165-1179.

    Franc, V. & Savchynskyy, B. (2017). "On the direction of ADAG."

Requires:
    manet library compiled and on PYTHONPATH.
    See libs/manet/ — compile with:
        cd libs/manet/manet/adag_solver/
        g++ -O3 -Wall -shared -std=c++11 -fPIC libadag.cpp -o libadag.so
        python3 build_adag_cffi.py
"""

import torch
import numpy as np
from manet.maxsum import adag


def _to_adag_inputs(unary_single, pairwise, edges):
    """Convert our PyTorch tensors to manet's adag numpy format.

    Our format:
        unary_single: [T, K]        — unary potentials for one sample
        pairwise:     [K, K]        — shared pairwise potentials
        edges:        [num_edges, 2] — edge list

    manet's adag format (see examples/map_inference.ipynb):
        Q: [nK, nT]                — unary (transposed)
        G: [nK, nK]                — pairwise (2D, single shared function)
        E: [3, nE]                 — each column is (node_i, node_j, pairwise_index)
    """
    Q = unary_single.detach().cpu().numpy().astype(np.float64).T  # [K, T]
    G = pairwise.detach().cpu().numpy().astype(np.float64)        # [K, K]

    nE = edges.shape[0]
    E_np = np.zeros((3, nE), dtype=int)
    E_np[0, :] = edges[:, 0].cpu().numpy()
    E_np[1, :] = edges[:, 1].cpu().numpy()
    E_np[2, :] = 0  # all edges use pairwise function index 0

    return Q, G, E_np


def adag_decode_single(unary_single, pairwise, edges):
    """Run ADAG on a single sample.

    Args:
        unary_single: [T, K]       — unary potentials
        pairwise:     [K, K]       — pairwise potentials
        edges:        [num_edges, 2]

    Returns:
        labels: [T] numpy int array
        energy: float
    """
    Q, G, E = _to_adag_inputs(unary_single, pairwise, edges)
    labels, energy = adag(Q, G, E)
    return labels, energy


def lp_decode(unary, pairwise, edges=None):
    """LP relaxation decoding via ADAG — drop-in replacement for viterbi_decode.

    If edges is None, assumes a chain graph (matching viterbi_decode interface).

    Args:
        unary:    [batch, T, K]  — unary potentials
        pairwise: [K, K]        — pairwise potentials
        edges:    [num_edges, 2] — edge list (optional, defaults to chain)

    Returns:
        y: [batch, T] (LongTensor) — decoded labeling
    """
    batch, T, K = unary.shape
    device = unary.device

    if edges is None:
        edges = _chain_edges(T, device)

    y = torch.zeros(batch, T, dtype=torch.long, device=device)
    for b in range(batch):
        labels, _ = adag_decode_single(unary[b], pairwise, edges)
        y[b] = torch.from_numpy(labels.astype(np.int64)).to(device)

    return y


def loss_augmented_lp(unary, pairwise, y_true, edges=None):
    """Loss-augmented LP relaxation for structured SVM training.

    Solves: y* = argmax_y [ F(y|x) + Delta(y, y_true) ]

    Args:
        unary:    [batch, T, K]  — unary potentials
        pairwise: [K, K]        — pairwise potentials
        y_true:   [batch, T]    — ground truth labels
        edges:    [num_edges, 2] — edge list (optional, defaults to chain)

    Returns:
        y_star: [batch, T] (LongTensor) — most-violating labeling
    """
    batch, T, n_classes = unary.shape

    if edges is None:
        edges = _chain_edges(T, unary.device)

    # Hamming loss augmentation: +1 for every class != y_true
    loss_term = torch.ones(batch, T, n_classes, device=unary.device)
    loss_term.scatter_(2, y_true.unsqueeze(-1), 0.0)

    augmented_unary = unary.detach() + loss_term
    return lp_decode(augmented_unary, pairwise.detach(), edges)


def _chain_edges(seq_len, device=None):
    """Build chain edge list (utility, same as m3n.chain_edges)."""
    src = torch.arange(seq_len - 1, device=device)
    dst = torch.arange(1, seq_len, device=device)
    return torch.stack([src, dst], dim=1)