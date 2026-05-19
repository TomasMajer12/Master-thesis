"""Equivalence test for the batched lp_m3n_loss.
"""

import torch

from mnlearn.learning import lp_m3n_loss


# ---------------------------------------------------------------------------
# Reference: hand-coded single-example loss (the pre-vectorisation logic).
# ---------------------------------------------------------------------------

def _per_example_loss(unary_t, pairwise, y_true_t, edges, phi_t):
    """Hand-coded single-example reference. Returns a scalar tensor."""
    T, K = unary_t.shape
    E = edges.shape[0]
    src = edges[:, 0]
    dst = edges[:, 1]

    hamming_aug = torch.ones(T, K, device=unary_t.device, dtype=unary_t.dtype) / T
    hamming_aug.scatter_(1, y_true_t.unsqueeze(1), 0.0)
    aug_unary = unary_t + hamming_aug

    idx_src = src.unsqueeze(1).expand(E, K)
    idx_dst = dst.unsqueeze(1).expand(E, K)
    node_phi = torch.zeros(T, K, device=unary_t.device, dtype=unary_t.dtype)
    node_phi = node_phi.scatter_add(0, idx_src, phi_t[0])
    node_phi = node_phi.scatter_add(0, idx_dst, phi_t[1])
    node_scores = aug_unary - node_phi

    u_nodes = node_scores.max(dim=1).values.sum()
    edge_scores = (pairwise.unsqueeze(0)
                   + phi_t[0].unsqueeze(2)
                   + phi_t[1].unsqueeze(1))
    u_edges = edge_scores.reshape(E, K * K).max(dim=1).values.sum()

    t_idx = torch.arange(T, device=unary_t.device)
    unary_score = unary_t[t_idx, y_true_t].sum()
    pw_score = pairwise[y_true_t[src], y_true_t[dst]].sum()
    score_true = unary_score + pw_score

    return u_nodes + u_edges - score_true


# ---------------------------------------------------------------------------
# Toy problem builder.
# ---------------------------------------------------------------------------

def _toy_problem(B: int = 3, T: int = 5, K: int = 3, E: int = 4,
                 seed: int = 0):
    """Build a small (unary, pairwise, y_true, edges, phi) tuple.

    Uses a simple chain edge structure ``[(0,1), (1,2), ...]`` so that
    every edge endpoint is a valid node index in ``[0, T)``.
    """
    assert E + 1 <= T, "edge structure needs T >= E + 1"
    g = torch.Generator().manual_seed(seed)
    unary    = torch.randn(B, T, K, generator=g, requires_grad=True)
    pairwise = torch.randn(K, K,    generator=g, requires_grad=True)
    y_true   = torch.randint(0, K, (B, T), generator=g)
    edges    = torch.stack(
        [torch.arange(E), torch.arange(1, E + 1)], dim=1,
    ).long()
    phi      = torch.randn(B, 2, E, K, generator=g, requires_grad=True)
    return unary, pairwise, y_true, edges, phi


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_lp_m3n_batched_matches_per_example_forward():
    """Per-example losses from the batched function must match the
    hand-coded per-example reference (atol=1e-5 for fp32)."""
    unary, pairwise, y_true, edges, phi = _toy_problem(seed=0)

    batched_losses = lp_m3n_loss(unary, pairwise, y_true, edges, phi)  # [B]

    ref_losses = torch.stack([
        _per_example_loss(unary[b], pairwise, y_true[b], edges, phi[b])
        for b in range(unary.shape[0])
    ])

    assert batched_losses.shape == ref_losses.shape, (
        f"shape mismatch: batched {tuple(batched_losses.shape)} "
        f"vs ref {tuple(ref_losses.shape)}"
    )
    assert torch.allclose(batched_losses, ref_losses, atol=1e-5), (
        f"batched: {batched_losses.detach()}\nref: {ref_losses.detach()}"
    )


def test_lp_m3n_batched_matches_per_example_gradients():
    """Gradients of sum(loss) w.r.t. unary, pairwise, and phi must match
    those produced by the hand-coded per-example reference."""
    # --- batched path ---
    u_b, p_b, y_b, e_b, phi_b = _toy_problem(seed=1)
    lp_m3n_loss(u_b, p_b, y_b, e_b, phi_b).sum().backward()
    g_unary_b = u_b.grad.clone()
    g_pair_b  = p_b.grad.clone()
    g_phi_b   = phi_b.grad.clone()

    # --- reference path (rebuild fresh leaves with the same seed) ---
    u_r, p_r, y_r, e_r, phi_r = _toy_problem(seed=1)
    total = sum(
        _per_example_loss(u_r[b], p_r, y_r[b], e_r, phi_r[b])
        for b in range(u_r.shape[0])
    )
    total.backward()

    assert torch.allclose(g_unary_b, u_r.grad,   atol=1e-5)
    assert torch.allclose(g_pair_b,  p_r.grad,   atol=1e-5)
    assert torch.allclose(g_phi_b,   phi_r.grad, atol=1e-5)


def test_lp_m3n_batched_handles_singleton_batch():
    """B=1 is a corner case worth pinning down."""
    unary, pairwise, y_true, edges, phi = _toy_problem(B=1, seed=2)
    out = lp_m3n_loss(unary, pairwise, y_true, edges, phi)
    assert out.shape == (1,)
    ref = _per_example_loss(unary[0], pairwise, y_true[0], edges, phi[0])
    assert torch.allclose(out[0], ref, atol=1e-5)


if __name__ == "__main__":
    test_lp_m3n_batched_matches_per_example_forward();   print("PASS: lp m3n batched matches per example forward")
    test_lp_m3n_batched_matches_per_example_gradients(); print("PASS: lp m3n batched matches per example gradients")
    test_lp_m3n_batched_handles_singleton_batch();       print("PASS: lp m3n batched handles singleton batch")
    print("\nAll tests passed.")
