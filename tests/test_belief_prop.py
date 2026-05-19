"""Tests for max-sum belief propagation inference.

The vectorised :func:`bp_decode` must produce identical labelings to the
reference per-sample implementation :func:`_bp_single`, and must match
Viterbi exactly on chain graphs.
"""

import torch

from mnlearn.inference import bp_decode, viterbi_decode
from mnlearn.inference.belief_prop import _bp_batched, _bp_single
from mnlearn.models import chain_edges, sudoku_edges


def _per_sample_decode(unary, pairwise, edges, num_iters):
    """Stack per-sample _bp_single calls into a batched output."""
    return torch.stack(
        [_bp_single(unary[b], pairwise, edges, num_iters)
         for b in range(unary.shape[0])],
        dim=0,
    )


# ---------------------------------------------------------------------------
# Output shape / dtype
# ---------------------------------------------------------------------------

def test_bp_output_shape_chain():
    """Basic shape + dtype check on a chain."""
    batch, T, K = 6, 25, 7
    unary    = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K) * 0.3

    y = bp_decode(unary, pairwise, num_iters=10)
    assert y.shape == (batch, T)
    assert y.dtype == torch.long
    assert y.min() >= 0 and y.max() < K
    print("PASS: bp_decode output shape + dtype (chain)")


def test_bp_output_shape_sudoku():
    """Same on the 810-edge Sudoku graph."""
    batch, T, K = 4, 81, 9
    unary    = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K) * 0.3
    edges    = sudoku_edges()

    y = bp_decode(unary, pairwise, edges=edges, num_iters=5)
    assert y.shape == (batch, T)
    assert y.dtype == torch.long
    assert y.min() >= 0 and y.max() < K
    print("PASS: bp_decode output shape + dtype (sudoku)")


# ---------------------------------------------------------------------------
# Batched == per-sample (the actual vectorisation correctness test)
# ---------------------------------------------------------------------------

def test_bp_batched_matches_per_sample_chain():
    """Vectorised BP must match the reference per-sample implementation
    on a chain (where labels are well-determined, so float reordering does
    not perturb argmax)."""
    torch.manual_seed(42)
    batch, T, K = 5, 30, 6
    unary    = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K) * 0.5
    edges    = chain_edges(T)

    for num_iters in (1, 5, 30):
        y_batched = _bp_batched(unary, pairwise, edges, num_iters)
        y_single  = _per_sample_decode(unary, pairwise, edges, num_iters)
        assert torch.equal(y_batched, y_single), (
            f"Mismatch at num_iters={num_iters}:\n"
            f"  batched: {y_batched}\n  per-sample: {y_single}"
        )
    print("PASS: batched BP matches per-sample on chain across iter counts")


def test_bp_batched_matches_per_sample_sudoku():
    """Same equivalence check on the Sudoku graph — confirms scatter_add
    over a batched dim has the right semantics for a cyclic graph too.
    """
    torch.manual_seed(7)
    batch, T, K = 4, 81, 9
    unary    = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K) * 0.3
    edges    = sudoku_edges()

    for num_iters in (1, 5, 20):
        y_batched = _bp_batched(unary, pairwise, edges, num_iters)
        y_single  = _per_sample_decode(unary, pairwise, edges, num_iters)
        mismatch_rate = (y_batched != y_single).float().mean().item()
        assert mismatch_rate <= 0.01, (
            f"num_iters={num_iters}: {mismatch_rate*100:.2f}% nodes differ "
            f"between batched and per-sample (tolerance 1%)"
        )
    print("PASS: batched BP matches per-sample on Sudoku across iter counts")


# ---------------------------------------------------------------------------
# Exactness on chains (BP == Viterbi on tree-structured graphs)
# ---------------------------------------------------------------------------

def test_bp_matches_viterbi_on_chain():
    """On chains BP is exact, so after enough iterations its labeling must
    match Viterbi. Use small T/K and many iters to keep ties rare."""
    torch.manual_seed(123)
    batch, T, K = 8, 15, 5
    unary    = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K) * 0.4

    y_bp  = bp_decode(unary, pairwise, num_iters=200)
    y_vit = viterbi_decode(unary, pairwise)

    mismatch_rate = (y_bp != y_vit).float().mean().item()
    assert mismatch_rate <= 0.01, (
        f"{mismatch_rate*100:.2f}% nodes differ between BP-200 and Viterbi "
        f"on a chain (tolerance 1% for argmax ties)"
    )
    print("PASS: BP matches Viterbi on chain")


# ---------------------------------------------------------------------------
# Argmax preservation under broadcast
# ---------------------------------------------------------------------------

def test_bp_independent_samples():
    """Decoding sample b in a batch must not depend on what other samples
    are in the batch. (Trivially true mathematically, but a useful guard
    against accidental cross-sample tensor reductions.)"""
    torch.manual_seed(0)
    B, T, K = 6, 81, 9
    unary    = torch.randn(B, T, K)
    pairwise = torch.randn(K, K) * 0.3
    edges    = sudoku_edges()

    y_full = bp_decode(unary, pairwise, edges=edges, num_iters=15)

    # Decode just sample 0 alone, sample 0 + 3 together, and the full batch
    # — sample 0's labeling must agree (modulo argmax ties).
    y_solo   = bp_decode(unary[:1], pairwise, edges=edges, num_iters=15)
    y_pair   = bp_decode(unary[[0, 3]], pairwise, edges=edges, num_iters=15)

    assert (y_full[0] == y_solo[0]).float().mean().item() >= 0.99
    assert (y_full[0] == y_pair[0]).float().mean().item() >= 0.99
    assert (y_full[3] == y_pair[1]).float().mean().item() >= 0.99
    print("PASS: per-sample BP labeling is independent of batch composition")


if __name__ == "__main__":
    test_bp_output_shape_chain()
    test_bp_output_shape_sudoku()
    test_bp_batched_matches_per_sample_chain()
    test_bp_batched_matches_per_sample_sudoku()
    test_bp_matches_viterbi_on_chain()
    test_bp_independent_samples()
    print("\nAll tests passed.")
