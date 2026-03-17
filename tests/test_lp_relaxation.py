"""Tests for LP relaxation (max-sum diffusion) inference."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.inference import viterbi_decode, loss_augmented_viterbi
from src.inference import lp_decode, loss_augmented_lp, max_sum_diffusion
from src.models import chain_edges


def test_lp_matches_viterbi_on_chain():
    """LP relaxation is tight on chains — must match Viterbi exactly."""
    torch.manual_seed(42)
    batch, T, K = 5, 20, 10

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)
    edges = chain_edges(T)

    y_viterbi = viterbi_decode(unary, pairwise)
    y_lp = lp_decode(unary, pairwise, edges)

    assert torch.equal(y_viterbi, y_lp), (
        f"Mismatch!\nViterbi: {y_viterbi}\nLP:      {y_lp}\n"
        f"Differ at: {(y_viterbi != y_lp).nonzero()}"
    )
    print("PASS: LP matches Viterbi on chain")


def test_lp_matches_viterbi_many_seeds():
    """Test agreement across multiple random seeds."""
    batch, T, K = 3, 15, 8
    edges = chain_edges(T)

    for seed in range(10):
        torch.manual_seed(seed * 17 + 3)
        unary = torch.randn(batch, T, K)
        pairwise = torch.randn(K, K)

        y_viterbi = viterbi_decode(unary, pairwise)
        y_lp = lp_decode(unary, pairwise, edges)

        assert torch.equal(y_viterbi, y_lp), (
            f"Mismatch at seed {seed}!\n"
            f"Differ at: {(y_viterbi != y_lp).nonzero()}"
        )
    print("PASS: LP matches Viterbi across 10 random seeds")


def test_loss_augmented_lp_matches_viterbi():
    """Loss-augmented LP must match loss-augmented Viterbi on chains."""
    torch.manual_seed(99)
    batch, T, K = 4, 15, 6
    edges = chain_edges(T)

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)
    y_true = torch.randint(0, K, (batch, T))

    y_vit = loss_augmented_viterbi(unary, pairwise, y_true)
    y_lp = loss_augmented_lp(unary, pairwise, y_true, edges=edges)

    assert torch.equal(y_vit, y_lp), (
        f"Mismatch!\nViterbi: {y_vit}\nLP:      {y_lp}\n"
        f"Differ at: {(y_vit != y_lp).nonzero()}"
    )
    print("PASS: loss-augmented LP matches loss-augmented Viterbi on chain")


def test_lp_matches_bruteforce():
    """LP relaxation must match brute-force on small inputs."""
    import itertools
    torch.manual_seed(42)
    batch, T, K = 2, 4, 3  # K^T = 81

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)
    edges = chain_edges(T)

    y_lp = lp_decode(unary, pairwise, edges)

    # Brute force
    all_labelings = list(itertools.product(range(K), repeat=T))
    y_brute = torch.zeros(batch, T, dtype=torch.long)
    for b in range(batch):
        best_score = float('-inf')
        for labeling in all_labelings:
            y = torch.tensor(labeling, dtype=torch.long)
            score = sum(unary[b, t, y[t]].item() for t in range(T))
            score += sum(pairwise[y[t], y[t + 1]].item() for t in range(T - 1))
            if score > best_score:
                best_score = score
                y_brute[b] = y

    assert torch.equal(y_lp, y_brute), (
        f"Mismatch!\nLP:          {y_lp}\nBrute force: {y_brute}"
    )
    print("PASS: LP matches brute-force")


def test_convergence_info():
    """Check that max_sum_diffusion returns convergence diagnostics."""
    torch.manual_seed(7)
    batch, T, K = 2, 10, 5
    edges = chain_edges(T)

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)

    y, info = max_sum_diffusion(unary, pairwise, edges)

    assert 'converged_at' in info
    assert 'max_change' in info
    assert info['converged_at'] <= 200, "Should converge on a chain"
    assert info['max_change'] < 1e-5, f"Should converge tightly, got {info['max_change']}"
    print(f"PASS: converged in {info['converged_at']} iterations "
          f"(max_change={info['max_change']:.2e})")


def test_lp_output_shape():
    """Basic shape and dtype check."""
    batch, T, K = 8, 30, 25
    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)
    edges = chain_edges(T)

    y = lp_decode(unary, pairwise, edges)
    assert y.shape == (batch, T)
    assert y.dtype == torch.long
    assert y.min() >= 0 and y.max() < K
    print("PASS: output shape and dtype")


if __name__ == "__main__":
    test_lp_output_shape()
    test_lp_matches_bruteforce()
    test_lp_matches_viterbi_on_chain()
    test_lp_matches_viterbi_many_seeds()
    test_loss_augmented_lp_matches_viterbi()
    test_convergence_info()
    print("\nAll LP relaxation tests passed.")