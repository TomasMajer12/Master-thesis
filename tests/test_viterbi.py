"""Tests for Viterbi inference."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.inference import viterbi_decode, loss_augmented_viterbi


def brute_force_decode(unary, pairwise):
    """Enumerate all labelings and pick the best. Only feasible for tiny inputs."""
    batch, T, K = unary.shape
    best_y = torch.zeros(batch, T, dtype=torch.long)

    # Generate all K^T labelings
    import itertools
    all_labelings = list(itertools.product(range(K), repeat=T))

    for b in range(batch):
        best_score = float('-inf')
        for labeling in all_labelings:
            y = torch.tensor(labeling, dtype=torch.long)
            score = sum(unary[b, t, y[t]].item() for t in range(T))
            score += sum(pairwise[y[t], y[t + 1]].item() for t in range(T - 1))
            if score > best_score:
                best_score = score
                best_y[b] = y
    return best_y


def old_viterbi_decode(unary, pairwise):
    """Loop-based Viterbi from the old SimpleM3N.py code, for comparison."""
    batch, seq_len, num_classes = unary.shape
    y_pred = torch.zeros(batch, seq_len, dtype=torch.long)

    pw = pairwise.detach()

    for b in range(batch):
        dp = torch.zeros(seq_len, num_classes)
        backpointer = torch.zeros(seq_len, num_classes, dtype=torch.long)

        dp[0] = unary[b, 0].detach()

        for t in range(1, seq_len):
            for curr_class in range(num_classes):
                scores = dp[t - 1] + pw[:, curr_class] + unary[b, t, curr_class].detach()
                dp[t, curr_class] = torch.max(scores)
                backpointer[t, curr_class] = torch.argmax(scores)

        y_pred[b, -1] = torch.argmax(dp[-1])
        for t in range(seq_len - 2, -1, -1):
            y_pred[b, t] = backpointer[t + 1, y_pred[b, t + 1]]

    return y_pred


def test_viterbi_vs_bruteforce():
    """Viterbi must match brute-force on small inputs."""
    torch.manual_seed(42)
    batch, T, K = 3, 4, 3  # K^T = 81 labelings — tiny

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)

    y_viterbi = viterbi_decode(unary, pairwise)
    y_brute = brute_force_decode(unary, pairwise)

    assert torch.equal(y_viterbi, y_brute), (
        f"Mismatch!\nViterbi:     {y_viterbi}\nBrute force: {y_brute}"
    )
    print("PASS: viterbi matches brute-force")


def test_viterbi_vs_old_code():
    """New vectorized Viterbi must match the old loop-based implementation."""
    torch.manual_seed(7)
    batch, T, K = 5, 20, 10

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)

    y_new = viterbi_decode(unary, pairwise)
    y_old = old_viterbi_decode(unary, pairwise)

    assert torch.equal(y_new, y_old), (
        f"Mismatch!\nNew: {y_new}\nOld: {y_old}"
    )
    print("PASS: viterbi matches old loop-based code")


def test_loss_augmented_vs_bruteforce():
    """Loss-augmented Viterbi must find argmax of (score + Hamming loss)."""
    torch.manual_seed(99)
    batch, T, K = 2, 4, 3

    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)
    y_true = torch.randint(0, K, (batch, T))

    # Our implementation
    y_star = loss_augmented_viterbi(unary, pairwise, y_true)

    # Brute-force: enumerate and find argmax of (score + hamming)
    import itertools
    all_labelings = list(itertools.product(range(K), repeat=T))
    y_brute = torch.zeros(batch, T, dtype=torch.long)

    for b in range(batch):
        best_score = float('-inf')
        for labeling in all_labelings:
            y = torch.tensor(labeling, dtype=torch.long)
            score = sum(unary[b, t, y[t]].item() for t in range(T))
            score += sum(pairwise[y[t], y[t + 1]].item() for t in range(T - 1))
            hamming = (y != y_true[b]).float().sum().item()
            total = score + hamming
            if total > best_score:
                best_score = total
                y_brute[b] = y

    assert torch.equal(y_star, y_brute), (
        f"Mismatch!\nLoss-aug Viterbi: {y_star}\nBrute force:      {y_brute}"
    )
    print("PASS: loss-augmented viterbi matches brute-force")


def test_loss_augmented_returns_true_when_no_violation():
    """If the model perfectly scores y_true, loss-augmented should still find it
    (or something equally good). At minimum, it should never return a score
    lower than score(y_true) + 0 (since Hamming loss of y_true is 0)."""
    torch.manual_seed(0)
    K = 3
    T = 5
    batch = 2

    # Make unary strongly prefer y_true
    y_true = torch.randint(0, K, (batch, T))
    unary = torch.zeros(batch, T, K)
    unary.scatter_(2, y_true.unsqueeze(-1), 100.0)  # huge score for true class

    pairwise = torch.randn(K, K) * 0.01  # weak pairwise

    y_star = loss_augmented_viterbi(unary, pairwise, y_true)
    # With such strong unary, the augmented inference should return y_true
    assert torch.equal(y_star, y_true), (
        f"Expected y_true but got {y_star}"
    )
    print("PASS: loss-augmented returns y_true when model is confident")


def test_viterbi_output_shape():
    """Basic shape check."""
    batch, T, K = 8, 30, 25
    unary = torch.randn(batch, T, K)
    pairwise = torch.randn(K, K)

    y = viterbi_decode(unary, pairwise)
    assert y.shape == (batch, T)
    assert y.dtype == torch.long
    assert y.min() >= 0 and y.max() < K
    print("PASS: output shape and dtype")


if __name__ == "__main__":
    test_viterbi_output_shape()
    test_viterbi_vs_bruteforce()
    test_viterbi_vs_old_code()
    test_loss_augmented_vs_bruteforce()
    test_loss_augmented_returns_true_when_no_violation()
    print("\nAll tests passed.")
