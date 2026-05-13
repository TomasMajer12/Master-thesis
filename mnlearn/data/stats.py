"""
Statistics for the Sudoku and HMC datasets.

Sudoku
-----------------------
    sudoku_blank_rate(quizzes) -> float
    sudoku_clues_per_puzzle(quizzes) -> np.ndarray [N]
    sudoku_clue_rate_per_cell(quizzes) -> np.ndarray [81]
    sudoku_digit_distribution(arr) -> np.ndarray [10]
    sudoku_difficulty_histogram(quizzes, bins=None) -> (counts, edges)

HMC
------------------
    hmc_state_distribution(labels, K) -> np.ndarray [K]
    hmc_empirical_transition(labels, K) -> np.ndarray [K, K]
    hmc_emission_accuracy(obs, labels) -> float
    hmc_empirical_p_self(labels) -> float
    hmc_run_length_distribution(labels) -> np.ndarray

Shared
------
    mnist_pool_usage(image_indices, key_arr, num_digits)
        -> dict[int, np.ndarray]
"""

from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    """Convert torch.Tensor or array-like to a numpy ndarray (no copy if avoidable)."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def sudoku_blank_rate(quizzes) -> float:
    """Fraction of cells that are blank (digit 0) across all puzzles.

    For the canonical bryanpark/sudoku dataset, expected ≈ 0.55 (≈ 36
    clues out of 81 cells per puzzle).
    """
    q = _np(quizzes)
    return float((q == 0).mean())


def sudoku_clues_per_puzzle(quizzes) -> np.ndarray:
    """Number of given clues (non-blank cells) per puzzle. Shape [N]."""
    q = _np(quizzes)
    return (q != 0).sum(axis=-1).astype(np.int64)


def sudoku_clue_rate_per_cell(quizzes) -> np.ndarray:
    """Per-position fraction of puzzles where the cell holds a clue.

    Args:
        quizzes: shape [N, 81] int array (or [N, 9, 9]; will be flattened).

    Returns:
        np.ndarray of shape [81] float64. Entry r*9+c is the fraction of
        puzzles whose cell (r, c) is non-blank.
    """
    q = _np(quizzes).reshape(-1, 81)
    return (q != 0).mean(axis=0).astype(np.float64)


def sudoku_digit_distribution(arr) -> np.ndarray:
    """Histogram over digits 0..9. Shape [10].

    Use on quizzes (digit 0 = blank) or solutions (digits 1-9 only).
    """
    a = _np(arr).ravel()
    return np.bincount(a, minlength=10)


def sudoku_difficulty_histogram(quizzes, bins=None):
    """Histogram of clues-per-puzzle.

    Returns (counts, bin_edges). Default bins span the observed range
    with width 2.
    """
    clues = sudoku_clues_per_puzzle(quizzes)
    if bins is None:
        lo, hi = int(clues.min()), int(clues.max())
        bins = np.arange(lo, hi + 2, 2)
    counts, edges = np.histogram(clues, bins=bins)
    return counts, edges


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def hmc_state_distribution(labels, K: int) -> np.ndarray:
    """Empirical marginal distribution over hidden states. Shape [K].

    Expected ≈ uniform 1/K (initial state is uniform; transition is
    symmetric).
    """
    labels = _np(labels).ravel()
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    return counts / counts.sum()


def hmc_empirical_transition(labels, K: int) -> np.ndarray:
    """Row-stochastic empirical transition matrix [K, K].

    ``M[i, j] = P̂(y_{t+1} = j | y_t = i)``. Rows over states never
    visited fall back to a zero row.
    """
    labels = _np(labels)
    flat_from = labels[:, :-1].ravel()
    flat_to   = labels[:,  1:].ravel()
    M = np.zeros((K, K), dtype=np.int64)
    np.add.at(M, (flat_from, flat_to), 1)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return M / rs


def hmc_emission_accuracy(obs, labels) -> float:
    """Fraction of (obs, label) pairs where obs == label.

    Empirical estimate of p_emit. Expected ≈ p_emit (e.g. 0.7).
    """
    return float((_np(obs) == _np(labels)).mean())


def hmc_empirical_p_self(labels) -> float:
    """Empirical self-transition rate.

    Expected ≈ p_self (e.g. 0.7).
    """
    labels = _np(labels)
    return float((labels[:, :-1] == labels[:, 1:]).mean())


def hmc_run_length_distribution(labels) -> np.ndarray:
    """Run lengths in the hidden-state sequences.

    A 'run' is a maximal stretch of consecutive equal states. With
    self-transition probability p_self, the expected run length is
    ``1 / (1 - p_self)``.

    Returns a flat 1D array — one entry per run, all sequences merged.
    """
    labels = _np(labels)
    out: list[int] = []
    for seq in labels:
        change = np.diff(seq) != 0
        boundaries = np.concatenate([[0], np.where(change)[0] + 1, [len(seq)]])
        out.extend(np.diff(boundaries).tolist())
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def mnist_pool_usage(image_indices, key_arr, num_digits: int) -> dict[int, np.ndarray]:
    """Per-digit usage histogram over the MNIST image-pool indices.

    For each digit ``d`` ∈ ``range(num_digits)``, returns an array
    ``counts`` where ``counts[i]`` is how many cells that hold digit ``d``
    used MNIST image index ``i``.

    Useful for spotting overrepresentation: ideally counts[i] is roughly
    equal across i, with no zeros (every pool image appears at least once).

    Args:
        image_indices: per-cell MNIST image index, shape matching key_arr.
        key_arr:       per-cell digit (e.g. quizzes for Sudoku, obs_indices
                       for HMC).
        num_digits:    iterate digits in ``range(num_digits)`` (use 10 for
                       both Sudoku — digit 0 will yield an empty array
                       because blanks have no MNIST image — and HMC).

    Returns:
        dict mapping digit → 1-D numpy histogram (length = max image
        index seen for that digit, plus 1; empty array if the digit
        never appears in key_arr).
    """
    img = _np(image_indices).ravel()
    key = _np(key_arr).ravel()
    out: dict[int, np.ndarray] = {}
    for d in range(num_digits):
        mask = key == d
        if not mask.any():
            out[d] = np.zeros(0, dtype=np.int64)
            continue
        chosen = img[mask]
        out[d] = np.bincount(chosen, minlength=int(chosen.max()) + 1)
    return out