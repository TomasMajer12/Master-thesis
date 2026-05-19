"""
Tests for the hard-coded Sudoku solver (mnlearn.baselines.sudoku_solver).

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mnlearn.baselines import solve, solve_batch


# ---------------------------------------------------------------------------
# Fixtures: hand-baked puzzles
# ---------------------------------------------------------------------------

# The canonical Wikipedia "Sudoku" example. Unique solution.
WIKI_QUIZ = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
], dtype=np.int64)

WIKI_SOLUTION = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
], dtype=np.int64)


def _is_valid_solution(grid: np.ndarray) -> bool:
    """Validate that ``grid`` is a complete, well-formed Sudoku solution."""
    if grid.shape != (9, 9):
        return False
    if not ((grid >= 1) & (grid <= 9)).all():
        return False
    expected = set(range(1, 10))
    for r in range(9):
        if set(grid[r].tolist()) != expected:
            return False
    for c in range(9):
        if set(grid[:, c].tolist()) != expected:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(grid[br:br + 3, bc:bc + 3].flatten().tolist()) != expected:
                return False
    return True


# ---------------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------------

def test_solve_wiki_example_matches_known_solution():
    """The Wikipedia puzzle has a unique solution; solver must recover it."""
    out = solve(WIKI_QUIZ)
    assert out is not None
    assert out.shape == (9, 9)
    np.testing.assert_array_equal(out, WIKI_SOLUTION)


def test_solve_accepts_flat_and_grid_shapes_equivalently():
    """[81] and [9, 9] are accepted; both yield equivalent solutions."""
    out_grid = solve(WIKI_QUIZ)
    out_flat = solve(WIKI_QUIZ.reshape(81))
    assert out_grid is not None and out_flat is not None
    assert out_grid.shape == (9, 9)
    assert out_flat.shape == (81,)
    np.testing.assert_array_equal(out_flat.reshape(9, 9), out_grid)


def test_solve_empty_grid_returns_a_valid_sudoku():
    """An all-blank quiz is solvable; the solver should return some valid grid."""
    out = solve(np.zeros((9, 9), dtype=np.int64))
    assert out is not None
    assert _is_valid_solution(out)


def test_solve_returns_none_on_clue_conflict():
    """Two equal clues in the same row -> solver must report infeasibility."""
    bad_quiz = WIKI_QUIZ.copy()
    bad_quiz[0, 2] = 5  # row 0 already has a 5 at column 0
    assert solve(bad_quiz) is None


def test_solve_returns_none_on_column_conflict():
    """Two equal clues in the same column -> infeasible."""
    bad_quiz = WIKI_QUIZ.copy()
    bad_quiz[2, 0] = 5  # column 0 already has a 5 at row 0
    assert solve(bad_quiz) is None


def test_solve_returns_none_on_box_conflict():
    """Two equal clues in the same 3x3 box -> infeasible."""
    bad_quiz = WIKI_QUIZ.copy()
    bad_quiz[1, 1] = 5  # top-left box already contains 5 at (0, 0)
    assert solve(bad_quiz) is None


def test_solve_rejects_wrong_shape():
    with pytest.raises(ValueError):
        solve(np.zeros((8, 9), dtype=np.int64))
    with pytest.raises(ValueError):
        solve(np.zeros(80, dtype=np.int64))


def test_solve_rejects_out_of_range_value():
    bad_quiz = WIKI_QUIZ.copy()
    bad_quiz[0, 2] = 10
    with pytest.raises(ValueError):
        solve(bad_quiz)


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def test_solve_batch_flat_shapes():
    """[N, 81] in -> [N, 81] out, with feasibility flags."""
    quizzes = np.stack([WIKI_QUIZ.reshape(81)] * 3, axis=0)
    sols, feas = solve_batch(quizzes)
    assert sols.shape == (3, 81)
    assert feas.shape == (3,)
    assert feas.all()
    for i in range(3):
        np.testing.assert_array_equal(
            sols[i].reshape(9, 9), WIKI_SOLUTION
        )


def test_solve_batch_grid_shapes():
    """[N, 9, 9] in -> [N, 9, 9] out, with feasibility flags."""
    quizzes = np.stack([WIKI_QUIZ] * 2, axis=0)
    sols, feas = solve_batch(quizzes)
    assert sols.shape == (2, 9, 9)
    assert feas.all()
    np.testing.assert_array_equal(sols[0], WIKI_SOLUTION)


def test_solve_batch_handles_mixed_feasibility():
    """A batch with one infeasible entry: zeros and feasible=False there."""
    bad = WIKI_QUIZ.copy()
    bad[0, 2] = 5  # row conflict
    quizzes = np.stack([WIKI_QUIZ, bad, WIKI_QUIZ], axis=0)
    sols, feas = solve_batch(quizzes)
    assert feas.tolist() == [True, False, True]
    np.testing.assert_array_equal(sols[1], np.zeros((9, 9), dtype=np.int64))
    np.testing.assert_array_equal(sols[0], WIKI_SOLUTION)


def test_solve_batch_rejects_wrong_shape():
    with pytest.raises(ValueError):
        solve_batch(np.zeros((3, 80), dtype=np.int64))
    with pytest.raises(ValueError):
        solve_batch(np.zeros((3, 9, 8), dtype=np.int64))


# ---------------------------------------------------------------------------
# Integration test: round-trip on real Kaggle puzzles
# ---------------------------------------------------------------------------
# Skipped when benchmarks/sudoku/ has not been built. When available, this
# is the strongest correctness anchor: 50 puzzles drawn from the same data
# the OCR baseline will be evaluated on.

_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "sudoku"


@pytest.mark.skipif(
    not (_BENCH_DIR / "test" / "quizzes.pt").is_file(),
    reason="benchmarks/sudoku not built; run build_sudoku_benchmark first",
)
def test_round_trip_on_benchmark_sample():
    """Solve 50 Kaggle puzzles and check against ground-truth solutions."""
    import torch  # local import; only used in this opt-in integration test
    quizzes = torch.load(
        _BENCH_DIR / "test" / "quizzes.pt", weights_only=True
    ).numpy()
    solutions = torch.load(
        _BENCH_DIR / "test" / "solutions.pt", weights_only=True
    ).numpy()

    n = min(50, len(quizzes))
    for i in range(n):
        out = solve(quizzes[i].reshape(9, 9))
        assert out is not None, f"solver failed on benchmark puzzle {i}"
        np.testing.assert_array_equal(
            out, solutions[i].reshape(9, 9),
            err_msg=f"benchmark puzzle {i}: solver output differs from ground truth",
        )


if __name__ == "__main__":
    test_solve_wiki_example_matches_known_solution();       print("PASS: solve wiki example matches known solution")
    test_solve_accepts_flat_and_grid_shapes_equivalently(); print("PASS: solve accepts flat and grid shapes equivalently")
    test_solve_empty_grid_returns_a_valid_sudoku();         print("PASS: solve empty grid returns a valid sudoku")
    test_solve_returns_none_on_clue_conflict();             print("PASS: solve returns none on clue conflict")
    test_solve_returns_none_on_column_conflict();           print("PASS: solve returns none on column conflict")
    test_solve_returns_none_on_box_conflict();              print("PASS: solve returns none on box conflict")
    test_solve_rejects_wrong_shape();                       print("PASS: solve rejects wrong shape")
    test_solve_rejects_out_of_range_value();                print("PASS: solve rejects out of range value")
    test_solve_batch_flat_shapes();                         print("PASS: solve batch flat shapes")
    test_solve_batch_grid_shapes();                         print("PASS: solve batch grid shapes")
    test_solve_batch_handles_mixed_feasibility();           print("PASS: solve batch handles mixed feasibility")
    test_solve_batch_rejects_wrong_shape();                 print("PASS: solve batch rejects wrong shape")
    test_round_trip_on_benchmark_sample();                  print("PASS: round trip on benchmark sample")
    print("\nAll tests passed.")
