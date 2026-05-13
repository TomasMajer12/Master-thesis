"""
Hard-coded Sudoku solver: constraint propagation + backtracking.

Used by the two-stage OCR baseline (mnlearn.baselines.two_stage):

    OCR predicts the digit of each clue cell ->
    solver completes the grid given those predicted clues ->
    final 0/1 error vs. ground-truth solution.

Algorithm
---------
Standard depth-first search augmented by constraint propagation and the
Minimum Remaining Values (MRV) heuristic; see Russell and Norvig (2020,
Ch. 6) for the technique class and Norvig (2006) for the Sudoku-specific
formulation this implementation follows.

1. Maintain a [9, 9, 9] boolean ``candidates`` array.
   ``candidates[r, c, d]`` = "digit (d+1) is still possible at (r, c)".
2. Apply each clue: assign it, eliminate that digit from the cell's row,
   column, and 3x3 box. Fail fast if two clues already conflict.
3. Naked-singles propagation: any cell with exactly one remaining candidate
   gets assigned. Iterate until a full sweep produces no new singletons.
4. If the grid is full, return it. Otherwise pick the empty cell with the
   FEWEST remaining candidates (MRV), try each candidate in ascending order,
   and recurse on a deep copy of the candidate state.
5. If every branch contradicts, return None.

References
----------
- Russell, S. and Norvig, P. (2020). Artificial Intelligence: A Modern
  Approach, 4th ed., Chapter 6 (Constraint Satisfaction Problems).
- Norvig, P. (2006). Solving Every Sudoku Puzzle.
  https://norvig.com/sudoku.html
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(quiz: np.ndarray) -> np.ndarray | None:
    """Solve a Sudoku puzzle.

    Args:
        quiz: shape [9, 9] or [81], integer dtype. Values in {0..9};
              0 = blank, 1..9 = clue digit.

    Returns:
        Solved grid as np.ndarray with the SAME shape as input (int64),
        all entries in {1..9}. Returns None if no solution exists (e.g. the
        clues already violate row/column/box constraints — the typical
        signature of an OCR error in the two-stage pipeline).
    """
    quiz_arr = np.asarray(quiz)
    flat = quiz_arr.reshape(-1)
    if flat.size != 81:
        raise ValueError(
            f"sudoku_solver.solve expects 81 cells (shape [9, 9] or [81]), "
            f"got shape {quiz_arr.shape}"
        )

    grid = flat.astype(np.int64).reshape(9, 9).copy()
    candidates = np.ones((9, 9, 9), dtype=bool)

    # --- Apply clues, fail fast on initial conflict ---
    for r in range(9):
        for c in range(9):
            d = int(grid[r, c])
            if d == 0:
                continue
            if not (1 <= d <= 9):
                raise ValueError(
                    f"sudoku_solver.solve: cell ({r}, {c}) has out-of-range "
                    f"value {d}; expected 0..9."
                )
            if not candidates[r, c, d - 1]:
                return None  # this clue contradicts an earlier clue
            _assign(grid, candidates, r, c, d)

    if not _propagate(grid, candidates):
        return None
    if not _search(grid, candidates):
        return None

    return grid.reshape(quiz_arr.shape).astype(np.int64)


def solve_batch(quizzes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve N puzzles.

    Args:
        quizzes: shape [N, 81] or [N, 9, 9], integer dtype.

    Returns:
        solutions: same shape as input (int64). Zeros for puzzles where the
                   solver failed (use the ``feasible`` flag to distinguish).
        feasible:  bool array [N]; True where a solution was found.
    """
    arr = np.asarray(quizzes)
    if arr.ndim == 2 and arr.shape[1] == 81:
        per_puzzle_shape: tuple[int, ...] = (81,)
    elif arr.ndim == 3 and arr.shape[1:] == (9, 9):
        per_puzzle_shape = (9, 9)
    else:
        raise ValueError(
            f"sudoku_solver.solve_batch expects shape [N, 81] or [N, 9, 9], "
            f"got {arr.shape}"
        )

    n = arr.shape[0]
    solutions = np.zeros(arr.shape, dtype=np.int64)
    feasible = np.zeros(n, dtype=bool)
    for i in range(n):
        sol = solve(arr[i])
        if sol is not None:
            solutions[i] = sol.reshape(per_puzzle_shape)
            feasible[i] = True
    return solutions, feasible


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign(grid: np.ndarray, candidates: np.ndarray,
            r: int, c: int, d: int) -> None:
    """Place digit d at cell (r, c), strip d from row/col/box peers.

    ``grid`` and ``candidates`` are mutated in place. Caller has already
    verified that ``candidates[r, c, d-1]`` is True.
    """
    di = d - 1
    grid[r, c] = d
    # Strip d from row, column, and 3x3 box (this also touches (r, c),
    # which we re-set immediately below so the assigned cell keeps
    # exactly one candidate).
    candidates[r, :, di] = False
    candidates[:, c, di] = False
    br, bc = (r // 3) * 3, (c // 3) * 3
    candidates[br:br + 3, bc:bc + 3, di] = False
    # Lock the cell to the assigned digit only.
    candidates[r, c, :] = False
    candidates[r, c, di] = True


def _propagate(grid: np.ndarray, candidates: np.ndarray) -> bool:
    """Naked-singles fixed-point. Returns False on contradiction.

    Loops until a full sweep produces no new assignments. A cell with zero
    remaining candidates is an immediate contradiction; a cell with exactly
    one remaining candidate gets assigned and triggers another sweep.
    """
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r, c] != 0:
                    continue
                cands = np.flatnonzero(candidates[r, c])
                if cands.size == 0:
                    return False
                if cands.size == 1:
                    _assign(grid, candidates, r, c, int(cands[0]) + 1)
                    changed = True
    return True


def _search(grid: np.ndarray, candidates: np.ndarray) -> bool:
    """Recursive backtracking with MRV. Mutates grid + candidates.

    Returns True if a complete grid was found (left in ``grid``); False if
    the current partial assignment is unsatisfiable.
    """
    # MRV: empty cell with the fewest remaining candidates.
    best_r, best_c, best_count = -1, -1, 10
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                continue
            cnt = int(candidates[r, c].sum())
            if cnt < best_count:
                best_r, best_c, best_count = r, c, cnt
                if cnt <= 1:
                    break
        if best_count <= 1:
            break

    if best_r == -1:
        return True  # no empty cells left -> solved

    if best_count == 0:
        return False  # contradiction at the most-constrained cell

    for d in (np.flatnonzero(candidates[best_r, best_c]) + 1).tolist():
        # Snapshot before trying this branch; restore on failure so the
        # caller sees an unmodified state and can try the next digit.
        grid_snap = grid.copy()
        cand_snap = candidates.copy()

        _assign(grid, candidates, best_r, best_c, int(d))
        if _propagate(grid, candidates) and _search(grid, candidates):
            return True

        grid[:] = grid_snap
        candidates[:] = cand_snap

    return False
