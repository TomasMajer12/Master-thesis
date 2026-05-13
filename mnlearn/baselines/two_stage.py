"""
Two-stage OCR + solver evaluation pipeline.

Given a trained OCR digit classifier and a visual Sudoku test split,
:func:`evaluate_two_stage` runs the full pipeline (OCR -> hard-coded
solver -> compare to ground truth) and reports:

  * ``zero_one``           — fraction of puzzles whose final grid does
                             NOT match the solution (solver failures
                             count as wrong).
  * ``solver_feasibility`` — fraction of puzzles where the solver
                             returned some grid.
  * ``ocr_clue_error``     — fraction of clue cells where OCR's argmax
                             disagreed with the original clue digit
                             (a diagnostic isolating OCR-component
                             quality from end-to-end pipeline error).

The function is split into two halves:

  * :func:`evaluate_two_stage_from_predictions` — pure: takes
    pre-computed OCR predictions and runs solver + scoring.
  * :func:`evaluate_two_stage` — convenience wrapper that runs the OCR
    model over ``X_visual`` (in chunks) before calling the pure half.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .sudoku_solver import solve


# ---------------------------------------------------------------------------
# Pure scoring half
# ---------------------------------------------------------------------------

def evaluate_two_stage_from_predictions(
    ocr_preds: torch.Tensor,
    quizzes:   torch.Tensor,
    solutions: torch.Tensor,
) -> dict:
    """Score the two-stage pipeline given pre-computed OCR predictions.

    Args:
        ocr_preds: ``[N, 81]`` LongTensor of 0-indexed digit predictions
                   (values in ``{0..8}``). The OCR may produce predictions
                   for blank cells too, but those are ignored — the
                   original ``quizzes`` mask decides which cells become
                   clues for the solver.
        quizzes:   ``[N, 81]`` LongTensor; 1-indexed (``0`` = blank,
                   ``1..9`` = clue).
        solutions: ``[N, 81]`` LongTensor; 1-indexed full ground-truth
                   solutions (values in ``{1..9}``).

    Returns:
        ``{'zero_one': float, 'solver_feasibility': float,
           'ocr_clue_error': float}`` — all in ``[0, 1]``.
    """
    if ocr_preds.shape != quizzes.shape or quizzes.shape != solutions.shape:
        raise ValueError(
            f"shape mismatch: ocr_preds={tuple(ocr_preds.shape)}, "
            f"quizzes={tuple(quizzes.shape)}, solutions={tuple(solutions.shape)}"
        )
    if quizzes.ndim != 2 or quizzes.shape[1] != 81:
        raise ValueError(
            f"expected [N, 81] tensors, got shape {tuple(quizzes.shape)}"
        )

    N = quizzes.shape[0]
    ocr_preds_cpu = ocr_preds.detach().cpu().long()
    quizzes_cpu   = quizzes.detach().cpu().long()
    solutions_cpu = solutions.detach().cpu().long()

    # --- OCR-component diagnostic: per-clue-cell classification error ---
    is_clue = quizzes_cpu != 0
    if is_clue.any():
        true_clue_0idx = quizzes_cpu[is_clue] - 1
        pred_clue_0idx = ocr_preds_cpu[is_clue]
        ocr_clue_error = (pred_clue_0idx != true_clue_0idx).float().mean().item()
    else:
        ocr_clue_error = 0.0

    # --- Build the OCR-corrupted quiz (1-indexed for the solver) ---
    # Where the original quiz had a clue, replace the digit with OCR's prediction.
    # Where the original quiz was blank, keep blank.
    predicted_quiz = torch.zeros_like(quizzes_cpu)
    predicted_quiz[is_clue] = ocr_preds_cpu[is_clue] + 1   # 0-idx -> 1-idx

    # --- Run solver per puzzle, score against ground truth ---
    quiz_np = predicted_quiz.numpy()
    sol_np  = solutions_cpu.numpy()

    feasible_count = 0
    correct_count  = 0
    for i in range(N):
        result = solve(quiz_np[i].reshape(9, 9))
        if result is None:
            continue  # zero_one = 1 for this puzzle; no fallback grid
        feasible_count += 1
        if np.array_equal(result.flatten(), sol_np[i]):
            correct_count += 1

    return {
        "zero_one":           1.0 - correct_count / max(N, 1),
        "solver_feasibility":       feasible_count / max(N, 1),
        "ocr_clue_error":     ocr_clue_error,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper that runs the OCR model
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_two_stage(
    ocr_model: nn.Module,
    X_visual:  torch.Tensor,
    quizzes:   torch.Tensor,
    solutions: torch.Tensor,
    chunk_size: int = 64,
    device:    torch.device | None = None,
) -> dict:
    """Run OCR over ``X_visual`` then score the two-stage pipeline.

    Args:
        ocr_model:  ``nn.Module`` mapping ``[batch, 1, 28, 28] -> [batch, K]``
                    raw logits (the contract from
                    :func:`mnlearn.baselines.build_ocr_model`).
        X_visual:   ``[N, 81, 1, 28, 28]`` per-cell rendered images.
        quizzes:    ``[N, 81]`` 1-indexed quiz (``0`` = blank).
        solutions:  ``[N, 81]`` 1-indexed solutions (``1..9``).
        chunk_size: number of *puzzles* per forward pass. Each puzzle has
                    81 cells, so a chunk_size of 64 forwards 64*81 = 5184
                    images at a time.
        device:     where to run the OCR forward. Defaults to whatever
                    device the model is on.

    Returns: see :func:`evaluate_two_stage_from_predictions`.
    """
    if device is None:
        device = next(ocr_model.parameters()).device
    ocr_model.eval()

    N = X_visual.shape[0]
    if X_visual.dim() != 5 or X_visual.shape[1] != 81:
        raise ValueError(
            f"X_visual must have shape [N, 81, 1, 28, 28], got "
            f"{tuple(X_visual.shape)}"
        )

    # Forward all cells in chunks of `chunk_size` puzzles. The model has the
    # per-image contract, so we flatten the (puzzle, cell) axes for the
    # forward pass and reshape the predictions back to [N, 81].
    image_shape = X_visual.shape[2:]   # (1, 28, 28)
    flat_X = X_visual.reshape(N * 81, *image_shape)

    pred_chunks = []
    cells_per_chunk = chunk_size * 81
    for i in range(0, N * 81, cells_per_chunk):
        chunk = flat_X[i:i + cells_per_chunk].to(device, non_blocking=True)
        logits = ocr_model(chunk)
        pred_chunks.append(logits.argmax(dim=-1).cpu())
    ocr_preds = torch.cat(pred_chunks, dim=0).reshape(N, 81)

    return evaluate_two_stage_from_predictions(ocr_preds, quizzes, solutions)
