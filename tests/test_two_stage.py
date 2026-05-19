"""
Tests for the two-stage OCR + solver evaluation pipeline.

Covered:
    - evaluate_two_stage_from_predictions:
        * perfect-OCR oracle -> zero_one == 0, feasibility == 1, ocr_clue_error == 0
        * wrong-by-one OCR puzzle -> the corrupted puzzle reports zero_one = 1
        * always-class-zero OCR -> low solver_feasibility (most puzzles infeasible)
        * shape-mismatch and shape-shape errors
    - evaluate_two_stage (full wrapper):
        * a model that returns logits hard-wired to the perfect prediction
          reproduces the pure-half result on a small benchmark sample.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mnlearn.baselines import (
    evaluate_two_stage,
    evaluate_two_stage_from_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures: small benchmark sample
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "sudoku"


@pytest.fixture(scope="module")
def benchmark_sample():
    """Load 8 (quiz, solution) pairs from benchmarks/sudoku/test."""
    if not (_BENCH_DIR / "test" / "quizzes.pt").is_file():
        pytest.skip("benchmarks/sudoku not built; run build_sudoku_benchmark first")
    quizzes   = torch.load(_BENCH_DIR / "test" / "quizzes.pt",   weights_only=True)
    solutions = torch.load(_BENCH_DIR / "test" / "solutions.pt", weights_only=True)
    n = 8
    return quizzes[:n].long(), solutions[:n].long()


# ---------------------------------------------------------------------------
# evaluate_two_stage_from_predictions: pure scoring half
# ---------------------------------------------------------------------------

def test_perfect_ocr_yields_zero_error(benchmark_sample):
    """OCR predicts each clue's true digit -> solver must reproduce ground truth."""
    quizzes, solutions = benchmark_sample
    # 0-indexed predictions matching the original quiz at clue cells.
    # At blank cells the value is irrelevant (mask drops them); use any digit.
    perfect = (quizzes - 1).clamp(min=0)

    metrics = evaluate_two_stage_from_predictions(perfect, quizzes, solutions)

    assert metrics["zero_one"]            == 0.0
    assert metrics["solver_feasibility"]  == 1.0
    assert metrics["ocr_clue_error"]      == 0.0


def test_one_corrupted_clue_breaks_one_puzzle(benchmark_sample):
    """Corrupt one clue in puzzle 0; that puzzle's 0/1 must flip to wrong.

    The remaining 7 puzzles use perfect OCR and stay correct, so 0/1 = 1/8.
    Solver feasibility may drop or stay at 1.0 depending on whether the
    corruption produces an infeasible quiz; either way 0/1 is the right
    metric to assert.
    """
    quizzes, solutions = benchmark_sample
    preds = (quizzes - 1).clamp(min=0).clone()

    # Find the first clue cell of puzzle 0 and bump it to a different digit.
    clue_cells = (quizzes[0] != 0).nonzero(as_tuple=True)[0]
    cell = int(clue_cells[0].item())
    true = int(preds[0, cell].item())
    preds[0, cell] = (true + 1) % 9

    metrics = evaluate_two_stage_from_predictions(preds, quizzes, solutions)

    n = quizzes.shape[0]
    assert metrics["zero_one"] >= 1.0 / n - 1e-9
    assert metrics["zero_one"] <= 1.0 / n + 1e-9
    # Exactly one cell out of all clue cells was corrupted.
    total_clues = int((quizzes != 0).sum().item())
    assert metrics["ocr_clue_error"] == pytest.approx(1.0 / total_clues)


def test_always_zero_ocr_produces_high_failure_rate(benchmark_sample):
    """OCR predicts class 0 (digit 1) everywhere -> most puzzles infeasible.

    Every clue is rewritten to '1', which (almost surely) collides with
    other clues in the same row/column/box. solver_feasibility should be
    well below 1; zero_one should be ~1.
    """
    quizzes, solutions = benchmark_sample
    preds = torch.zeros_like(quizzes)   # 0-indexed -> "1" at every cell

    metrics = evaluate_two_stage_from_predictions(preds, quizzes, solutions)

    assert metrics["zero_one"] >= 0.5
    assert metrics["solver_feasibility"] < 0.5


def test_rejects_shape_mismatch():
    quizzes   = torch.zeros(2, 81, dtype=torch.long)
    solutions = torch.zeros(2, 81, dtype=torch.long)
    bad_preds = torch.zeros(3, 81, dtype=torch.long)
    with pytest.raises(ValueError):
        evaluate_two_stage_from_predictions(bad_preds, quizzes, solutions)


def test_rejects_wrong_per_puzzle_dim():
    quizzes   = torch.zeros(2, 80, dtype=torch.long)
    solutions = torch.zeros(2, 80, dtype=torch.long)
    preds     = torch.zeros(2, 80, dtype=torch.long)
    with pytest.raises(ValueError):
        evaluate_two_stage_from_predictions(preds, quizzes, solutions)


# ---------------------------------------------------------------------------
# evaluate_two_stage: full wrapper with a model
# ---------------------------------------------------------------------------

class _IdentityIndexedOcrModel(nn.Module):
    """Returns one-hot logits at a class encoded by a unique input pixel.

    Test trick: the test crafts ``X_visual`` so that ``X_visual[n, c, 0, 0, 0]``
    encodes the desired prediction class for cell ``c`` of puzzle ``n``.
    The model's forward reads pixel ``[0, 0, 0]`` of each input and emits
    a one-hot logit at that integer class. This lets us drive the OCR
    pathway through the full ``evaluate_two_stage`` wrapper without
    actually training anything.
    """

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
        # A trainable parameter keeps next(model.parameters()) functional
        # (evaluate_two_stage uses it to detect the device).
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28]. Read the encoded class index from a fixed pixel.
        cls = x[:, 0, 0, 0].long().clamp(0, self.num_classes - 1)
        logits = torch.full(
            (x.shape[0], self.num_classes), -10.0, device=x.device,
        )
        logits[torch.arange(x.shape[0], device=x.device), cls] = 10.0
        return logits


def test_full_wrapper_with_perfect_oracle(benchmark_sample):
    """Encode perfect predictions in a fixed pixel; full wrapper agrees with pure half."""
    quizzes, solutions = benchmark_sample
    n = quizzes.shape[0]

    # Build X_visual whose [n, c, 0, 0, 0] pixel encodes the desired prediction.
    perfect = (quizzes - 1).clamp(min=0)
    X_visual = torch.zeros(n, 81, 1, 28, 28)
    X_visual[..., 0, 0, 0] = perfect.float()

    model = _IdentityIndexedOcrModel(num_classes=9)
    metrics = evaluate_two_stage(model, X_visual, quizzes, solutions, chunk_size=4)

    assert metrics["zero_one"]            == 0.0
    assert metrics["solver_feasibility"]  == 1.0
    assert metrics["ocr_clue_error"]      == 0.0


def test_full_wrapper_rejects_wrong_X_shape():
    model = _IdentityIndexedOcrModel()
    bad_X = torch.zeros(2, 80, 1, 28, 28)
    quizzes   = torch.zeros(2, 81, dtype=torch.long)
    solutions = torch.zeros(2, 81, dtype=torch.long)
    with pytest.raises(ValueError):
        evaluate_two_stage(model, bad_X, quizzes, solutions)


if __name__ == "__main__":
    # Load the benchmark sample once and pass it to the four fixture-using
    # tests (mirrors pytest's module-scoped `benchmark_sample` fixture).
    if not (_BENCH_DIR / "test" / "quizzes.pt").is_file():
        raise SystemExit(
            "benchmarks/sudoku not built; run build_sudoku_benchmark first"
        )
    _quizzes   = torch.load(_BENCH_DIR / "test" / "quizzes.pt",   weights_only=True)
    _solutions = torch.load(_BENCH_DIR / "test" / "solutions.pt", weights_only=True)
    _sample = (_quizzes[:8].long(), _solutions[:8].long())

    test_perfect_ocr_yields_zero_error(_sample);              print("PASS: perfect ocr yields zero error")
    test_one_corrupted_clue_breaks_one_puzzle(_sample);       print("PASS: one corrupted clue breaks one puzzle")
    test_always_zero_ocr_produces_high_failure_rate(_sample); print("PASS: always zero ocr produces high failure rate")
    test_rejects_shape_mismatch();                            print("PASS: rejects shape mismatch")
    test_rejects_wrong_per_puzzle_dim();                      print("PASS: rejects wrong per puzzle dim")
    test_full_wrapper_with_perfect_oracle(_sample);           print("PASS: full wrapper with perfect oracle")
    test_full_wrapper_rejects_wrong_X_shape();                print("PASS: full wrapper rejects wrong X shape")
    print("\nAll tests passed.")
