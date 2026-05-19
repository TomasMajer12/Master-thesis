"""
Tests for the OCR baseline helpers (mnlearn.baselines.ocr).

Coverage:
    - extract_clue_pairs: shape, label encoding (0-indexed), correct masking,
      shape-mismatch errors.
    - build_ocr_model: produces a per-image (not per-cell-of-puzzle) model
      from a BaselineArchitectureCfg loaded from ocr_lenet5.yaml.
    - The OCR model's output is raw logits ([batch, num_classes]) — not
      softmax-normalised — so CrossEntropyLoss can apply log-softmax itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mnlearn.baselines import build_ocr_model, extract_clue_pairs
from mnlearn.config import load_baseline_config


# ---------------------------------------------------------------------------
# extract_clue_pairs
# ---------------------------------------------------------------------------

def test_extract_clue_pairs_shapes_and_labels():
    """Two synthetic puzzles with known clue cells -> correct (X, Y)."""
    # 2 puzzles, 81 cells each, single-channel 28x28 images.
    X_visual = torch.zeros(2, 81, 1, 28, 28)
    quizzes = torch.zeros(2, 81, dtype=torch.long)

    # Puzzle 0: clue digit 5 at cell 0; clue digit 9 at cell 80.
    quizzes[0, 0]  = 5
    quizzes[0, 80] = 9
    X_visual[0, 0,  0, 0, 0] = 5.0   # tag the image so we can identify it
    X_visual[0, 80, 0, 0, 0] = 9.0

    # Puzzle 1: clue digit 1 at cell 40.
    quizzes[1, 40] = 1
    X_visual[1, 40, 0, 0, 0] = 1.0

    X_clue, Y_clue = extract_clue_pairs(X_visual, quizzes)

    assert X_clue.shape == (3, 1, 28, 28)
    assert Y_clue.shape == (3,)
    assert Y_clue.dtype == torch.long
    # Labels are 0-indexed: 5 -> 4, 9 -> 8, 1 -> 0.
    assert sorted(Y_clue.tolist()) == [0, 4, 8]
    # Tagged pixels survive the mask (mapping (label -> tag value)).
    tag_by_label = {int(Y_clue[i].item()): float(X_clue[i, 0, 0, 0].item())
                    for i in range(3)}
    assert tag_by_label == {0: 1.0, 4: 5.0, 8: 9.0}


def test_extract_clue_pairs_drops_blanks():
    X_visual = torch.zeros(1, 81, 1, 28, 28)
    quizzes = torch.zeros(1, 81, dtype=torch.long)
    # No clues -> no clue pairs.
    X_clue, Y_clue = extract_clue_pairs(X_visual, quizzes)
    assert X_clue.shape == (0, 1, 28, 28)
    assert Y_clue.shape == (0,)


def test_extract_clue_pairs_rejects_mismatched_N():
    X_visual = torch.zeros(2, 81, 1, 28, 28)
    quizzes = torch.zeros(3, 81, dtype=torch.long)
    with pytest.raises(ValueError):
        extract_clue_pairs(X_visual, quizzes)


def test_extract_clue_pairs_rejects_mismatched_per_puzzle_dim():
    X_visual = torch.zeros(2, 81, 1, 28, 28)
    quizzes = torch.zeros(2, 80, dtype=torch.long)
    with pytest.raises(ValueError):
        extract_clue_pairs(X_visual, quizzes)


# ---------------------------------------------------------------------------
# build_ocr_model
# ---------------------------------------------------------------------------

_OCR_LENET5_YAML = (
    Path(__file__).resolve().parents[1]
    / "configs" / "architectures" / "ocr_lenet5.yaml"
)


def _make_minimal_ocr_experiment_yaml(tmp_path: Path) -> Path:
    """Write a baseline experiment YAML that references ocr_lenet5.yaml."""
    import yaml
    exp = {
        "experiment": {"name": "ocr_smoke", "seed": 0},
        "architecture": str(_OCR_LENET5_YAML.resolve()),
        "data": {
            "task": "sudoku",
            "mode": "visual",
            "paths": {"benchmark": "benchmarks/sudoku", "mnist": "mnist_data"},
            "train_size": 100,
            "val_size":   100,
            "test_size":  100,
            "batch_size": 32,
        },
        "training": {
            "loss": "cross_entropy",
            "optimizer": {"type": "adam", "lr": 0.001},
            "num_epochs": 1,
        },
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(exp, sort_keys=False))
    return p


def test_build_ocr_model_consumes_per_image_input(tmp_path):
    """OCR model has the per-image contract; no per-cell reshape wrapper."""
    cfg = load_baseline_config(_make_minimal_ocr_experiment_yaml(tmp_path))
    model = build_ocr_model(cfg.architecture)

    x = torch.zeros(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 9)


def test_build_ocr_model_outputs_raw_logits(tmp_path):
    """No terminal softmax in the OCR YAML -> outputs are NOT a probability simplex.

    With a non-softmax tail and randomly-initialised weights, summing over the
    class dimension will (almost surely) not equal 1.
    """
    cfg = load_baseline_config(_make_minimal_ocr_experiment_yaml(tmp_path))
    model = build_ocr_model(cfg.architecture)
    model.eval()
    with torch.no_grad():
        # Use a non-zero input so the network produces a meaningful output
        # (an all-zero input through an untrained CNN can give surprisingly
        # uniform pre-activations).
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
    sums = out.sum(dim=-1)
    # Probabilities would sum to 1; raw logits will not (gap from 1 well above 1e-3).
    assert (sums - 1.0).abs().max().item() > 1e-3


if __name__ == "__main__":
    from _fixtures import fixtures
    test_extract_clue_pairs_shapes_and_labels();                 print("PASS: extract clue pairs shapes and labels")
    test_extract_clue_pairs_drops_blanks();                      print("PASS: extract clue pairs drops blanks")
    test_extract_clue_pairs_rejects_mismatched_N();              print("PASS: extract clue pairs rejects mismatched N")
    test_extract_clue_pairs_rejects_mismatched_per_puzzle_dim(); print("PASS: extract clue pairs rejects mismatched per puzzle dim")
    with fixtures() as tp: test_build_ocr_model_consumes_per_image_input(tp);             print("PASS: build ocr model consumes per image input")
    with fixtures() as tp: test_build_ocr_model_outputs_raw_logits(tp);                   print("PASS: build ocr model outputs raw logits")
    print("\nAll tests passed.")
