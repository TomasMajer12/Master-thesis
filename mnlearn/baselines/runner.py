"""Runner for the two-stage OCR + solver baseline.

Mirrors :func:`mnlearn.learning.train`: load a YAML config, set the seed,
build data + model + trainer, fit, evaluate on test, dump artifacts. The
artifact set (``config.yaml``, ``history.json``, ``results.json``,
``model.pt``) has the same shape as the M3N runner.

Shared runtime helpers (seeding, device resolution, output-dir
management, artifact saving) live in :mod:`mnlearn.learning.runtime`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mnlearn.config import BaselineConfig, load_baseline_config
from mnlearn.data import load_sudoku_benchmark
from mnlearn.learning import ClassifierTrainer
from mnlearn.learning.builders import build_scheduler
from mnlearn.learning.runtime import (
    resolve_device,
    resolve_output_dir,
    save_artifacts,
    set_seed,
    with_output_dir,
)

from .ocr import build_ocr_model, extract_clue_pairs
from .two_stage import evaluate_two_stage


def train_baseline(
    config: BaselineConfig | str | Path,
) -> tuple[dict, dict]:
    """Run one baseline experiment end-to-end.

    Args:
        config: a loaded :class:`BaselineConfig` or a path to a baseline
                experiment YAML.

    Returns:
        ``(history, test_metrics)`` — ``history`` is the trainer's
        per-epoch log on per-cell OCR error; ``test_metrics`` is the
        dict from :func:`mnlearn.baselines.evaluate_two_stage`.
    """
    cfg = config if isinstance(config, BaselineConfig) else load_baseline_config(config)

    set_seed(cfg.experiment.seed)
    device = resolve_device(cfg.experiment.device)
    output_dir = resolve_output_dir(cfg.experiment.name, cfg.experiment.output_dir)

    if cfg.logging.verbose:
        print(f"[baseline] {cfg.experiment.name}  device={device}  output={output_dir}")

    # --- Load benchmark with quizzes + solutions exposed -----------------
    # ``mnlearn.data.builders.build_datasets`` only returns ``(X, Y)`` (Y =
    # solution - 1). The two-stage pipeline needs the original 1-indexed
    # quizzes (to identify clue cells) and 1-indexed solutions (for scoring),
    # so we go through ``load_sudoku_benchmark`` directly here.
    base = Path.cwd()
    bench_path = (base / cfg.data.paths["benchmark"]).resolve()
    mnist_root = (base / cfg.data.paths["mnist"]).resolve()
    bench = load_sudoku_benchmark(
        str(bench_path), mode="visual", mnist_root=str(mnist_root),
    )

    train_X, train_quiz, _          = _materialize_visual_split(bench["train"], cfg.data.train_size)
    val_X,   val_quiz,   _          = _materialize_visual_split(bench["val"],   cfg.data.val_size)
    test_X,  test_quiz,  test_sol   = _materialize_visual_split(bench["test"],  cfg.data.test_size)

    # --- Train OCR classifier on per-cell clue images --------------------
    train_clue_X, train_clue_Y = extract_clue_pairs(train_X, train_quiz)
    val_clue_X,   val_clue_Y   = extract_clue_pairs(val_X,   val_quiz)

    if cfg.logging.verbose:
        print(
            f"[baseline] clue cells: train={train_clue_X.shape[0]}, "
            f"val={val_clue_X.shape[0]}"
        )

    ocr_model = build_ocr_model(cfg.architecture).to(device)
    trainer = ClassifierTrainer(
        ocr_model,
        lr           = cfg.training.optimizer.lr,
        weight_decay = cfg.training.optimizer.weight_decay,
        optimizer    = cfg.training.optimizer.type,
        device       = device,
    )
    trainer.scheduler = build_scheduler(trainer.optimizer, cfg.training.scheduler)

    history = trainer.fit(
        train_clue_X, train_clue_Y,
        val_clue_X,   val_clue_Y,
        config=_baseline_fit_kwargs(
            cfg.training,
            batch_size=cfg.data.batch_size,
            verbose=cfg.logging.verbose,
        ),
    )

    # --- Test phase: two-stage evaluation on full puzzles ----------------
    test_metrics = evaluate_two_stage(
        ocr_model, test_X, test_quiz, test_sol, device=device,
    )

    # --- Persist artifacts -----------------------------------------------
    cfg_resolved = with_output_dir(cfg, str(output_dir))
    save_artifacts(cfg_resolved, output_dir, ocr_model, history, test_metrics)

    if cfg.logging.verbose:
        print(
            f"[baseline] done  zero_one={test_metrics['zero_one']:.4f}  "
            f"solver_feasibility={test_metrics['solver_feasibility']:.4f}  "
            f"ocr_clue_error={test_metrics['ocr_clue_error']:.4f}"
        )

    return history, test_metrics


# ---------------------------------------------------------------------------
# Local helpers (baseline-specific data shaping)
# ---------------------------------------------------------------------------

def _materialize_visual_split(ds, n: int) -> tuple[Any, Any, Any]:
    """Materialise the first ``n`` examples as (X, quizzes, solutions).

    ``X`` is the dense visual tensor ``[n, 81, 1, 28, 28]``. ``quizzes``
    and ``solutions`` are 1-indexed (``0`` = blank, ``1..9`` = digit) —
    they come straight from the underlying :class:`VisualSudokuDataset`
    attributes and are NOT shifted to 0-index here. Downstream code
    (``extract_clue_pairs``, ``evaluate_two_stage``) handles the
    1-indexed / 0-indexed conversions explicitly.
    """
    n = min(n, len(ds))
    X = torch.zeros(n, 81, 1, 28, 28)
    for i in range(n):
        X[i], _ = ds[i]   # __getitem__ returns (image, solution-1); ignore the y
    quizzes   = ds.quizzes[:n].clone().long()
    solutions = ds.solutions[:n].clone().long()
    return X, quizzes, solutions


def _baseline_fit_kwargs(
    train_cfg, batch_size: int, verbose: bool,
) -> dict:
    """Adapter from ``BaselineTrainingCfg`` to ``ClassifierTrainer.fit``'s dict shape."""
    return {
        "num_epochs": train_cfg.num_epochs,
        "batch_size": batch_size,
        "eval_every": train_cfg.eval_every,
        "patience":   train_cfg.early_stopping.patience,
        "min_delta":  train_cfg.early_stopping.min_delta,
        "verbose":    verbose,
    }
