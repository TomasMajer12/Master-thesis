"""Single entry point for YAML-driven structured-prediction experiments.

``train(cfg)`` runs one experiment end-to-end: load the config, set the
seed, build data + model + Trainer, fit, evaluate on test, dump artifacts.
Returns a result dict containing config, history, test_metrics, model,
trainer, and the resolved output_dir — notebooks can pull whatever they
need without re-running the experiment.

The artifact set on disk (``config.yaml``, ``history.json``,
``results.json``, ``model.pt``)

Sweeps live in user code: call :func:`train` repeatedly with a
``dataclasses.replace``-d ``Config`` to vary one knob at a time.

OCR baselines have their own runner in
:mod:`mnlearn.baselines.runner`;
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mnlearn.config import Config, load_config
from mnlearn.data import build_datasets
from mnlearn.models import build_model

from .runtime import (
    resolve_device,
    resolve_output_dir,
    save_artifacts,
    set_seed,
    with_output_dir,
)
from .trainer import Trainer


def train(
    cfg: Config | str | Path,
    edges_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run one experiment end-to-end and return a result dict.

    Args:
        cfg:             a :class:`Config` or a path to an experiment YAML.
        edges_override:  if given, replaces the graph from ``cfg.architecture.graph``.
                         Use this when the constraint graph is constructed
                         programmatically (e.g. parameter-sweep notebooks).

    Returns:
        ``{"config", "history", "test_metrics", "model", "trainer", "output_dir"}``.
    """
    cfg = cfg if isinstance(cfg, Config) else load_config(cfg)

    set_seed(cfg.experiment.seed)
    device = resolve_device(cfg.experiment.device)
    output_dir = resolve_output_dir(cfg.experiment.name, cfg.experiment.output_dir)

    if cfg.logging.verbose:
        print(f"[run] {cfg.experiment.name}  device={device}  output={output_dir}")

    # --- Build pipeline ----------------------------------------------------
    base = Path.cwd()
    data = build_datasets(cfg.data, base_dir=base)
    model, edges = build_model(cfg.architecture, base_dir=base)

    if edges_override is not None:
        if cfg.logging.verbose:
            print(f"[run] edges_override provided; ignoring YAML graph "
                  f"({cfg.architecture.graph.type!r})")
        edges = edges_override

    # Move model + edges onto the resolved device. Train / val / test
    # tensors stay on CPU and are transferred per-batch / per-chunk by
    # the Trainer;
    model = model.to(device)
    edges = edges.to(device)

    n_train = data["train"][0].shape[0]
    trainer = Trainer(
        model=model, edges=edges, cfg=cfg.training,
        n_train=n_train, device=device,
    )

    # --- Fit + final evaluation -------------------------------------------
    history = trainer.fit(
        train_data    = data["train"],
        val_data      = data["val"],
        num_epochs    = cfg.training.num_epochs,
        batch_size    = cfg.data.batch_size,
        eval_every    = cfg.training.eval_every,
        monitor       = cfg.training.early_stopping.monitor,
        patience      = cfg.training.early_stopping.patience,
        min_delta     = cfg.training.early_stopping.min_delta,
        verbose       = cfg.logging.verbose,
        print_metrics = cfg.logging.print_metrics,
    )

    test_x, test_y = data["test"]
    test_metrics = trainer.metrics(test_x, test_y)

    # --- Persist artifacts ------------------------------------------------
    cfg_resolved = with_output_dir(cfg, str(output_dir))
    save_artifacts(cfg_resolved, output_dir, model, history, test_metrics)

    if cfg.logging.verbose:
        print(f"[run] done  test_hamming={test_metrics['hamming']:.4f}  "
              f"test_zero_one={test_metrics['zero_one']:.4f}")

    return {
        "config":       cfg_resolved,
        "history":      history,
        "test_metrics": test_metrics,
        "model":        model,
        "trainer":      trainer,
        "output_dir":   output_dir,
    }
