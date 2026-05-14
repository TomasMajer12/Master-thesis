"""Shared runtime helpers for experiment runners.

Used by both the structured-prediction runner
(:func:`mnlearn.learning.train`) and the OCR baseline runner
(:func:`mnlearn.baselines.train_baseline`). The helpers are intentionally
generic over the config dataclass type — both ``Config`` and
``BaselineConfig`` carry an ``experiment.output_dir`` field, so the same
``with_output_dir`` and ``save_artifacts`` work for both.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Seeding / device / output-dir resolution
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG the project touches so a run is reproducible.

    Covers:
      * Python's ``random`` module — used internally by some PyTorch and
        torchvision data utilities.
      * NumPy's legacy global RNG (``np.random.seed``) — used by the data
        builders (e.g. MNIST-image-index assignment in
        ``mnlearn.data.sudoku``).
      * PyTorch's CPU RNG (``torch.manual_seed``).
      * PyTorch's CUDA RNG on every device (``torch.cuda.manual_seed_all``)
        — no-op on CPU-only machines.
      * cuDNN's deterministic flag (``torch.backends.cudnn.deterministic``)
        plus ``benchmark=False`` so convolutions pick the same algorithm
        every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def resolve_device(device_str: str) -> torch.device:
    """Map ``"auto" | "cpu" | "cuda"`` to a concrete :class:`torch.device`."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def resolve_output_dir(name: str, override: str) -> Path:
    """Return the directory where artifacts should be written.

    If ``override`` is set, use it verbatim; otherwise default to
    ``results/{name}``. The directory is created if it doesn't exist.
    """
    out = Path(override) if override else Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def with_output_dir(cfg, output_dir: str):
    """Return a copy of ``cfg`` with the resolved output_dir baked in."""
    return replace(cfg, experiment=replace(cfg.experiment, output_dir=output_dir))


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(cfg, output_dir: Path, model,
                   history: dict, test_metrics: dict) -> None:
    """Write the standard artifact set: config.yaml, history.json,
    results.json, model.pt.

    ``cfg`` may be either a :class:`Config` (M3N) or a
    :class:`BaselineConfig` (OCR baseline). Both are dumped via
    ``yaml.safe_dump(asdict(cfg))``
    """
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False),
    )

    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2, default=json_default)

    with (output_dir / "results.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    torch.save(model.state_dict(), output_dir / "model.pt")


def json_default(o: Any):
    """Fallback serialiser for non-JSON-native values in the history dict.
    Handles NumPy scalars and PyTorch tensors. Anything else raises
    TypeError
    """
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
