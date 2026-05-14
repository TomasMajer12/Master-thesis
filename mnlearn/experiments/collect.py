"""Aggregate experiment artifacts across runs into a tidy DataFrame.

The runner-on-disk layout is one directory per run, each containing
``config.yaml``, ``history.json``, ``results.json``, ``model.pt`` (see
:mod:`mnlearn.learning.runtime`). :func:`collect_results` reads every
matching run and returns a wide DataFrame — one row per run — that
notebooks can ``groupby`` and plot from.

The reader is permissive: any of ``history.json`` / ``results.json``
may be missing (e.g. an interrupted run); the corresponding columns
just become NaN. Bad / unreadable directories are skipped with a
warning.
"""

from __future__ import annotations

import json
import logging
import re
from glob import glob as _glob
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_results(
    pattern: str | Iterable[str],
    *,
    parse_name: bool = True,
) -> pd.DataFrame:
    """Load every run directory matching ``pattern`` into one DataFrame.

    Args:
        pattern:    a glob pattern (or an iterable of patterns) pointing
                    at run directories. Each directory should contain
                    ``results.json`` (test-phase metrics) and optionally
                    ``history.json`` (training trajectory).
        parse_name: when True, also extract common sweep keys from each
                    run's directory name (``n``, ``seed``, ``lambda``,
                    ``wd``) using :func:`parse_run_name`. Notebooks that
                    use non-standard names can pass ``parse_name=False``
                    and parse manually.

    Returns:
        DataFrame, one row per run, with at minimum:
          * ``name``       — directory name
          * ``output_dir`` — full path
          * ``loss``       — from history (``m3n_hinge`` / ``lp_m3n`` /
                             ``classifier``) when available
          * any keys present in ``results.json`` (e.g. ``hamming``,
            ``zero_one``, ``solver_feasibility``, ``ocr_clue_error``)
          * history summary fields when available: ``best_epoch``,
            ``best_monitor_value`` / ``best_val_error``, ``early_stopped``,
            ``final_epoch``, ``total_seconds``
          * sweep-key columns extracted from the name (when ``parse_name=True``)

    Missing files / unreadable directories are logged and skipped.
    """
    if isinstance(pattern, str):
        patterns = [pattern]
    else:
        patterns = list(pattern)

    paths: list[Path] = []
    for pat in patterns:
        for p in _glob(pat):
            path = Path(p)
            if path.is_dir():
                paths.append(path)

    rows: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        row = _read_run(path)
        if row is None:
            continue
        if parse_name:
            row.update(parse_run_name(path.name))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Stable column order: identity columns first, then sweep keys, then
    # metrics + summary. Anything else trailing.
    leading = [c for c in ("name", "loss", "n", "seed", "lambda", "wd",
                           "weight_decay", "weight_decay_phi")
               if c in df.columns]
    other = [c for c in df.columns if c not in leading and c != "output_dir"]
    return df[leading + other + ["output_dir"]]


def parse_run_name(name: str) -> dict[str, Any]:
    """Extract common sweep keys from a run-directory name.

    Recognises (case-insensitive) the suffixes ``_n<int>``,
    ``_seed<int>``, ``_lambda<float>``, ``_wd<float>``,
    ``_weight_decay<float>``, ``_weight_decay_phi<float>``. Anything not
    matched is silently skipped — non-standard names just produce an
    empty dict.

    Numeric values are returned as int when integer-valued, else float.

    Examples:
        >>> parse_run_name("lpm3n_visual_sudoku_n1000_seed3")
        {'n': 1000, 'seed': 3}
        >>> parse_run_name("lpm3n_visual_sudoku_n100_lambda0.001_seed0")
        {'n': 100, 'lambda': 0.001, 'seed': 0}
    """
    out: dict[str, Any] = {}
    # Each pattern: ``_<key><value>``. Order matters — match longer keys first
    # so ``weight_decay_phi`` doesn't get parsed as ``weight_decay``.
    patterns = [
        ("weight_decay_phi", r"_weight_decay_phi([0-9eE.+-]+)"),
        ("weight_decay",     r"_weight_decay([0-9eE.+-]+)"),
        ("wd",               r"_wd([0-9eE.+-]+)"),
        ("lambda",           r"_lambda([0-9eE.+-]+)"),
        ("n",                r"_n(\d+)(?=_|$)"),
        ("seed",             r"_seed(\d+)(?=_|$)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1)
        out[key] = _coerce_number(raw)
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _read_run(path: Path) -> dict[str, Any] | None:
    """Read one run directory. Returns None if there's nothing useful."""
    row: dict[str, Any] = {"name": path.name, "output_dir": str(path)}

    results_path = path / "results.json"
    history_path = path / "history.json"

    have_anything = False

    if results_path.is_file():
        try:
            results = json.loads(results_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("collect_results: failed to read %s: %s", results_path, e)
        else:
            if isinstance(results, dict):
                row.update({k: _scalarise(v) for k, v in results.items()})
                have_anything = True

    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("collect_results: failed to read %s: %s", history_path, e)
        else:
            if isinstance(history, dict):
                row.update(_summarise_history(history))
                have_anything = True

    return row if have_anything else None


def _summarise_history(history: dict) -> dict[str, Any]:
    """Pull the per-run summary fields out of a history dict.

    Tolerant to shape: handles both the unified Trainer history
    (``best_monitor_value``) and the ClassifierTrainer history
    (``best_val_error``).
    """
    out: dict[str, Any] = {}
    for key in ("loss", "best_epoch", "best_monitor_value",
                "best_val_error", "early_stopped", "monitor"):
        if key in history:
            out[key] = history[key]
    epochs = history.get("epoch")
    if isinstance(epochs, list) and epochs:
        # final_epoch = the last epoch number reached. Truthful indicator
        # of training duration (early-stop pulls it below num_epochs).
        out["final_epoch"] = int(epochs[-1])
    secs = history.get("epoch_seconds")
    if isinstance(secs, list) and secs:
        out["total_seconds"] = float(sum(secs))
    return out


def _scalarise(v: Any) -> Any:
    """Best-effort: leave scalars alone, stringify lists/dicts.

    test-metric dicts written by ``train`` / ``train_baseline`` are flat
    scalars, but if a custom evaluator ever wrote a nested value we
    don't want pandas to barf — turn it into a string the user can
    parse manually.
    """
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return json.dumps(v)


def _coerce_number(s: str) -> int | float:
    """Cast a numeric string to int when possible, else float."""
    try:
        i = int(s)
        if str(i) == s:
            return i
    except ValueError:
        pass
    return float(s)
