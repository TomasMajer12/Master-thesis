"""Plotting helpers.

Two shapes covered:

  * :func:`plot_learning_curve` — group a results DataFrame by an
    ``x`` column (typically training-set size or λ), reduce over seeds
    to mean ± std, plot one curve per ``hue`` group.
  * :func:`plot_run_history` — single-run trajectory: per-epoch metrics
    from a history dict.

Both return a matplotlib ``Axes`` so the notebook caller can title /
legend / decorate. Defaults are chosen for thesis figures: tight
margins, log-scale x where the values span orders of magnitude, std
shown as a shaded band rather than error bars (cleaner overlays).
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aggregated learning / sweep curves
# ---------------------------------------------------------------------------

def plot_learning_curve(
    df:        pd.DataFrame,
    *,
    x:         str,
    y:         str,
    hue:       str | None = None,
    ax:        plt.Axes | None = None,
    log_x:     bool = False,
    band:      bool = True,
    label:     str | None = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot mean(y) vs x with std shown as a band, optionally split by ``hue``.

    Args:
        df:    a results DataFrame as produced by
               :func:`mnlearn.experiments.collect_results`. Must have
               columns ``x`` and ``y``; if ``hue`` is given, that column too.
        x:     column to use as the curve's x axis (e.g. ``"n"`` for a
               training-set learning curve, ``"lambda"`` for a λ sweep).
        y:     metric column on the y axis (e.g. ``"zero_one"``).
        hue:   optional column to split into multiple curves (e.g.
               ``"loss"`` to overlay m3n_hinge / lp_m3n / classifier).
        ax:    target axes; one is created if None.
        log_x: log-scale the x axis (useful for λ sweeps spanning decades).
        band:  show std as a translucent shaded band (True) or as
               classical error bars (False).
        label: legend label when ``hue`` is None; ignored otherwise.
        **plot_kwargs: forwarded to ``ax.plot`` / ``ax.errorbar``.

    Returns:
        The matplotlib Axes (so the notebook can title / legend it).
    """
    _require_columns(df, [x, y] + ([hue] if hue is not None else []))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if hue is None:
        _plot_one_curve(ax, df, x=x, y=y, band=band, label=label, **plot_kwargs)
    else:
        for key, group in df.groupby(hue):
            _plot_one_curve(ax, group, x=x, y=y, band=band,
                            label=str(key), **plot_kwargs)
        ax.legend(title=hue)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    return ax


def _plot_one_curve(
    ax:    plt.Axes,
    df:    pd.DataFrame,
    *,
    x:     str,
    y:     str,
    band:  bool,
    label: str | None,
    **plot_kwargs,
) -> None:
    """Plot one mean ± std curve from a single group's rows."""
    grouped = (
        df.groupby(x)[y]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(x)
    )
    xs   = grouped[x].to_numpy()
    mean = grouped["mean"].to_numpy()
    std  = grouped["std"].fillna(0.0).to_numpy()

    if band:
        line, = ax.plot(xs, mean, label=label, **plot_kwargs)
        color = line.get_color()
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)
    else:
        ax.errorbar(xs, mean, yerr=std, label=label, capsize=3, **plot_kwargs)


# ---------------------------------------------------------------------------
# Single-run training trajectory
# ---------------------------------------------------------------------------

def plot_run_history(
    history:  dict,
    *,
    metrics:  Iterable[str] | None = None,
    splits:   Iterable[str] = ("train", "val"),
    ax:       plt.Axes | None = None,
) -> plt.Axes:
    """Plot training-trajectory curves for one run.

    Works on both the unified Trainer history (with ``train_metrics`` /
    ``val_metrics`` lists of dicts) and the ClassifierTrainer history
    (with flat ``train_error`` / ``val_error`` arrays).

    Args:
        history: history dict returned by ``Trainer.fit`` or
                 ``ClassifierTrainer.fit``, or loaded from
                 ``history.json``.
        metrics: which metric keys to plot. Defaults: every key
                 available in ``val_metrics[0]`` (unified trainer) or
                 ``["error", "loss"]`` (classifier).
        splits:  which splits to plot per metric (``"train"`` / ``"val"``).
                 Train traces are shown dashed.
        ax:      target axes; one is created if None.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    epochs = history.get("epoch")
    if not epochs:
        raise ValueError("history has no 'epoch' field — nothing to plot")
    epochs = np.asarray(epochs)

    is_unified = "val_metrics" in history and isinstance(history["val_metrics"], list)

    if is_unified:
        if metrics is None:
            metrics = list(history["val_metrics"][0].keys()) if history["val_metrics"] else []
        for split in splits:
            series_key = f"{split}_metrics"
            if series_key not in history:
                continue
            for metric_name in metrics:
                ys = [d.get(metric_name) for d in history[series_key]]
                if any(v is None for v in ys):
                    continue
                style = "--" if split == "train" else "-"
                ax.plot(epochs, ys, style,
                        label=f"{split}/{metric_name}")
    else:
        # Classifier-style history: flat arrays with "<split>_<metric>" keys.
        if metrics is None:
            metrics = ["error", "loss"]
        for split in splits:
            for metric_name in metrics:
                key = f"{split}_{metric_name}"
                if key not in history:
                    continue
                style = "--" if split == "train" else "-"
                ax.plot(epochs, history[key], style, label=key)

    if "best_epoch" in history and history["best_epoch"]:
        ax.axvline(history["best_epoch"], color="black", linestyle=":",
                   alpha=0.4, label=f"best epoch ({history['best_epoch']})")

    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )
