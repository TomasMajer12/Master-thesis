"""Aggregation + plotting helpers for experiments.

Experiments themselves run from notebooks (``train`` /
``train_baseline`` in a ``for`` loop over (N, seed, λ, ...)). The
notebook then loads results back through these helpers:

    >>> from mnlearn.experiments import collect_results, plot_learning_curve
    >>> df = collect_results("results/lpm3n_visual_sudoku_n*_seed*/")
    >>> plot_learning_curve(df, x="n", y="zero_one", hue="seed")

Splits responsibility:
  * ``collect`` — turn a glob of run directories into a tidy DataFrame.
  * ``plot``    — common figures (learning curves, single-run history).
"""

from .collect import collect_results, parse_run_name
from .plot import plot_learning_curve, plot_run_history

__all__ = [
    "collect_results",
    "parse_run_name",
    "plot_learning_curve",
    "plot_run_history",
]
