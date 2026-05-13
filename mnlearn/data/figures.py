"""
Thesis figures for Chapter 3 — Datasets.

Each function takes data (or a dataset) plus a save path, applies a
consistent thesis matplotlib style, and writes a PNG. Reproducibility is
inherited from the benchmark seeds stored in ``config.json``.

Public API
----------
Sudoku
    plot_sudoku_symbolic_example(quiz, solution, save_path, ...)
    plot_sudoku_visual_example(visual_dataset, idx, save_path, ...)
    plot_sudoku_difficulty(quizzes, save_path, ...)
    plot_sudoku_clue_rate_per_cell(quizzes, save_path, ...)
    plot_mnist_pool_usage_sudoku(image_indices, quizzes, save_path, ...)

HMC
    plot_hmc_symbolic_example(obs, labels, save_path, ...)
    plot_hmc_visual_example(visual_dataset, idx, save_path, ...)
    plot_hmc_transition(labels, K, save_path, ...)
    plot_mnist_pool_usage_hmc(image_indices, obs, K, save_path, ...)

Notes
-----
Output is always PNG at ``DPI=200``. Figure widths are sized to a
``\\textwidth`` of ~6.0 in (close to the CTU master-thesis text width).
"""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from . import stats


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_THESIS_WIDTH = 6.0   # inches (≈ \textwidth in the CTU thesis class)
_DPI          = 200

_RC_PARAMS = {
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       _DPI,
    "savefig.dpi":      _DPI,
    "savefig.bbox":     "tight",
    "axes.grid":        False,
}

# Colors shared across figures.
_BAR_COLOR    = "#4c72b0"   # steelblue
_ACCENT_COLOR = "#dd8452"   # warm orange (clue borders, expected lines)
_HIGHLIGHT    = "#c44e52"   # red (mismatch markers)


def _apply_style() -> None:
    plt.rcParams.update(_RC_PARAMS)


def _save(fig, save_path: str | os.PathLike) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# ---------------------------------------------------------------------------
# Sudoku — sample renders
# ---------------------------------------------------------------------------

def plot_sudoku_symbolic_example(
    quiz, solution, save_path,
    show_solution: bool = False,
    title: str | None = None,
):
    """Typeset a 9×9 Sudoku grid with 3×3 box boundaries.

    Args:
        quiz:          [81] or [9, 9] int array; 0 = blank, 1-9 = clue.
        solution:      [81] or [9, 9] int array; full solution.
        save_path:     output PNG path.
        show_solution: if True, render the filled-in solution; clue digits
                       are bold, originally-blank cells are normal weight.
                       If False, render only the puzzle clues (rest blank).
        title:         optional figure title.
    """
    _apply_style()
    quiz     = _np(quiz).reshape(9, 9)
    solution = _np(solution).reshape(9, 9)
    grid = solution if show_solution else quiz

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(9):
        for c in range(9):
            v = grid[r, c]
            if v == 0:
                continue
            is_clue = quiz[r, c] != 0
            ax.text(
                c + 0.5, 8 - r + 0.5, str(int(v)),
                ha="center", va="center",
                fontsize=14,
                fontweight="bold" if is_clue else "normal",
                color="black" if is_clue else "#555555",
            )

    # Cell-level grid plus thick 3×3 box lines.
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.5
        ax.axhline(k, color="black", linewidth=lw)
        ax.axvline(k, color="black", linewidth=lw)

    if title is not None:
        ax.set_title(title)

    return _save(fig, save_path)


def plot_sudoku_visual_example(
    visual_dataset, idx: int, save_path,
    show_clue_borders: bool = True,
    title: str | None = None,
):
    """Render a 9×9 grid of MNIST images for one Sudoku puzzle.

    Composites the 81 per-cell 28×28 images into a single 252×252 array
    and overlays a thin grid (cell boundaries) plus thick 3×3 box lines.
    Optionally marks clue cells (non-blank in the puzzle) with a colored
    border.

    Args:
        visual_dataset:    VisualSudokuDataset.
        idx:               puzzle index.
        save_path:         output PNG path.
        show_clue_borders: highlight clue cells with a colored border.
        title:             optional figure title.
    """
    _apply_style()
    x, _ = visual_dataset[idx]                     # x : [81, 1, 28, 28]
    quiz = visual_dataset.quizzes[idx].view(9, 9)
    img4d = x.view(9, 9, 28, 28).numpy()

    composite = np.zeros((9 * 28, 9 * 28), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            composite[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img4d[r, c]

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH * 0.6, _THESIS_WIDTH * 0.6))
    ax.imshow(composite, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    # Grid lines (thick on 3x3 box boundaries, thin elsewhere).
    for k in range(10):
        lw    = 2.0 if k % 3 == 0 else 0.4
        color = "white" if k % 3 == 0 else "#888888"
        ax.axhline(k * 28 - 0.5, color=color, linewidth=lw)
        ax.axvline(k * 28 - 0.5, color=color, linewidth=lw)

    if show_clue_borders:
        for r in range(9):
            for c in range(9):
                if quiz[r, c] > 0:
                    rect = Rectangle(
                        (c * 28 - 0.5, r * 28 - 0.5), 28, 28,
                        fill=False, edgecolor=_ACCENT_COLOR, linewidth=1.5,
                    )
                    ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    return _save(fig, save_path)


def plot_sudoku_difficulty(
    quizzes, save_path,
    train_size: int | None = None,
):
    """Histogram of clues per puzzle.

    Annotates the figure with mean / std / min / max of the clue count.
    Intended for §3.2.5 (Sudoku statistics).
    """
    _apply_style()
    counts, edges = stats.sudoku_difficulty_histogram(quizzes)
    clues = stats.sudoku_clues_per_puzzle(quizzes)

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH, 3.0))
    ax.bar(
        edges[:-1], counts, width=np.diff(edges), align="edge",
        color=_BAR_COLOR, edgecolor="black", alpha=0.85,
    )
    ax.set_xlabel("clues per puzzle")
    ax.set_ylabel("number of puzzles")
    ax.text(
        0.97, 0.97,
        f"mean = {clues.mean():.1f}\n"
        f"std  = {clues.std():.1f}\n"
        f"min  = {clues.min()}\n"
        f"max  = {clues.max()}",
        transform=ax.transAxes,
        ha="right", va="top",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="black", alpha=0.85),
    )
    if train_size is not None:
        ax.set_title(f"Sudoku difficulty ($N = {train_size:,}$)")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, save_path)


def plot_sudoku_clue_rate_per_cell(
    quizzes, save_path,
    train_size: int | None = None,
):
    """9×9 heatmap of per-cell clue rate.

    Renders ``stats.sudoku_clue_rate_per_cell`` as a 9×9 heatmap with
    thick white lines marking the 3×3 box boundaries. Useful as a
    uniformity anchor for §3.2.5 — every cell should have approximately
    the same clue rate ≈ 1 − blank_rate.
    """
    _apply_style()
    rate = stats.sudoku_clue_rate_per_cell(quizzes).reshape(9, 9)
    mean = float(rate.mean())

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH * 0.6, _THESIS_WIDTH * 0.6))
    im = ax.imshow(rate, cmap="viridis")

    # Thick 3×3 box boundaries.
    for k in (3, 6):
        ax.axhline(k - 0.5, color="white", linewidth=2.0)
        ax.axvline(k - 0.5, color="white", linewidth=2.0)

    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(range(1, 10))
    ax.set_yticklabels(range(1, 10))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    plt.colorbar(im, ax=ax, fraction=0.046, label="clue rate")

    if train_size is not None:
        ax.set_title(
            f"per-cell clue rate (mean = {mean:.4f}, $N = {train_size:,}$)"
        )
    else:
        ax.set_title(f"per-cell clue rate (mean = {mean:.4f})")

    return _save(fig, save_path)


def plot_mnist_pool_usage_sudoku(
    image_indices, quizzes, save_path,
    train_size: int | None = None,
):
    """Boxplot of MNIST-image reuse counts for Sudoku visual variant.

    For each digit 1–9, shows the distribution of how many cells used the
    same source MNIST image. A uniform-target reference line is overlaid
    (expected reuse = total cells with that digit / pool size).
    """
    _apply_style()
    usage = stats.mnist_pool_usage(image_indices, quizzes, num_digits=10)

    digits = list(range(1, 10))
    box_data = [usage[d] for d in digits]

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH, 3.5))
    bp = ax.boxplot(
        box_data, tick_labels=digits, showmeans=True, meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=_BAR_COLOR, alpha=0.85, edgecolor="black"),
        medianprops=dict(color="black"),
        meanprops=dict(color=_ACCENT_COLOR, linewidth=1.5),
    )
    ax.set_xlabel("digit")
    ax.set_ylabel("reuse count per MNIST image")
    if train_size is not None:
        ax.set_title(f"MNIST pool usage in Sudoku train split ($N = {train_size:,}$)")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# HMC — sample renders
# ---------------------------------------------------------------------------

def plot_hmc_symbolic_example(
    obs, labels, save_path,
    K: int | None = None,
    title: str | None = None,
):
    """Render an HMC sequence as a two-row strip: obs (top), labels (bottom).

    Cells where ``obs[t] != labels[t]`` are highlighted in the top row to
    visualise the emission noise.

    Args:
        obs:      [T] int array of observation indices.
        labels:   [T] int array of true hidden states.
        save_path: output PNG path.
        K:        label-set size. Used for context only (not currently rendered).
        title:    optional figure title.
    """
    _apply_style()
    obs    = _np(obs)
    labels = _np(labels)
    T = len(obs)

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH, 1.7))
    GAP = 0.1
    for t in range(T):
        is_noisy = obs[t] != labels[t]
        ax.add_patch(Rectangle(
            (t, 1 + GAP), 1, 1,
            facecolor=_HIGHLIGHT if is_noisy else "white",
            edgecolor="black", linewidth=0.6,
            alpha=0.4 if is_noisy else 1.0,
        ))
        ax.text(t + 0.5, 1.5 + GAP, str(int(obs[t])),
                ha="center", va="center", fontsize=10)
        ax.add_patch(Rectangle(
            (t, 0), 1, 1,
            facecolor="white", edgecolor="black", linewidth=0.6,
        ))
        ax.text(t + 0.5, 0.5, str(int(labels[t])),
                ha="center", va="center", fontsize=10)

    ax.text(-0.3, 1.5 + GAP, r"obs $x_t$",
            ha="right", va="center", fontsize=10)
    ax.text(-0.3, 0.5, r"label $y_t$",
            ha="right", va="center", fontsize=10)
    ax.text(
        T / 2, -0.45,
        r"red shading marks positions where $x_t \neq y_t$ (emission noise)",
        ha="center", va="center", fontsize=8, style="italic", color="#666666",
    )

    ax.set_xlim(-2.5, T + 0.2)
    ax.set_ylim(-0.7, 2 + GAP + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, pad=8)

    return _save(fig, save_path)


def plot_hmc_visual_example(
    visual_dataset, idx: int, save_path,
    title: str | None = None,
):
    """Render a visual HMC sequence: MNIST images on top, labels below.

    Cells where ``obs != labels`` get a red border to visualise
    emission noise.
    """
    _apply_style()
    x, y = visual_dataset[idx]                     # x : [T, 1, 28, 28]
    obs  = visual_dataset.obs_indices[idx].numpy()
    labels = y.numpy()
    T = x.shape[0]

    fig, axes = plt.subplots(
        2, T, figsize=(_THESIS_WIDTH, 1.9),
        gridspec_kw={"height_ratios": [1, 0.45]},
    )
    for t in range(T):
        # Top row: observation image.
        axes[0, t].imshow(x[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])
        is_noisy = obs[t] != labels[t]
        for spine in axes[0, t].spines.values():
            spine.set_visible(is_noisy)
            if is_noisy:
                spine.set_edgecolor(_HIGHLIGHT)
                spine.set_linewidth(1.4)
        # Bottom row: ground-truth label as text.
        axes[1, t].text(
            0.5, 0.5, str(int(labels[t])),
            ha="center", va="center", fontsize=10,
        )
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])
        for spine in axes[1, t].spines.values():
            spine.set_visible(False)

    # Row labels via ylabel on the leftmost subplots.
    axes[0, 0].set_ylabel(
        r"obs $x_t$", rotation=0, ha="right", va="center",
        labelpad=15, fontsize=9,
    )
    axes[1, 0].set_ylabel(
        r"label $y_t$", rotation=0, ha="right", va="center",
        labelpad=15, fontsize=9,
    )

    plt.subplots_adjust(left=0.10, right=0.99, wspace=0.05, hspace=0.05,
                        bottom=0.18)
    fig.text(
        0.5, 0.05,
        r"red border marks positions where $x_t \neq y_t$ (emission noise)",
        ha="center", va="center", fontsize=8, style="italic", color="#666666",
    )

    if title is not None:
        fig.suptitle(title, y=1.02)

    return _save(fig, save_path)


def plot_hmc_transition(
    labels, K: int, save_path,
    train_size: int | None = None,
):
    """Empirical transition matrix heatmap.

    Renders ``stats.hmc_empirical_transition`` as a viridis heatmap with
    the from-state and to-state axes labeled. Uniform off-diagonal entries
    and a heavy diagonal indicate the configured ``p_self``.
    """
    _apply_style()
    Tmat = stats.hmc_empirical_transition(labels, K)

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH * 0.55, _THESIS_WIDTH * 0.55))
    im = ax.imshow(Tmat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel(r"to state $j$")
    ax.set_ylabel(r"from state $i$")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    plt.colorbar(im, ax=ax, fraction=0.046)

    if train_size is not None:
        ax.set_title(
            rf"empirical $\hat{{P}}(y_{{t+1}}\mid y_t)$  ($N = {train_size:,}$)"
        )
    else:
        ax.set_title(r"empirical $\hat{P}(y_{t+1}\mid y_t)$")

    return _save(fig, save_path)


def plot_mnist_pool_usage_hmc(
    image_indices, obs, K: int, save_path,
    train_size: int | None = None,
):
    """Boxplot of MNIST-image reuse counts for HMC visual variant.

    Mirrors :func:`plot_mnist_pool_usage_sudoku`, but uses the HMC
    observation indices (digits 0..K-1) as the per-cell key.
    """
    _apply_style()
    usage = stats.mnist_pool_usage(image_indices, obs, num_digits=K)

    digits = [d for d in range(K) if len(usage[d]) > 0]
    box_data = [usage[d] for d in digits]

    fig, ax = plt.subplots(figsize=(_THESIS_WIDTH, 3.5))
    ax.boxplot(
        box_data, tick_labels=digits, showmeans=True, meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=_BAR_COLOR, alpha=0.85, edgecolor="black"),
        medianprops=dict(color="black"),
        meanprops=dict(color=_ACCENT_COLOR, linewidth=1.5),
    )
    ax.set_xlabel("observation digit")
    ax.set_ylabel("reuse count per MNIST image")
    if train_size is not None:
        ax.set_title(f"MNIST pool usage in HMC train split ($N = {train_size:,}$)")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, save_path)