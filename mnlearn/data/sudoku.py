"""
Sudoku dataset: download + load + datasets + benchmark build/load.

Source
------
Park (2018), "1 million Sudoku games", Kaggle, dataset `bryanpark/sudoku`,
licensed CC0. CSV format: two columns ('quizzes', 'solutions'), each an
81-char string of digits 0-9; '0' means blank cell.

Two input modalities are supported:
    - SymbolicSudokuDataset: per-cell one-hot encoding (blank → zero vector).
    - VisualSudokuDataset:   per-cell MNIST image of the digit (blank → zero image).

Public API
----------
    ensure_sudoku_csv(csv_path=None) -> Path
    load_sudoku_csv(csv_path, num_puzzles, seed) -> (quizzes, solutions)
    SymbolicSudokuDataset, VisualSudokuDataset
    build_sudoku_benchmark(output_dir, modes, ...) -> None
    load_sudoku_benchmark(output_dir, mode, mnist_root=None) -> dict
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .mnist_pool import MNISTPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical-dataset reference values
# ---------------------------------------------------------------------------
# bryanpark/sudoku on Kaggle.
KAGGLE_DATASET = "bryanpark/sudoku"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def ensure_sudoku_csv(csv_path: str | os.PathLike | None = None) -> Path:
    """Return a path to the Sudoku CSV.

    If ``csv_path`` is given, it must point to an existing file with the
    expected schema (columns 'quizzes' and 'solutions'; first row 81 digits
    each). Schema violations raise ValueError.

    If ``csv_path`` is None, the file is downloaded via ``kagglehub`` from
    the Kaggle dataset ``bryanpark/sudoku``. kagglehub manages its own
    cache (default ``~/.cache/kagglehub/``; override with
    ``KAGGLEHUB_CACHE``). Requires ``kagglehub`` (``pip install
    mnlearn[download]``) and Kaggle credentials
    (https://www.kaggle.com/docs/api).

    Returns:
        Path to a sudoku.csv file.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Sudoku CSV not found at {path}. "
                f"Either fix the path or omit csv_path to auto-download."
            )
        _validate_sudoku_csv(path)
        return path

    return _kagglehub_download_sudoku()


def _kagglehub_download_sudoku() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required to auto-download the Sudoku dataset.\n"
            "  Install:  pip install mnlearn[download]\n"
            "  Or:       pip install kagglehub\n"
            f"  Or pass csv_path=<path> after downloading manually from\n"
            f"           {KAGGLE_URL}"
        ) from e

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_file = dataset_dir / "sudoku.csv"
    if not csv_file.exists():
        # Defensive: if Kaggle ever republishes under a different filename.
        candidates = list(dataset_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV found in kagglehub download at {dataset_dir}. "
                f"Inspect that directory and pass csv_path explicitly."
            )
        csv_file = candidates[0]
    _validate_sudoku_csv(csv_file)
    return csv_file


def _validate_sudoku_csv(path: Path) -> None:
    """Schema check: 'quizzes'+'solutions' columns, first row is 81 digits.

    Raises ValueError on schema violation (downstream code can't parse it).
    Provenance against the canonical bryanpark file is intentionally NOT
    enforced here — use ``verify_sudoku_csv(path)`` for that opt-in check.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"quizzes", "solutions"} - set(reader.fieldnames):
            raise ValueError(
                f"Sudoku CSV at {path} is missing required columns. "
                f"Expected 'quizzes' and 'solutions', got {reader.fieldnames!r}."
            )
        try:
            first = next(reader)
        except StopIteration:
            raise ValueError(f"Sudoku CSV at {path} has no data rows.")
    for col in ("quizzes", "solutions"):
        v = first[col]
        if len(v) != 81 or not v.isdigit():
            raise ValueError(
                f"Sudoku CSV at {path}: '{col}' field is not 81 digits "
                f"(got len={len(v)}, value={v!r})."
            )


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def load_sudoku_csv(
    csv_path: str | os.PathLike,
    num_puzzles: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``num_puzzles`` random rows from the Kaggle Sudoku CSV.

    Two-pass: first pass counts rows, second pass reads only the selected
    indices. Constant memory.

    Args:
        csv_path:    path to a Kaggle-format Sudoku CSV (cols: quizzes, solutions).
        num_puzzles: number of rows to sample (capped at total rows).
        seed:        RNG seed for reproducible sampling.

    Returns:
        quizzes:   np.ndarray [N, 81] int64 — puzzle assignments (0=blank).
        solutions: np.ndarray [N, 81] int64 — full solutions (1-9).
    """
    csv_path = Path(csv_path)

    with open(csv_path, newline="") as f:
        total = sum(1 for _ in csv.DictReader(f))

    rng = np.random.default_rng(seed)
    selected = set(rng.choice(total, size=min(num_puzzles, total), replace=False))

    quiz_list, sol_list = [], []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i in selected:
                quiz_list.append([int(c) for c in row["quizzes"]])
                sol_list.append([int(c) for c in row["solutions"]])

    return (
        np.asarray(quiz_list, dtype=np.int64),
        np.asarray(sol_list,  dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class SymbolicSudokuDataset(Dataset):
    """Sudoku with one-hot per-cell input.

    Per cell:
        blank (quiz=0)        → 9-dim zero vector
        digit d (quiz∈{1..9}) → one-hot at index d-1

    Returns per sample:
        x: [81, 9]  float32
        y: [81]     int64, values in {0..8} (solution digit minus 1)
    """

    def __init__(self, quizzes, solutions):
        self.quizzes   = torch.as_tensor(quizzes,   dtype=torch.long)
        self.solutions = torch.as_tensor(solutions, dtype=torch.long)

    def __len__(self):
        return len(self.quizzes)

    def __getitem__(self, idx):
        quiz = self.quizzes[idx]
        x = torch.zeros(81, 9)
        nz = quiz > 0
        if nz.any():
            x[nz] = F.one_hot(quiz[nz] - 1, 9).float()
        y = self.solutions[idx] - 1
        return x, y

    def get_all_tensors(self):
        """Return the full dataset as dense (X, Y) tensors."""
        N = len(self.quizzes)
        X = torch.zeros(N, 81, 9)
        nz = self.quizzes > 0
        if nz.any():
            X[nz] = F.one_hot(self.quizzes[nz] - 1, 9).float()
        Y = self.solutions - 1
        return X, Y


class VisualSudokuDataset(Dataset):
    """Sudoku with MNIST-image per-cell input.

    Each non-blank cell digit is rendered as a randomly chosen MNIST image
    of that digit (image choice fixed via ``image_indices`` for
    reproducibility). Blank cells are rendered as zero images.

    Returns per sample:
        x: [81, 1, 28, 28]  float32
        y: [81]             int64
    """

    def __init__(self, quizzes, solutions, mnist_pool: MNISTPool,
                 seed: int = 0, image_indices=None):
        self.quizzes   = torch.as_tensor(quizzes,   dtype=torch.long)
        self.solutions = torch.as_tensor(solutions, dtype=torch.long)
        self.mnist_pool = mnist_pool

        if image_indices is not None:
            self.image_indices = torch.as_tensor(image_indices, dtype=torch.long)
        else:
            quiz_np = np.asarray(quizzes)
            rng = np.random.default_rng(seed)
            img_idx = np.zeros(quiz_np.shape, dtype=np.int64)
            for digit in range(1, 10):
                mask = quiz_np == digit
                count = int(mask.sum())
                if count > 0:
                    img_idx[mask] = rng.integers(mnist_pool.pool_size(digit), size=count)
            self.image_indices = torch.from_numpy(img_idx)

    def __len__(self):
        return len(self.quizzes)

    def __getitem__(self, idx):
        quiz    = self.quizzes[idx]          # [81]
        img_idx = self.image_indices[idx]    # [81]
        # Gather per digit class (9 iterations) instead of per cell (81).
        x = torch.zeros(81, 1, 28, 28)
        for d in range(1, 10):
            mask = quiz == d
            if mask.any():
                pool = self.mnist_pool.images_by_digit[d]    # [N_d, 1, 28, 28]
                x[mask] = pool[img_idx[mask]]
        y = self.solutions[idx] - 1
        return x, y


# ---------------------------------------------------------------------------
# Benchmark build / load
# ---------------------------------------------------------------------------

# File names — single source of truth for the on-disk schema.
_QUIZZES_FILE   = "quizzes.pt"
_SOLUTIONS_FILE = "solutions.pt"
_VISUAL_INDICES = "visual_image_indices.pt"
_CONFIG_FILE    = "config.json"
_EXAMPLES_FILE  = "examples.json"

_VALID_MODES = {"symbolic", "visual"}


def build_sudoku_benchmark(
    output_dir: str | os.PathLike,
    modes: tuple[str, ...] = ("symbolic", "visual"),
    csv_path: str | os.PathLike | None = None,
    num_puzzles: int = 50_000,
    train_size: int = 30_000,
    val_size:   int = 10_000,
    test_size:  int = 10_000,
    puzzle_seed: int = 42,
    mnist_seed:  int = 200,
    mnist_root:  str | os.PathLike | None = None,
    num_examples: int = 5,
) -> None:
    """Generate train/val/test splits and write them under ``output_dir``.

    Layout written::

        output_dir/
            config.json
            examples.json
            train/
                quizzes.pt              [train_size, 81]   int64
                solutions.pt            [train_size, 81]   int64
                visual_image_indices.pt [train_size, 81]   int64   (only if 'visual' in modes)
            val/  ...
            test/ ...

    The MNIST image index assignment uses **disjoint pools per split**:
    train and val draw from disjoint partitions of the MNIST training
    set, test draws from the MNIST test set. This prevents train/val
    image leakage and guarantees test images were never seen during
    training or model selection.

    Args:
        output_dir:       destination directory (created if needed).
        modes:            non-empty subset of {"symbolic", "visual"}. If
                          'visual' is absent, MNIST is not loaded at all.
        csv_path:         path to the Sudoku CSV; None → auto-download via
                          ``ensure_sudoku_csv``.
        num_puzzles:      total puzzles to draw from the CSV
                          (= sum of split sizes).
        train_size, val_size, test_size: split sizes; must sum to num_puzzles.
        puzzle_seed:      seed for puzzle sampling and split shuffling.
        mnist_seed:       base seed for MNIST image assignment.
        mnist_root:       MNIST cache directory; None uses the package default.
        num_examples:     number of example puzzles to dump into examples.json.
    """
    modes = tuple(modes)
    if not modes or any(m not in _VALID_MODES for m in modes):
        raise ValueError(f"modes must be a non-empty subset of {_VALID_MODES}, got {modes}")
    if train_size + val_size + test_size != num_puzzles:
        raise ValueError(
            f"split sizes ({train_size}+{val_size}+{test_size}) must sum to "
            f"num_puzzles ({num_puzzles})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Puzzles ----
    csv_path = ensure_sudoku_csv(csv_path)
    logger.info("[sudoku] sampling %d puzzles from %s", num_puzzles, csv_path)
    quizzes, solutions = load_sudoku_csv(csv_path, num_puzzles, seed=puzzle_seed)

    # Permute before splitting (puzzle_seed → reproducible).
    rng = np.random.default_rng(puzzle_seed)
    perm = rng.permutation(len(quizzes))
    quizzes, solutions = quizzes[perm], solutions[perm]

    split_ranges = {
        "train": (0,                     train_size),
        "val":   (train_size,            train_size + val_size),
        "test":  (train_size + val_size, num_puzzles),
    }

    # ---- MNIST pools (only if visual requested) ----
    mnist_train_pool = mnist_test_pool = None
    digit_split_points: dict[int, int] = {}
    if "visual" in modes:
        mnist_train_pool = MNISTPool(train=True,  root=mnist_root)
        mnist_test_pool  = MNISTPool(train=False, root=mnist_root)
        # Train / val use disjoint partitions of MNIST train; test uses MNIST test.
        train_fraction = train_size / (train_size + val_size)
        for d in range(1, 10):
            digit_split_points[d] = int(mnist_train_pool.pool_size(d) * train_fraction)

    # ---- Per-split write ----
    seed_offsets = {"train": 0, "val": 1, "test": 2}
    examples: dict[str, list[dict]] = {}

    for split, (start, end) in split_ranges.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        sq, ss = quizzes[start:end], solutions[start:end]

        torch.save(torch.from_numpy(sq), split_dir / _QUIZZES_FILE)
        torch.save(torch.from_numpy(ss), split_dir / _SOLUTIONS_FILE)

        image_indices = None
        if "visual" in modes:
            image_indices = _assign_mnist_indices(
                sq, split,
                mnist_train_pool, mnist_test_pool, digit_split_points,
                mnist_seed + seed_offsets[split],
            )
            torch.save(torch.from_numpy(image_indices),
                       split_dir / _VISUAL_INDICES)

        # Tiny human-inspectable sample.
        n_show = min(num_examples, len(sq))
        examples[split] = [
            {
                "quiz":     sq[i].tolist(),
                "solution": ss[i].tolist(),
                **({"image_indices": image_indices[i].tolist()}
                   if image_indices is not None else {}),
            }
            for i in range(n_show)
        ]

    # ---- Top-level config + summary ----
    config = {
        "task":         "sudoku",
        "modes":        list(modes),
        "csv_path":     str(Path(csv_path).resolve()),
        "num_puzzles":  num_puzzles,
        "train_size":   train_size,
        "val_size":     val_size,
        "test_size":    test_size,
        "puzzle_seed":  puzzle_seed,
        "mnist_seed":   mnist_seed,
        "num_nodes":    81,
        "num_labels":   9,
    }
    (output_dir / _CONFIG_FILE).write_text(json.dumps(config, indent=2))
    (output_dir / _EXAMPLES_FILE).write_text(json.dumps(examples, indent=2))

    logger.info("[sudoku] wrote benchmark to %s/", output_dir)
    for split, (start, end) in split_ranges.items():
        logger.info("  %5s: %d puzzles", split, end - start)


def _assign_mnist_indices(
    quizzes: np.ndarray,
    split: str,
    mnist_train_pool: MNISTPool,
    mnist_test_pool:  MNISTPool,
    digit_split_points: dict[int, int],
    seed: int,
) -> np.ndarray:
    """Pick a reproducible MNIST image index per non-blank cell."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(quizzes, dtype=np.int64)
    for d in range(1, 10):
        mask = quizzes == d
        n = int(mask.sum())
        if n == 0:
            continue
        if split == "train":
            lo, hi = 0, digit_split_points[d]
        elif split == "val":
            lo, hi = digit_split_points[d], mnist_train_pool.pool_size(d)
        else:  # test
            lo, hi = 0, mnist_test_pool.pool_size(d)
        out[mask] = rng.integers(lo, hi, size=n)
    return out


def load_sudoku_benchmark(
    output_dir: str | os.PathLike,
    mode: str = "symbolic",
    mnist_root: str | os.PathLike | None = None,
) -> dict:
    """Load a previously-built Sudoku benchmark for one modality.

    Args:
        output_dir: directory where ``build_sudoku_benchmark`` wrote.
        mode:       "symbolic" or "visual".
        mnist_root: only used if mode='visual'; MNIST cache directory.

    Returns:
        ``{"train": Dataset, "val": Dataset, "test": Dataset, "config": dict}``,
        where Dataset is :class:`SymbolicSudokuDataset` or
        :class:`VisualSudokuDataset`.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    output_dir = Path(output_dir)
    config = json.loads((output_dir / _CONFIG_FILE).read_text())

    # Refuse if the requested mode wasn't built.
    built_modes = config.get("modes", ["symbolic", "visual"])
    if mode not in built_modes:
        raise ValueError(
            f"benchmark at {output_dir} was built with modes={built_modes}; "
            f"cannot load mode={mode!r}. Re-run build_sudoku_benchmark with "
            f"modes including {mode!r}."
        )

    mnist_train_pool = mnist_test_pool = None
    if mode == "visual":
        mnist_train_pool = MNISTPool(train=True,  root=mnist_root)
        mnist_test_pool  = MNISTPool(train=False, root=mnist_root)

    out: dict = {"config": config}
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        quizzes   = torch.load(split_dir / _QUIZZES_FILE,   weights_only=True)
        solutions = torch.load(split_dir / _SOLUTIONS_FILE, weights_only=True)

        if mode == "symbolic":
            out[split] = SymbolicSudokuDataset(quizzes, solutions)
        else:
            image_indices = torch.load(split_dir / _VISUAL_INDICES, weights_only=True)
            pool = mnist_test_pool if split == "test" else mnist_train_pool
            out[split] = VisualSudokuDataset(quizzes, solutions, pool,
                                             image_indices=image_indices)
    return out
