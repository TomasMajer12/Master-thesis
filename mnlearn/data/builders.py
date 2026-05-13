"""Build train/val/test tensor datasets from a :class:`DataCfg`.

Wraps the existing benchmark loaders and materialises the requested
number of examples into contiguous ``(X, Y)`` tensors. Returns a flat
dict so trainers can index ``data['train']`` etc. without caring about
task or input mode.

Visual datasets synthesise per-cell MNIST images on the fly; we
materialise them once up-front so training does not pay that cost on
every batch.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mnlearn.config.schema import DataCfg

from .sudoku import load_sudoku_benchmark
from .hmc    import load_hmc_benchmark


def build_datasets(cfg: DataCfg, base_dir: Path | str | None = None) -> dict[str, tuple]:
    """Materialise the train/val/test splits as dense tensors.

    Returns:
        ``{'train': (X, Y), 'val': (X, Y), 'test': (X, Y)}``

    The shapes are:
        symbolic sudoku: X=[N, 81, 9],          Y=[N, 81]
        visual sudoku:   X=[N, 81, 1, 28, 28],  Y=[N, 81]
        symbolic hmc:    X=[N, T,  K],          Y=[N, T]
        visual hmc:      X=[N, T,  1, 28, 28],  Y=[N, T]
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    if cfg.task == "sudoku":
        return _build_sudoku(cfg, base)
    if cfg.task == "hmc":
        return _build_hmc(cfg, base)
    raise ValueError(f"Unknown data.task={cfg.task!r}")


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def _build_sudoku(cfg: DataCfg, base: Path) -> dict[str, tuple]:
    bench_path = (base / cfg.paths["benchmark"]).resolve()

    if cfg.mode == "visual":
        mnist_root = (base / cfg.paths["mnist"]).resolve()
        data = load_sudoku_benchmark(
            str(bench_path), mode="visual", mnist_root=str(mnist_root),
        )
        sizes = (cfg.train_size, cfg.val_size, cfg.test_size)
        return {
            split: _materialize_visual(
                data[split], n, image_shape=(1, 28, 28), seq_len=81,
            )
            for split, n in zip(("train", "val", "test"), sizes)
        }

    if cfg.mode == "symbolic":
        data = load_sudoku_benchmark(str(bench_path), mode="symbolic")
        return {
            "train": _materialize_symbolic(data["train"], cfg.train_size),
            "val":   _materialize_symbolic(data["val"],   cfg.val_size),
            "test":  _materialize_symbolic(data["test"],  cfg.test_size),
        }

    raise ValueError(f"Unknown data.mode={cfg.mode!r}")


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def _build_hmc(cfg: DataCfg, base: Path) -> dict[str, tuple]:
    bench_path = (base / cfg.paths["benchmark"]).resolve()

    if cfg.mode == "visual":
        mnist_root = (base / cfg.paths["mnist"]).resolve()
        data = load_hmc_benchmark(
            str(bench_path), mode="visual", mnist_root=str(mnist_root),
        )
        seq_len = data["config"]["seq_len"]
        sizes = (cfg.train_size, cfg.val_size, cfg.test_size)
        return {
            split: _materialize_visual(
                data[split], n, image_shape=(1, 28, 28), seq_len=seq_len,
            )
            for split, n in zip(("train", "val", "test"), sizes)
        }

    if cfg.mode == "symbolic":
        data = load_hmc_benchmark(str(bench_path), mode="symbolic")
        return {
            "train": _materialize_symbolic(data["train"], cfg.train_size),
            "val":   _materialize_symbolic(data["val"],   cfg.val_size),
            "test":  _materialize_symbolic(data["test"],  cfg.test_size),
        }

    raise ValueError(f"Unknown data.mode={cfg.mode!r}")


# ---------------------------------------------------------------------------
# Materialisation helpers
# ---------------------------------------------------------------------------

def _materialize_symbolic(ds: Dataset, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first n examples of a symbolic dataset as dense tensors."""
    n = min(n, len(ds))
    X_full, Y_full = ds.get_all_tensors()
    return X_full[:n], Y_full[:n]


def _materialize_visual(ds: Dataset, n: int, image_shape: tuple,
                        seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first n examples of a visual dataset as dense tensors.

    Each ``__getitem__`` of a visual dataset synthesises images on the fly;
    for in-memory training we want a single contiguous tensor. The shape
    of one element is ``[seq_len, *image_shape]``.
    """
    n = min(n, len(ds))
    X = torch.zeros(n, seq_len, *image_shape)
    Y = torch.zeros(n, seq_len, dtype=torch.long)
    for i in range(n):
        x_i, y_i = ds[i]
        X[i] = x_i
        Y[i] = y_i
    return X, Y
