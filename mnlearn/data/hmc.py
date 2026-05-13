"""
Hidden Markov Chain (HMC) dataset: generate + datasets + benchmark build/load.

Model
-----
Hidden states Y form a K-state Markov chain::

    P(y_t = y_{t-1})                 = p_self
    P(y_t = j | y_{t-1} = i, j != i) = (1 - p_self) / (K - 1)

Observations X are noisy emissions::

    P(x_t = y_t)                     = p_emit
    P(x_t = j | y_t = i, j != i)     = (1 - p_emit) / (K - 1)

Initial state is uniform over K classes.

Two input modalities are supported:
    - SymbolicHMCDataset: per-step one-hot of the observation index.
    - VisualHMCDataset:   per-step MNIST image of the observed digit.

Public API
----------
    generate_hmc_sequences(num_samples, seq_len, num_states, p_self, p_emit, seed)
        Sample (obs_indices, labels) arrays.

    SymbolicHMCDataset, VisualHMCDataset
        torch.utils.data.Dataset subclasses.

    build_hmc_benchmark(output_dir, modes, ...) -> None
        Generate and save train/val/test splits to disk.

    load_hmc_benchmark(output_dir, mode, mnist_root=None) -> dict
        Load a previously-saved benchmark for one modality.
"""

from __future__ import annotations

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
# Generation
# ---------------------------------------------------------------------------

def generate_hmc_sequences(
    num_samples: int,
    seq_len: int,
    num_states: int = 10,
    p_self: float = 0.7,
    p_emit: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate HMC sequences with the explicit transition / emission loop.

    Run once per benchmark build, so the explicit loop is preferred over a
    vectorised version for readability. Uses an explicit RNG instance for
    reproducibility.

    Args:
        num_samples: number of sequences to generate.
        seq_len:     length of each sequence (T).
        num_states:  size of the label set (K).
        p_self:      self-transition probability (higher → more persistent
                     hidden states).
        p_emit:      correct-emission probability (higher → less observation
                     noise).
        seed:        RNG seed.

    Returns:
        obs_indices: np.ndarray [N, T] int64 — observed digits in {0..K-1}.
        labels:      np.ndarray [N, T] int64 — true hidden states.
    """
    rng = np.random.default_rng(seed)

    labels      = np.zeros((num_samples, seq_len), dtype=np.int64)
    obs_indices = np.zeros((num_samples, seq_len), dtype=np.int64)

    for n in range(num_samples):
        # Initial state: uniform.
        labels[n, 0] = rng.integers(num_states)

        # Hidden-state Markov chain.
        for t in range(1, seq_len):
            if rng.random() < p_self:
                labels[n, t] = labels[n, t - 1]
            else:
                choices = [s for s in range(num_states) if s != labels[n, t - 1]]
                labels[n, t] = rng.choice(choices)

        # Observations.
        for t in range(seq_len):
            if rng.random() < p_emit:
                obs_indices[n, t] = labels[n, t]
            else:
                choices = [s for s in range(num_states) if s != labels[n, t]]
                obs_indices[n, t] = rng.choice(choices)

    return obs_indices, labels


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class SymbolicHMCDataset(Dataset):
    """HMC with one-hot per-step input.

    Returns per sample:
        x: [T, K]  float32 — one-hot encoding of obs_indices
        y: [T]     int64
    """

    def __init__(self, obs_indices, labels, num_states: int = 10):
        self.obs_indices = torch.as_tensor(obs_indices, dtype=torch.long)
        self.labels      = torch.as_tensor(labels,      dtype=torch.long)
        self.num_states  = num_states

    def __len__(self):
        return len(self.obs_indices)

    def __getitem__(self, idx):
        x = F.one_hot(self.obs_indices[idx], self.num_states).float()
        y = self.labels[idx]
        return x, y

    def get_all_tensors(self):
        """Return the full dataset as dense (X, Y) tensors."""
        X = F.one_hot(self.obs_indices, self.num_states).float()
        Y = self.labels
        return X, Y


class VisualHMCDataset(Dataset):
    """HMC with MNIST-image per-step input.

    Each observation index 0..K-1 is rendered as a randomly chosen MNIST
    image of that digit (image choice fixed via ``image_indices`` for
    reproducibility).

    Returns per sample:
        x: [T, 1, 28, 28]  float32
        y: [T]             int64
    """

    def __init__(self, obs_indices, labels, mnist_pool: MNISTPool,
                 seed: int = 0, image_indices=None):
        self.obs_indices = torch.as_tensor(obs_indices, dtype=torch.long)
        self.labels      = torch.as_tensor(labels,      dtype=torch.long)
        self.mnist_pool  = mnist_pool

        if image_indices is not None:
            self.image_indices = torch.as_tensor(image_indices, dtype=torch.long)
        else:
            obs_np = np.asarray(obs_indices)
            rng = np.random.default_rng(seed)
            img_idx = np.zeros(obs_np.shape, dtype=np.int64)
            num_states = len(mnist_pool.images_by_digit)
            for d in range(num_states):
                mask = obs_np == d
                count = int(mask.sum())
                if count > 0:
                    img_idx[mask] = rng.integers(mnist_pool.pool_size(d), size=count)
            self.image_indices = torch.from_numpy(img_idx)

    def __len__(self):
        return len(self.obs_indices)

    def __getitem__(self, idx):
        obs     = self.obs_indices[idx]      # [T]
        img_idx = self.image_indices[idx]    # [T]
        T = obs.shape[0]
        # Gather per digit class (K iterations) instead of per step (T).
        x = torch.zeros(T, 1, 28, 28)
        for d, pool in self.mnist_pool.images_by_digit.items():
            mask = obs == d
            if mask.any():
                x[mask] = pool[img_idx[mask]]
        y = self.labels[idx]
        return x, y


# ---------------------------------------------------------------------------
# Benchmark build / load
# ---------------------------------------------------------------------------

# File names — single source of truth for the on-disk schema.
_OBS_FILE       = "obs_indices.pt"
_LABELS_FILE    = "labels.pt"
_VISUAL_INDICES = "visual_image_indices.pt"
_CONFIG_FILE    = "config.json"
_EXAMPLES_FILE  = "examples.json"

_VALID_MODES = {"symbolic", "visual"}


def build_hmc_benchmark(
    output_dir: str | os.PathLike,
    modes: tuple[str, ...] = ("symbolic", "visual"),
    num_samples: int = 50_000,
    seq_len: int = 30,
    num_states: int = 10,
    p_self: float = 0.7,
    p_emit: float = 0.7,
    train_size: int = 30_000,
    val_size:   int = 10_000,
    test_size:  int = 10_000,
    hmc_seed:   int = 42,
    mnist_seed: int = 123,
    mnist_root: str | os.PathLike | None = None,
    num_examples: int = 5,
) -> None:
    """Generate train/val/test HMC splits and write them under ``output_dir``.

    Layout written::

        output_dir/
            config.json
            examples.json
            train/
                obs_indices.pt          [train_size, T]   int64
                labels.pt               [train_size, T]   int64
                visual_image_indices.pt [train_size, T]   int64   (only if 'visual' in modes)
            val/  ...
            test/ ...

    The MNIST image index assignment uses **disjoint pools per split**:
    train and val draw from disjoint partitions of the MNIST training
    set, test draws from the MNIST test set.

    Args:
        output_dir:   destination directory (created if needed).
        modes:        non-empty subset of {"symbolic", "visual"}. If
                      'visual' is absent, MNIST is not loaded at all.
        num_samples:  total sequences to generate (= sum of split sizes).
        seq_len:      sequence length T.
        num_states:   label-set size K (and observation alphabet size).
        p_self:       self-transition probability.
        p_emit:       correct-emission probability.
        train_size, val_size, test_size: split sizes; must sum to num_samples.
        hmc_seed:     seed for sequence generation.
        mnist_seed:   base seed for MNIST image assignment.
        mnist_root:   MNIST cache directory; None uses the package default.
        num_examples: number of example sequences to dump into examples.json.
    """
    modes = tuple(modes)
    if not modes or any(m not in _VALID_MODES for m in modes):
        raise ValueError(f"modes must be a non-empty subset of {_VALID_MODES}, got {modes}")
    if train_size + val_size + test_size != num_samples:
        raise ValueError(
            f"split sizes ({train_size}+{val_size}+{test_size}) must sum to "
            f"num_samples ({num_samples})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sequences ----
    logger.info("[hmc] generating %d sequences (T=%d, K=%d)",
                num_samples, seq_len, num_states)
    obs_indices, labels = generate_hmc_sequences(
        num_samples, seq_len, num_states, p_self, p_emit, hmc_seed,
    )

    split_ranges = {
        "train": (0,                     train_size),
        "val":   (train_size,            train_size + val_size),
        "test":  (train_size + val_size, num_samples),
    }

    # ---- MNIST pools (only if visual requested) ----
    mnist_train_pool = mnist_test_pool = None
    digit_split_points: dict[int, int] = {}
    if "visual" in modes:
        mnist_train_pool = MNISTPool(train=True,  root=mnist_root)
        mnist_test_pool  = MNISTPool(train=False, root=mnist_root)
        # Train / val use disjoint partitions of MNIST train; test uses MNIST test.
        train_fraction = train_size / (train_size + val_size)
        for d in range(num_states):
            digit_split_points[d] = int(mnist_train_pool.pool_size(d) * train_fraction)

    # ---- Per-split write ----
    seed_offsets = {"train": 0, "val": 1, "test": 2}
    examples: dict[str, list[dict]] = {}

    for split, (start, end) in split_ranges.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        so, sl = obs_indices[start:end], labels[start:end]

        torch.save(torch.from_numpy(so), split_dir / _OBS_FILE)
        torch.save(torch.from_numpy(sl), split_dir / _LABELS_FILE)

        image_indices = None
        if "visual" in modes:
            image_indices = _assign_mnist_indices(
                so, split, num_states,
                mnist_train_pool, mnist_test_pool, digit_split_points,
                mnist_seed + seed_offsets[split],
            )
            torch.save(torch.from_numpy(image_indices),
                       split_dir / _VISUAL_INDICES)

        n_show = min(num_examples, len(so))
        examples[split] = [
            {
                "obs_indices": so[i].tolist(),
                "labels":      sl[i].tolist(),
                **({"image_indices": image_indices[i].tolist()}
                   if image_indices is not None else {}),
            }
            for i in range(n_show)
        ]

    # ---- Top-level config + examples ----
    config = {
        "task":         "hmc",
        "modes":        list(modes),
        "num_samples":  num_samples,
        "seq_len":      seq_len,
        "num_states":   num_states,
        "p_self":       p_self,
        "p_emit":       p_emit,
        "train_size":   train_size,
        "val_size":     val_size,
        "test_size":    test_size,
        "hmc_seed":     hmc_seed,
        "mnist_seed":   mnist_seed,
    }
    (output_dir / _CONFIG_FILE).write_text(json.dumps(config, indent=2))
    (output_dir / _EXAMPLES_FILE).write_text(json.dumps(examples, indent=2))

    logger.info("[hmc] wrote benchmark to %s/", output_dir)
    for split, (start, end) in split_ranges.items():
        logger.info("  %5s: %d sequences", split, end - start)


def _assign_mnist_indices(
    obs_indices: np.ndarray,
    split: str,
    num_states: int,
    mnist_train_pool: MNISTPool,
    mnist_test_pool:  MNISTPool,
    digit_split_points: dict[int, int],
    seed: int,
) -> np.ndarray:
    """Pick a reproducible MNIST image index per observation."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(obs_indices, dtype=np.int64)
    for d in range(num_states):
        mask = obs_indices == d
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


def load_hmc_benchmark(
    output_dir: str | os.PathLike,
    mode: str = "symbolic",
    mnist_root: str | os.PathLike | None = None,
) -> dict:
    """Load a previously-built HMC benchmark for one modality.

    Args:
        output_dir: directory where ``build_hmc_benchmark`` wrote.
        mode:       "symbolic" or "visual".
        mnist_root: only used if mode='visual'; MNIST cache directory.

    Returns:
        ``{"train": Dataset, "val": Dataset, "test": Dataset, "config": dict}``,
        where Dataset is :class:`SymbolicHMCDataset` or
        :class:`VisualHMCDataset`.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    output_dir = Path(output_dir)
    config = json.loads((output_dir / _CONFIG_FILE).read_text())

    # Refuse if the requested mode wasn't built.
    # Legacy benchmarks predate the 'modes' key; assume both modes were built.
    built_modes = config.get("modes", ["symbolic", "visual"])
    if mode not in built_modes:
        raise ValueError(
            f"benchmark at {output_dir} was built with modes={built_modes}; "
            f"cannot load mode={mode!r}. Re-run build_hmc_benchmark with "
            f"modes including {mode!r}."
        )

    num_states = config["num_states"]

    mnist_train_pool = mnist_test_pool = None
    if mode == "visual":
        mnist_train_pool = MNISTPool(train=True,  root=mnist_root)
        mnist_test_pool  = MNISTPool(train=False, root=mnist_root)

    out: dict = {"config": config}
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        obs_indices = torch.load(split_dir / _OBS_FILE,    weights_only=True)
        labels      = torch.load(split_dir / _LABELS_FILE, weights_only=True)

        if mode == "symbolic":
            out[split] = SymbolicHMCDataset(obs_indices, labels, num_states)
        else:
            image_indices = torch.load(split_dir / _VISUAL_INDICES, weights_only=True)
            pool = mnist_test_pool if split == "test" else mnist_train_pool
            out[split] = VisualHMCDataset(obs_indices, labels, pool,
                                          image_indices=image_indices)
    return out