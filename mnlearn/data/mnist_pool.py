"""
Pre-indexed pool of MNIST images grouped by digit (0-9).

Used by the visual datasets (`VisualSudokuDataset`, `VisualHMCDataset`) to
map observation indices to actual MNIST images.

The default ``root`` resolves to a per-user cache directory
(e.g. ``~/.cache/mnlearn/mnist`` on Linux/Mac, ``%LOCALAPPDATA%\\mnlearn\\mnist``
on Windows) so the package works correctly regardless of cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torchvision import datasets, transforms


def _default_mnist_root() -> Path:
    """Cross-platform user cache directory for MNIST.

    Prefers ``platformdirs`` if installed; otherwise falls back to
    sensible per-OS defaults (``$LOCALAPPDATA`` on Windows,
    ``$XDG_CACHE_HOME`` or ``~/.cache`` on POSIX).
    """
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("mnlearn")) / "mnist"
    except ImportError:
        if os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA",
                                       Path.home() / "AppData" / "Local"))
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME",
                                       Path.home() / ".cache"))
        return base / "mnlearn" / "mnist"


class MNISTPool:
    """Pool of MNIST images indexed by digit class.

    Loads the full MNIST dataset and groups images by their label (0-9).
    Images are stored as float32 tensors normalised to [0, 1].

    Args:
        train: if True, use MNIST training set (60k images);
               if False, use the test set (10k images).
        root:  directory to download/load MNIST data. None → user cache
               (see :func:`_default_mnist_root`).
    """

    def __init__(self, train: bool = True, root: str | os.PathLike | None = None):
        if root is None:
            root = _default_mnist_root()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        dataset = datasets.MNIST(
            root=str(root), train=train, download=True,
            transform=transforms.ToTensor(),
        )

        by_digit: dict[int, list[torch.Tensor]] = {d: [] for d in range(10)}
        for img, label in dataset:
            by_digit[label].append(img)

        # images_by_digit[d] has shape [N_d, 1, 28, 28]
        self.images_by_digit = {
            d: torch.stack(imgs) for d, imgs in by_digit.items()
        }

    def pool_size(self, digit: int) -> int:
        """Number of available images for a given digit."""
        return len(self.images_by_digit[digit])