"""Convenience wrapper for inference with a trained M3N model.

A finished ``mnlearn.learning.train`` run writes a self-contained set of
artifacts under ``results/<experiment.name>/``:

* ``config.yaml`` — resolved config (architecture inlined).
* ``model.pt``    — the trained ``state_dict``.

This module provides :func:`load_predictor` which collapses the
re-create-architecture / load-state / pick-decoder / eval-mode ritual
into a single call, returning a callable :class:`Predictor` that maps
input tensors to predicted labellings.

The decoder is selected automatically from the YAML's
``training.inference.eval`` field: ``viterbi`` for chains, ``bp`` for
general graphs.  Decoder keyword arguments (e.g. ``bp_iters``) are read
from ``training.inference.params``.

For advanced use the underlying ``M3N`` model, edge list, and decoder
function remain exposed as attributes of the returned object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mnlearn.config import load_config
from mnlearn.inference import bp_decode, viterbi_decode
from mnlearn.learning.runtime import resolve_device
from mnlearn.models import build_model


class Predictor:
    """Callable wrapper around a trained M3N model.

    Constructed by :func:`load_predictor`.  Calling the instance as
    ``predictor(X)`` runs the full forward + decode pipeline; calling
    ``predictor.unary(X)`` returns the intermediate unary potentials.
    """

    def __init__(
        self,
        model,                                       # mnlearn.models.M3N
        edges: torch.Tensor,                         # [E, 2]
        eval_mode: str,                              # "viterbi" or "bp"
        decoder_params: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        self.model = model.eval()
        self.edges = edges
        self.eval_mode = eval_mode
        self.decoder_params = dict(decoder_params or {})
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def unary(self, X: torch.Tensor) -> torch.Tensor:
        """Forward through the backbone only.

        Returns the per-node unary potentials of shape ``[B, V, K]``.
        """
        X = X.to(self.device)
        return self.model.unary(X)

    @torch.no_grad()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Forward + decode.  Returns predicted labelling ``[B, V]`` (long)."""
        unary = self.unary(X)
        if self.eval_mode == "viterbi":
            return viterbi_decode(unary, self.model.pairwise)
        if self.eval_mode == "bp":
            # YAML key ``bp_iters`` maps to ``bp_decode``'s ``num_iters`` kwarg.
            # Matches the translation done in ``mnlearn.learning.builders``.
            kwargs = dict(self.decoder_params)
            if "bp_iters" in kwargs:
                kwargs["num_iters"] = kwargs.pop("bp_iters")
            return bp_decode(unary, self.model.pairwise, self.edges, **kwargs)
        raise ValueError(
            f"Unknown eval mode {self.eval_mode!r}; expected 'viterbi' or 'bp'."
        )


def load_predictor(run_dir: str | Path, device: str = "auto") -> Predictor:
    """Load a trained M3N predictor from a ``train()`` run directory.

    Args:
        run_dir: Path containing ``config.yaml`` and ``model.pt``.
            Typically ``results/<experiment.name>/``.
        device: ``"auto"`` (use CUDA if available, else CPU), ``"cpu"``,
            or ``"cuda"``.

    Returns:
        A callable :class:`Predictor`.  Usage::

            pred  = load_predictor("results/lpm3n_visual_sudoku")
            y_hat = pred(X)               # [B, V] long tensor
            unary = pred.unary(X)         # [B, V, K] float tensor

    The decoder is selected from the YAML's ``training.inference.eval``
    field, and its parameters from ``training.inference.params``.  To
    override either, build a :class:`Predictor` directly.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    dev = resolve_device(device)

    model, edges = build_model(cfg.architecture)
    state = torch.load(run_dir / "model.pt", weights_only=True, map_location=dev)
    model.load_state_dict(state)
    model = model.to(dev)
    edges = edges.to(dev)

    # Map "lp" used at training time onto the chosen eval decoder.
    return Predictor(
        model           = model,
        edges           = edges,
        eval_mode       = cfg.training.inference.eval,
        decoder_params  = dict(cfg.training.inference.params or {}),
        device          = dev,
    )
