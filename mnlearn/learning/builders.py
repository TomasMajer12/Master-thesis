"""Builders for optimizer / scheduler / inference.

Each builder takes a typed config dataclass and returns a ready-to-use
runtime object. These are leaf utilities — they don't call each other.

Inference compatibility (only certain pairs are wired today):

    loss=lp_m3n      -> inference.train must be 'lp'
                        (the LP-M3N loss has its own augmented inference
                        baked in; ``train_fn`` is returned as ``None``)
    loss=m3n_hinge   -> inference.train must be 'viterbi'
                        (only Viterbi has a loss-augmented variant today)

Eval (decoding) is independent and supports {viterbi, bp}.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import optim
from torch.optim import lr_scheduler

from mnlearn.config.schema import (
    InferenceCfg,
    SchedulerCfg,
)
from mnlearn.inference import bp_decode, loss_augmented_viterbi, viterbi_decode


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: optim.Optimizer,
                    cfg: SchedulerCfg) -> lr_scheduler.LRScheduler | None:
    """Return ``None`` for ``type='none'``; otherwise a torch LRScheduler."""
    if cfg.type == "none":
        return None
    if cfg.type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.params.get("T_max", 100),
        )
    if cfg.type == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.params.get("step_size", 30),
            gamma=cfg.params.get("gamma", 0.1),
        )
    if cfg.type == "exp":
        return lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.params.get("gamma", 0.95),
        )
    if cfg.type == "lambda":
        # Hyperbolic decay: lr_t = lr_0 * offset / (offset + epoch).
        offset = cfg.params.get("offset", 100)
        return lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: offset / (offset + e),
        )
    raise ValueError(f"Unknown scheduler.type={cfg.type!r}")


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def build_inference(cfg: InferenceCfg, edges: torch.LongTensor
                   ) -> tuple[Callable | None, Callable]:
    """Return ``(train_fn, eval_fn)`` matching the inference config.

    ``train_fn`` is ``None`` when no inference oracle is needed at training
    time (i.e. for LP-M3N, whose loss has the augmentation built in).
    """
    train_fn = _build_train_inference(cfg.train, edges)
    eval_fn  = _build_eval_inference(cfg.eval,  edges, cfg.params)
    return train_fn, eval_fn


def _build_train_inference(name: str, edges: torch.LongTensor) -> Callable | None:
    if name == "lp":
        # LP-M3N has its own augmented inference inside the loss.
        return None
    if name == "viterbi":
        # Loss-augmented Viterbi assumes a chain graph — edges are implicit.
        return loss_augmented_viterbi
    raise ValueError(
        f"No loss-augmented inference implemented for inference.train={name!r}. "
        f"Currently supported: 'viterbi' (chain), 'lp' (LP-M3N)."
    )


def _build_eval_inference(name: str, edges: torch.LongTensor,
                          params: dict) -> Callable:
    if name == "viterbi":
        return viterbi_decode
    if name == "bp":
        num_iters = params.get("bp_iters", 50)
        def _bp(unary, pairwise):
            return bp_decode(unary, pairwise, edges=edges, num_iters=num_iters)
        return _bp
    raise ValueError(
        f"No decode implementation for inference.eval={name!r}. "
        f"Currently supported: 'viterbi', 'bp'."
    )
