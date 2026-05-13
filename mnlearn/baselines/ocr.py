"""
OCR digit classifier for the two-stage Sudoku baseline.

Trains a per-cell digit classifier on clue images extracted from the
visual Sudoku training set (cells where the original quiz has a non-zero
digit). At evaluation time the classifier predicts a digit at every clue
cell of a test puzzle, and the resulting (corrupted) quiz is fed into
the hard-coded solver in :mod:`mnlearn.baselines.sudoku_solver`.

Reuses the same backbone factory as the M3N predictor
(:func:`mnlearn.models.backbones.build_backbone`); the architecture
must output raw logits (Linear tail) so ``CrossEntropyLoss`` doesn't
double-apply softmax, enforced at YAML-load time by
:func:`mnlearn.config.validate_baseline`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mnlearn.config.schema import BaselineArchitectureCfg
from mnlearn.models.backbones import ConfigBackbone, WrappedBackbone, build_backbone


# ---------------------------------------------------------------------------
# Clue extraction
# ---------------------------------------------------------------------------

def extract_clue_pairs(
    X_visual: torch.Tensor,
    quizzes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten clue-cell images and 0-indexed digit labels.

    Args:
        X_visual: ``[N, 81, 1, 28, 28]`` FloatTensor — per-cell rendered images
                  (blank cells are zero images).
        quizzes:  ``[N, 81]`` LongTensor — values in ``{0..9}``; 0 = blank.

    Returns:
        ``X_clue: [M, 1, 28, 28]`` — images of clue cells only.
        ``Y_clue: [M]``            — class labels (``quiz - 1``, in ``{0..8}``).
        Where ``M = (quizzes != 0).sum()``.

    The blank-cell images are dropped entirely (they are constant zeros and
    carry no signal); the OCR classifier never sees them. The original
    ``quizzes`` tensor is the source of truth for which cells are clues —
    the two-stage pipeline uses the same mask at test time.
    """
    if X_visual.shape[0] != quizzes.shape[0]:
        raise ValueError(
            f"X_visual and quizzes must agree on N (got {X_visual.shape[0]} "
            f"vs {quizzes.shape[0]})"
        )
    if X_visual.dim() < 3 or X_visual.shape[1] != quizzes.shape[1]:
        raise ValueError(
            f"X_visual and quizzes must agree on the per-puzzle dimension "
            f"(got X_visual.shape={tuple(X_visual.shape)}, "
            f"quizzes.shape={tuple(quizzes.shape)})"
        )

    mask = quizzes != 0                        # [N, 81] bool
    X_clue = X_visual[mask]                    # [M, *image_shape]
    Y_clue = (quizzes[mask] - 1).long()        # [M]
    return X_clue, Y_clue


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_ocr_model(arch_cfg: BaselineArchitectureCfg) -> nn.Module:
    """Build a per-image digit classifier from a :class:`BaselineArchitectureCfg`.

    The backbone is constructed via the same factory as the M3N predictor
    (:func:`mnlearn.models.backbones.build_backbone`)
    The per-cell-of-puzzle reshape wrapper that the M3N path needs is
    stripped here because the OCR classifier consumes flat
    ``[batch, 1, 28, 28]`` images, not ``[batch, 81, 1, 28, 28]`` sequences.

    Returns:
        ``nn.Module`` mapping ``[batch, *image_shape] ->
        [batch, num_classes]`` of raw logits.
    """
    backbone = build_backbone(arch_cfg.backbone, arch_cfg.num_classes)

    if isinstance(backbone, ConfigBackbone):
        # ConfigBackbone wraps an inner ``nn.Sequential`` that already has the
        # right per-image contract; the outer reshape only matters for the
        # per-cell-of-puzzle case.
        return backbone.net

    if isinstance(backbone, WrappedBackbone):
        # The wrapped feature extractor takes a per-image input; combine with
        # its trainable head into a single Sequential for the OCR contract.
        return nn.Sequential(backbone.features, backbone.head)

    raise TypeError(
        f"Unsupported backbone type for OCR: {type(backbone).__name__}"
    )
