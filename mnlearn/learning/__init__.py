"""Public API of the learning subsystem.

This package exposes the unified :class:`Trainer` that handles both
structured training paths used in the thesis:

* ``cfg.loss = "m3n_hinge"`` — structured hinge with Viterbi
  loss-augmented inference (chain graphs).
* ``cfg.loss = "lp_m3n"``    — LP-relaxed M3N with per-example dual
  variables (general graphs, including Sudoku).

The top-level entry point is :func:`mnlearn.learning.train`, which
loads a YAML config, dispatches to the right loss, and writes the
results directory described in :mod:`mnlearn.learning.runner`.
"""

from .builders import build_inference, build_scheduler
from .classifier_trainer import ClassifierTrainer
from .evaluation import hamming_distance, hamming_loss, zero_one_loss
from .lp_m3n import lp_m3n_loss
from .predictor import Predictor, load_predictor
from .runner import train
from .structured_svm import structured_hinge_loss
from .trainer import EarlyStopping, Trainer

__all__ = [
    # Loss functions
    "structured_hinge_loss",
    "lp_m3n_loss",
    # Evaluation metrics / utilities
    "hamming_loss",
    "hamming_distance",
    "zero_one_loss",
    # Unified structured trainer
    "Trainer",
    "EarlyStopping",
    # Cross-entropy classifier trainer (used by the OCR baseline)
    "ClassifierTrainer",
    # Builders
    "build_scheduler",
    "build_inference",
    # Top-level runner
    "train",
    # Inference convenience wrapper
    "Predictor",
    "load_predictor",
]
