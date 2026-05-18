"""Dataclass definitions for the experiment configuration schema.

Each YAML field maps to a frozen dataclass. The schema is purely
structural — no validation logic lives here. Cross-field and value
checks are performed in :mod:`mnlearn.config.validate` so every problem
in the user's config can be reported in a single error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneCfg:
    type: str                                       # "config" | "torchvision" | "wrapped"
    layers: list[dict] | None = None                # used when type == "config"
    name: str | None = None                         # used when type in {"torchvision", "wrapped"}
    pretrained: bool = False
    freeze: bool = False
    feature_dim: int | None = None


@dataclass(frozen=True)
class GraphCfg:
    type: str                                       # "sudoku" | "chain" | "edges_file" | "inline"
    seq_len: int | None = None                      # for "chain"
    path: str | None = None                         # for "edges_file" (relative to experiment file)
    edges: list[list[int]] | None = None            # for "inline"


@dataclass(frozen=True)
class PairwiseCfg:
    init_scale: float = 0.1


@dataclass(frozen=True)
class ArchitectureCfg:
    num_classes: int
    backbone: BackboneCfg
    graph: GraphCfg
    pairwise: PairwiseCfg = field(default_factory=PairwiseCfg)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataCfg:
    task: str                                       # "sudoku" | "hmc"
    mode: str                                       # "symbolic" | "visual"
    paths: dict[str, str]
    train_size: int
    val_size: int
    test_size: int
    batch_size: int


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizerCfg:
    type: str                                       # "adam" | "sgd"
    lr: float
    weight_decay: float = 0.0
    # Separate L2 penalty applied only to the per-example phi variables
    # of LP-M3N.
    weight_decay_phi: float = 0.0
    # Standard deviation of Gaussian noise added to phi at init.
    phi_init_std: float = 0.0
    # Separate learning rate for the per-example phi variables. With
    # joint gradient descent on (model, phi)
    lr_phi: float = 0.0
    # Separate learning rate for the pairwise M3N matrix W. Sentinel
    # 0.0 means "same as lr_model"; >0 gives the pairwise a distinct
    # learning rate to compensate for CNN-vs-W training asymmetry at
    # low N (CNN converges faster, leaving W under-trained).
    lr_pairwise: float = 0.0


@dataclass(frozen=True)
class SchedulerCfg:
    type: str = "none"                              # "none" | "lambda" | "cosine" | "step" | "exp"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceCfg:
    train: str                                      # "lp" | "viterbi"
    eval: str                                       # "viterbi" | "bp"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EarlyStoppingCfg:
    # Dotted path into the per-epoch history record. Examples:
    #   "val_metrics.hamming", "val_metrics.zero_one",
    #   "diagnostics.phi_norm", "train_loss".
    # Whether the path resolves to a real value is task-driven and
    # checked at runtime by the Trainer (different tasks expose different
    # metric keys); the validator only rejects malformed paths.
    monitor: str = "val_metrics.hamming"
    patience: int = 10
    min_delta: float = 0.001


@dataclass(frozen=True)
class TrainingCfg:
    loss: str                                       # "m3n_hinge" | "lp_m3n"
    inference: InferenceCfg
    optimizer: OptimizerCfg
    num_epochs: int
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    eval_every: int = 1
    early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)


# ---------------------------------------------------------------------------
# Experiment / logging / top level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoggingCfg:
    verbose: bool = True
    # Dotted paths into the per-epoch history record to print at each
    # eval (e.g. ["train_loss", "val_metrics.zero_one", "val_metrics.hamming",
    # "diagnostics.phi_norm"]). When ``None``, the trainer prints
    # ``train_loss`` plus all keys in ``val_metrics``.
    print_metrics: list[str] | None = None


@dataclass(frozen=True)
class ExperimentCfg:
    name: str
    seed: int = 42
    device: str = "auto"                            # "auto" | "cpu" | "cuda"
    output_dir: str = ""                            # if empty, run-time defaults to results/{name}


@dataclass(frozen=True)
class Config:
    experiment: ExperimentCfg
    architecture: ArchitectureCfg
    data: DataCfg
    training: TrainingCfg
    logging: LoggingCfg = field(default_factory=LoggingCfg)


# ---------------------------------------------------------------------------
# Baseline (non-structured) experiment schema
# ---------------------------------------------------------------------------
# A separate parallel schema for two-stage / non-structured baselines.
# Reuses ExperimentCfg / DataCfg / LoggingCfg / BackboneCfg / OptimizerCfg /
# SchedulerCfg unchanged. Drops architecture.graph, architecture.pairwise,
# and training.inference because they have no meaning when there is no
# Markov-network decision layer.

@dataclass(frozen=True)
class BaselineArchitectureCfg:
    """Architecture for a non-structured baseline (no graph, no pairwise)."""
    num_classes: int
    backbone: BackboneCfg


@dataclass(frozen=True)
class BaselineEarlyStoppingCfg:
    monitor: str = "val_error"                      # only "val_error" wired in ClassifierTrainer.fit
    patience: int = 10
    min_delta: float = 0.001


@dataclass(frozen=True)
class BaselineTrainingCfg:
    """Training config for a baseline (no inference oracle / decoder)."""
    loss: str                                       # "cross_entropy"
    optimizer: OptimizerCfg
    num_epochs: int
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    eval_every: int = 1
    early_stopping: BaselineEarlyStoppingCfg = field(default_factory=BaselineEarlyStoppingCfg)


@dataclass(frozen=True)
class BaselineConfig:
    experiment: ExperimentCfg
    architecture: BaselineArchitectureCfg
    data: DataCfg
    training: BaselineTrainingCfg
    logging: LoggingCfg = field(default_factory=LoggingCfg)
