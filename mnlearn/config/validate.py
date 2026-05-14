"""Cross-field validation of a constructed :class:`Config`.

All checks are collected and raised together as a single
:class:`ConfigValidationError` so the user sees every problem at once
rather than fixing them one-by-one.
"""

from __future__ import annotations

from .schema import BaselineConfig, Config


# Allowed string values for enum-like fields. Kept as module-level constants
# so error messages list them in a stable, sorted order.
_BACKBONE_TYPES = {"config", "torchvision", "wrapped"}
_GRAPH_TYPES    = {"sudoku", "chain", "edges_file", "inline"}
_DATA_TASKS     = {"sudoku", "hmc"}
_DATA_MODES     = {"symbolic", "visual"}
_LOSSES         = {"m3n_hinge", "lp_m3n"}
_OPTIMIZERS     = {"adam", "sgd"}
_SCHEDULERS     = {"none", "cosine", "step", "exp", "lambda"}
_TRAIN_INFERENCE = {"lp", "viterbi"}
_EVAL_INFERENCE  = {"viterbi", "bp"}
_DEVICES         = {"auto", "cpu", "cuda"}


def _is_valid_monitor_path(path: str) -> bool:
    """Dotted path of identifiers (``a.b.c``).

    The validator only rejects malformed strings. Whether the path
    resolves to an actual metric is task-driven (different losses
    expose different keys) and checked at runtime by ``Trainer.fit``.
    """
    if not path:
        return False
    parts = path.split(".")
    return all(p.isidentifier() for p in parts)


class ConfigValidationError(ValueError):
    """Raised when a config has one or more problems.

    Attributes:
        errors: list of one-line problem descriptions, each prefixed by
                the dotted path to the offending field.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        msg = "Config validation failed:\n  - " + "\n  - ".join(self.errors)
        super().__init__(msg)


# Map from training.loss to the inference.train values that have a
# matching loss-augmented inference implementation. Edit needed when a new
# loss-augmented variant is wired up in mnlearn/learning/builders.py.
_VALID_LOSS_TRAIN_PAIRS = {
    "m3n_hinge": {"viterbi"},
    "lp_m3n":    {"lp"},
}


def validate(cfg: Config) -> None:
    """Validate `cfg`. Raises ConfigValidationError if any problems exist."""
    errors: list[str] = []
    _check_experiment(cfg, errors)
    _check_data(cfg, errors)
    _check_architecture(cfg, errors)
    _check_training(cfg, errors)
    _check_loss_inference_compat(cfg, errors)
    _check_loss_specific_optimizer_fields(cfg, errors)

    if errors:
        raise ConfigValidationError(errors)


def _check_loss_specific_optimizer_fields(cfg: Config, errors: list[str]) -> None:
    """Enforce that LP-M3N-only optimiser fields are zero for non-LP-M3N losses.

    ``weight_decay_phi`` and ``phi_init_std`` only have meaning for the
    per-example phi bank introduced by LP-M3N. Letting them sit non-zero
    in an ``m3n_hinge`` config silently does nothing — and a stale value
    that does nothing is a future debugging trap. We require an explicit
    ``0.0`` (or absence in YAML) for all other losses.
    """
    loss = cfg.training.loss
    opt  = cfg.training.optimizer
    if loss == "lp_m3n":
        return
    if loss not in _LOSSES:
        # Already reported by _check_training; skip.
        return
    if opt.weight_decay_phi != 0.0:
        errors.append(
            f"training.optimizer.weight_decay_phi={opt.weight_decay_phi} "
            f"is only meaningful for loss='lp_m3n'; for loss={loss!r} it "
            f"must be 0.0 (or omitted)."
        )
    if opt.phi_init_std != 0.0:
        errors.append(
            f"training.optimizer.phi_init_std={opt.phi_init_std} "
            f"is only meaningful for loss='lp_m3n'; for loss={loss!r} it "
            f"must be 0.0 (or omitted)."
        )
    if opt.lr_phi != 0.0:
        errors.append(
            f"training.optimizer.lr_phi={opt.lr_phi} "
            f"is only meaningful for loss='lp_m3n'; for loss={loss!r} it "
            f"must be 0.0 (or omitted)."
        )


def _check_loss_inference_compat(cfg: Config, errors: list[str]) -> None:
    """Reject (loss, inference.train) combinations with no implementation."""
    loss = cfg.training.loss
    train_inf = cfg.training.inference.train
    valid = _VALID_LOSS_TRAIN_PAIRS.get(loss)
    if valid is None:
        # Unknown loss — already reported by _check_training; skip.
        return
    if train_inf not in valid:
        errors.append(
            f"training.inference.train={train_inf!r} is not implemented for "
            f"training.loss={loss!r}. Valid choices: {sorted(valid)}"
        )


# ---------------------------------------------------------------------------
# Per-section checks
# ---------------------------------------------------------------------------

def _check_experiment(cfg: Config, errors: list[str]) -> None:
    e = cfg.experiment
    if e.device not in _DEVICES:
        errors.append(f"experiment.device={e.device!r} not in {sorted(_DEVICES)}")
    if e.seed < 0:
        errors.append(f"experiment.seed must be >= 0 (got {e.seed})")
    if not e.name:
        errors.append("experiment.name must be a non-empty string")


def _check_data(cfg: Config, errors: list[str]) -> None:
    d = cfg.data
    if d.task not in _DATA_TASKS:
        errors.append(f"data.task={d.task!r} not in {sorted(_DATA_TASKS)}")
    if d.mode not in _DATA_MODES:
        errors.append(f"data.mode={d.mode!r} not in {sorted(_DATA_MODES)}")
    if d.task == "sudoku" and "benchmark" not in d.paths:
        errors.append("data.paths.benchmark is required when data.task='sudoku'")
    for fname, val in [("train_size", d.train_size), ("val_size", d.val_size),
                       ("test_size", d.test_size), ("batch_size", d.batch_size)]:
        if val <= 0:
            errors.append(f"data.{fname} must be > 0 (got {val})")


def _check_architecture(cfg: Config, errors: list[str]) -> None:
    a = cfg.architecture
    if a.num_classes <= 0:
        errors.append(f"architecture.num_classes must be > 0 (got {a.num_classes})")

    bb = a.backbone
    if bb.type not in _BACKBONE_TYPES:
        errors.append(
            f"architecture.backbone.type={bb.type!r} not in {sorted(_BACKBONE_TYPES)}"
        )
    elif bb.type == "config":
        if not bb.layers:
            errors.append("architecture.backbone.layers is required when type='config'")
        else:
            # When the final layer is a plain Linear, its out_features must equal
            # num_classes — otherwise unary potentials will have the wrong shape.
            # For non-linear final layers (e.g. softmax), we cannot statically
            # verify and rely on the build-time shape check instead.
            last = bb.layers[-1]
            if last.get("type") == "linear" and "out_features" in last:
                if last["out_features"] != a.num_classes:
                    errors.append(
                        f"architecture.backbone: final linear layer has "
                        f"out_features={last['out_features']}, but "
                        f"architecture.num_classes={a.num_classes}"
                    )
    elif bb.type in {"torchvision", "wrapped"}:
        if not bb.name:
            errors.append(f"architecture.backbone.name is required when type={bb.type!r}")
        if bb.feature_dim is None or bb.feature_dim <= 0:
            errors.append(
                f"architecture.backbone.feature_dim must be > 0 when type={bb.type!r}"
            )

    g = a.graph
    if g.type not in _GRAPH_TYPES:
        errors.append(f"architecture.graph.type={g.type!r} not in {sorted(_GRAPH_TYPES)}")
    elif g.type == "chain":
        if g.seq_len is None or g.seq_len < 2:
            errors.append("architecture.graph.seq_len must be >= 2 when type='chain'")
    elif g.type == "edges_file":
        if not g.path:
            errors.append("architecture.graph.path is required when type='edges_file'")
    elif g.type == "inline":
        if not g.edges:
            errors.append("architecture.graph.edges is required when type='inline'")

    if a.pairwise.init_scale < 0:
        errors.append(
            f"architecture.pairwise.init_scale must be >= 0 (got {a.pairwise.init_scale})"
        )


def _check_training(cfg: Config, errors: list[str]) -> None:
    t = cfg.training
    if t.loss not in _LOSSES:
        errors.append(f"training.loss={t.loss!r} not in {sorted(_LOSSES)}")
    if t.optimizer.type not in _OPTIMIZERS:
        errors.append(
            f"training.optimizer.type={t.optimizer.type!r} not in {sorted(_OPTIMIZERS)}"
        )
    if t.optimizer.lr <= 0:
        errors.append(f"training.optimizer.lr must be > 0 (got {t.optimizer.lr})")
    if t.optimizer.weight_decay < 0:
        errors.append(
            f"training.optimizer.weight_decay must be >= 0 (got {t.optimizer.weight_decay})"
        )
    if t.scheduler.type not in _SCHEDULERS:
        errors.append(
            f"training.scheduler.type={t.scheduler.type!r} not in {sorted(_SCHEDULERS)}"
        )
    if t.inference.train not in _TRAIN_INFERENCE:
        errors.append(
            f"training.inference.train={t.inference.train!r} not in {sorted(_TRAIN_INFERENCE)}"
        )
    if t.inference.eval not in _EVAL_INFERENCE:
        errors.append(
            f"training.inference.eval={t.inference.eval!r} not in {sorted(_EVAL_INFERENCE)}"
        )
    if t.num_epochs <= 0:
        errors.append(f"training.num_epochs must be > 0 (got {t.num_epochs})")
    if t.eval_every <= 0:
        errors.append(f"training.eval_every must be > 0 (got {t.eval_every})")
    if not _is_valid_monitor_path(t.early_stopping.monitor):
        errors.append(
            f"training.early_stopping.monitor={t.early_stopping.monitor!r}: "
            f"must be a dotted path of identifiers, e.g. "
            f"'val_metrics.hamming', 'diagnostics.phi_norm', 'train_loss'."
        )
    if t.early_stopping.patience < 0:
        errors.append(
            f"training.early_stopping.patience must be >= 0 "
            f"(got {t.early_stopping.patience})"
        )
    if t.early_stopping.min_delta < 0:
        errors.append(
            f"training.early_stopping.min_delta must be >= 0 "
            f"(got {t.early_stopping.min_delta})"
        )


# ---------------------------------------------------------------------------
# Baseline validation
# ---------------------------------------------------------------------------
# Parallel validator for BaselineConfig. Allowed values are intentionally
# narrower than for Config (no graph / pairwise / inference, only
# cross_entropy loss for now).

_BASELINE_LOSSES   = {"cross_entropy"}
_BASELINE_MONITORS = {"val_error"}


def validate_baseline(cfg: BaselineConfig) -> None:
    """Validate `cfg` (BaselineConfig). Raises ConfigValidationError on any problem."""
    errors: list[str] = []
    _check_baseline_experiment(cfg, errors)
    _check_baseline_data(cfg, errors)
    _check_baseline_architecture(cfg, errors)
    _check_baseline_training(cfg, errors)

    if errors:
        raise ConfigValidationError(errors)


def _check_baseline_experiment(cfg: BaselineConfig, errors: list[str]) -> None:
    e = cfg.experiment
    if e.device not in _DEVICES:
        errors.append(f"experiment.device={e.device!r} not in {sorted(_DEVICES)}")
    if e.seed < 0:
        errors.append(f"experiment.seed must be >= 0 (got {e.seed})")
    if not e.name:
        errors.append("experiment.name must be a non-empty string")


def _check_baseline_data(cfg: BaselineConfig, errors: list[str]) -> None:
    d = cfg.data
    # The two-stage OCR baseline only makes sense on visual sudoku.
    # Relax this if/when we add a baseline that runs elsewhere.
    if d.task != "sudoku":
        errors.append(
            f"data.task={d.task!r}: baseline only supports task='sudoku' "
            f"(the two-stage OCR baseline is sudoku-specific)"
        )
    if d.mode != "visual":
        errors.append(
            f"data.mode={d.mode!r}: baseline only supports mode='visual' "
            f"(OCR is a digit-image classifier)"
        )
    if "benchmark" not in d.paths:
        errors.append("data.paths.benchmark is required for the visual baseline")
    for fname, val in [("train_size", d.train_size), ("val_size", d.val_size),
                       ("test_size", d.test_size), ("batch_size", d.batch_size)]:
        if val <= 0:
            errors.append(f"data.{fname} must be > 0 (got {val})")


def _check_baseline_architecture(cfg: BaselineConfig, errors: list[str]) -> None:
    a = cfg.architecture
    if a.num_classes <= 0:
        errors.append(f"architecture.num_classes must be > 0 (got {a.num_classes})")

    bb = a.backbone
    if bb.type not in _BACKBONE_TYPES:
        errors.append(
            f"architecture.backbone.type={bb.type!r} not in {sorted(_BACKBONE_TYPES)}"
        )
    elif bb.type == "config":
        if not bb.layers:
            errors.append("architecture.backbone.layers is required when type='config'")
        else:
            # The baseline must NOT end in a softmax / log-softmax:
            last = bb.layers[-1]
            last_type = str(last.get("type", "")).lower()
            if last_type in {"softmax", "logsoftmax", "log_softmax"}:
                errors.append(
                    f"architecture.backbone: final layer is {last['type']!r}; "
                    f"the baseline trainer uses CrossEntropyLoss which applies "
                    f"log-softmax internally. Drop the terminal softmax."
                )
            if last_type == "linear" and "out_features" in last:
                if last["out_features"] != a.num_classes:
                    errors.append(
                        f"architecture.backbone: final linear layer has "
                        f"out_features={last['out_features']}, but "
                        f"architecture.num_classes={a.num_classes}"
                    )
    elif bb.type in {"torchvision", "wrapped"}:
        if not bb.name:
            errors.append(f"architecture.backbone.name is required when type={bb.type!r}")
        if bb.feature_dim is None or bb.feature_dim <= 0:
            errors.append(
                f"architecture.backbone.feature_dim must be > 0 when type={bb.type!r}"
            )


def _check_baseline_training(cfg: BaselineConfig, errors: list[str]) -> None:
    t = cfg.training
    if t.loss not in _BASELINE_LOSSES:
        errors.append(
            f"training.loss={t.loss!r} not in {sorted(_BASELINE_LOSSES)}"
        )
    if t.optimizer.type not in _OPTIMIZERS:
        errors.append(
            f"training.optimizer.type={t.optimizer.type!r} not in {sorted(_OPTIMIZERS)}"
        )
    if t.optimizer.lr <= 0:
        errors.append(f"training.optimizer.lr must be > 0 (got {t.optimizer.lr})")
    if t.optimizer.weight_decay < 0:
        errors.append(
            f"training.optimizer.weight_decay must be >= 0 (got {t.optimizer.weight_decay})"
        )
    if t.scheduler.type not in _SCHEDULERS:
        errors.append(
            f"training.scheduler.type={t.scheduler.type!r} not in {sorted(_SCHEDULERS)}"
        )
    if t.num_epochs <= 0:
        errors.append(f"training.num_epochs must be > 0 (got {t.num_epochs})")
    if t.eval_every <= 0:
        errors.append(f"training.eval_every must be > 0 (got {t.eval_every})")
    if t.early_stopping.monitor not in _BASELINE_MONITORS:
        errors.append(
            f"training.early_stopping.monitor={t.early_stopping.monitor!r} "
            f"not in {sorted(_BASELINE_MONITORS)}"
        )
    if t.early_stopping.patience < 0:
        errors.append(
            f"training.early_stopping.patience must be >= 0 "
            f"(got {t.early_stopping.patience})"
        )
    if t.early_stopping.min_delta < 0:
        errors.append(
            f"training.early_stopping.min_delta must be >= 0 "
            f"(got {t.early_stopping.min_delta})"
        )
