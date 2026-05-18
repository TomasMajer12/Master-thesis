"""YAML -> :class:`Config` loader and dumper.

The experiment file references its architecture by *path* — relative to
the experiment file's own directory — so architecture YAMLs are reusable
across runs without absolute paths leaking into configs.

For reproducibility, :func:`dump_config` writes the merged config (with
the architecture inlined as a dict) so a single artifact records every
hyperparameter actually used. The loader accepts both forms — a string
path or an inline dict — so dumped configs round-trip back through
:func:`load_config`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .schema import (
    ArchitectureCfg,
    BackboneCfg,
    BaselineArchitectureCfg,
    BaselineConfig,
    BaselineEarlyStoppingCfg,
    BaselineTrainingCfg,
    Config,
    DataCfg,
    EarlyStoppingCfg,
    ExperimentCfg,
    GraphCfg,
    InferenceCfg,
    LoggingCfg,
    OptimizerCfg,
    PairwiseCfg,
    SchedulerCfg,
    TrainingCfg,
)
from .validate import ConfigValidationError, validate, validate_baseline


def load_config(experiment_path: str | Path) -> Config:
    """Load and validate an experiment config from a YAML file.

    `experiment_path` points at the experiment YAML, which must reference
    its architecture either by relative path (``architecture: ../arch.yaml``)
    or inline as a dict (the form produced by :func:`dump_config`).
    """
    experiment_path = Path(experiment_path).resolve()
    raw = _read_yaml(experiment_path)

    raw["architecture"] = _resolve_architecture(raw, experiment_path)

    cfg = _bind(raw)
    validate(cfg)
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write `cfg` as a single YAML file with the architecture inlined."""
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))


# ---------------------------------------------------------------------------
# Architecture resolution
# ---------------------------------------------------------------------------

def _resolve_architecture(raw: dict, experiment_path: Path) -> dict:
    arch_ref = raw.get("architecture")
    if arch_ref is None:
        raise ConfigValidationError(
            ["experiment file is missing top-level 'architecture' field"]
        )
    if isinstance(arch_ref, str):
        arch_path = (experiment_path.parent / arch_ref).resolve()
        return _read_yaml(arch_path)
    if isinstance(arch_ref, dict):
        return arch_ref
    raise ConfigValidationError([
        f"experiment.architecture must be a path (str) or an inline dict, "
        f"got {type(arch_ref).__name__}"
    ])


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return data or {}


# ---------------------------------------------------------------------------
# dict -> dataclass binding
# ---------------------------------------------------------------------------

def _bind(raw: dict) -> Config:
    return Config(
        experiment   = _experiment(raw.get("experiment", {})),
        architecture = _architecture(raw["architecture"]),
        data         = _data(raw.get("data", {})),
        training     = _training(raw.get("training", {})),
        logging      = _logging(raw.get("logging", {})),
    )


def _experiment(d: dict) -> ExperimentCfg:
    return ExperimentCfg(
        name       = d.get("name", ""),
        seed       = int(d.get("seed", 42)),
        device     = d.get("device", "auto"),
        output_dir = d.get("output_dir", ""),
    )


def _architecture(d: dict) -> ArchitectureCfg:
    return ArchitectureCfg(
        num_classes = int(d["num_classes"]),
        backbone    = _backbone(d["backbone"]),
        graph       = _graph(d["graph"]),
        pairwise    = _pairwise(d.get("pairwise", {})),
    )


def _backbone(d: dict) -> BackboneCfg:
    return BackboneCfg(
        type        = d["type"],
        layers      = list(d["layers"]) if "layers" in d else None,
        name        = d.get("name"),
        pretrained  = bool(d.get("pretrained", False)),
        freeze      = bool(d.get("freeze", False)),
        feature_dim = d.get("feature_dim"),
    )


def _graph(d: dict) -> GraphCfg:
    return GraphCfg(
        type    = d["type"],
        seq_len = d.get("seq_len"),
        path    = d.get("path"),
        edges   = d.get("edges"),
    )


def _pairwise(d: dict) -> PairwiseCfg:
    return PairwiseCfg(init_scale=float(d.get("init_scale", 0.1)))


def _data(d: dict) -> DataCfg:
    return DataCfg(
        task       = d["task"],
        mode       = d["mode"],
        paths      = dict(d.get("paths", {})),
        train_size = int(d["train_size"]),
        val_size   = int(d["val_size"]),
        test_size  = int(d["test_size"]),
        batch_size = int(d["batch_size"]),
    )


def _training(d: dict) -> TrainingCfg:
    return TrainingCfg(
        loss            = d["loss"],
        inference       = _inference(d["inference"]),
        optimizer       = _optimizer(d["optimizer"]),
        num_epochs      = int(d["num_epochs"]),
        scheduler       = _scheduler(d.get("scheduler", {"type": "none"})),
        eval_every      = int(d.get("eval_every", 1)),
        early_stopping  = _early_stopping(d.get("early_stopping", {})),
    )


def _optimizer(d: dict) -> OptimizerCfg:
    return OptimizerCfg(
        type             = d["type"],
        lr               = float(d["lr"]),
        weight_decay     = float(d.get("weight_decay", 0.0)),
        weight_decay_phi = float(d.get("weight_decay_phi", 0.0)),
        phi_init_std     = float(d.get("phi_init_std", 0.0)),
        lr_phi           = float(d.get("lr_phi", 0.0)),
        lr_pairwise      = float(d.get("lr_pairwise", 0.0)),
    )


def _scheduler(d: dict) -> SchedulerCfg:
    return SchedulerCfg(
        type   = d.get("type", "none"),
        params = dict(d.get("params", {})),
    )


def _inference(d: dict) -> InferenceCfg:
    return InferenceCfg(
        train  = d["train"],
        eval   = d["eval"],
        params = dict(d.get("params", {})),
    )


def _early_stopping(d: dict) -> EarlyStoppingCfg:
    return EarlyStoppingCfg(
        monitor   = d.get("monitor", "val_metrics.hamming"),
        patience  = int(d.get("patience", 10)),
        min_delta = float(d.get("min_delta", 0.001)),
    )


def _logging(d: dict) -> LoggingCfg:
    print_metrics = d.get("print_metrics")
    if print_metrics is not None:
        print_metrics = list(print_metrics)
    return LoggingCfg(
        verbose       = bool(d.get("verbose", True)),
        print_metrics = print_metrics,
    )


# ---------------------------------------------------------------------------
# Baseline config loader / dumper
# ---------------------------------------------------------------------------
# Same on-disk layout as load_config: the experiment YAML references its
# architecture either by relative path or as an inline dict (the form that
# dump_baseline_config produces).

def load_baseline_config(experiment_path: str | Path) -> BaselineConfig:
    """Load and validate a baseline experiment config from a YAML file.

    Parallel to :func:`load_config` but binds the simpler
    :class:`BaselineConfig` schema (no graph / pairwise / inference).
    """
    experiment_path = Path(experiment_path).resolve()
    raw = _read_yaml(experiment_path)
    raw["architecture"] = _resolve_architecture(raw, experiment_path)
    cfg = _bind_baseline(raw)
    validate_baseline(cfg)
    return cfg


def dump_baseline_config(cfg: BaselineConfig, path: str | Path) -> None:
    """Write `cfg` (BaselineConfig) as a single YAML with architecture inlined."""
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))


def _bind_baseline(raw: dict) -> BaselineConfig:
    return BaselineConfig(
        experiment   = _experiment(raw.get("experiment", {})),
        architecture = _baseline_architecture(raw["architecture"]),
        data         = _data(raw.get("data", {})),
        training     = _baseline_training(raw.get("training", {})),
        logging      = _logging(raw.get("logging", {})),
    )


def _baseline_architecture(d: dict) -> BaselineArchitectureCfg:
    return BaselineArchitectureCfg(
        num_classes = int(d["num_classes"]),
        backbone    = _backbone(d["backbone"]),
    )


def _baseline_training(d: dict) -> BaselineTrainingCfg:
    return BaselineTrainingCfg(
        loss            = d["loss"],
        optimizer       = _optimizer(d["optimizer"]),
        num_epochs      = int(d["num_epochs"]),
        scheduler       = _scheduler(d.get("scheduler", {"type": "none"})),
        eval_every      = int(d.get("eval_every", 1)),
        early_stopping  = _baseline_early_stopping(d.get("early_stopping", {})),
    )


def _baseline_early_stopping(d: dict) -> BaselineEarlyStoppingCfg:
    return BaselineEarlyStoppingCfg(
        monitor   = d.get("monitor", "val_error"),
        patience  = int(d.get("patience", 10)),
        min_delta = float(d.get("min_delta", 0.001)),
    )
