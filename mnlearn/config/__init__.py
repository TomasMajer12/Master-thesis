"""Experiment configuration: schema, loader, validator.

The package is intentionally torch-free — it can be imported and used
in isolation (e.g. for config-validation CI checks) without pulling in
the rest of the project.
"""

from .loader import dump_baseline_config, dump_config, load_baseline_config, load_config
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

__all__ = [
    # M3N schema
    "Config",
    "ExperimentCfg",
    "ArchitectureCfg",
    "BackboneCfg",
    "GraphCfg",
    "PairwiseCfg",
    "DataCfg",
    "TrainingCfg",
    "OptimizerCfg",
    "SchedulerCfg",
    "InferenceCfg",
    "EarlyStoppingCfg",
    "LoggingCfg",
    # Baseline schema
    "BaselineConfig",
    "BaselineArchitectureCfg",
    "BaselineTrainingCfg",
    "BaselineEarlyStoppingCfg",
    # Loaders / dumpers / validation
    "load_config",
    "dump_config",
    "load_baseline_config",
    "dump_baseline_config",
    "validate",
    "validate_baseline",
    "ConfigValidationError",
]
