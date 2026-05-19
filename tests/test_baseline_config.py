"""
Tests for the BaselineConfig schema, loader, and validator.

Coverage:
    - Round-trip: write YAML, load_baseline_config, dump_baseline_config,
      reload, verify equivalence.
    - Architecture-by-relative-path resolution (matching load_config).
    - Validation rejects unsupported task / mode / loss / monitor /
      softmax-tailed architectures.
    - Defaults applied when optional sections are omitted.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from mnlearn.config import (
    BaselineConfig,
    ConfigValidationError,
    dump_baseline_config,
    load_baseline_config,
)


# ---------------------------------------------------------------------------
# Helpers: minimal valid baseline experiment YAML
# ---------------------------------------------------------------------------

_VALID_ARCHITECTURE = {
    "num_classes": 9,
    "backbone": {
        "type": "config",
        "layers": [
            {"type": "flatten", "start_dim": 1},
            {"type": "linear", "in_features": 784, "out_features": 9},
        ],
    },
}

_VALID_EXPERIMENT_BASE = {
    "experiment": {"name": "test_baseline", "seed": 0},
    "data": {
        "task": "sudoku",
        "mode": "visual",
        "paths": {"benchmark": "benchmarks/sudoku", "mnist": "mnist_data"},
        "train_size": 100,
        "val_size":   100,
        "test_size":  100,
        "batch_size": 32,
    },
    "training": {
        "loss": "cross_entropy",
        "optimizer": {"type": "adam", "lr": 0.001},
        "num_epochs": 5,
    },
}


def _write_experiment_yaml(tmp_path: Path,
                           overrides: dict | None = None,
                           inline_arch: bool = True) -> Path:
    """Write a baseline experiment YAML and return its path.

    By default the architecture is inlined (mimicking dump_baseline_config).
    Pass ``inline_arch=False`` to write a separate architecture YAML
    referenced by relative path. Pass ``overrides={"architecture": ...}``
    to replace the default architecture wholesale (shallow merge into the
    inline default — useful for testing single-field violations).
    """
    experiment = {**_VALID_EXPERIMENT_BASE}

    # Seed the default architecture FIRST so overrides can replace it.
    if inline_arch:
        experiment["architecture"] = dict(_VALID_ARCHITECTURE)
    else:
        arch_path = tmp_path / "arch.yaml"
        arch_path.write_text(yaml.safe_dump(_VALID_ARCHITECTURE, sort_keys=False))
        experiment["architecture"] = "arch.yaml"

    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(experiment.get(k), dict):
                experiment[k] = {**experiment[k], **v}
            else:
                experiment[k] = v

    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(experiment, sort_keys=False))
    return exp_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_load_minimal_valid_config(tmp_path):
    cfg = load_baseline_config(_write_experiment_yaml(tmp_path))

    assert isinstance(cfg, BaselineConfig)
    assert cfg.experiment.name == "test_baseline"
    assert cfg.architecture.num_classes == 9
    assert cfg.architecture.backbone.type == "config"
    assert cfg.training.loss == "cross_entropy"
    assert cfg.training.early_stopping.monitor == "val_error"


def test_load_resolves_architecture_by_relative_path(tmp_path):
    """Same relative-path resolution as load_config."""
    cfg = load_baseline_config(_write_experiment_yaml(tmp_path, inline_arch=False))
    assert cfg.architecture.num_classes == 9


def test_dump_then_reload_round_trips(tmp_path):
    """dump_baseline_config produces YAML that load_baseline_config can re-read."""
    cfg = load_baseline_config(_write_experiment_yaml(tmp_path))

    out_path = tmp_path / "dumped.yaml"
    dump_baseline_config(cfg, out_path)
    cfg2 = load_baseline_config(out_path)

    # All fields preserved.
    assert asdict(cfg) == asdict(cfg2)


def test_defaults_applied_when_optional_sections_omitted(tmp_path):
    cfg = load_baseline_config(_write_experiment_yaml(tmp_path))
    # No 'logging' in the YAML -> default LoggingCfg.
    assert cfg.logging.verbose is True
    # No 'scheduler' in training -> 'none'.
    assert cfg.training.scheduler.type == "none"
    # No 'early_stopping' -> defaults.
    assert cfg.training.early_stopping.patience == 10
    assert cfg.training.early_stopping.min_delta == 0.001


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

def test_validate_rejects_non_sudoku_task(tmp_path):
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path, overrides={"data": {"task": "hmc"}},
        ))
    assert "data.task" in str(exc.value)


def test_validate_rejects_symbolic_mode(tmp_path):
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path, overrides={"data": {"mode": "symbolic"}},
        ))
    assert "data.mode" in str(exc.value)


def test_validate_rejects_unknown_loss(tmp_path):
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path, overrides={"training": {"loss": "m3n_hinge"}},
        ))
    assert "training.loss" in str(exc.value)


def test_validate_rejects_softmax_tailed_architecture(tmp_path):
    """Final Softmax / LogSoftmax must be flagged at config-load time."""
    bad_arch = {
        "num_classes": 9,
        "backbone": {
            "type": "config",
            "layers": [
                {"type": "flatten", "start_dim": 1},
                {"type": "linear", "in_features": 784, "out_features": 9},
                {"type": "Softmax", "dim": 1},
            ],
        },
    }
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path, overrides={"architecture": bad_arch},
        ))
    assert "Softmax" in str(exc.value) or "softmax" in str(exc.value).lower()


def test_validate_rejects_logsoftmax_tailed_architecture(tmp_path):
    bad_arch = {
        "num_classes": 9,
        "backbone": {
            "type": "config",
            "layers": [
                {"type": "flatten", "start_dim": 1},
                {"type": "linear", "in_features": 784, "out_features": 9},
                {"type": "LogSoftmax", "dim": 1},
            ],
        },
    }
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path, overrides={"architecture": bad_arch},
        ))
    assert "softmax" in str(exc.value).lower()


def test_validate_rejects_mismatched_num_classes(tmp_path):
    bad_arch = {
        "num_classes": 9,
        "backbone": {
            "type": "config",
            "layers": [
                {"type": "flatten", "start_dim": 1},
                # out_features != num_classes
                {"type": "linear", "in_features": 784, "out_features": 10},
            ],
        },
    }
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path, overrides={"architecture": bad_arch},
        ))
    assert "out_features" in str(exc.value)


def test_validate_rejects_unknown_monitor(tmp_path):
    with pytest.raises(ConfigValidationError) as exc:
        load_baseline_config(_write_experiment_yaml(
            tmp_path,
            overrides={"training": {
                "loss": "cross_entropy",
                "optimizer": {"type": "adam", "lr": 0.001},
                "num_epochs": 5,
                "early_stopping": {"monitor": "val_hamming"},
            }},
        ))
    assert "monitor" in str(exc.value)


if __name__ == "__main__":
    from _fixtures import fixtures
    with fixtures() as tp: test_load_minimal_valid_config(tp);                       print("PASS: load minimal valid config")
    with fixtures() as tp: test_load_resolves_architecture_by_relative_path(tp);     print("PASS: load resolves architecture by relative path")
    with fixtures() as tp: test_dump_then_reload_round_trips(tp);                    print("PASS: dump then reload round trips")
    with fixtures() as tp: test_defaults_applied_when_optional_sections_omitted(tp); print("PASS: defaults applied when optional sections omitted")
    with fixtures() as tp: test_validate_rejects_non_sudoku_task(tp);                print("PASS: validate rejects non sudoku task")
    with fixtures() as tp: test_validate_rejects_symbolic_mode(tp);                  print("PASS: validate rejects symbolic mode")
    with fixtures() as tp: test_validate_rejects_unknown_loss(tp);                   print("PASS: validate rejects unknown loss")
    with fixtures() as tp: test_validate_rejects_softmax_tailed_architecture(tp);    print("PASS: validate rejects softmax tailed architecture")
    with fixtures() as tp: test_validate_rejects_logsoftmax_tailed_architecture(tp); print("PASS: validate rejects logsoftmax tailed architecture")
    with fixtures() as tp: test_validate_rejects_mismatched_num_classes(tp);         print("PASS: validate rejects mismatched num classes")
    with fixtures() as tp: test_validate_rejects_unknown_monitor(tp);                print("PASS: validate rejects unknown monitor")
    print("\nAll tests passed.")
