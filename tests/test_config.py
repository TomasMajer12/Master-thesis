"""Tests for the YAML config loader, schema, and validator."""

import tempfile
from pathlib import Path

import yaml

from mnlearn.config import (
    Config,
    ConfigValidationError,
    dump_config,
    load_config,
)


# ---------------------------------------------------------------------------
# Reusable minimal-valid YAML fragments
# ---------------------------------------------------------------------------

VALID_ARCHITECTURE = """
num_classes: 9

backbone:
  type: config
  layers:
    - {type: linear, in_features: 9, out_features: 9}

graph:
  type: sudoku

pairwise:
  init_scale: 0.1
"""


def _valid_experiment(arch_ref: str) -> str:
    # Note: f-string brace-escaping for the inline `{bp_iters: 5}` mapping.
    return f"""
experiment:
  name: test_run
  seed: 42

architecture: {arch_ref}

data:
  task: sudoku
  mode: symbolic
  paths:
    benchmark: benchmarks/sudoku
  train_size: 100
  val_size: 100
  test_size: 100
  batch_size: 16

training:
  loss: m3n_hinge
  inference:
    train: viterbi
    eval: viterbi
    params: {{bp_iters: 5}}
  optimizer:
    type: adam
    lr: 0.01
    weight_decay: 0.0
  num_epochs: 10
"""


def _write_pair(tmp: Path, exp_text: str, arch_text: str,
                arch_rel: str = "arch.yaml",
                exp_rel: str = "exp.yaml") -> tuple[Path, Path]:
    arch_path = (tmp / arch_rel).resolve()
    arch_path.parent.mkdir(parents=True, exist_ok=True)
    arch_path.write_text(arch_text)

    exp_path = (tmp / exp_rel).resolve()
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.write_text(exp_text)

    return exp_path, arch_path


def _expect_validation_error(fn, *, contains: str):
    try:
        fn()
    except ConfigValidationError as e:
        assert contains in str(e), f"Expected {contains!r} in error, got: {e}"
        return e
    raise AssertionError(f"Expected ConfigValidationError containing {contains!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_minimal_valid_config():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        exp_path, _ = _write_pair(tmp, _valid_experiment("arch.yaml"), VALID_ARCHITECTURE)
        cfg = load_config(exp_path)

        assert isinstance(cfg, Config)
        assert cfg.experiment.name == "test_run"
        assert cfg.architecture.num_classes == 9
        assert cfg.architecture.backbone.type == "config"
        assert cfg.architecture.graph.type == "sudoku"
        assert cfg.training.optimizer.type == "adam"
        assert cfg.training.inference.params["bp_iters"] == 5
        # Defaults that aren't in the YAML
        assert cfg.training.scheduler.type == "none"
        assert cfg.logging.verbose is True
    print("PASS: load_minimal_valid_config")


def test_architecture_include_relative_to_experiment_file():
    """Architecture path is resolved relative to the experiment file."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # Layout:
        #   tmp/archs/sudoku.yaml
        #   tmp/exps/run.yaml  (references ../archs/sudoku.yaml)
        _write_pair(
            tmp,
            exp_text=_valid_experiment("../archs/sudoku.yaml"),
            arch_text=VALID_ARCHITECTURE,
            arch_rel="archs/sudoku.yaml",
            exp_rel="exps/run.yaml",
        )
        cfg = load_config(tmp / "exps" / "run.yaml")
        assert cfg.architecture.num_classes == 9
    print("PASS: architecture_include_relative_to_experiment_file")


def test_dump_inlines_architecture_and_round_trips():
    """dump_config should produce a single self-contained file that loads back."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        exp_path, _ = _write_pair(tmp, _valid_experiment("arch.yaml"), VALID_ARCHITECTURE)
        cfg = load_config(exp_path)

        dump_path = tmp / "dumped.yaml"
        dump_config(cfg, dump_path)

        # The dump must include architecture as an inline dict.
        roundtripped_raw = yaml.safe_load(dump_path.read_text())
        assert isinstance(roundtripped_raw["architecture"], dict)
        assert roundtripped_raw["architecture"]["num_classes"] == 9

        # And load_config must accept that form.
        cfg2 = load_config(dump_path)
        assert cfg2.architecture.num_classes == cfg.architecture.num_classes
        assert cfg2.training.optimizer.lr == cfg.training.optimizer.lr
        assert cfg2.training.inference.params == cfg.training.inference.params
    print("PASS: dump_inlines_architecture_and_round_trips")


def test_invalid_backbone_type_rejected():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        bad_arch = VALID_ARCHITECTURE.replace("type: config", "type: nonsense")
        exp_path, _ = _write_pair(tmp, _valid_experiment("arch.yaml"), bad_arch)
        _expect_validation_error(
            lambda: load_config(exp_path),
            contains="architecture.backbone.type",
        )
    print("PASS: invalid_backbone_type_rejected")


def test_visual_mode_without_mnist_path_falls_back_to_user_cache():
    """Visual mode no longer requires an explicit ``mnist`` path.

    The library's :class:`MNISTPool` falls back to the OS user cache
    (``~/.cache/mnlearn/mnist`` via ``_default_mnist_root``) when no path
    is supplied, so the YAML pipeline accepts visual configs without
    ``data.paths.mnist``.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        exp = _valid_experiment("arch.yaml").replace("mode: symbolic", "mode: visual")
        exp_path, _ = _write_pair(tmp, exp, VALID_ARCHITECTURE)
        cfg = load_config(exp_path)
        assert cfg.data.mode == "visual"
        assert "mnist" not in cfg.data.paths
    print("PASS: visual_mode_without_mnist_path_falls_back_to_user_cache")


def test_final_layer_must_match_num_classes():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        bad_arch = VALID_ARCHITECTURE.replace(
            "out_features: 9", "out_features: 7", 1,
        )
        exp_path, _ = _write_pair(tmp, _valid_experiment("arch.yaml"), bad_arch)
        e = _expect_validation_error(
            lambda: load_config(exp_path),
            contains="num_classes",
        )
        assert "out_features" in str(e)
    print("PASS: final_layer_must_match_num_classes")


def test_torchvision_backbone_requires_name_and_feature_dim():
    """A torchvision backbone needs name + feature_dim — bare type is not enough."""
    bad_arch = """
num_classes: 9

backbone:
  type: torchvision

graph:
  type: sudoku
"""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        exp_path, _ = _write_pair(tmp, _valid_experiment("arch.yaml"), bad_arch)
        e = _expect_validation_error(
            lambda: load_config(exp_path),
            contains="architecture.backbone.name",
        )
        assert any("feature_dim" in err for err in e.errors)
    print("PASS: torchvision_backbone_requires_name_and_feature_dim")


def test_chain_graph_requires_seq_len():
    bad_arch = """
num_classes: 9

backbone:
  type: config
  layers:
    - {type: linear, in_features: 9, out_features: 9}

graph:
  type: chain
"""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        exp_path, _ = _write_pair(tmp, _valid_experiment("arch.yaml"), bad_arch)
        _expect_validation_error(
            lambda: load_config(exp_path),
            contains="architecture.graph.seq_len",
        )
    print("PASS: chain_graph_requires_seq_len")


def test_multiple_errors_reported_together():
    """Validator collects every problem rather than stopping at the first."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        bad_exp = _valid_experiment("arch.yaml")
        bad_exp = bad_exp.replace("type: adam", "type: not_a_real_optimizer")
        bad_exp = bad_exp.replace("num_epochs: 10", "num_epochs: -5")
        exp_path, _ = _write_pair(tmp, bad_exp, VALID_ARCHITECTURE)
        e = _expect_validation_error(
            lambda: load_config(exp_path),
            contains="optimizer.type",
        )
        assert len(e.errors) >= 2, f"Expected >=2 errors, got {len(e.errors)}: {e.errors}"
        assert any("num_epochs" in err for err in e.errors)
    print("PASS: multiple_errors_reported_together")


def test_missing_architecture_field_rejected():
    """Top-level 'architecture' is mandatory."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # Strip the architecture line
        exp_text = _valid_experiment("arch.yaml")
        exp_text = "\n".join(
            line for line in exp_text.splitlines() if not line.startswith("architecture:")
        )
        exp_path = tmp / "exp.yaml"
        exp_path.write_text(exp_text)
        # Architecture file still needs to exist so the failure is about the missing
        # *reference*, not the missing target.
        (tmp / "arch.yaml").write_text(VALID_ARCHITECTURE)
        _expect_validation_error(
            lambda: load_config(exp_path),
            contains="architecture",
        )
    print("PASS: missing_architecture_field_rejected")


if __name__ == "__main__":
    test_load_minimal_valid_config()
    test_architecture_include_relative_to_experiment_file()
    test_dump_inlines_architecture_and_round_trips()
    test_invalid_backbone_type_rejected()
    test_visual_mode_without_mnist_path_falls_back_to_user_cache()
    test_final_layer_must_match_num_classes()
    test_torchvision_backbone_requires_name_and_feature_dim()
    test_chain_graph_requires_seq_len()
    test_multiple_errors_reported_together()
    test_missing_architecture_field_rejected()
    print("\nAll tests passed.")
