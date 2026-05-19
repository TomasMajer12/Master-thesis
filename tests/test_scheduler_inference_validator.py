"""Tests for the scheduler / inference builders + config validator.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from mnlearn.config import ConfigValidationError, load_config
from mnlearn.config.schema import (
    GraphCfg,
    InferenceCfg,
    SchedulerCfg,
)
from mnlearn.learning import (
    build_inference,
    build_scheduler,
)
from mnlearn.models import build_graph


# ---------------------------------------------------------------------------
# Scheduler builder
# ---------------------------------------------------------------------------

def test_build_scheduler_none_returns_none():
    model = nn.Linear(3, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = build_scheduler(opt, SchedulerCfg(type="none"))
    assert sched is None


def test_build_scheduler_step():
    model = nn.Linear(3, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = build_scheduler(opt, SchedulerCfg(type="step", params={"step_size": 5, "gamma": 0.5}))
    assert sched is not None


def test_build_scheduler_lambda_decays_lr():
    """Hyperbolic decay: lr_t = lr_0 * offset / (offset + epoch).

    At epoch=0 the multiplier is 1.0 (lr unchanged). At epoch=offset the
    multiplier is 0.5 (lr halved). Used to replicate Kadlec (2022).
    """
    model = nn.Linear(3, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    sched = build_scheduler(opt, SchedulerCfg(type="lambda", params={"offset": 100}))
    assert sched is not None
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-9
    # Avoid PyTorch's "scheduler.step before optimizer.step" warning.
    opt.step()
    for _ in range(100):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 0.05) < 1e-9
    for _ in range(800):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# Inference builders
# ---------------------------------------------------------------------------

def test_build_inference_viterbi_pair():
    edges = build_graph(GraphCfg(type="chain", seq_len=8))
    train_fn, eval_fn = build_inference(
        InferenceCfg(train="viterbi", eval="viterbi", params={}), edges,
    )
    assert callable(train_fn)
    assert callable(eval_fn)


def test_build_inference_lp_train_returns_none():
    """LP-M3N has no inference oracle at training time."""
    edges = build_graph(GraphCfg(type="sudoku"))
    train_fn, eval_fn = build_inference(
        InferenceCfg(train="lp", eval="bp", params={"bp_iters": 5}), edges,
    )
    assert train_fn is None
    assert callable(eval_fn)


def test_build_inference_unsupported_train_raises():
    edges = build_graph(GraphCfg(type="chain", seq_len=8))
    try:
        build_inference(
            InferenceCfg(train="bp", eval="bp", params={}), edges,
        )
    except ValueError as e:
        assert "loss-augmented" in str(e)
        return
    raise AssertionError("Expected ValueError for unsupported train inference")


# ---------------------------------------------------------------------------
# Validator: (loss, inference.train) compatibility + per-loss strict rules
# ---------------------------------------------------------------------------

_VALID_ARCH_YAML = """
num_classes: 5
backbone:
  type: config
  layers:
    - {type: linear, in_features: 5, out_features: 5}
graph:
  type: chain
  seq_len: 8
pairwise:
  init_scale: 0.1
"""


def _exp_yaml(loss: str, train_inf: str, eval_inf: str = "viterbi",
              extra_optim: str = "") -> str:
    return f"""
experiment:
  name: t
  seed: 1

architecture: arch.yaml

data:
  task: hmc
  mode: symbolic
  paths: {{}}
  train_size: 10
  val_size: 10
  test_size: 10
  batch_size: 4

training:
  loss: {loss}
  inference:
    train: {train_inf}
    eval: {eval_inf}
  optimizer:
    type: adam
    lr: 0.01{extra_optim}
  num_epochs: 1
"""


def _expect_validation_error_substring(fn, substring: str):
    try:
        fn()
    except ConfigValidationError as e:
        assert substring in str(e), f"Expected {substring!r} in error, got: {e}"
        return e
    raise AssertionError(f"Expected ConfigValidationError containing {substring!r}")


def test_validator_rejects_lp_m3n_with_viterbi_train():
    """loss=lp_m3n requires inference.train='lp', not 'viterbi'."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "arch.yaml").write_text(_VALID_ARCH_YAML)
        exp_path = tmp / "exp.yaml"
        exp_path.write_text(_exp_yaml(loss="lp_m3n", train_inf="viterbi"))
        _expect_validation_error_substring(
            lambda: load_config(exp_path),
            substring="not implemented",
        )


def test_validator_rejects_m3n_hinge_with_lp_train():
    """loss=m3n_hinge has no LP-augmented inference; must use 'viterbi'."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "arch.yaml").write_text(_VALID_ARCH_YAML)
        exp_path = tmp / "exp.yaml"
        exp_path.write_text(_exp_yaml(loss="m3n_hinge", train_inf="lp"))
        _expect_validation_error_substring(
            lambda: load_config(exp_path),
            substring="not implemented",
        )


def test_validator_rejects_phi_fields_when_loss_is_not_lp_m3n():
    """weight_decay_phi / phi_init_std / lr_phi must be 0 for non-LP-M3N losses."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "arch.yaml").write_text(_VALID_ARCH_YAML)
        exp_path = tmp / "exp.yaml"
        exp_path.write_text(_exp_yaml(
            loss="m3n_hinge", train_inf="viterbi",
            extra_optim="\n    weight_decay_phi: 0.001"
                         "\n    phi_init_std: 0.5"
                         "\n    lr_phi: 0.01",
        ))
        e = _expect_validation_error_substring(
            lambda: load_config(exp_path),
            substring="weight_decay_phi",
        )
        assert any("phi_init_std" in err for err in e.errors)
        assert any("lr_phi" in err for err in e.errors)


def test_validator_accepts_dotted_monitor_path():
    """Any well-formed dotted-path identifier is accepted by the validator.

    Whether the path resolves to a real metric is a runtime check inside
    Trainer.fit, not a config-load check.
    """
    arch = _VALID_ARCH_YAML
    exp = """
experiment:
  name: t
  seed: 1
architecture: arch.yaml
data:
  task: hmc
  mode: symbolic
  paths: {}
  train_size: 10
  val_size: 10
  test_size: 10
  batch_size: 4
training:
  loss: m3n_hinge
  inference: {train: viterbi, eval: viterbi}
  optimizer: {type: adam, lr: 0.01}
  num_epochs: 1
  early_stopping:
    monitor: diagnostics.phi_norm
"""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "arch.yaml").write_text(arch)
        exp_path = tmp / "exp.yaml"
        exp_path.write_text(exp)
        cfg = load_config(exp_path)
        assert cfg.training.early_stopping.monitor == "diagnostics.phi_norm"


def test_validator_rejects_malformed_monitor_path():
    """Empty / non-identifier monitor strings are rejected."""
    arch = _VALID_ARCH_YAML
    exp = """
experiment:
  name: t
  seed: 1
architecture: arch.yaml
data:
  task: hmc
  mode: symbolic
  paths: {}
  train_size: 10
  val_size: 10
  test_size: 10
  batch_size: 4
training:
  loss: m3n_hinge
  inference: {train: viterbi, eval: viterbi}
  optimizer: {type: adam, lr: 0.01}
  num_epochs: 1
  early_stopping:
    monitor: "val metrics.hamming"
"""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "arch.yaml").write_text(arch)
        exp_path = tmp / "exp.yaml"
        exp_path.write_text(exp)
        _expect_validation_error_substring(
            lambda: load_config(exp_path),
            substring="dotted path",
        )


if __name__ == "__main__":
    test_build_scheduler_none_returns_none();                    print("PASS: build scheduler none returns none")
    test_build_scheduler_step();                                 print("PASS: build scheduler step")
    test_build_scheduler_lambda_decays_lr();                     print("PASS: build scheduler lambda decays lr")
    test_build_inference_viterbi_pair();                         print("PASS: build inference viterbi pair")
    test_build_inference_lp_train_returns_none();                print("PASS: build inference lp train returns none")
    test_build_inference_unsupported_train_raises();             print("PASS: build inference unsupported train raises")
    test_validator_rejects_lp_m3n_with_viterbi_train();          print("PASS: validator rejects lp m3n with viterbi train")
    test_validator_rejects_m3n_hinge_with_lp_train();            print("PASS: validator rejects m3n hinge with lp train")
    test_validator_rejects_phi_fields_when_loss_is_not_lp_m3n(); print("PASS: validator rejects phi fields when loss is not lp m3n")
    test_validator_accepts_dotted_monitor_path();                print("PASS: validator accepts dotted monitor path")
    test_validator_rejects_malformed_monitor_path();             print("PASS: validator rejects malformed monitor path")
    print("\nAll tests passed.")
