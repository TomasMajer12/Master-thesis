"""Tests for the shared runtime helpers (mnlearn.learning.runtime).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mnlearn.config.schema import (
    ArchitectureCfg,
    BackboneCfg,
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
from mnlearn.learning.runtime import (
    json_default,
    resolve_device,
    resolve_output_dir,
    save_artifacts,
    set_seed,
    with_output_dir,
)


# ---------------------------------------------------------------------------
# Fixture: a minimal Config to drive save_artifacts / with_output_dir
# ---------------------------------------------------------------------------

def _minimal_config(output_dir: str = "") -> Config:
    return Config(
        experiment   = ExperimentCfg(name="t", seed=0, device="cpu",
                                     output_dir=output_dir),
        architecture = ArchitectureCfg(
            num_classes = 3,
            backbone    = BackboneCfg(type="config", layers=[
                {"type": "linear", "in_features": 3, "out_features": 3},
            ]),
            graph       = GraphCfg(type="chain", seq_len=4),
            pairwise    = PairwiseCfg(init_scale=0.1),
        ),
        data = DataCfg(
            task="hmc", mode="symbolic", paths={},
            train_size=1, val_size=1, test_size=1, batch_size=1,
        ),
        training = TrainingCfg(
            loss="m3n_hinge",
            inference=InferenceCfg(train="viterbi", eval="viterbi"),
            optimizer=OptimizerCfg(type="adam", lr=0.01),
            num_epochs=1,
        ),
        logging = LoggingCfg(verbose=False),
    )


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_picks_available_backend():
    """auto -> cuda when available, else cpu. Test that *something* sensible comes back."""
    dev = resolve_device("auto")
    assert dev.type in {"cpu", "cuda"}
    assert dev.type == ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# resolve_output_dir
# ---------------------------------------------------------------------------

def test_resolve_output_dir_uses_override(tmp_path):
    target = tmp_path / "explicit_dir"
    out = resolve_output_dir(name="ignored", override=str(target))
    assert out == target
    assert out.is_dir()


def test_resolve_output_dir_defaults_to_results_subdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = resolve_output_dir(name="myrun", override="")
    assert out == Path("results") / "myrun"
    assert out.is_dir()


# ---------------------------------------------------------------------------
# with_output_dir
# ---------------------------------------------------------------------------

def test_with_output_dir_returns_copy_with_path_set():
    cfg = _minimal_config(output_dir="")
    cfg2 = with_output_dir(cfg, "results/x")

    assert cfg2 is not cfg
    assert cfg2.experiment.output_dir == "results/x"
    # Other fields preserved.
    assert cfg2.experiment.name == cfg.experiment.name
    assert cfg2.training.loss   == cfg.training.loss


# ---------------------------------------------------------------------------
# save_artifacts
# ---------------------------------------------------------------------------

def test_save_artifacts_writes_full_set(tmp_path):
    cfg = _minimal_config(output_dir=str(tmp_path))
    model = torch.nn.Linear(3, 3)
    history = {"epoch": [1, 2], "train_loss": [0.5, 0.3]}
    test_metrics = {"hamming": 0.1, "zero_one": 0.2}

    save_artifacts(cfg, tmp_path, model, history, test_metrics)

    for fname in ("config.yaml", "history.json", "results.json", "model.pt"):
        assert (tmp_path / fname).is_file(), f"missing {fname}"

    # config.yaml round-trips back through asdict comparison.
    on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert on_disk == asdict(cfg)

    # results.json matches the dict we passed.
    assert json.loads((tmp_path / "results.json").read_text()) == test_metrics


def test_save_artifacts_serialises_numpy_and_tensor_values(tmp_path):
    """history may contain numpy scalars / tensors; json_default handles them."""
    cfg = _minimal_config(output_dir=str(tmp_path))
    model = torch.nn.Linear(3, 3)
    history = {
        "np_float":   np.float64(0.5),
        "np_int":     np.int64(7),
        "tensor":     torch.tensor([1.0, 2.0]),
        "plain_list": [1, 2, 3],
    }
    save_artifacts(cfg, tmp_path, model, history, test_metrics={"hamming": 0.0})
    on_disk = json.loads((tmp_path / "history.json").read_text())
    assert on_disk["np_float"] == 0.5
    assert on_disk["np_int"]   == 7
    assert on_disk["tensor"]   == [1.0, 2.0]


# ---------------------------------------------------------------------------
# json_default
# ---------------------------------------------------------------------------

def test_json_default_handles_numpy_scalars():
    assert json_default(np.float32(1.5)) == 1.5
    assert json_default(np.int32(3))     == 3


def test_json_default_handles_torch_tensor():
    assert json_default(torch.tensor([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]


def test_json_default_raises_on_unsupported_type():
    class Weird:
        pass
    with pytest.raises(TypeError):
        json_default(Weird())


if __name__ == "__main__":
    import os, tempfile, contextlib

    class _Monkeypatch:
        """Minimal stand-in for pytest's monkeypatch fixture, supporting only
        the .chdir() and .setattr() methods used in this file."""
        def __init__(self):
            self._undo_stack = []
        def chdir(self, path):
            prev = Path.cwd()
            os.chdir(path)
            self._undo_stack.append(lambda: os.chdir(prev))
        def undo(self):
            while self._undo_stack:
                self._undo_stack.pop()()

    @contextlib.contextmanager
    def _fixtures(*, want_monkeypatch=False):
        with tempfile.TemporaryDirectory() as d:
            mp = _Monkeypatch() if want_monkeypatch else None
            try:
                yield (Path(d), mp) if want_monkeypatch else Path(d)
            finally:
                if mp is not None:
                    mp.undo()

    test_set_seed_makes_torch_deterministic();                print("PASS: set seed makes torch deterministic")
    test_set_seed_makes_numpy_deterministic();                print("PASS: set seed makes numpy deterministic")
    test_resolve_device_explicit_cpu();                       print("PASS: resolve device explicit cpu")
    test_resolve_device_auto_picks_available_backend();       print("PASS: resolve device auto picks available backend")

    with _fixtures() as tp:
        test_resolve_output_dir_uses_override(tp);            print("PASS: resolve output dir uses override")
    with _fixtures(want_monkeypatch=True) as (tp, mp):
        test_resolve_output_dir_defaults_to_results_subdir(tp, mp); print("PASS: resolve output dir defaults to results subdir")

    test_with_output_dir_returns_copy_with_path_set();        print("PASS: with output dir returns copy with path set")

    with _fixtures() as tp:
        test_save_artifacts_writes_full_set(tp);              print("PASS: save artifacts writes full set")
    with _fixtures() as tp:
        test_save_artifacts_serialises_numpy_and_tensor_values(tp); print("PASS: save artifacts serialises numpy and tensor values")

    test_json_default_handles_numpy_scalars();                print("PASS: json default handles numpy scalars")
    test_json_default_handles_torch_tensor();                 print("PASS: json default handles torch tensor")
    test_json_default_raises_on_unsupported_type();           print("PASS: json default raises on unsupported type")
    print("\nAll tests passed.")
