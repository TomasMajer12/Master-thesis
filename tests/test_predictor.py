"""Tests for ``mnlearn.learning.load_predictor``.

Verifies that the convenience helper produces predictions
bit-for-bit identical to the manual 5-step inference pattern
(rebuild architecture → load state → eval → unary → decode).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

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
from mnlearn.data import build_datasets
from mnlearn.inference import bp_decode, viterbi_decode
from mnlearn.learning import load_predictor, train
from mnlearn.learning.predictor import Predictor
from mnlearn.models import build_model


REPO_ROOT     = Path(__file__).resolve().parent.parent
HMC_BENCHMARK = REPO_ROOT / "benchmarks" / "hmc"


def _smoke_config(output_dir: Path, loss: str = "m3n_hinge") -> Config:
    """Mirror tests/test_runner.py::_smoke_config — chain HMC, 2 epochs."""
    inf = (InferenceCfg(train="viterbi", eval="viterbi")
           if loss == "m3n_hinge"
           else InferenceCfg(train="lp", eval="bp", params={"bp_iters": 5}))
    return Config(
        experiment = ExperimentCfg(name="predictor_smoke", seed=0,
                                   device="cpu", output_dir=str(output_dir)),
        architecture = ArchitectureCfg(
            num_classes = 10,
            backbone    = BackboneCfg(type="config", layers=[
                {"type": "linear", "in_features": 10, "out_features": 10}
            ]),
            graph       = GraphCfg(type="chain", seq_len=5),
            pairwise    = PairwiseCfg(init_scale=0.1),
        ),
        data = DataCfg(
            task="hmc", mode="symbolic",
            paths={"benchmark": str(HMC_BENCHMARK)},
            train_size=20, val_size=10, test_size=10, batch_size=5,
        ),
        training = TrainingCfg(
            loss            = loss,
            inference       = inf,
            optimizer       = OptimizerCfg(type="adam", lr=0.01, weight_decay=0.0),
            num_epochs      = 2,
            scheduler       = SchedulerCfg(type="none"),
            eval_every      = 1,
            early_stopping  = EarlyStoppingCfg(
                monitor="val_metrics.hamming", patience=5, min_delta=0.0),
        ),
        logging = LoggingCfg(verbose=False),
    )


def _manual_predict(run_dir: Path, X: torch.Tensor) -> torch.Tensor:
    """The 5-step pattern the README documents, by hand."""
    from mnlearn.config import load_config
    cfg = load_config(run_dir / "config.yaml")
    model, edges = build_model(cfg.architecture)
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        unary = model.unary(X)
        if cfg.training.inference.eval == "viterbi":
            return viterbi_decode(unary, model.pairwise)
        if cfg.training.inference.eval == "bp":
            # YAML key ``bp_iters`` maps to ``bp_decode``'s ``num_iters`` kwarg.
            kwargs = dict(cfg.training.inference.params or {})
            if "bp_iters" in kwargs:
                kwargs["num_iters"] = kwargs.pop("bp_iters")
            return bp_decode(unary, model.pairwise, edges, **kwargs)
    raise ValueError(f"unknown eval mode {cfg.training.inference.eval}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_predictor_returns_predictor_instance():
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(Path(tmp_str) / "run")
        result = train(cfg)
        pred = load_predictor(result["output_dir"], device="cpu")
        assert isinstance(pred, Predictor)
        assert pred.eval_mode == "viterbi"
        assert hasattr(pred, "model")
        assert hasattr(pred, "edges")


def test_load_predictor_matches_manual_path_viterbi():
    """Helper's output must equal the 5-step manual path on chain (viterbi)."""
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(Path(tmp_str) / "run", loss="m3n_hinge")
        result = train(cfg)
        run_dir = Path(result["output_dir"])

        # Pull a small test batch from the same benchmark.
        X, _ = build_datasets(cfg.data)["test"]
        X_batch = X[:4]

        y_helper = load_predictor(run_dir, device="cpu")(X_batch)
        y_manual = _manual_predict(run_dir, X_batch)

        assert torch.equal(y_helper, y_manual), (
            f"helper vs manual disagree:\n  helper={y_helper}\n  manual={y_manual}"
        )


def test_load_predictor_matches_manual_path_lp_m3n():
    """Same parity, but on the LP-M3N / BP code path."""
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(Path(tmp_str) / "run", loss="lp_m3n")
        result = train(cfg)
        run_dir = Path(result["output_dir"])

        X, _ = build_datasets(cfg.data)["test"]
        X_batch = X[:4]

        y_helper = load_predictor(run_dir, device="cpu")(X_batch)
        y_manual = _manual_predict(run_dir, X_batch)

        assert torch.equal(y_helper, y_manual)


def test_load_predictor_unary_method_returns_per_node_scores():
    """predictor.unary(X) gives access to [B, V, K] intermediate values.

    Shape is [batch, num_nodes, num_classes] where num_nodes comes from
    the input tensor (not the architecture's nominal seq_len, which may
    differ from the benchmark's actual sequence length).
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(Path(tmp_str) / "run")
        result = train(cfg)
        run_dir = Path(result["output_dir"])

        X, _ = build_datasets(cfg.data)["test"]
        pred = load_predictor(run_dir, device="cpu")

        unary = pred.unary(X[:3])
        assert unary.shape == (3, X.shape[1], cfg.architecture.num_classes)


def test_load_predictor_dispatches_correct_decoder():
    """eval_mode reflects training.inference.eval in the YAML."""
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(Path(tmp_str) / "run", loss="m3n_hinge")
        result = train(cfg)
        pred = load_predictor(result["output_dir"], device="cpu")
        assert pred.eval_mode == "viterbi"

    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(Path(tmp_str) / "run", loss="lp_m3n")
        result = train(cfg)
        pred = load_predictor(result["output_dir"], device="cpu")
        assert pred.eval_mode == "bp"


if __name__ == "__main__":
    test_load_predictor_returns_predictor_instance();      print("PASS: load_predictor returns Predictor instance")
    test_load_predictor_matches_manual_path_viterbi();     print("PASS: helper matches manual path (viterbi)")
    test_load_predictor_matches_manual_path_lp_m3n();      print("PASS: helper matches manual path (lp_m3n)")
    test_load_predictor_unary_method_returns_per_node_scores(); print("PASS: predictor.unary returns per-node scores")
    test_load_predictor_dispatches_correct_decoder();      print("PASS: eval_mode dispatches correctly")
    print("\nAll tests passed.")
