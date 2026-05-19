"""Smoke tests for the unified YAML-driven training entry point.

"""

import json
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
from mnlearn.learning import train


REPO_ROOT     = Path(__file__).resolve().parent.parent
HMC_BENCHMARK = REPO_ROOT / "benchmarks" / "hmc"


def _smoke_config(output_dir: Path,
                  num_states: int = 10,
                  seq_len: int = 5) -> Config:
    """Tiny symbolic-HMC config that exercises every builder."""
    return Config(
        experiment = ExperimentCfg(
            name       = "smoke",
            seed       = 0,
            device     = "cpu",
            output_dir = str(output_dir),
        ),
        architecture = ArchitectureCfg(
            num_classes = num_states,
            backbone    = BackboneCfg(type="config", layers=[
                {"type": "linear", "in_features": num_states, "out_features": num_states}
            ]),
            graph       = GraphCfg(type="chain", seq_len=seq_len),
            pairwise    = PairwiseCfg(init_scale=0.1),
        ),
        data = DataCfg(
            task       = "hmc",
            mode       = "symbolic",
            paths      = {"benchmark": str(HMC_BENCHMARK)},
            train_size = 20,
            val_size   = 10,
            test_size  = 10,
            batch_size = 5,
        ),
        training = TrainingCfg(
            loss            = "m3n_hinge",
            inference       = InferenceCfg(train="viterbi", eval="viterbi"),
            optimizer       = OptimizerCfg(type="adam", lr=0.01, weight_decay=0.0),
            num_epochs      = 2,
            scheduler       = SchedulerCfg(type="none"),
            eval_every      = 1,
            early_stopping  = EarlyStoppingCfg(
                monitor="val_metrics.hamming", patience=5, min_delta=0.0,
            ),
        ),
        logging = LoggingCfg(verbose=False),
    )


def test_train_returns_result_dict_with_expected_keys():
    """train(cfg) returns the full result dict the runner promises."""
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(output_dir=Path(tmp_str) / "run")
        result = train(cfg)

    assert set(result.keys()) == {
        "config", "history", "test_metrics", "model", "trainer", "output_dir",
    }
    assert result["config"].experiment.name == "smoke"
    assert isinstance(result["history"], dict)
    assert isinstance(result["test_metrics"], dict)


def test_train_writes_all_artifacts():
    """train() should create config.yaml, history.json, results.json, model.pt."""
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(output_dir=Path(tmp_str) / "run")
        result = train(cfg)
        out = Path(result["config"].experiment.output_dir)

        for name in ("config.yaml", "history.json", "results.json", "model.pt"):
            assert (out / name).is_file(), f"Missing artifact: {name}"

        # results.json content matches the returned test_metrics.
        with (out / "results.json").open() as f:
            saved = json.load(f)
        assert saved == result["test_metrics"]


def test_train_history_is_rich_and_per_epoch():
    """history has parallel arrays and dicts; metric keys are task-driven."""
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(output_dir=Path(tmp_str) / "run")
        result = train(cfg)
        history = result["history"]

    assert history["loss"] == "m3n_hinge"
    n = len(history["epoch"])
    assert n >= 1
    # Parallel arrays.
    for k in ("epoch_seconds", "lr", "train_loss",
              "train_metrics", "val_metrics", "diagnostics"):
        assert len(history[k]) == n, f"{k} length mismatch with epoch"

    # Per-split metric dicts have the expected keys.
    assert {"hamming", "zero_one"}.issubset(history["train_metrics"][-1].keys())
    assert {"hamming", "zero_one"}.issubset(history["val_metrics"][-1].keys())
    # Pairwise diagnostics always present; phi_norm absent for m3n_hinge.
    assert "pairwise_diag_mean" in history["diagnostics"][-1]
    assert "phi_norm" not in history["diagnostics"][-1]

    # Best-epoch summary populated.
    assert history["best_epoch"] >= 1
    assert history["monitor"] == "val_metrics.hamming"


def test_train_returns_test_metrics_in_unit_interval():
    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(output_dir=Path(tmp_str) / "run")
        result = train(cfg)
        m = result["test_metrics"]
        assert 0.0 <= m["hamming"]  <= 1.0
        assert 0.0 <= m["zero_one"] <= 1.0


def test_dumped_config_round_trips():
    """The dumped config.yaml should reload back into an equivalent Config."""
    from mnlearn.config import load_config

    with tempfile.TemporaryDirectory() as tmp_str:
        cfg = _smoke_config(output_dir=Path(tmp_str) / "run")
        result = train(cfg)
        out_path = Path(result["config"].experiment.output_dir) / "config.yaml"
        cfg2 = load_config(out_path)

        assert cfg2.architecture.num_classes == cfg.architecture.num_classes
        assert cfg2.training.optimizer.lr     == cfg.training.optimizer.lr
        assert cfg2.training.loss             == cfg.training.loss
        # output_dir was resolved at run time and persisted into the dump.
        assert cfg2.experiment.output_dir == str(Path(cfg.experiment.output_dir))


def test_edges_override_replaces_yaml_graph():
    """Passing edges_override should bypass the graph builder."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        cfg = _smoke_config(output_dir=tmp / "run")
        # Empty edge tensor: M3N still works (pairwise term contributes 0).
        custom_edges = torch.zeros((0, 2), dtype=torch.long)
        result = train(cfg, edges_override=custom_edges)
        assert (tmp / "run" / "model.pt").is_file()
        # The trainer was constructed with the override, not the YAML graph.
        assert result["trainer"].edges.shape == (0, 2)


def test_train_smoke_lp_m3n():
    """End-to-end LP-M3N smoke test on symbolic HMC.

    Mirrors the m3n_hinge smoke tests above but exercises the loss that
    produces every Sudoku result in Chapter 5: phi-bank allocation,
    LP-relaxed loss, BP eval (chain --> BP exact on tree). Catches any
    regression in the train --> artifact pipeline for the principal
    training path used on cyclic graphs.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        base = _smoke_config(output_dir=Path(tmp_str) / "run")
        cfg = Config(
            experiment   = base.experiment,
            architecture = base.architecture,
            data         = base.data,
            training     = TrainingCfg(
                loss            = "lp_m3n",
                inference       = InferenceCfg(train="lp", eval="bp",
                                               params={"bp_iters": 5}),
                optimizer       = OptimizerCfg(
                    type="adam", lr=0.01, weight_decay=0.0,
                    weight_decay_phi=0.0, phi_init_std=0.0,
                ),
                num_epochs      = 2,
                scheduler       = SchedulerCfg(type="none"),
                eval_every      = 1,
                early_stopping  = EarlyStoppingCfg(
                    monitor="val_metrics.hamming", patience=5, min_delta=0.0,
                ),
            ),
            logging      = base.logging,
        )
        result = train(cfg)

        # All artifacts written.
        out = Path(result["config"].experiment.output_dir)
        for name in ("config.yaml", "history.json", "results.json", "model.pt"):
            assert (out / name).is_file(), f"Missing artifact: {name}"

    # LP-M3N-specific assertions.
    history = result["history"]
    assert history["loss"] == "lp_m3n"
    # phi_norm must appear in diagnostics for LP-M3N (absent for m3n_hinge).
    assert "phi_norm" in history["diagnostics"][-1]
    # The phi-bank was allocated and persisted on the trainer.
    assert result["trainer"].phi_bank is not None
    assert len(result["trainer"].phi_bank) == cfg.data.train_size

    # Test metrics still in [0, 1].
    m = result["test_metrics"]
    assert 0.0 <= m["hamming"]  <= 1.0
    assert 0.0 <= m["zero_one"] <= 1.0


if __name__ == "__main__":
    test_train_returns_result_dict_with_expected_keys(); print("PASS: train returns result dict with expected keys")
    test_train_writes_all_artifacts();                   print("PASS: train writes all artifacts")
    test_train_history_is_rich_and_per_epoch();          print("PASS: train history is rich and per epoch")
    test_train_returns_test_metrics_in_unit_interval();  print("PASS: train returns test metrics in unit interval")
    test_dumped_config_round_trips();                    print("PASS: dumped config round trips")
    test_edges_override_replaces_yaml_graph();           print("PASS: edges override replaces yaml graph")
    test_train_smoke_lp_m3n();                           print("PASS: train smoke lp_m3n")
    print("\nAll tests passed.")
