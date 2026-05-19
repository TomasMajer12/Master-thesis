"""Tests for mnlearn.experiments (collect + plot helpers)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mnlearn.experiments import (  # noqa: E402
    collect_results,
    parse_run_name,
    plot_learning_curve,
    plot_run_history,
)


# ---------------------------------------------------------------------------
# Fixtures: synthesise a results/ tree the way runner.py would
# ---------------------------------------------------------------------------

def _write_run(root: Path, name: str, results: dict, history: dict | None = None) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results))
    if history is not None:
        (run_dir / "history.json").write_text(json.dumps(history))
    return run_dir


def _unified_history(epochs: list[int], val_zero_one: list[float]) -> dict:
    n = len(epochs)
    return {
        "loss":               "lp_m3n",
        "epoch":              epochs,
        "epoch_seconds":      [0.5] * n,
        "lr":                 [1e-3] * n,
        "train_loss":         [0.5 - 0.05 * i for i in range(n)],
        "train_metrics":      [{"hamming": 0.4 - 0.05 * i,
                                "zero_one": 0.9 - 0.05 * i}
                               for i in range(n)],
        "val_metrics":        [{"hamming": v + 0.02, "zero_one": v}
                               for v in val_zero_one],
        "diagnostics":        [{"phi_norm": 0.1 * i,
                                "pairwise_diag_mean":     -0.1,
                                "pairwise_off_diag_mean":  0.05}
                               for i in range(n)],
        "monitor":            "val_metrics.zero_one",
        "best_epoch":         epochs[-1],
        "best_monitor_value": val_zero_one[-1],
        "early_stopped":      False,
    }


# ---------------------------------------------------------------------------
# parse_run_name
# ---------------------------------------------------------------------------

def test_parse_run_name_extracts_n_and_seed():
    assert parse_run_name("lpm3n_visual_sudoku_n1000_seed3") == {"n": 1000, "seed": 3}


def test_parse_run_name_handles_lambda_and_floats():
    out = parse_run_name("lpm3n_visual_sudoku_n100_lambda0.001_seed0")
    assert out["n"] == 100
    assert out["seed"] == 0
    assert out["lambda"] == pytest.approx(0.001)


def test_parse_run_name_returns_empty_for_unrecognised():
    assert parse_run_name("custom_baseline_run") == {}


def test_parse_run_name_distinguishes_weight_decay_phi():
    out = parse_run_name("lpm3n_n100_weight_decay_phi0.01_seed1")
    assert out["weight_decay_phi"] == pytest.approx(0.01)
    assert "weight_decay" not in out  # must NOT also match the shorter key


# ---------------------------------------------------------------------------
# collect_results
# ---------------------------------------------------------------------------

def test_collect_results_returns_one_row_per_run(tmp_path):
    _write_run(tmp_path, "lpm3n_n100_seed0",
               results={"hamming": 0.3, "zero_one": 0.85},
               history=_unified_history([1, 2, 3], [0.9, 0.85, 0.85]))
    _write_run(tmp_path, "lpm3n_n100_seed1",
               results={"hamming": 0.32, "zero_one": 0.88},
               history=_unified_history([1, 2, 3], [0.92, 0.88, 0.88]))

    df = collect_results(str(tmp_path / "lpm3n_*"))
    assert len(df) == 2
    assert set(df["seed"]) == {0, 1}
    assert (df["n"] == 100).all()
    assert "hamming" in df.columns
    assert "zero_one" in df.columns


def test_collect_results_propagates_history_summary(tmp_path):
    _write_run(tmp_path, "run_n50_seed0",
               results={"hamming": 0.5, "zero_one": 0.95},
               history=_unified_history([1, 2], [0.95, 0.95]))
    df = collect_results(str(tmp_path / "*"))
    assert df.iloc[0]["loss"] == "lp_m3n"
    assert df.iloc[0]["best_epoch"] == 2
    assert df.iloc[0]["final_epoch"] == 2
    assert df.iloc[0]["total_seconds"] == pytest.approx(1.0)


def test_collect_results_skips_empty_dirs(tmp_path):
    """A directory with neither results.json nor history.json is dropped."""
    (tmp_path / "empty").mkdir()
    _write_run(tmp_path, "good", results={"hamming": 0.0, "zero_one": 0.0})
    df = collect_results(str(tmp_path / "*"))
    assert len(df) == 1
    assert df.iloc[0]["name"] == "good"


def test_collect_results_handles_classifier_history(tmp_path):
    """ClassifierTrainer history (flat keys) round-trips through collect_results."""
    _write_run(
        tmp_path, "baseline_n1000_seed0",
        results={"zero_one": 0.7, "solver_feasibility": 0.85, "ocr_clue_error": 0.12},
        history={
            "epoch":          [1, 2],
            "train_loss":     [0.5, 0.3],
            "train_error":    [0.4, 0.2],
            "val_error":      [0.45, 0.25],
            "val_loss":       [0.6, 0.4],
            "lr":             [1e-3, 1e-3],
            "best_val_error": 0.25,
            "best_epoch":     2,
            "early_stopped":  False,
        },
    )
    df = collect_results(str(tmp_path / "*"))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["best_val_error"] == pytest.approx(0.25)
    assert row["best_epoch"] == 2
    assert row["solver_feasibility"] == pytest.approx(0.85)


def test_collect_results_returns_empty_df_when_no_matches(tmp_path):
    df = collect_results(str(tmp_path / "definitely_no_match_*"))
    assert df.empty


def test_collect_results_handles_multiple_patterns(tmp_path):
    _write_run(tmp_path / "exp_a", "run_n10_seed0", results={"zero_one": 0.1})
    _write_run(tmp_path / "exp_b", "run_n20_seed0", results={"zero_one": 0.2})
    df = collect_results([str(tmp_path / "exp_a" / "*"),
                          str(tmp_path / "exp_b" / "*")])
    assert len(df) == 2
    assert set(df["n"]) == {10, 20}


# ---------------------------------------------------------------------------
# plot_learning_curve
# ---------------------------------------------------------------------------

def _learning_curve_df() -> pd.DataFrame:
    rows = []
    for n in (100, 1000, 5000):
        for seed in range(3):
            rows.append({
                "n":        n,
                "seed":     seed,
                "zero_one": 0.9 - 0.0001 * n + 0.01 * seed,
                "hamming":  0.4 - 0.00005 * n,
                "loss":     "lp_m3n",
            })
    return pd.DataFrame(rows)


def test_plot_learning_curve_returns_axes_and_plots():
    df = _learning_curve_df()
    ax = plot_learning_curve(df, x="n", y="zero_one")
    assert isinstance(ax, plt.Axes)
    # Three N values * one curve = at least three points on the line.
    lines = ax.get_lines()
    assert any(len(line.get_xdata()) == 3 for line in lines)
    plt.close(ax.figure)


def test_plot_learning_curve_with_hue_makes_one_curve_per_group():
    df = _learning_curve_df().copy()
    df.loc[df.index % 2 == 0, "loss"] = "m3n_hinge"
    ax = plot_learning_curve(df, x="n", y="zero_one", hue="loss")
    # Two distinct labels in the legend.
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.texts) == 2
    plt.close(ax.figure)


def test_plot_learning_curve_rejects_missing_column():
    df = _learning_curve_df()
    with pytest.raises(KeyError):
        plot_learning_curve(df, x="nonexistent", y="zero_one")
    with pytest.raises(KeyError):
        plot_learning_curve(df, x="n", y="nope")


def test_plot_learning_curve_log_x_sets_log_scale():
    df = _learning_curve_df()
    ax = plot_learning_curve(df, x="n", y="zero_one", log_x=True)
    assert ax.get_xscale() == "log"
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# plot_run_history
# ---------------------------------------------------------------------------

def test_plot_run_history_unified_history():
    history = _unified_history([1, 2, 3, 4, 5], [0.9, 0.8, 0.7, 0.65, 0.6])
    ax = plot_run_history(history)
    assert isinstance(ax, plt.Axes)
    # Curves drawn for at least val_metrics.{hamming, zero_one} = 2 lines
    # plus the best_epoch axvline.
    assert len(ax.get_lines()) >= 2
    plt.close(ax.figure)


def test_plot_run_history_classifier_history():
    history = {
        "epoch":          [1, 2, 3],
        "train_loss":     [0.6, 0.4, 0.3],
        "train_error":    [0.5, 0.3, 0.2],
        "val_error":      [0.55, 0.35, 0.25],
        "val_loss":       [0.65, 0.45, 0.35],
        "lr":             [1e-3] * 3,
        "best_epoch":     3,
        "early_stopped":  False,
    }
    ax = plot_run_history(history)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_plot_run_history_raises_on_empty_history():
    with pytest.raises(ValueError):
        plot_run_history({"epoch": []})


if __name__ == "__main__":
    from _fixtures import fixtures
    test_parse_run_name_extracts_n_and_seed();                     print("PASS: parse run name extracts n and seed")
    test_parse_run_name_handles_lambda_and_floats();               print("PASS: parse run name handles lambda and floats")
    test_parse_run_name_returns_empty_for_unrecognised();          print("PASS: parse run name returns empty for unrecognised")
    test_parse_run_name_distinguishes_weight_decay_phi();          print("PASS: parse run name distinguishes weight decay phi")
    with fixtures() as tp: test_collect_results_returns_one_row_per_run(tp);                print("PASS: collect results returns one row per run")
    with fixtures() as tp: test_collect_results_propagates_history_summary(tp);             print("PASS: collect results propagates history summary")
    with fixtures() as tp: test_collect_results_skips_empty_dirs(tp);                       print("PASS: collect results skips empty dirs")
    with fixtures() as tp: test_collect_results_handles_classifier_history(tp);             print("PASS: collect results handles classifier history")
    with fixtures() as tp: test_collect_results_returns_empty_df_when_no_matches(tp);       print("PASS: collect results returns empty df when no matches")
    with fixtures() as tp: test_collect_results_handles_multiple_patterns(tp);              print("PASS: collect results handles multiple patterns")
    test_plot_learning_curve_returns_axes_and_plots();             print("PASS: plot learning curve returns axes and plots")
    test_plot_learning_curve_with_hue_makes_one_curve_per_group(); print("PASS: plot learning curve with hue makes one curve per group")
    test_plot_learning_curve_rejects_missing_column();             print("PASS: plot learning curve rejects missing column")
    test_plot_learning_curve_log_x_sets_log_scale();               print("PASS: plot learning curve log x sets log scale")
    test_plot_run_history_unified_history();                       print("PASS: plot run history unified history")
    test_plot_run_history_classifier_history();                    print("PASS: plot run history classifier history")
    test_plot_run_history_raises_on_empty_history();               print("PASS: plot run history raises on empty history")
    print("\nAll tests passed.")
