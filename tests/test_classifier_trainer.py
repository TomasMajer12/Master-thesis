"""
Tests for ClassifierTrainer.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from mnlearn.learning import ClassifierTrainer


# ---------------------------------------------------------------------------
# Synthetic dataset: K-class blobs in 2D, linearly separable
# ---------------------------------------------------------------------------

def _make_blobs(n_per_class: int = 200, num_classes: int = 3, seed: int = 0):
    """Generate (X, y) with one Gaussian blob per class."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([
        [+3.0,  0.0],
        [-1.5, +2.6],
        [-1.5, -2.6],
    ])[:num_classes]
    Xs, Ys = [], []
    for k in range(num_classes):
        x = centers[k] + 0.4 * torch.randn(n_per_class, 2, generator=g)
        y = torch.full((n_per_class,), k, dtype=torch.long)
        Xs.append(x); Ys.append(y)
    return torch.cat(Xs), torch.cat(Ys)


def _build_model(num_classes: int = 3) -> nn.Module:
    """Tiny MLP, raw-logit output (no softmax — CE applies it)."""
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def test_classifier_trainer_converges_on_separable_blobs():
    torch.manual_seed(0)
    X_train, Y_train = _make_blobs(n_per_class=200, seed=1)
    X_val,   Y_val   = _make_blobs(n_per_class=100, seed=2)

    model = _build_model()
    trainer = ClassifierTrainer(model, lr=0.05, optimizer="adam", device="cpu")

    history = trainer.fit(
        X_train, Y_train, X_val, Y_val,
        config={"num_epochs": 30, "batch_size": 64, "eval_every": 1,
                "patience": 30, "min_delta": 0.0, "verbose": False},
    )

    # Linearly-separable 3-class blobs should be very easy.
    assert history["best_val_error"] < 0.05, (
        f"convergence failed: best_val_error={history['best_val_error']}"
    )
    assert history["best_epoch"] >= 1


# ---------------------------------------------------------------------------
# History / evaluate / predict shapes
# ---------------------------------------------------------------------------

def test_history_dict_has_expected_keys():
    torch.manual_seed(0)
    X, Y = _make_blobs(n_per_class=50)
    model = _build_model()
    trainer = ClassifierTrainer(model, lr=0.01, device="cpu")

    history = trainer.fit(
        X, Y, X, Y,
        config={"num_epochs": 3, "batch_size": 32, "eval_every": 1,
                "patience": 3, "min_delta": 0.0, "verbose": False},
    )

    expected = {"epoch", "train_loss", "train_error", "val_error",
                "val_loss", "lr", "best_val_error", "best_epoch",
                "early_stopped"}
    assert expected.issubset(history.keys())
    assert len(history["epoch"]) == len(history["val_error"])
    assert all(isinstance(e, int) for e in history["epoch"])


def test_evaluate_returns_error_and_loss():
    torch.manual_seed(0)
    X, Y = _make_blobs(n_per_class=50)
    trainer = ClassifierTrainer(_build_model(), lr=0.01, device="cpu")
    metrics = trainer.evaluate(X, Y)
    assert set(metrics.keys()) == {"error", "loss"}
    assert 0.0 <= metrics["error"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_predict_returns_long_tensor_of_correct_shape():
    torch.manual_seed(0)
    X, _ = _make_blobs(n_per_class=20)
    trainer = ClassifierTrainer(_build_model(), lr=0.01, device="cpu")
    preds = trainer.predict(X)
    assert preds.shape == (X.shape[0],)
    assert preds.dtype == torch.long
    assert (preds >= 0).all() and (preds < 3).all()


def test_predict_proba_returns_normalised_distribution():
    torch.manual_seed(0)
    X, _ = _make_blobs(n_per_class=20)
    trainer = ClassifierTrainer(_build_model(), lr=0.01, device="cpu")
    probs = trainer.predict_proba(X)
    assert probs.shape == (X.shape[0], 3)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        ClassifierTrainer(_build_model(), optimizer="rmsprop")


def test_sgd_optimizer_dispatch():
    trainer = ClassifierTrainer(_build_model(), lr=0.01, optimizer="sgd",
                                device="cpu")
    assert isinstance(trainer.optimizer, torch.optim.SGD)


# ---------------------------------------------------------------------------
# Best-state restoration
# ---------------------------------------------------------------------------

def test_fit_restores_best_state():
    """After fit(), model params should equal those of the best-val-error epoch.

    We can't snapshot the best epoch's params from outside (the trainer
    holds them privately), so we verify the post-fit val_error matches
    history['best_val_error'] up to evaluation noise (< 1e-6).
    """
    torch.manual_seed(0)
    X_train, Y_train = _make_blobs(n_per_class=100, seed=1)
    X_val,   Y_val   = _make_blobs(n_per_class=100, seed=2)

    model = _build_model()
    trainer = ClassifierTrainer(model, lr=0.05, device="cpu")

    history = trainer.fit(
        X_train, Y_train, X_val, Y_val,
        config={"num_epochs": 10, "batch_size": 32, "eval_every": 1,
                "patience": 10, "min_delta": 0.0, "verbose": False},
    )

    final = trainer.evaluate(X_val, Y_val)
    assert math.isclose(final["error"], history["best_val_error"], abs_tol=1e-6)


if __name__ == "__main__":
    test_classifier_trainer_converges_on_separable_blobs(); print("PASS: classifier trainer converges on separable blobs")
    test_history_dict_has_expected_keys();                  print("PASS: history dict has expected keys")
    test_evaluate_returns_error_and_loss();                 print("PASS: evaluate returns error and loss")
    test_predict_returns_long_tensor_of_correct_shape();    print("PASS: predict returns long tensor of correct shape")
    test_predict_proba_returns_normalised_distribution();   print("PASS: predict proba returns normalised distribution")
    test_unknown_optimizer_raises();                        print("PASS: unknown optimizer raises")
    test_sgd_optimizer_dispatch();                          print("PASS: sgd optimizer dispatch")
    test_fit_restores_best_state();                         print("PASS: fit restores best state")
    print("\nAll tests passed.")
