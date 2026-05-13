"""Tests for the learning module: structured loss, evaluation, unified Trainer."""

import torch

from mnlearn.config.schema import (
    EarlyStoppingCfg,
    InferenceCfg,
    OptimizerCfg,
    SchedulerCfg,
    TrainingCfg,
)
from mnlearn.inference import loss_augmented_viterbi
from mnlearn.learning import (
    Trainer,
    hamming_loss,
    structured_hinge_loss,
    zero_one_loss,
)
from mnlearn.models import ConfigBackbone, M3N, chain_edges


def _linear_backbone(input_dim: int, num_classes: int) -> ConfigBackbone:
    """Replacement for the deleted LinearBackbone convenience class."""
    return ConfigBackbone([{"type": "linear",
                            "in_features": input_dim,
                            "out_features": num_classes}])


def _hinge_training_cfg(
    *,
    lr: float = 0.05,
    num_epochs: int = 10,
    eval_every: int = 5,
    patience: int = 100,
) -> TrainingCfg:
    """A minimal m3n_hinge TrainingCfg for unit-test Trainer construction."""
    return TrainingCfg(
        loss            = "m3n_hinge",
        inference       = InferenceCfg(train="viterbi", eval="viterbi"),
        optimizer       = OptimizerCfg(type="adam", lr=lr, weight_decay=0.0),
        num_epochs      = num_epochs,
        scheduler       = SchedulerCfg(type="none"),
        eval_every      = eval_every,
        early_stopping  = EarlyStoppingCfg(
            monitor="val_metrics.hamming", patience=patience, min_delta=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def test_hamming_loss():
    y_pred = torch.tensor([[0, 1, 2], [1, 1, 1]])
    y_true = torch.tensor([[0, 1, 0], [1, 0, 1]])
    # 2 wrong out of 6 = 0.333...
    loss = hamming_loss(y_pred, y_true)
    assert abs(loss - 2/6) < 1e-6, f"Expected {2/6}, got {loss}"
    assert hamming_loss(y_true, y_true) == 0.0


def test_zero_one_loss():
    y_pred = torch.tensor([[0, 1, 2], [1, 0, 1]])
    y_true = torch.tensor([[0, 1, 2], [1, 0, 1]])
    assert zero_one_loss(y_pred, y_true) == 0.0  # both samples correct

    y_pred = torch.tensor([[0, 1, 2], [1, 1, 1]])
    y_true = torch.tensor([[0, 1, 2], [1, 0, 1]])
    # First sample correct, second wrong -> 0.5
    assert abs(zero_one_loss(y_pred, y_true) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Structured hinge loss
# ---------------------------------------------------------------------------

def test_loss_zero_when_perfect():
    """If y_true is already the highest-scoring labeling, loss should be 0."""
    torch.manual_seed(0)
    K, T, batch = 3, 5, 2

    backbone = _linear_backbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    y_true = torch.randint(0, K, (batch, T))

    with torch.no_grad():
        model.backbone.net[0].weight.zero_()
        model.backbone.net[0].bias.zero_()
        model.pairwise.zero_()

    unary = torch.zeros(batch, T, K)
    unary.scatter_(2, y_true.unsqueeze(-1), 100.0)

    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    assert loss.item() == 0.0, f"Expected 0, got {loss.item()}"


def test_loss_nonnegative():
    """Structured hinge loss must always be >= 0."""
    torch.manual_seed(42)
    K, T, batch = 5, 10, 4

    backbone = _linear_backbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    x = torch.randn(batch, T, K)
    y_true = torch.randint(0, K, (batch, T))
    unary = model.unary(x)

    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    assert loss.item() >= 0


def test_loss_gradient_flow():
    """Gradients must flow from loss back to model parameters."""
    torch.manual_seed(0)
    K, T, batch = 4, 6, 3

    backbone = _linear_backbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    x = torch.randn(batch, T, K)
    y_true = torch.randint(0, K, (batch, T))

    unary = model.unary(x)
    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_loss_includes_hamming():
    """Loss is F(y*) + Δ(y*, y_true) - F(y_true), normalised by T."""
    torch.manual_seed(7)
    K, T, batch = 3, 4, 1

    backbone = _linear_backbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    x = torch.randn(batch, T, K)
    y_true = torch.randint(0, K, (batch, T))
    unary = model.unary(x)

    y_star = loss_augmented_viterbi(unary, model.pairwise, y_true)
    with torch.no_grad():
        s_star = model.score(unary, y_star, edges)
        s_true = model.score(unary, y_true, edges)
        ham = (y_star != y_true).float().mean(dim=1)
        expected = torch.clamp(s_star + ham - s_true, min=0.0).mean().item()

    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    assert abs(loss.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# Unified Trainer (m3n_hinge path)
# ---------------------------------------------------------------------------

def test_trainer_smoke_m3n_hinge():
    """Trainer should run an m3n_hinge fit without errors and produce a history."""
    torch.manual_seed(42)
    K, T = 5, 8
    N_train, N_val = 20, 10

    backbone = _linear_backbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    trainer = Trainer(
        model=model, edges=edges,
        cfg=_hinge_training_cfg(num_epochs=10, eval_every=5),
        n_train=N_train, device=torch.device("cpu"),
    )

    train_x = torch.randn(N_train, T, K)
    train_y = torch.randint(0, K, (N_train, T))
    val_x   = torch.randn(N_val, T, K)
    val_y   = torch.randint(0, K, (N_val, T))

    history = trainer.fit(
        train_data=(train_x, train_y), val_data=(val_x, val_y),
        num_epochs=10, batch_size=10, eval_every=5,
        monitor="val_metrics.hamming",
        patience=100, min_delta=0.0, verbose=False,
    )

    # New rich history shape:
    assert history["task"] == "m3n_hinge"
    assert len(history["epoch"]) > 0
    assert len(history["train_metrics"]) == len(history["epoch"])
    assert "hamming"  in history["val_metrics"][-1]
    assert "zero_one" in history["val_metrics"][-1]
    assert history["best_monitor_value"] < float("inf")
    assert history["best_epoch"] >= 1
    # Diagnostics include pairwise stats; phi_norm absent for m3n_hinge.
    assert "pairwise_diag_mean" in history["diagnostics"][-1]
    assert "phi_norm"     not in history["diagnostics"][-1]


# ---------------------------------------------------------------------------
# Unified Trainer (lp_m3n path)
# ---------------------------------------------------------------------------

def _lp_training_cfg(num_epochs: int = 5) -> TrainingCfg:
    return TrainingCfg(
        loss            = "lp_m3n",
        inference       = InferenceCfg(train="lp", eval="viterbi"),
        optimizer       = OptimizerCfg(type="adam", lr=0.05, weight_decay=0.0,
                                       weight_decay_phi=0.0, phi_init_std=0.0),
        num_epochs      = num_epochs,
        scheduler       = SchedulerCfg(type="none"),
        eval_every      = 1,
        early_stopping  = EarlyStoppingCfg(
            monitor="val_metrics.hamming", patience=100, min_delta=0.0,
        ),
    )


def test_trainer_smoke_lp_m3n():
    """LP-M3N path: phi bank built; diagnostics expose phi_norm."""
    torch.manual_seed(42)
    K, T = 4, 6
    N_train, N_val = 8, 4

    backbone = _linear_backbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    trainer = Trainer(
        model=model, edges=edges,
        cfg=_lp_training_cfg(num_epochs=3),
        n_train=N_train, device=torch.device("cpu"),
    )
    assert trainer.phi_bank is not None
    assert len(trainer.phi_bank) == N_train

    train_x = torch.randn(N_train, T, K)
    train_y = torch.randint(0, K, (N_train, T))
    val_x   = torch.randn(N_val, T, K)
    val_y   = torch.randint(0, K, (N_val, T))

    history = trainer.fit(
        train_data=(train_x, train_y), val_data=(val_x, val_y),
        num_epochs=3, batch_size=4, eval_every=1,
        monitor="val_metrics.hamming",
        patience=10, min_delta=0.0, verbose=False,
    )

    assert history["task"] == "lp_m3n"
    assert len(history["epoch"]) >= 1
    # phi_norm must appear in LP-M3N diagnostics.
    assert "phi_norm" in history["diagnostics"][-1]


# ---------------------------------------------------------------------------
# lr_phi: separate learning rate for the phi parameter group
# ---------------------------------------------------------------------------

def test_lp_m3n_optimizer_has_two_param_groups_with_separate_lrs():
    """When lr_phi > 0, the phi group uses lr_phi; the model group uses lr."""
    torch.manual_seed(0)
    cfg = TrainingCfg(
        loss            = "lp_m3n",
        inference       = InferenceCfg(train="lp", eval="viterbi"),
        optimizer       = OptimizerCfg(type="adam", lr=0.001, weight_decay=0.0,
                                       weight_decay_phi=0.0, phi_init_std=0.0,
                                       lr_phi=0.05),
        num_epochs      = 1,
        scheduler       = SchedulerCfg(type="none"),
        eval_every      = 1,
        early_stopping  = EarlyStoppingCfg(monitor="val_metrics.hamming",
                                           patience=1, min_delta=0.0),
    )
    backbone = _linear_backbone(3, 3)
    model = M3N(backbone, 3)
    edges = chain_edges(4)

    trainer = Trainer(model=model, edges=edges, cfg=cfg,
                      n_train=2, device=torch.device("cpu"))

    groups = trainer.optimizer.param_groups
    assert len(groups) == 2, "lp_m3n optimizer should have 2 param groups"
    assert groups[0]["lr"] == 0.001, "model group lr should be cfg.lr"
    assert groups[1]["lr"] == 0.05,  "phi group lr should be cfg.lr_phi"


def test_lp_m3n_lr_phi_zero_falls_back_to_lr():
    """Sentinel: lr_phi=0 means 'same as lr_model' (preserves prior behaviour)."""
    torch.manual_seed(0)
    cfg = TrainingCfg(
        loss            = "lp_m3n",
        inference       = InferenceCfg(train="lp", eval="viterbi"),
        optimizer       = OptimizerCfg(type="adam", lr=0.01, weight_decay=0.0,
                                       weight_decay_phi=0.0, phi_init_std=0.0,
                                       lr_phi=0.0),  # sentinel
        num_epochs      = 1,
        scheduler       = SchedulerCfg(type="none"),
        eval_every      = 1,
        early_stopping  = EarlyStoppingCfg(monitor="val_metrics.hamming",
                                           patience=1, min_delta=0.0),
    )
    backbone = _linear_backbone(3, 3)
    model = M3N(backbone, 3)
    edges = chain_edges(4)

    trainer = Trainer(model=model, edges=edges, cfg=cfg,
                      n_train=2, device=torch.device("cpu"))

    groups = trainer.optimizer.param_groups
    assert groups[0]["lr"] == 0.01
    assert groups[1]["lr"] == 0.01, "lr_phi=0 should fall back to lr"


def test_m3n_hinge_optimizer_has_one_param_group():
    """No phi bank for m3n_hinge; only the model param group exists."""
    torch.manual_seed(0)
    backbone = _linear_backbone(3, 3)
    model = M3N(backbone, 3)
    edges = chain_edges(4)
    trainer = Trainer(
        model=model, edges=edges,
        cfg=_hinge_training_cfg(),
        n_train=2, device=torch.device("cpu"),
    )
    assert len(trainer.optimizer.param_groups) == 1
    assert trainer.phi_bank is None
