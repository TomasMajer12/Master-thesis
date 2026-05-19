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
    assert history["loss"] == "m3n_hinge"
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

    assert history["loss"] == "lp_m3n"
    assert len(history["epoch"]) >= 1
    # phi_norm must appear in LP-M3N diagnostics.
    assert "phi_norm" in history["diagnostics"][-1]


# ---------------------------------------------------------------------------
# lr_phi: separate learning rate for the phi parameter group
# ---------------------------------------------------------------------------

def test_lp_m3n_optimizer_has_three_param_groups_with_separate_lrs():
    """Group layout: [0] backbone @ lr, [1] pairwise W @ lr_pairwise,
    [2] phi bank @ lr_phi. When lr_phi > 0 it overrides the model lr."""
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
    assert len(groups) == 3, "lp_m3n optimizer should have 3 param groups"
    assert groups[0]["lr"] == 0.001, "backbone group lr should be cfg.lr"
    assert groups[1]["lr"] == 0.001, "pairwise group lr falls back to cfg.lr when lr_pairwise=0"
    assert groups[2]["lr"] == 0.05,  "phi group lr should be cfg.lr_phi"


def test_lp_m3n_lr_phi_zero_falls_back_to_lr():
    """Sentinel: lr_phi=0 (and lr_pairwise=0) means 'same as lr_model'."""
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
    assert len(groups) == 3
    assert groups[0]["lr"] == 0.01, "backbone uses cfg.lr"
    assert groups[1]["lr"] == 0.01, "pairwise lr_pairwise=0 falls back to lr"
    assert groups[2]["lr"] == 0.01, "phi lr_phi=0 falls back to lr"


def test_lp_m3n_lr_pairwise_independent_from_lr():
    """Regression test for the lr_pairwise loader bug: lr_pairwise > 0 must
    propagate to the pairwise param group and override the backbone's lr.
    The earlier bug (lr_pairwise silently defaulting to 0.0 inside the YAML
    loader) would have made this assertion fail."""
    torch.manual_seed(0)
    cfg = TrainingCfg(
        loss            = "lp_m3n",
        inference       = InferenceCfg(train="lp", eval="viterbi"),
        optimizer       = OptimizerCfg(type="adam", lr=0.001, weight_decay=0.0,
                                       weight_decay_phi=0.0, phi_init_std=0.0,
                                       lr_pairwise=0.01, lr_phi=0.01),
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
    assert groups[0]["lr"] == 0.001, "backbone uses lr"
    assert groups[1]["lr"] == 0.01,  "pairwise uses lr_pairwise (10x backbone)"
    assert groups[2]["lr"] == 0.01,  "phi uses lr_phi"


def test_m3n_hinge_optimizer_has_two_param_groups():
    """No phi bank for m3n_hinge, but the pairwise W still gets its own
    group so lr_pairwise can be set independently."""
    torch.manual_seed(0)
    backbone = _linear_backbone(3, 3)
    model = M3N(backbone, 3)
    edges = chain_edges(4)
    trainer = Trainer(
        model=model, edges=edges,
        cfg=_hinge_training_cfg(),
        n_train=2, device=torch.device("cpu"),
    )
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2, "m3n_hinge optimizer should have 2 param groups (backbone + pairwise)"
    assert trainer.phi_bank is None


if __name__ == "__main__":
    test_hamming_loss();                                            print("PASS: hamming_loss")
    test_zero_one_loss();                                           print("PASS: zero_one_loss")
    test_loss_zero_when_perfect();                                  print("PASS: loss zero when perfect")
    test_loss_nonnegative();                                        print("PASS: loss non-negative")
    test_loss_gradient_flow();                                      print("PASS: loss gradient flow")
    test_loss_includes_hamming();                                   print("PASS: loss includes hamming")
    test_trainer_smoke_m3n_hinge();                                 print("PASS: trainer smoke m3n_hinge")
    test_trainer_smoke_lp_m3n();                                    print("PASS: trainer smoke lp_m3n")
    test_lp_m3n_optimizer_has_three_param_groups_with_separate_lrs(); print("PASS: lp_m3n optimizer has three param groups with separate lrs")
    test_lp_m3n_lr_phi_zero_falls_back_to_lr();                       print("PASS: lp_m3n lr_phi=0 falls back to lr")
    test_lp_m3n_lr_pairwise_independent_from_lr();                    print("PASS: lp_m3n lr_pairwise propagates to pairwise group")
    test_m3n_hinge_optimizer_has_two_param_groups();                  print("PASS: m3n_hinge optimizer has two param groups")
    print("\nAll tests passed.")
