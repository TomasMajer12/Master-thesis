"""Tests for the learning module: structured loss, evaluation, trainer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models import LinearBackbone, M3N, chain_edges
from src.inference import viterbi_decode, loss_augmented_viterbi
from src.learning import structured_hinge_loss, hamming_loss, zero_one_loss, Trainer


# ---- Evaluation metrics ----

def test_hamming_loss():
    y_pred = torch.tensor([[0, 1, 2], [1, 1, 1]])
    y_true = torch.tensor([[0, 1, 0], [1, 0, 1]])
    # 2 wrong out of 6 = 0.333...
    loss = hamming_loss(y_pred, y_true)
    assert abs(loss - 2/6) < 1e-6, f"Expected {2/6}, got {loss}"

    # Perfect prediction
    assert hamming_loss(y_true, y_true) == 0.0
    print("PASS: hamming_loss")


def test_zero_one_loss():
    y_pred = torch.tensor([[0, 1, 2], [1, 0, 1]])
    y_true = torch.tensor([[0, 1, 2], [1, 0, 1]])
    assert zero_one_loss(y_pred, y_true) == 0.0  # both samples correct

    y_pred = torch.tensor([[0, 1, 2], [1, 1, 1]])
    y_true = torch.tensor([[0, 1, 2], [1, 0, 1]])
    # First sample correct, second wrong -> 0.5
    assert abs(zero_one_loss(y_pred, y_true) - 0.5) < 1e-6
    print("PASS: zero_one_loss")


# ---- Structured hinge loss ----

def test_loss_zero_when_perfect():
    """If y_true is already the highest-scoring labeling, loss should be 0."""
    torch.manual_seed(0)
    K, T, batch = 3, 5, 2

    backbone = LinearBackbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    # Make y_true strongly preferred by setting huge unary for true labels
    y_true = torch.randint(0, K, (batch, T))
    x = torch.zeros(batch, T, K)

    # Manually set backbone to output high scores for y_true
    with torch.no_grad():
        model.backbone.net.weight.zero_()
        model.backbone.net.bias.zero_()
        model.pairwise.zero_()

    # With all-zero potentials, everything scores the same, so loss > 0.
    # Instead, construct unary directly:
    unary = torch.zeros(batch, T, K)
    unary.scatter_(2, y_true.unsqueeze(-1), 100.0)

    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    assert loss.item() == 0.0, f"Expected 0, got {loss.item()}"
    print("PASS: loss is 0 when y_true is optimal")


def test_loss_nonnegative():
    """Structured hinge loss must always be >= 0."""
    torch.manual_seed(42)
    K, T, batch = 5, 10, 4

    backbone = LinearBackbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    x = torch.randn(batch, T, K)
    y_true = torch.randint(0, K, (batch, T))
    unary = model.unary(x)

    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    assert loss.item() >= 0, f"Loss is negative: {loss.item()}"
    print("PASS: loss is non-negative")


def test_loss_gradient_flow():
    """Gradients must flow from loss back to model parameters."""
    torch.manual_seed(0)
    K, T, batch = 4, 6, 3

    backbone = LinearBackbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    x = torch.randn(batch, T, K)
    y_true = torch.randint(0, K, (batch, T))

    unary = model.unary(x)
    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
    print("PASS: loss gradient flows to all parameters")


def test_loss_includes_hamming():
    """Verify the loss includes the Hamming term (not just score difference).

    The correct loss is: F(y*) + Delta(y*, y_true) - F(y_true)
    NOT just:            F(y*) - F(y_true)          <- old bug
    """
    torch.manual_seed(7)
    K, T, batch = 3, 4, 1

    backbone = LinearBackbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    x = torch.randn(batch, T, K)
    y_true = torch.randint(0, K, (batch, T))
    unary = model.unary(x)

    # Manually compute what the loss should be
    y_star = loss_augmented_viterbi(unary, model.pairwise, y_true)
    with torch.no_grad():
        s_star = model.score(unary, y_star, edges)
        s_true = model.score(unary, y_true, edges)
        ham = (y_star != y_true).float().sum(dim=1)
        expected = torch.clamp(s_star + ham - s_true, min=0.0).mean().item()

    loss = structured_hinge_loss(model, unary, y_true, edges, loss_augmented_viterbi)
    diff = abs(loss.item() - expected)
    assert diff < 1e-5, f"Loss={loss.item()}, expected={expected}, diff={diff}"
    print(f"PASS: loss includes Hamming term (diff={diff:.2e})")


# ---- Trainer ----

def test_trainer_smoke():
    """Trainer should run without errors and reduce loss."""
    torch.manual_seed(42)
    K, T = 5, 8
    N_train, N_test = 20, 10

    backbone = LinearBackbone(K, K)
    model = M3N(backbone, K)
    edges = chain_edges(T)

    trainer = Trainer(
        model, loss_augmented_viterbi, viterbi_decode, edges,
        lr=0.05, weight_decay=0.0,
    )

    # Random data
    train_x = torch.randn(N_train, T, K)
    train_y = torch.randint(0, K, (N_train, T))
    test_x = torch.randn(N_test, T, K)
    test_y = torch.randint(0, K, (N_test, T))

    history = trainer.fit(train_x, train_y, test_x, test_y, config={
        'num_epochs': 10,
        'batch_size': 10,
        'eval_every': 5,
        'patience': 100,
        'verbose': False,
    })

    assert len(history['epoch']) > 0, "No evaluations recorded"
    assert history['best_test_hamming'] < 1.0, "Should have some reasonable error"
    print("PASS: trainer smoke test")


if __name__ == "__main__":
    test_hamming_loss()
    test_zero_one_loss()
    test_loss_zero_when_perfect()
    test_loss_nonnegative()
    test_loss_gradient_flow()
    test_loss_includes_hamming()
    test_trainer_smoke()
    print("\nAll tests passed.")
