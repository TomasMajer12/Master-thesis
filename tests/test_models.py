"""Verify that the new M3N model computes scores correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models import (
    LinearBackbone, MLPBackbone, ConfigBackbone, WrappedBackbone,
    build_backbone, M3N, chain_edges,
)


def test_backbone_shapes():
    """Backbones should map [batch, nodes, dim] -> [batch, nodes, classes]."""
    batch, nodes, dim, classes = 4, 10, 25, 25

    for backbone in [LinearBackbone(dim, classes), MLPBackbone(dim, classes, (32,))]:
        x = torch.randn(batch, nodes, dim)
        out = backbone(x)
        assert out.shape == (batch, nodes, classes), f"Got {out.shape}"
    print("PASS: backbone shapes")


def test_chain_edges():
    """chain_edges(T) should produce T-1 edges: (0,1), (1,2), ..., (T-2,T-1)."""
    edges = chain_edges(5)
    expected = torch.tensor([[0,1],[1,2],[2,3],[3,4]])
    assert torch.equal(edges, expected), f"Got {edges}"
    print("PASS: chain_edges")


def test_score_manual():
    """Compare M3N.score against a manual loop (like the old code)."""
    torch.manual_seed(0)
    batch, seq_len, num_states = 2, 5, 3

    backbone = LinearBackbone(num_states, num_states)
    model = M3N(backbone, num_states)
    edges = chain_edges(seq_len)

    x = torch.randn(batch, seq_len, num_states)
    y = torch.randint(0, num_states, (batch, seq_len))

    # Score via our vectorized method
    unary = model.unary(x)
    score_vec = model.score(unary, y, edges)

    # Score via explicit loops (the old way)
    score_loop = torch.zeros(batch)
    for b in range(batch):
        for t in range(seq_len):
            score_loop[b] += unary[b, t, y[b, t]]
            if t > 0:
                score_loop[b] += model.pairwise[y[b, t-1], y[b, t]]

    diff = (score_vec - score_loop).abs().max().item()
    assert diff < 1e-5, f"Score mismatch: max diff = {diff}"
    print(f"PASS: score matches loop (max diff = {diff:.2e})")


def test_score_gradient():
    """Verify that gradients flow through score back to backbone and pairwise."""
    backbone = LinearBackbone(10, 5)
    model = M3N(backbone, 5)
    edges = chain_edges(8)

    x = torch.randn(2, 8, 10)
    y = torch.randint(0, 5, (2, 8))

    unary = model.unary(x)
    scores = model.score(unary, y, edges)
    loss = scores.sum()
    loss.backward()

    # Check that backbone parameters got gradients
    for name, p in model.backbone.named_parameters():
        assert p.grad is not None, f"No gradient for backbone.{name}"

    # Check that pairwise got gradients
    assert model.pairwise.grad is not None, "No gradient for pairwise"
    print("PASS: gradients flow correctly")


def test_config_backbone():
    """ConfigBackbone should build layers from a spec list."""
    batch, nodes, classes = 4, 10, 25
    spec = [
        {"type": "linear", "in_features": 25, "out_features": 64},
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "in_features": 64, "out_features": classes},
    ]
    backbone = ConfigBackbone(spec)
    x = torch.randn(batch, nodes, 25)
    out = backbone(x)
    assert out.shape == (batch, nodes, classes), f"Got {out.shape}"
    print("PASS: config backbone")


def test_config_backbone_torch_nn_fallback():
    """ConfigBackbone should accept any torch.nn class name."""
    spec = [
        {"type": "linear", "in_features": 10, "out_features": 32},
        {"type": "GELU"},   # not in our registry, but exists in torch.nn
        {"type": "linear", "in_features": 32, "out_features": 5},
    ]
    backbone = ConfigBackbone(spec)
    out = backbone(torch.randn(2, 4, 10))
    assert out.shape == (2, 4, 5)
    print("PASS: config backbone torch.nn fallback")


def test_wrapped_backbone():
    """WrappedBackbone should adapt any feature extractor to the contract."""
    feature_extractor = nn.Sequential(
        nn.Linear(25, 64),
        nn.ReLU(),
    )
    backbone = WrappedBackbone(feature_extractor, feature_dim=64, num_classes=10)
    out = backbone(torch.randn(4, 10, 25))
    assert out.shape == (4, 10, 10)
    print("PASS: wrapped backbone")


def test_build_backbone_factory():
    """build_backbone should create backbones from config dicts."""
    configs = [
        {"type": "linear", "input_dim": 25, "num_classes": 10},
        {"type": "mlp", "input_dim": 25, "num_classes": 10, "hidden_dims": [64, 32]},
        {"type": "config", "layers": [
            {"type": "linear", "in_features": 25, "out_features": 10},
        ]},
    ]
    for cfg in configs:
        backbone = build_backbone(cfg)
        out = backbone(torch.randn(2, 5, 25))
        assert out.shape == (2, 5, 10), f"Failed for {cfg['type']}: {out.shape}"
    print("PASS: build_backbone factory")


if __name__ == "__main__":
    test_backbone_shapes()
    test_chain_edges()
    test_score_manual()
    test_score_gradient()
    test_config_backbone()
    test_config_backbone_torch_nn_fallback()
    test_wrapped_backbone()
    test_build_backbone_factory()
    print("\nAll tests passed.")
