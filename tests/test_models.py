"""Tests for backbones, M3N scoring, and graph constructors."""

import torch
import torch.nn as nn

from mnlearn.config.schema import BackboneCfg, GraphCfg, ArchitectureCfg, PairwiseCfg
from mnlearn.models import (
    ConfigBackbone,
    M3N,
    WrappedBackbone,
    build_backbone,
    build_graph,
    build_model,
    chain_edges,
    sudoku_edges,
)


# ---------------------------------------------------------------------------
# Backbone shape contract
# ---------------------------------------------------------------------------

def _linear_layers(dim: int, classes: int) -> list[dict]:
    return [{"type": "linear", "in_features": dim, "out_features": classes}]


def _mlp_layers(dim: int, classes: int, hidden: int) -> list[dict]:
    return [
        {"type": "linear", "in_features": dim, "out_features": hidden},
        {"type": "relu"},
        {"type": "linear", "in_features": hidden, "out_features": classes},
    ]


def test_backbone_shapes():
    """ConfigBackbone should map [batch, nodes, dim] -> [batch, nodes, classes]."""
    batch, nodes, dim, classes = 4, 10, 25, 25

    for layers in (_linear_layers(dim, classes), _mlp_layers(dim, classes, 32)):
        backbone = ConfigBackbone(layers)
        x = torch.randn(batch, nodes, dim)
        out = backbone(x)
        assert out.shape == (batch, nodes, classes), f"Got {out.shape}"
    print("PASS: backbone shapes")


# ---------------------------------------------------------------------------
# Graph constructors
# ---------------------------------------------------------------------------

def test_chain_edges():
    edges = chain_edges(5)
    expected = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert torch.equal(edges, expected), f"Got {edges}"
    print("PASS: chain_edges")


def test_sudoku_edges_count():
    edges = sudoku_edges()
    # 9 rows x C(9,2) + 9 cols x C(9,2) + 9 boxes x C(9,2) - overlap
    # Known result: 810 unique constraint pairs
    assert edges.shape == (810, 2), f"Got {edges.shape}"
    print("PASS: sudoku_edges_count")


# ---------------------------------------------------------------------------
# M3N scoring
# ---------------------------------------------------------------------------

def test_score_manual():
    """Compare M3N.score against a manual loop."""
    torch.manual_seed(0)
    batch, seq_len, num_states = 2, 5, 3

    backbone = ConfigBackbone(_linear_layers(num_states, num_states))
    model = M3N(backbone, num_states)
    edges = chain_edges(seq_len)

    x = torch.randn(batch, seq_len, num_states)
    y = torch.randint(0, num_states, (batch, seq_len))

    unary = model.unary(x)
    score_vec = model.score(unary, y, edges)

    score_loop = torch.zeros(batch)
    for b in range(batch):
        for t in range(seq_len):
            score_loop[b] += unary[b, t, y[b, t]]
            if t > 0:
                score_loop[b] += model.pairwise[y[b, t - 1], y[b, t]]

    diff = (score_vec - score_loop).abs().max().item()
    assert diff < 1e-5, f"Score mismatch: max diff = {diff}"
    print(f"PASS: score matches loop (max diff = {diff:.2e})")


def test_score_gradient():
    """Gradients must flow through score back to backbone and pairwise."""
    backbone = ConfigBackbone(_linear_layers(10, 5))
    model = M3N(backbone, 5)
    edges = chain_edges(8)

    x = torch.randn(2, 8, 10)
    y = torch.randint(0, 5, (2, 8))

    unary = model.unary(x)
    scores = model.score(unary, y, edges)
    loss = scores.sum()
    loss.backward()

    for name, p in model.backbone.named_parameters():
        assert p.grad is not None, f"No gradient for backbone.{name}"
    assert model.pairwise.grad is not None, "No gradient for pairwise"
    print("PASS: gradients flow correctly")


# ---------------------------------------------------------------------------
# ConfigBackbone variants
# ---------------------------------------------------------------------------

def test_config_backbone_arbitrary_layers():
    """ConfigBackbone should build any layer combination from a spec list."""
    batch, nodes, classes = 4, 10, 25
    spec = [
        {"type": "linear", "in_features": 25, "out_features": 64},
        {"type": "relu"},
        {"type": "dropout", "p": 0.1},
        {"type": "linear", "in_features": 64, "out_features": classes},
    ]
    backbone = ConfigBackbone(spec)
    out = backbone(torch.randn(batch, nodes, 25))
    assert out.shape == (batch, nodes, classes), f"Got {out.shape}"
    print("PASS: config_backbone_arbitrary_layers")


def test_config_backbone_torch_nn_fallback():
    """Layer types not in the alias map should fall through to torch.nn."""
    spec = [
        {"type": "linear", "in_features": 10, "out_features": 32},
        {"type": "GELU"},                    # not aliased; comes from torch.nn directly
        {"type": "linear", "in_features": 32, "out_features": 5},
    ]
    backbone = ConfigBackbone(spec)
    out = backbone(torch.randn(2, 4, 10))
    assert out.shape == (2, 4, 5)
    print("PASS: config_backbone_torch_nn_fallback")


def test_wrapped_backbone():
    feature_extractor = nn.Sequential(nn.Linear(25, 64), nn.ReLU())
    backbone = WrappedBackbone(feature_extractor, feature_dim=64, num_classes=10)
    out = backbone(torch.randn(4, 10, 25))
    assert out.shape == (4, 10, 10)
    print("PASS: wrapped_backbone")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def test_build_backbone_config():
    cfg = BackboneCfg(type="config", layers=_linear_layers(25, 10))
    backbone = build_backbone(cfg, num_classes=10)
    out = backbone(torch.randn(2, 5, 25))
    assert out.shape == (2, 5, 10)
    print("PASS: build_backbone_config")


def test_build_graph_each_type():
    """build_graph dispatches to the right constructor for each type."""
    # sudoku
    edges = build_graph(GraphCfg(type="sudoku"))
    assert edges.shape == (810, 2)

    # chain
    edges = build_graph(GraphCfg(type="chain", seq_len=5))
    assert edges.shape == (4, 2)

    # inline
    edges = build_graph(GraphCfg(type="inline", edges=[[0, 1], [1, 2], [2, 0]]))
    assert torch.equal(edges, torch.tensor([[0, 1], [1, 2], [2, 0]]))
    print("PASS: build_graph_each_type")


def test_build_model_assembles_m3n_and_edges():
    arch = ArchitectureCfg(
        num_classes = 9,
        backbone    = BackboneCfg(type="config", layers=_linear_layers(9, 9)),
        graph       = GraphCfg(type="sudoku"),
        pairwise    = PairwiseCfg(init_scale=0.1),
    )
    model, edges = build_model(arch)
    assert isinstance(model, M3N)
    assert model.num_classes == 9
    assert edges.shape == (810, 2)
    # End-to-end: a forward pass yields the expected unary shape.
    out = model.unary(torch.randn(2, 81, 9))
    assert out.shape == (2, 81, 9)
    print("PASS: build_model_assembles_m3n_and_edges")


if __name__ == "__main__":
    test_backbone_shapes()
    test_chain_edges()
    test_sudoku_edges_count()
    test_score_manual()
    test_score_gradient()
    test_config_backbone_arbitrary_layers()
    test_config_backbone_torch_nn_fallback()
    test_wrapped_backbone()
    test_build_backbone_config()
    test_build_graph_each_type()
    test_build_model_assembles_m3n_and_edges()
    print("\nAll tests passed.")
