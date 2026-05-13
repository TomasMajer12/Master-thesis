"""Backbone networks that map raw input to unary potentials.

Every backbone follows the same contract:
    Input:  x of shape [batch, num_nodes, *input_shape]
    Output: unary potentials of shape [batch, num_nodes, num_classes]

Each node (sequence position / Sudoku cell) is processed independently
through the same network — parameter sharing across positions.

Two ways to build a backbone:

    1. ``ConfigBackbone(layers)`` — list of layer dicts becomes an
       ``nn.Sequential``. Used for fully custom architectures (the
       common case for this thesis).

    2. ``WrappedBackbone(feature_extractor, feature_dim, num_classes)``
       — adapt any existing ``nn.Module`` (e.g. a torchvision trunk) to
       the per-node contract by appending a linear head.

The :func:`build_backbone` factory dispatches between these based on a
:class:`BackboneCfg` from the config schema.
"""

from __future__ import annotations

import importlib

import torch.nn as nn

from mnlearn.config.schema import BackboneCfg


# ---------------------------------------------------------------------------
# Backbone classes
# ---------------------------------------------------------------------------

class ConfigBackbone(nn.Module):
    """Backbone built from a list of layer specifications.

    Each layer spec is a dict like ``{"type": "linear", "in_features": 25,
    "out_features": 9}``. The ``type`` value is looked up in :mod:`torch.nn`
    (lowercase aliases for common layers are accepted), and the remaining
    keys are passed as keyword arguments.

    Example::

        ConfigBackbone([
            {"type": "linear", "in_features": 25, "out_features": 64},
            {"type": "relu"},
            {"type": "linear", "in_features": 64, "out_features": 25},
        ])
    """

    def __init__(self, layer_specs: list[dict]):
        super().__init__()
        layers = [_build_layer(spec) for spec in layer_specs]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch, num_nodes = x.shape[0], x.shape[1]
        node_input = x.reshape(-1, *x.shape[2:])      # merge batch & nodes
        out = self.net(node_input)                     # [batch*nodes, num_classes]
        return out.reshape(batch, num_nodes, -1)


class WrappedBackbone(nn.Module):
    """Adapt any ``nn.Module`` to the backbone contract.

    The wrapped ``feature_extractor`` is expected to map a single-node
    input to a feature vector of size ``feature_dim``. A trainable linear
    head maps those features to ``num_classes`` unary scores.
    """

    def __init__(self, feature_extractor: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.features = feature_extractor
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        batch, num_nodes = x.shape[0], x.shape[1]
        node_input = x.reshape(-1, *x.shape[2:])
        features = self.features(node_input)           # [batch*nodes, feature_dim]
        out = self.head(features)                      # [batch*nodes, num_classes]
        return out.reshape(batch, num_nodes, -1)


# ---------------------------------------------------------------------------
# Layer dispatch
# ---------------------------------------------------------------------------

_LAYER_ALIASES = {
    "linear":      "Linear",
    "relu":        "ReLU",
    "leaky_relu":  "LeakyReLU",
    "sigmoid":     "Sigmoid",
    "softmax":     "Softmax",
    "tanh":        "Tanh",
    "dropout":     "Dropout",
    "batchnorm1d": "BatchNorm1d",
    "batchnorm2d": "BatchNorm2d",
    "conv2d":      "Conv2d",
    "maxpool2d":   "MaxPool2d",
    "flatten":     "Flatten",
}


def _build_layer(layer_spec: dict) -> nn.Module:
    spec = dict(layer_spec)
    type_name = spec.pop("type")
    type_name = _LAYER_ALIASES.get(type_name, type_name)

    if not hasattr(nn, type_name):
        raise ValueError(
            f"Unknown layer type {type_name!r}. "
            f"Must be a class in torch.nn (or one of: {list(_LAYER_ALIASES)})."
        )
    return getattr(nn, type_name)(**spec)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backbone(cfg: BackboneCfg, num_classes: int) -> nn.Module:
    """Build a backbone from a :class:`BackboneCfg`.

    The ``num_classes`` argument is used by the ``torchvision`` and
    ``wrapped`` branches to size the output head. For ``config`` it is
    accepted but unused (the layer list is fully self-describing).
    """
    if cfg.type == "config":
        assert cfg.layers is not None, "build_backbone(config): cfg.layers must be set"
        return ConfigBackbone(cfg.layers)

    if cfg.type == "torchvision":
        return _build_torchvision_backbone(cfg, num_classes)

    if cfg.type == "wrapped":
        return _build_wrapped_backbone(cfg, num_classes)

    raise ValueError(f"Unknown backbone.type={cfg.type!r}")


def _build_torchvision_backbone(cfg: BackboneCfg, num_classes: int) -> WrappedBackbone:
    """Load a torchvision model, strip its classification head, wrap it."""
    import torchvision.models as tvm

    if not hasattr(tvm, cfg.name or ""):
        raise ValueError(f"torchvision.models has no attribute {cfg.name!r}")
    factory = getattr(tvm, cfg.name)

    # Pass weights="DEFAULT" to load pretrained, None to start from scratch.
    model = factory(weights="DEFAULT" if cfg.pretrained else None)

    # Replace the final classification layer with Identity so the trunk
    # outputs raw features. Different torchvision families use different
    # attribute names — we handle the two most common (.fc for ResNet/etc,
    # .classifier for VGG/AlexNet/etc).
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    else:
        raise ValueError(
            f"torchvision model {cfg.name!r} has neither .fc nor .classifier — "
            f"add a custom branch in _build_torchvision_backbone to handle it."
        )

    if cfg.freeze:
        for p in model.parameters():
            p.requires_grad_(False)

    return WrappedBackbone(model, feature_dim=cfg.feature_dim, num_classes=num_classes)


def _build_wrapped_backbone(cfg: BackboneCfg, num_classes: int) -> WrappedBackbone:
    """Load a user-supplied feature extractor by dotted import path.

    ``cfg.name`` may be either ``"pkg.module.Class"`` or
    ``"pkg.module:Class"``. The class is instantiated with no arguments
    (must accept ``__init__()`` with no required args).
    """
    name = cfg.name or ""
    if ":" in name:
        module_path, attr = name.split(":", 1)
    else:
        module_path, _, attr = name.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"wrapped backbone name must be a dotted path 'pkg.module.Class' "
            f"or 'pkg.module:Class', got {name!r}"
        )

    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    feature_extractor = cls()

    if cfg.freeze:
        for p in feature_extractor.parameters():
            p.requires_grad_(False)

    return WrappedBackbone(feature_extractor, feature_dim=cfg.feature_dim, num_classes=num_classes)
