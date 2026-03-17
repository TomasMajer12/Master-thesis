"""
Backbone networks that map raw input features to unary potentials (per-node class scores).

Every backbone follows the same contract:
    Input:  x of shape [batch, num_nodes, input_dim]
    Output: unary potentials of shape [batch, num_nodes, num_classes]

Each node (position in the sequence / cell in the grid) is processed independently
through the same network — parameter sharing across positions.

Three ways to get a backbone:
    1. Use LinearBackbone / MLPBackbone directly (convenience classes)
    2. Build from config via build_backbone(config) — specify layers as a list
    3. Wrap any existing PyTorch nn.Module via wrap_backbone(module, num_classes)
"""

import torch.nn as nn


# ---------------------------------------------------------------------------
# Convenience classes (for quick use without config)
# ---------------------------------------------------------------------------

class LinearBackbone(nn.Module):
    """Single linear layer:  score = W @ x + b"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        batch, num_nodes, dim = x.shape
        out = self.net(x.reshape(-1, dim))
        return out.reshape(batch, num_nodes, -1)


class MLPBackbone(nn.Module):
    """Multi-layer perceptron:  input -> hidden layers with ReLU -> class scores"""

    def __init__(self, input_dim, num_classes, hidden_dims=(64,)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch, num_nodes, dim = x.shape
        out = self.net(x.reshape(-1, dim))
        return out.reshape(batch, num_nodes, -1)


# ---------------------------------------------------------------------------
# Config-driven backbone builder
# ---------------------------------------------------------------------------

# Registry of supported layer types -> PyTorch constructors.
# Each entry maps a string name to (nn.Module class, list of expected kwargs).
# This makes configs readable while staying close to raw PyTorch.
LAYER_REGISTRY = {
    "linear":      nn.Linear,       # args: in_features, out_features
    "relu":        nn.ReLU,
    "leaky_relu":  nn.LeakyReLU,    # args: negative_slope (optional)
    "sigmoid":     nn.Sigmoid,
    "tanh":        nn.Tanh,
    "dropout":     nn.Dropout,      # args: p
    "batchnorm1d": nn.BatchNorm1d,  # args: num_features
    "conv2d":      nn.Conv2d,       # args: in_channels, out_channels, kernel_size, ...
    "maxpool2d":   nn.MaxPool2d,    # args: kernel_size
    "flatten":     nn.Flatten,      # args: start_dim (optional)
}


def _build_layer(layer_spec):
    """Build a single nn.Module from a layer specification dict.

    A layer_spec is a dict like:
        {"type": "linear", "in_features": 784, "out_features": 128}
        {"type": "relu"}
        {"type": "conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3}

    If the type is not in LAYER_REGISTRY, we try to look it up in torch.nn
    directly, so you can use any PyTorch layer by its class name:
        {"type": "GELU"}
        {"type": "BatchNorm2d", "num_features": 32}
    """
    spec = dict(layer_spec)  # copy so we don't mutate the config
    type_name = spec.pop("type")

    # Try our registry first (lowercase names)
    if type_name in LAYER_REGISTRY:
        cls = LAYER_REGISTRY[type_name]
        return cls(**spec)

    # Fall back to any class in torch.nn
    if hasattr(nn, type_name):
        cls = getattr(nn, type_name)
        return cls(**spec)

    raise ValueError(
        f"Unknown layer type '{type_name}'. "
        f"Available in registry: {list(LAYER_REGISTRY.keys())}. "
        f"You can also use any torch.nn class name directly."
    )


class ConfigBackbone(nn.Module):
    """Backbone built from a list of layer specifications.

    Example config (YAML):
        backbone:
          type: config
          layers:
            - {type: linear, in_features: 25, out_features: 64}
            - {type: relu}
            - {type: linear, in_features: 64, out_features: 25}

    Example config (for a CNN processing images):
        backbone:
          type: config
          layers:
            - {type: conv2d, in_channels: 1, out_channels: 32, kernel_size: 3, padding: 1}
            - {type: relu}
            - {type: maxpool2d, kernel_size: 2}
            - {type: flatten, start_dim: 1}
            - {type: linear, in_features: 6272, out_features: 10}
    """

    def __init__(self, layer_specs):
        super().__init__()
        layers = [_build_layer(spec) for spec in layer_specs]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch, num_nodes = x.shape[0], x.shape[1]
        node_input = x.reshape(-1, *x.shape[2:])    # merge batch & nodes
        out = self.net(node_input)                   # [batch*nodes, num_classes]
        return out.reshape(batch, num_nodes, -1)


# ---------------------------------------------------------------------------
# Wrapper for existing PyTorch modules
# ---------------------------------------------------------------------------

class WrappedBackbone(nn.Module):
    """Adapt any nn.Module to the backbone contract.

    Takes a module whose forward maps a single input to a feature vector,
    and adds an output head (linear layer) to produce num_classes scores.

    Args:
        feature_extractor: nn.Module mapping single-node input to a feature
                           vector of size `feature_dim`
        feature_dim:       output size of the feature extractor
        num_classes:       number of output classes
    """

    def __init__(self, feature_extractor, feature_dim, num_classes):
        super().__init__()
        self.features = feature_extractor
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        batch, num_nodes = x.shape[0], x.shape[1]
        node_input = x.reshape(-1, *x.shape[2:])
        features = self.features(node_input)         # [batch*nodes, feature_dim]
        out = self.head(features)                    # [batch*nodes, num_classes]
        return out.reshape(batch, num_nodes, -1)


# ---------------------------------------------------------------------------
# Factory function — single entry point for config-driven creation
# ---------------------------------------------------------------------------

def build_backbone(config):
    """Create a backbone from a config dict.

    Supported config formats:

    1. Linear:
        {"type": "linear", "input_dim": 25, "num_classes": 25}

    2. MLP:
        {"type": "mlp", "input_dim": 25, "num_classes": 25, "hidden_dims": [64, 32]}

    3. Config (arbitrary layer list):
        {"type": "config", "layers": [
            {"type": "linear", "in_features": 25, "out_features": 64},
            {"type": "relu"},
            {"type": "linear", "in_features": 64, "out_features": 25}
        ]}
    """
    backbone_type = config["type"]

    if backbone_type == "linear":
        return LinearBackbone(config["input_dim"], config["num_classes"])

    if backbone_type == "mlp":
        return MLPBackbone(
            config["input_dim"],
            config["num_classes"],
            hidden_dims=tuple(config.get("hidden_dims", [64])),
        )

    if backbone_type == "config":
        return ConfigBackbone(config["layers"])

    raise ValueError(f"Unknown backbone type: '{backbone_type}'")
