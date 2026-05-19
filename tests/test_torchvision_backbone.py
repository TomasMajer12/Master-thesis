"""Smoke + contract test for the torchvision backbone path.

"""

import pytest

torchvision = pytest.importorskip("torchvision")

import torch

from mnlearn.config.schema import BackboneCfg
from mnlearn.models.backbones import WrappedBackbone, build_backbone


def _make_cfg(pretrained: bool, freeze: bool) -> BackboneCfg:
    return BackboneCfg(
        type="torchvision",
        name="resnet18",
        pretrained=pretrained,
        freeze=freeze,
        feature_dim=512,
        layers=None,
    )


def test_torchvision_backbone_builds_with_pretrained_weights():
    """resnet18 + pretrained=True loads, strips .fc, wraps with fresh head."""
    cfg = _make_cfg(pretrained=True, freeze=False)
    backbone = build_backbone(cfg, num_classes=9)

    assert isinstance(backbone, WrappedBackbone)
    # The classification head was stripped to Identity in the trunk.
    assert isinstance(backbone.features.fc, torch.nn.Identity)
    # A fresh Linear(feature_dim, num_classes) head sits on top.
    assert isinstance(backbone.head, torch.nn.Linear)
    assert backbone.head.in_features  == 512
    assert backbone.head.out_features == 9


def test_torchvision_backbone_respects_per_node_contract():
    """[B, V, C, H, W] -> [B, V, num_classes]. pretrained=False to skip download."""
    cfg = _make_cfg(pretrained=False, freeze=False)
    backbone = build_backbone(cfg, num_classes=9)

    x = torch.randn(2, 4, 3, 64, 64)
    out = backbone(x)
    assert out.shape == (2, 4, 9)


def test_torchvision_backbone_freeze_flag():
    """freeze=True freezes trunk parameters; head stays trainable."""
    cfg = _make_cfg(pretrained=False, freeze=True)
    backbone = build_backbone(cfg, num_classes=9)

    for name, p in backbone.features.named_parameters():
        assert not p.requires_grad, f"frozen trunk param {name} is still trainable"
    for p in backbone.head.parameters():
        assert p.requires_grad, "head should stay trainable"


if __name__ == "__main__":
    test_torchvision_backbone_builds_with_pretrained_weights(); print("PASS: torchvision backbone builds with pretrained weights")
    test_torchvision_backbone_respects_per_node_contract();     print("PASS: torchvision backbone respects per node contract")
    test_torchvision_backbone_freeze_flag();                    print("PASS: torchvision backbone freeze flag")
    print("\nAll tests passed.")
