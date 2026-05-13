"""Composition factory for the M3N predictor.

Given an :class:`ArchitectureCfg` from the experiment YAML, ``build_model``
constructs the backbone via :func:`mnlearn.models.backbones.build_backbone`,
the edge list via :func:`mnlearn.models.graph.build_graph`, and wraps them
in an :class:`M3N` instance.

Mirrors the ``builders`` convention used by :mod:`mnlearn.data` and
:mod:`mnlearn.learning`: one ``builders.py`` per subpackage carrying the
cfg-driven composition logic.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .backbones import build_backbone
from .graph import build_graph
from .m3n import M3N


def build_model(arch_cfg, base_dir: Path | str | None = None) -> tuple[M3N, torch.Tensor]:
    """Build an :class:`M3N` and its edge list from an :class:`ArchitectureCfg`.

    Args:
        arch_cfg: :class:`mnlearn.config.schema.ArchitectureCfg`.
        base_dir: used by :func:`build_graph` to resolve relative
                  ``edges_file`` paths. ``None`` → resolve against cwd.

    Returns:
        ``(model, edges)`` — model is on CPU; caller is responsible for
        ``.to(device)``. ``edges`` is a ``[E, 2]`` LongTensor on CPU.
    """
    backbone = build_backbone(arch_cfg.backbone, arch_cfg.num_classes)
    edges = build_graph(arch_cfg.graph, base_dir=base_dir)
    model = M3N(
        backbone,
        num_classes=arch_cfg.num_classes,
        pairwise_init_scale=arch_cfg.pairwise.init_scale,
    )
    return model, edges
