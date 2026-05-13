"""Constructors for the M3N constraint graph.

The graph is just an edge list passed at runtime to ``M3N.score`` and to
inference functions. It is not owned by the model — the same M3N works
for chain, grid, or arbitrary graphs.

Three ways to obtain edges:

    1. Built-in: :func:`chain_edges`, :func:`sudoku_edges`.
    2. From config: :func:`build_graph` dispatches on ``GraphCfg.type``.
    3. Programmatic: build a ``LongTensor`` of shape ``[E, 2]`` yourself
       and pass it as ``edges_override`` to :func:`mnlearn.learning.train`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mnlearn.config.schema import GraphCfg


def chain_edges(seq_len: int, device=None) -> torch.LongTensor:
    """Edge list for a chain graph: ``0-1-2-...-(T-1)``.

    Returns a LongTensor of shape ``[seq_len - 1, 2]``.
    """
    src = torch.arange(seq_len - 1, device=device)
    dst = torch.arange(1, seq_len, device=device)
    return torch.stack([src, dst], dim=1)


def sudoku_edges(device=None) -> torch.LongTensor:
    """Edge list for a 9x9 Sudoku constraint graph.

    Cells are numbered 0..80 in row-major order: ``cell (r, c) -> r*9 + c``.
    Edges connect every pair of cells that share a row, column, or 3x3
    box. Returns a LongTensor of shape ``[810, 2]``.
    """
    edge_set = set()

    def add(u, v):
        if u != v:
            edge_set.add((min(u, v), max(u, v)))

    for r in range(9):
        for c1 in range(9):
            for c2 in range(c1 + 1, 9):
                add(r * 9 + c1, r * 9 + c2)        # row
                add(c1 * 9 + r, c2 * 9 + r)        # col

    for br in range(3):
        for bc in range(3):
            cells = [
                (br * 3 + dr) * 9 + (bc * 3 + dc)
                for dr in range(3)
                for dc in range(3)
            ]
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    add(cells[i], cells[j])

    return torch.tensor(sorted(edge_set), dtype=torch.long, device=device)


def build_graph(cfg: GraphCfg, base_dir: Path | str | None = None) -> torch.LongTensor:
    """Build an edge list from a :class:`GraphCfg`.

    For ``edges_file`` the ``cfg.path`` is resolved relative to ``base_dir``
    if provided and the path is not already absolute.
    """
    if cfg.type == "sudoku":
        return sudoku_edges()

    if cfg.type == "chain":
        assert cfg.seq_len is not None, "build_graph(chain): cfg.seq_len must be set"
        return chain_edges(cfg.seq_len)

    if cfg.type == "edges_file":
        assert cfg.path, "build_graph(edges_file): cfg.path must be set"
        path = Path(cfg.path)
        if base_dir is not None and not path.is_absolute():
            path = Path(base_dir) / path
        edges_np = np.load(path)
        return torch.from_numpy(edges_np).long()

    if cfg.type == "inline":
        assert cfg.edges, "build_graph(inline): cfg.edges must be a non-empty list"
        return torch.tensor(cfg.edges, dtype=torch.long)

    raise ValueError(f"Unknown graph.type={cfg.type!r}")
