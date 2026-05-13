"""mnlearn.models — predictor model, backbones, and graph constructors."""

from .backbones import ConfigBackbone, WrappedBackbone, build_backbone
from .builders  import build_model
from .graph     import build_graph, chain_edges, sudoku_edges
from .m3n       import M3N

__all__ = [
    # Core predictor
    "M3N",
    # Backbones + factory
    "ConfigBackbone", "WrappedBackbone", "build_backbone",
    # Graph constructors
    "build_graph", "chain_edges", "sudoku_edges",
    # Composition factory
    "build_model",
]
