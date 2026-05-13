"""mnlearn.data — datasets, benchmark builders, and statistics."""

from .mnist_pool import MNISTPool

from .sudoku import (
    ensure_sudoku_csv,
    load_sudoku_csv,
    SymbolicSudokuDataset,
    VisualSudokuDataset,
    build_sudoku_benchmark,
    load_sudoku_benchmark,
)
from .hmc import (
    generate_hmc_sequences,
    SymbolicHMCDataset,
    VisualHMCDataset,
    build_hmc_benchmark,
    load_hmc_benchmark,
)
from .builders import build_datasets
from . import stats
from . import figures

__all__ = [
    # mnist
    "MNISTPool",
    # sudoku
    "ensure_sudoku_csv", "load_sudoku_csv",
    "SymbolicSudokuDataset", "VisualSudokuDataset",
    "build_sudoku_benchmark", "load_sudoku_benchmark",
    # hmc
    "generate_hmc_sequences",
    "SymbolicHMCDataset", "VisualHMCDataset",
    "build_hmc_benchmark", "load_hmc_benchmark",
    # builders
    "build_datasets",
    # stats / figures namespaces
    "stats",
    "figures",
]