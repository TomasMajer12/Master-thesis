# mnlearn

End-to-end joint training of neural networks and Markov-network predictors,
developed for the master's thesis _Visual Sudoku Solver_ (FEL CVUT).

Supports both:
- **Symbolic inputs** — numerical grid encodings (one-hot per cell).
- **Visual inputs** — MNIST-style cell images for visual Sudoku and HMC.

Two training paths are implemented behind a single `Trainer` class (dispatched by the YAML `training.loss` field):
- **Oracle structured hinge** (`loss: m3n_hinge`) — for chain graphs; uses Viterbi for loss-augmented inference.
- **LP-relaxed M3N** (`loss: lp_m3n`) — for general graphs (e.g. the 9×9 Sudoku constraint graph), with per-example dual variables; no inference oracle required at training time.

## Installation

```bash
git clone <repo>
cd Master-thesis
pip install -e .
```

For development (tests):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from mnlearn.config import load_config
from mnlearn.learning import train

cfg = load_config("configs/experiments/lpm3n_visual_sudoku.yaml")
result = train(cfg)
print(result["test_metrics"])  # {"hamming": ..., "zero_one": ...}
```

`train()` returns a dict with `config`, `history`, `test_metrics`, `model`, `trainer`, and `output_dir`.

Each `train()` run writes:

```
results/<experiment.name>/
    config.yaml      # resolved config (architecture inlined)
    history.json     # per-epoch trajectories
    results.json     # final test Hamming + 0/1 loss
    model.pt         # state_dict
```

## Data

The library covers two tasks (Sudoku and HMC) with two modalities each (symbolic and visual). MNIST is downloaded automatically by `torchvision` on first use; the Sudoku CSV is downloaded from Kaggle on demand.

### Setup

To enable Sudoku auto-download, install the optional `download` extra and set up Kaggle credentials (free; see https://www.kaggle.com/docs/api):

```bash
pip install -e ".[download]"
```

Alternatively, download the [bryanpark/sudoku CSV](https://www.kaggle.com/datasets/bryanpark/sudoku) manually and pass `csv_path=...` to `ensure_sudoku_csv`.

### Building benchmarks

```python
from mnlearn.data import build_sudoku_benchmark, build_hmc_benchmark

build_sudoku_benchmark(
    output_dir="benchmarks/sudoku",
    modes=("symbolic", "visual"),
    num_puzzles=50_000,
    train_size=30_000, val_size=10_000, test_size=10_000,
)

build_hmc_benchmark(
    output_dir="benchmarks/hmc",
    modes=("symbolic", "visual"),
    num_samples=50_000,
    seq_len=30, num_states=10, p_self=0.7, p_emit=0.7,
    train_size=30_000, val_size=10_000, test_size=10_000,
)
```

End-to-end smoke test and chapter-3 statistics: see [`notebooks/00_validate_data_pipeline.ipynb`](notebooks/00_validate_data_pipeline.ipynb).

### Loading

```python
from mnlearn.data import load_sudoku_benchmark

data = load_sudoku_benchmark("benchmarks/sudoku", mode="symbolic")
data["train"]            # SymbolicSudokuDataset
data["config"]           # generation parameters
```

Statistics for chapter 3 are exposed under `mnlearn.data.stats`.

## Repository layout

| Path | Contents |
|---|---|
| `mnlearn/` | Library source (models, learning, inference, data, config, baselines, experiments). |
| `configs/architectures/` | Backbone + graph specs (e.g. `sudoku_lenet5.yaml`). |
| `configs/experiments/` | Experiment specs (data + training + architecture reference). |
| `notebooks/` | End-to-end experiments and sweeps for each thesis section. |
| `tests/` | pytest test suite. |
| `benchmarks/` | Pre-generated train/val/test splits for HMC and Sudoku. |
| `mnist_data/` | MNIST images (used for visual variants). |
| `results/` | Experiment outputs — one folder per run. |

## Tests

```bash
pytest tests/
```
