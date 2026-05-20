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

## Inference with a trained model

A finished `train()` run stores the model state in `model.pt` and the resolved config in `config.yaml`. The `load_predictor` helper handles the rebuild-model + load-state + pick-decoder + eval-mode ritual:

```python
from mnlearn.data     import build_datasets
from mnlearn.config   import load_config
from mnlearn.learning import load_predictor

# 1. Load the trained predictor (decoder is chosen from the YAML).
predictor = load_predictor("results/lpm3n_visual_sudoku")

# 2. Get some input data — here we reuse the benchmark's test split.
cfg = load_config("results/lpm3n_visual_sudoku/config.yaml")
X, y_true = build_datasets(cfg.data)["test"]      # X: [N, 81, 1, 28, 28] for visual Sudoku

# 3. Predict.
y_pred = predictor(X[:8])                          # [8, 81] long tensor

# 4. Compare to ground truth.
hamming = (y_pred != y_true[:8]).float().mean().item()
print(f"per-cell Hamming on 8 test puzzles: {hamming:.4f}")
```

`predictor.unary(X)` returns the intermediate `[B, V, K]` unary potentials if you need them; `predictor.model`, `predictor.edges`, and `predictor.eval_mode` are also exposed for advanced use (e.g. running a different decoder than the YAML default).

## Example experiment YAML

The canonical configuration — LP-M³N on visual Sudoku with the LeNet-5 backbone. Save as `my_experiment.yaml` and pass to `train()`.

```yaml
experiment:
  name:   my_visual_sudoku_run
  seed:   42
  device: auto

architecture: configs/architectures/sudoku_lenet5.yaml   # or inline as a dict

data:
  task: sudoku
  mode: visual
  paths:
    benchmark: benchmarks/sudoku
    mnist:     mnist_data
  train_size: 1500
  val_size:   1000
  test_size:  1000
  batch_size: 32

training:
  loss: lp_m3n
  inference:
    train: lp
    eval:  bp
    params: {bp_iters: 50}
  optimizer:
    type:         adam
    lr:           0.001
    lr_pairwise:  0.01           # 10× lr; rebalances CNN-vs-W training dynamics
    lr_phi:       0.01           # 10× lr; compensates for once-per-epoch phi updates
    weight_decay: 0.001
  num_epochs: 1000
  scheduler:
    type:   lambda
    params: {offset: 500}        # hyperbolic decay; lr halves at epoch 500
  eval_every: 10
  early_stopping:
    monitor:   val_metrics.zero_one
    patience:  25
    min_delta: 0.0
```

For the full grammar see [`docs/yaml_reference.md`](docs/yaml_reference.md). For other working configs (symbolic HMC, the OCR baseline) look in `configs/experiments/`.

## Custom graph structures

The library ships with `chain` and `sudoku` graph builders. For any other graph (a CRF over a sentence, a segmentation grid, an arbitrary factor graph), there are two routes.

### 1. Inline edges in the YAML

For small or one-off graphs, list the edges directly under the `graph` block:

```yaml
architecture:
  num_classes: 5
  backbone: {type: config, layers: [...]}
  graph:
    type: inline
    edges: [[0, 1], [1, 2], [2, 0], [2, 3]]      # 4 edges over 4 nodes
  pairwise: {init_scale: 0.1}
```

Edges are stored as a `[E, 2]` tensor of node-index pairs (undirected: each edge listed once). The same `M3N` predictor handles arbitrary topologies — only the edge list changes.

### 2. Edges file on disk

For larger graphs, serialise the `[E, 2]` `LongTensor` once and reference it from the YAML:

```python
import torch
edges = torch.tensor([[0, 1], [1, 2], [2, 3], ...], dtype=torch.long)
torch.save(edges, "my_graph.pt")
```

```yaml
graph:
  type: edges_file
  path: my_graph.pt              # resolved relative to the architecture YAML
```


This bypasses the YAML graph builder entirely and feeds the edge tensor directly to the Trainer.

### Notes

- Node indices in the edge list must lie in `[0, num_nodes)` where `num_nodes` is determined by the input shape (e.g. 81 for Sudoku, `seq_len` for a chain).

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
