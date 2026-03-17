# Project Structure

## Directory Layout

```
Master-thesis/
├── configs/                        # YAML/JSON experiment configs
│   ├── hmc_symbolic.yaml           # HMC benchmark, symbolic input
│   ├── hmc_visual.yaml             # HMC benchmark, MNIST images
│   ├── sudoku_symbolic.yaml        # Sudoku benchmark, symbolic input
│   └── sudoku_visual.yaml          # Sudoku benchmark, visual input
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                       # Data generation & loading
│   │   ├── __init__.py
│   │   ├── hmc.py                  # HMC sequence generation
│   │   ├── sudoku.py               # Sudoku puzzle generation + solutions
│   │   ├── mnist_pool.py           # MNIST image pool
│   │   └── datasets.py             # PyTorch Dataset classes (symbolic + visual, for both HMC and Sudoku)
│   │
│   ├── models/                     # Neural network backbones + M3N wrapper
│   │   ├── __init__.py
│   │   ├── backbones.py            # Backbone networks: Linear, MLP, CNN (for images)
│   │   └── m3n.py                  # M3N model: unary potentials (from backbone) + pairwise potentials
│   │
│   ├── inference/                  # MAP inference methods
│   │   ├── __init__.py
│   │   ├── viterbi.py              # Viterbi for chain structures
│   │   ├── lp_relaxation.py        # LP relaxation for general graphs
│   │   └── bruteforce.py           # Brute-force for small problems / validation
│   │
│   ├── learning/                   # Training loop + structured loss
│   │   ├── __init__.py
│   │   ├── structured_svm.py       # Structured hinge loss + loss-augmented inference
│   │   ├── trainer.py              # Training loop with early stopping, logging, checkpointing
│   │   └── evaluation.py           # Hamming loss, 0/1 loss, learning curves
│   │
│   └── baselines/                  # Baseline methods
│       ├── __init__.py
│       ├── forward_backward.py     # Optimal Bayes classifier for HMC
│       └── two_stage_sudoku.py     # OCR + constraint solver for visual Sudoku
│
├── scripts/                        # Entry points
│   ├── train.py                    # Main training script (config-driven)
│   ├── evaluate.py                 # Evaluation + plots
│   └── generate_benchmark.py       # Generate & save benchmark datasets
│
├── benchmarks/                     # Pre-generated benchmark data (saved .pt files)
│   ├── hmc/
│   │   ├── config.json
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── sudoku/
│       ├── config.json
│       ├── train/
│       ├── val/
│       └── test/
│
├── results/                        # Experiment outputs (models, plots, summaries)
│   └── ...
│
└── requirements.txt
```

## Key Design Decisions

### 1. Graph structure abstraction (`inference/`)
Viterbi handles chains (HMC), LP relaxation handles general graphs (Sudoku 9x9 grid
with row/col/box constraints). Both expose the same interface:
`infer(unary_potentials, pairwise_potentials, graph) -> labels`.

### 2. M3N model is backbone-agnostic (`models/m3n.py`)
Takes any PyTorch backbone that maps input -> unary potentials. The pairwise potentials
are a learnable parameter matrix. This covers all architecture combos:
- Chain + symbolic + sequence -> Linear/MLP backbone + Viterbi
- Chain + visual + sequence -> CNN backbone + Viterbi
- General graph + symbolic -> Linear backbone + LP
- General graph + visual + single image -> CNN backbone + LP

### 3. Config-driven experiments (`configs/`)
One config file defines everything - dataset, backbone architecture, graph structure,
inference method, training hyperparameters. `scripts/train.py` reads it and wires
everything together.

### 4. Clean separation of inference vs learning
The structured SVM loss calls inference internally (loss-augmented inference), but
inference methods are standalone and reusable for prediction.

## Migration from Current Code

| Current                                      | Proposed                                                          |
|----------------------------------------------|-------------------------------------------------------------------|
| `dataset_HMC.py` (duplicated)                | Consolidated into `src/data/hmc.py` + `src/data/datasets.py`     |
| `SimpleM3N.py` + `NeuralM3N.py`              | `src/models/backbones.py` (Linear, MLP) + `src/models/m3n.py`    |
| `M3NTrainer` (mixed inference + loss + train) | Split: `src/inference/viterbi.py`, `src/learning/structured_svm.py`, `src/learning/trainer.py` |
| No LP inference                              | `src/inference/lp_relaxation.py` (NEW)                            |
| No Sudoku                                    | `src/data/sudoku.py` + `src/baselines/two_stage_sudoku.py` (NEW) |
| Hardcoded `main.py`                          | Config-driven `scripts/train.py`                                  |

## Implementation Priority

1. **Core refactor**: `src/models/`, `src/inference/viterbi.py`, `src/learning/` - restructure existing working code
2. **LP relaxation inference** - the main new algorithmic contribution
3. **Sudoku data generation** + benchmark
4. **Visual/CNN backbone** for image inputs
5. **Two-stage Sudoku baseline**
6. **Evaluation & plots** (learning curves, Hamming + 0/1 loss)
