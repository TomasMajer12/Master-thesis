## Directory Structure

```
Implementation/
├── config/                          # Configuration files
│   ├── default_config.yaml         # Default settings for all experiments
│   └── experiments/                # Experiment-specific configurations
│       ├── linear_m3n.yaml
│       ├── mlp_m3n.yaml
│       └── sudoku_cnn.yaml
│
├── models/                         # Neural network models
│   ├── __init__.py                 # Model factory and exports
│   ├── base_m3n.py                 # Abstract base class for M3N models
│   ├── configurable_m3n.py         # Main configurable M3N implementation
│   ├── architecture_builder.py     # Builds NN from YAML config
│   └── pairwise_potentials.py      # Different pairwise potential types
│
├── data/                           # Dataset generation and loading
│   ├── __init__.py                 # Dataset factory and exports
│   ├── base_dataset.py             # Abstract dataset class
│   ├── hmc_dataset.py              # Hidden Markov Chain data generator
│   └── sudoku_dataset.py           # Sudoku puzzle dataset (future)
│
├── training/                        # Training logic
│   ├── __init__.py                 # Trainer exports
│   ├── trainer.py                  # Main M3N trainer with structured SVM
│   ├── callbacks.py                # Training callbacks (early stopping, etc.)
│   └── losses.py                   # Loss functions (structured hinge, etc.)
│
├── inference/                      # Prediction algorithms
│   ├── __init__.py                 # Inference exports
│   ├── viterbi.py                  # Viterbi decoding for chain structures
│   ├── forward_backward.py         # Forward-backward algorithm
│   └── belief_propagation.py       # For general graph structures (future)
│
├── utils/                          # Utility functions
│   ├── __init__.py                 # Utility exports
│   ├── config_loader.py            # YAML configuration loading and merging
│   ├── metrics.py                  # Evaluation metrics (Hamming loss, accuracy)
│   └── visualization.py            # Plotting and result visualization
│
├── experiments/                    # Experiment runners and evaluation
│   ├── run_experiment.py           # Main experiment runner script
│   └── evaluate_model.py           # Model evaluation and comparison
│
├── notebooks/                       # Jupyter notebooks for interactive work
│   ├── 01_hmc_exploration.ipynb
│   ├── 02_architecture_testing.ipynb
│   └── 03_sudoku_experiments.ipynb
│  
└── tests/                           # Unit tests
    ├── test_models.py
    ├── test_data.py
    ├── test_training.py
    └── test_inference.py
```
