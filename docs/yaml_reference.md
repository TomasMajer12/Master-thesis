# YAML configuration reference

Every experiment in this project is driven by a YAML file. There are
two kinds, both under `configs/`:

| Kind | Where | Schema | Loader | Runner |
|---|---|---|---|---|
| Architecture | `configs/architectures/<name>.yaml` | reused inside experiment YAMLs | (referenced by relative path) | — |
| M3N experiment | `configs/experiments/<name>.yaml` | [`Config`](../mnlearn/config/schema.py) | `mnlearn.config.load_config` | `mnlearn.learning.train` |

The baseline OCR pipeline uses a parallel schema
(`BaselineConfig` / `load_baseline_config` / `train_baseline`) with
similar grammar; see the example configs in
`configs/experiments/baseline_*.yaml` and the dataclass definitions in
`mnlearn/config/schema.py` for details.

An experiment YAML references its architecture either by **relative
path** (`architecture: ../architectures/sudoku_lenet5.yaml`) or as an
**inline dict** (the form produced by `dump_config` when an artifact
is saved). Both forms round-trip through the loader.

Running an experiment:

```python
from mnlearn.learning import train
result = train("configs/experiments/lpm3n_visual_sudoku.yaml")
```

---

## 1. Architecture YAML

Defines the neural backbone and (for M3N) the graph structure plus the
pairwise-potential init scale.

| Field | Type | Required | Notes |
|---|---|---|---|
| `num_classes` | int > 0 | ✓ | Number of class labels per node (K). |
| `backbone.type` | str | ✓ | `config` ｜ `torchvision` ｜ `wrapped` |
| `backbone.layers` | list of dicts | when `type=config` | Layer specs (see below). |
| `backbone.name` | str | when `type=torchvision`/`wrapped` | torchvision model name OR dotted import path (`pkg.module.Class` or `pkg.module:Class`). |
| `backbone.pretrained` | bool | no (default false) | torchvision only. |
| `backbone.freeze` | bool | no (default false) | Freeze the wrapped feature extractor. |
| `backbone.feature_dim` | int > 0 | when `type=torchvision`/`wrapped` | Output dim of the wrapped feature extractor. |
| `graph.type` | str | ✓ for M3N (must be **absent** for OCR baseline) | `chain` ｜ `sudoku` ｜ `edges_file` ｜ `inline` |
| `graph.seq_len` | int ≥ 2 | when `graph.type=chain` | Chain length T. |
| `graph.path` | str | when `graph.type=edges_file` | Path to a `.pt` file holding `[E, 2]` edges, relative to the architecture YAML. |
| `graph.edges` | list of `[src, dst]` pairs | when `graph.type=inline` | — |
| `pairwise.init_scale` | float ≥ 0 | no (default 0.1) | Std of initial pairwise weights. M3N only. |

### Backbone layer specs (`backbone.type == config`)

Each layer is a dict whose `type` key names a class in `torch.nn`.
Common lowercase aliases are accepted; anything else is looked up by
exact case in `torch.nn`. Remaining keys are passed as keyword arguments.

**Aliases:** `linear`, `relu`, `leaky_relu`, `sigmoid`, `tanh`,
`dropout`, `batchnorm1d`, `batchnorm2d`, `conv2d`, `maxpool2d`,
`flatten`. **Anything else:** by exact name (e.g. `Softmax`, `GELU`,
`LayerNorm`).

The final layer must produce `num_classes` outputs. When the final
layer is a plain `linear`, the validator checks
`out_features == num_classes` at load time.

For full working examples, see
`configs/architectures/sudoku_lenet5.yaml` (M3N, has terminal softmax)
and `configs/architectures/ocr_lenet5.yaml` (OCR baseline, no softmax).

---

## 2. M3N experiment YAML

Schema: [`Config`](../mnlearn/config/schema.py). Loader:
`mnlearn.config.load_config`. Runner: `mnlearn.learning.train`.

```yaml
experiment:                        # block: ExperimentCfg
  name:       <str, required>      # used as the artifact subdirectory
  seed:       <int ≥ 0, default 42>
  device:     <auto | cpu | cuda, default auto>
  output_dir: <str, default "">    # "" → results/{name}

architecture: <relative path str OR inline dict>

data:                              # block: DataCfg
  task:       <sudoku | hmc, required>
  mode:       <symbolic | visual, required>
  paths:                           # required for the chosen task/mode
    benchmark: <str>               # required when task=sudoku
    mnist:     <str>               # optional when mode=visual; falls back to user cache
  train_size: <int > 0, required>  # capped at the available split size
  val_size:   <int > 0, required>
  test_size:  <int > 0, required>
  batch_size: <int > 0, required>

training:                          # block: TrainingCfg
  loss: <m3n_hinge | lp_m3n, required>

  inference:                       # block: InferenceCfg
    train: <viterbi | lp, required>   # m3n_hinge → viterbi; lp_m3n → lp
    eval:  <viterbi | bp, required>
    params: {bp_iters: <int, default 50>}

  optimizer:                       # block: OptimizerCfg
    type:             <adam | sgd, required>
    lr:               <float > 0, required>      # backbone (and fallback for lr_pairwise / lr_phi)
    weight_decay:     <float ≥ 0, default 0.0>
    lr_pairwise:      <float ≥ 0, default 0.0>   # pairwise-W lr; 0 = fall back to lr
    lr_phi:           <float ≥ 0, default 0.0>   # phi lr; 0 = fall back to lr (lp_m3n only)
    weight_decay_phi: <float ≥ 0, default 0.0>   # MUST be 0 unless loss=lp_m3n
    phi_init_std:     <float ≥ 0, default 0.0>   # MUST be 0 unless loss=lp_m3n

  num_epochs: <int > 0, required>

  scheduler:                       # block: SchedulerCfg, default {type: none}
    type:   <none | cosine | step | exp | lambda>
    params: <dict, scheduler-specific>     # e.g. lambda: {offset: 100}

  eval_every: <int > 0, default 1>

  early_stopping:                  # block: EarlyStoppingCfg
    monitor:   <dotted path str, default "val_metrics.hamming">
    patience:  <int ≥ 0, default 10>
    min_delta: <float ≥ 0, default 0.001>

logging:                           # block: LoggingCfg
  verbose:       <bool, default true>
  print_metrics: <list[str] | null, default null>   # dotted paths to print per eval
```

### Validator constraints

The validator emits every problem in a single error. Notable
cross-field rules:

- `loss=m3n_hinge` requires `inference.train=viterbi`.
- `loss=lp_m3n` requires `inference.train=lp`.
- For any `loss != lp_m3n`, `weight_decay_phi` and `phi_init_std` must
  be `0.0` (or omitted).
- The final `linear` layer's `out_features` must equal `num_classes`.
- `task=sudoku` requires `paths.benchmark`. `mode=visual` does **not**
  require `paths.mnist` — when absent, MNIST falls back to the user
  cache directory.
- `monitor` must be a dotted path of identifiers. Whether the path
  resolves to a real metric is checked at runtime by the Trainer.

### Monitor / print_metrics paths

Both `monitor` and entries in `print_metrics` are dotted paths into
the per-epoch record built inside `Trainer.fit`. Useful values:

- `val_metrics.hamming` (default monitor) — per-cell error on validation.
- `val_metrics.zero_one` — per-puzzle error on validation.
- `train_loss` — mean training loss for the epoch.
- `diagnostics.phi_norm` — mean L2 norm of the per-example phi (LP-M3N only).
- `diagnostics.pairwise_diag_mean` / `pairwise_off_diag_mean` — pairwise W shape.

### Working examples

Two reference experiment YAMLs live next to this doc:

- `configs/experiments/m3n_hinge_hmc_symbolic.yaml` — chain hinge.
- `configs/experiments/lpm3n_visual_sudoku.yaml` — visual Sudoku LP-M3N.

`dataclasses.replace` is the idiomatic way to vary one knob at a time:

```python
from dataclasses import replace
from mnlearn.config import load_config
from mnlearn.learning import train

base = load_config("configs/experiments/lpm3n_visual_sudoku.yaml")
for n in (1000, 2000, 5000):
    cfg = replace(
        base,
        experiment = replace(base.experiment, name=f"lpm3n_visual_sudoku_n{n}"),
        data       = replace(base.data,       train_size=n),
    )
    train(cfg)
```

---

## 3. Artifacts written by every run

Every successful run produces, under `output_dir`:

| File | Contents |
|---|---|
| `config.yaml` | The resolved config with the architecture inlined and `experiment.output_dir` set. Round-trips through `load_config`. |
| `history.json` | The full per-epoch history dict (parallel arrays + per-epoch dicts for train/val metrics + diagnostics). |
| `results.json` | The test-phase metrics dict. |
| `model.pt` | `model.state_dict()` of the fitted model. |

`config.yaml` is the reproducibility anchor: every figure or table in
the thesis points to one of these run directories.
