# Architecture

How the pieces of `mnlearn` fit together. For YAML field reference see
[`yaml_reference.md`](yaml_reference.md); for theoretical background see
the thesis.

## 1. Overview

The library has four layers wired together by YAML configs:

```
configs/*.yaml ──► mnlearn.config ──► Config dataclass
                                            │
                                            ▼
              ┌──── mnlearn.data ─── (X, Y) tensors
              │
              ▼
        mnlearn.models.M3N ── unary, pairwise W
              │
              ▼  loss + inference
      mnlearn.learning.Trainer ◄── mnlearn.inference
              │
              ▼
       results/<name>/{config.yaml, history.json, results.json, model.pt}
```

Each layer is independent: the model does not know about the loss, the
loss does not know about the data, and inference does not know about
training. The `Trainer` wires them and the YAML runner
(`mnlearn.learning.runner.train`) wires the wire.

## 2. Model (`mnlearn.models`)

The predictor is the `M3N` class in `mnlearn/models/m3n.py`. It scores a
labeling `y` of a graph `(V, E)` given input `x` as:

    F(y | x) = Σ_v  unary(x_v)[y_v]  +  Σ_{(u,v) ∈ E}  W[y_u, y_v]

- **Unary potentials** come from a backbone network. The backbone is
  generic over input shape — see `mnlearn/models/backbones.py` for the
  three flavours (`ConfigBackbone`, `WrappedBackbone`, plus a
  torchvision dispatcher).
- **Pairwise potentials** are a single learned `[K, K]` matrix `W`,
  shared across every edge. The matrix is unconstrained (not forced
  symmetric); the rationale is in the thesis §4.2.2 — the M3N is
  generic over graph orientation, so committing to `W = Wᵀ` would
  mis-parameterise directed graphs (e.g. HMC transitions).
- **Graph structure is external.** The edge list is a `[E, 2]`
  LongTensor passed at runtime to `model.score` and to inference
  functions. The same `M3N` instance works for chain graphs, the
  Sudoku constraint graph, or any user-supplied edge list. Constructors
  live in `mnlearn/models/graph.py` (`chain_edges`, `sudoku_edges`,
  `build_graph`).

A complete predictor is assembled by `build_model(arch_cfg)` in
`mnlearn/models/builders.py` from an `ArchitectureCfg`.

## 3. Loss (`mnlearn.learning`)

Two training paths are implemented; the loss is selected by
`training.loss` in the YAML:

| `loss` | Where defined | Inner inference | Graph |
|---|---|---|---|
| `m3n_hinge` | `mnlearn/learning/structured_svm.py` | loss-augmented Viterbi | chains only |
| `lp_m3n`    | `mnlearn/learning/lp_m3n.py`        | none (LP-relaxation baked in) | any graph |

**`m3n_hinge`** implements the standard structured-hinge identity
`L = max(0, F(y*) + Δ(y*, y) − F(y, y))` with `y* = loss-augmented
Viterbi`. Restricted to chains because Viterbi is the only
loss-augmented decoder we ship.

**`lp_m3n`** implements the LP-relaxed M3N loss of Franc & Yermakov
(2021), Eq. (24). Each training example carries its own dual variable
`φ_i ∈ R^{2 × E × K}` ("phi bank"). The loss decomposes into local
argmax operations, so no inference oracle is needed at training time —
which is what makes it usable on cyclic graphs like Sudoku.

The phi bank is allocated by `Trainer` in `_build_phi_bank` (one
`[2, E, K]` tensor per training example) and added as a third
optimizer parameter group with its own learning rate `lr_phi`. The
best-epoch model snapshot stored in `model.pt` does **not** include
the phi bank: phi stays at the last epoch's state. This is harmless
because eval-time decoding uses only `model.unary(x)` and
`model.pairwise`.

Both losses use the normalised Hamming task loss `Δ(y, y') =
(1/T) · |{v : y_v ≠ y'_v}|`. The normalisation keeps the total task
loss in `[0, 1]` and prevents the unary shortcut pathology at the
LP-relaxation level (see thesis §2.4).

## 4. Inference (`mnlearn.inference`)

Two MAP decoders are implemented:

| `inference.eval` | Function | Exactness | Graph |
|---|---|---|---|
| `viterbi` | `viterbi_decode` | exact | chains only |
| `bp`      | `bp_decode`      | exact on trees, approximate on cycles | any graph |

`viterbi_decode` is standard O(T·K²) DP, fully vectorised over batch
and class. `bp_decode` is batched max-sum belief propagation with
message normalisation (subtract max per message to avoid drift). Both
are pure functions — they take `(unary, pairwise, edges)` and return
`y`.

The `(loss, inference.train)` pairing is constrained by what is wired
in `mnlearn/learning/builders.py::build_inference`:

- `loss = m3n_hinge` ⇒ `inference.train = viterbi` (loss-augmented).
- `loss = lp_m3n`    ⇒ `inference.train = lp` (no oracle; returned as
  `None` and the LP-augmented decoder is baked into the loss).

The validator (`mnlearn.config.validate`) rejects any other pairing at
YAML load time.

## 5. Training (`mnlearn.learning.runner`)

The end-to-end entry point is `train(cfg)` in
`mnlearn/learning/runner.py`. One call runs:

```python
def train(cfg):
    set_seed(cfg.experiment.seed)
    device = resolve_device(cfg.experiment.device)
    data         = build_datasets(cfg.data)
    model, edges = build_model(cfg.architecture)
    trainer      = Trainer(model, edges.to(device), cfg.training, n_train)
    history      = trainer.fit(data["train"], data["val"], ...)
    test_metrics = trainer.metrics(*data["test"])
    save_artifacts(cfg, output_dir, model, history, test_metrics)
    return {"config", "history", "test_metrics", "model", "trainer",
            "output_dir"}
```

The `Trainer.optimizer` splits model parameters into up to three groups
with independent learning rates (see thesis §4.3.2 for the rationale):

- **Group 0** — backbone parameters, `lr = optimizer.lr`.
- **Group 1** — pairwise matrix `W`, `lr = optimizer.lr_pairwise`
  (sentinel `0.0` → same as `lr`).
- **Group 2** — phi bank (only for `lp_m3n`),
  `lr = optimizer.lr_phi` (sentinel `0.0` → same as `lr`).

All groups decay synchronously through the LR scheduler. Eval happens
every `eval_every` epochs; the LR scheduler is stepped *after* the
eval block so the logged `lr` matches the value actually used during
the epoch's training pass. Early stopping monitors a dotted path into
the per-epoch record (default `val_metrics.hamming`); on improvement,
`model.state_dict()` is deep-copied as the new best-epoch snapshot.

## 6. Reproducibility

`mnlearn.learning.runtime.set_seed(seed)` seeds Python's `random`,
NumPy, PyTorch's CPU RNG, PyTorch's CUDA RNG on every device, and sets
`torch.backends.cudnn.deterministic = True` and `benchmark = False`. It
is called by both `train` and `train_baseline` before any data or
model is constructed.

Every run writes a self-contained `config.yaml` next to its artifacts
with the architecture inlined and `experiment.output_dir` baked in.
This file round-trips back through `load_config` — re-running it
reproduces the same training trajectory (modulo non-determinism in
non-cuDNN GPU kernels, which are not currently constrained).

## 7. Baselines (`mnlearn.baselines`)

A parallel non-structured pipeline implements the two-stage OCR +
solver baseline. The baseline uses a separate config schema
`BaselineConfig` that omits `architecture.graph`,
`architecture.pairwise`, and `training.inference` (none of them apply
when there is no Markov-network decision layer).

End-to-end runner:

- `train_baseline(cfg)` in `mnlearn/baselines/runner.py`.
- Trains an OCR digit classifier with `nn.CrossEntropyLoss` on
  *clue cells* of visual Sudoku (extracted by `extract_clue_pairs`).
- At test time, runs `evaluate_two_stage`, which forwards the OCR over
  every cell, builds an OCR-corrupted quiz at clue positions, feeds it
  to the hard-coded constraint-propagation + backtracking solver in
  `mnlearn/baselines/sudoku_solver.py`, and compares the solver output
  to the ground-truth solution.
- Reports three metrics: `zero_one` (puzzles wrong), `solver_feasibility`
  (solver found *some* grid), `ocr_clue_error` (OCR digit error at clue
  cells, diagnostic).

The Forward-Backward Bayes-optimal classifier
(`mnlearn/baselines/forward_backward.py`) is a separate reference
predictor for symbolic HMC; it is not exercised by `train_baseline`
and is used directly from the HMC tightness notebooks.

## 8. Module map

| Module | Owns |
|---|---|
| `mnlearn.config`       | YAML schema, loader, dumper, validator. Torch-free. |
| `mnlearn.data`         | Datasets (Sudoku, HMC) + benchmark build/load + chapter-3 stats |
| `mnlearn.models`       | `M3N` predictor + backbones + graph constructors |
| `mnlearn.inference`    | Viterbi + Belief Propagation decoders |
| `mnlearn.learning`     | Unified `Trainer`, structured losses, evaluation, runner |
| `mnlearn.baselines`    | Two-stage OCR+solver pipeline + Forward-Backward classifier |
| `mnlearn.experiments`  | Result-collection DataFrame + plotting helpers (used by notebooks) |
