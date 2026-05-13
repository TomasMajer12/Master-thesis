# TODO

> Status: ✅ done, 🔁 in progress, ⏳ pending, ❓ undecided.

## Library / packaging

- ⏳ YAML-driven experiment + architecture configs.
- ⏳ Migrate `scripts/train_hmc.py` and `scripts/baseline_sudoku_ocr.py` to YAML configs (currently still hardcoded `CONFIG = {}` dicts).
- ⏳ Replace `print` with `logging` in `mnlearn/`.
- ⏳ Type hints on public-API signatures (m3n, backbones, trainers, losses).
- ⏳ Add `docs/architecture.md` explaining model / loss / inference relationships.
- ⏳ **§4.5 prerequisite** — Add `mnlearn/baselines/ocr.py`: OCR classifier training loop (per-cell cross-entropy on visual MNIST clue images). Currently no consolidated OCR baseline in the library; the `scripts/baseline_sudoku_ocr.py` referenced in the TODO above is stale (scripts/ folder no longer exists).
- ⏳ **§4.5 prerequisite** — Add `mnlearn/baselines/sudoku_solver.py`: hard-coded Sudoku solver (constraint propagation + backtracking). Suggested public API: `solve(quiz: np.ndarray) -> np.ndarray | None`.
- ⏳ **§4.5 prerequisite** — Add `mnlearn/baselines/two_stage.py`: pipeline gluing OCR predictions into the hard-coded solver, plus evaluation helpers.
- ⏳ **§4.6.2 prerequisite** — Populate `configs/experiments/` with the actual experiment configurations referenced in Chapter 5 (currently empty).
- ⏳ **§4.6.4 prerequisite** — Add a single `set_all_seeds(seed)` helper covering NumPy, PyTorch (CPU + CUDA), and `random`. Verify whether one exists; add if not.
- ⏳ **§4.2.1 prerequisite** — Add `configs/architectures/hmc_lenet5.yaml` (or similar) — chain-targeted backbone configuration matching the LeNet-5 + HMC visual-chain training experiments. (Currently only Sudoku-targeted YAMLs exist.)

## Architectures

- ✅ Configurable backbone via YAML (`ConfigBackbone`).
- ✅ LeNet5 architecture YAML matching reference thesis.
- ⏳ Verify the `WrappedBackbone` path with a torchvision pretrained backbone (currently unexercised).
- ❓ Single-image-input variant (one image of full grid + segmentation), instead of per-cell sequence.

## Learning algorithms

- ✅ Structured hinge loss with Viterbi-based loss-augmented inference (chain).
- ✅ LP-M3N loss with per-example dual variables φ (general graph).
- ⏳ Add φ-norm regularisation knob (already wired through `weight_decay_phi`; keep tracking).
- ⏳ Tightness validation experiment: LP-M3N vs oracle structured-hinge on a chain (sanity check).
- ❓ Generalize `lp_m3n_loss` (and `structured_hinge_loss`) to accept a user-supplied **node-decomposable** task-loss matrix `L ∈ R^{K×K}` with `L[c,c]=0`, `L[c,a]≥0`. Default stays `(1/T)·(1−eye(K))` (normalized Hamming). Two-line change in the augmentation:
  `aug = unary + L[y_true]`. Schema needs a new field + validator (non-negative, zero diagonal). Adds an axis of variation for weighted-Hamming experiments. **Not implementable:** 0/1 over the full output, F1, BLEU, edit distance — these are non-decomposable and incompatible with margin-rescaling at training time (theorem, not missing feature). Document this in the docstring + thesis §2.4.
- ❓ Document the `inference_fn` contract on `structured_hinge_loss`: it **must** be a loss-augmented decoder (Hamming bonus baked in before argmax), otherwise the structured-hinge identity `L = F(y*) + Δ(y,y*) − F(y,y)` silently breaks. Currently enforced by the config validator (`loss: m3n_hinge` ↔ `inference.train: viterbi`), so safe in practice. Revisit if `structured_hinge_loss` is ever exposed outside the YAML pipeline.

## Inference

- ✅ Viterbi (chain).
- ✅ Belief Propagation (general graph).
- ⏳ Re-enable ADAG / LP-relaxation inference via `manet` library at test time.
- ⏳ Three-decoder comparison on test set: greedy / BP / ADAG.

## Experiments

- ✅ Visual Sudoku M3N+CNN learning-curve sweep (`train_sizes ∈ {1000, 2000, 5000}`).
- ✅ Two-stage OCR baseline learning-curve sweep.
- 🔁 LeNet5 + λ sweep at N=100 (currently running).
- ⏳ Long run with best-λ from the sweep.
- ⏳ Reference-matched comparison: same test-set size (100 puzzles) and architecture (LeNet5) as Kadlec 2022.
- ⏳ Final report-table eval with BP + ADAG decoders side-by-side.

## Tests

- ✅ pytest suite for models, builders, config, learning, run_experiment, viterbi.
- ⏳ Fix or remove `tests/test_lp_relaxation.py` (currently broken — imports nonexistent `max_sum_diffusion`).
- ⏳ Add LP-M3N tightness + finite-difference gradient tests.

## Thesis writeup

- ⏳ Method chapter: M3N + LP relaxation derivation, joint NN+MN training algorithm.
- ⏳ Implementation chapter: library structure, design decisions, reproducibility.
- ⏳ Experiments chapter: setup, metrics (Hamming + 0/1), learning curves, sweep results.
- ⏳ Discussion: φ-saturation finding, softmax-on-unary mitigation, comparison with Kadlec 2022 and SATNet.
- ⏳ Final results table: M3N+LeNet5 vs OCR+solver baseline vs reference thesis.
- ❓ Theoretical background: add Belief Propagation subsection in §2.3 (currently invoked in introduction and §2.3 closing but never defined). Add only once the inference path on cyclic graphs is locked.
- ⏳ §2.2 intro paragraph: rephrase to reduce "Markov Network" repetition. Current and previously-attempted rewrites read awkwardly — needs a fresh take.
- ⏳ §2.2.3 Sudoku-MN paragraph: revisit the soft-vs-hard pairwise pin-down once the final parameterization in §2.5 is locked. Currently states the thesis "uses a learned soft preference" — confirm this is still accurate, or refine wording (e.g., explicit hard-mask + learned residual, etc.).
- ⏳ Method chapter (§3): describe the inference algorithm used at evaluation time on cyclic graphs (BP, ADAG, or whichever is finalized). §2.3 background defers this choice via "...the specific algorithm is discussed in Chapter~\ref{chap:method}".
- ⏳ §2.4 LP-M3N derivation: switch Hamming loss to the **normalized** form (per-component bonus $1/N$ instead of $1$) to match the implementation in `mnlearn/learning/lp_m3n.py` (which uses `1/T` to avoid the unary-shortcut pathology). Apply this when the §2.4 section pass starts.
- ⏳ §2.5.2 figure asset: download the LeNet-5 architecture diagram from https://d2l.ai/chapter_convolutional-neural-networks/lenet.html#lenet (cited as `d2l-lenet`) and place at `thesis/figures/lenet5.{png,pdf}`. The `\includegraphics` reference is already wired up.
- ⏳ §2.5.2 / Method: describe the single-image (full-grid) backbone variant once the corresponding experiment is designed. Out of scope per item above; relevant architectures: fully-convolutional with $9 \times 9 \times K$ output head, or torchvision backbone + grid-pooling head.
- ⏳ Method chapter LeNet-5 details: bring the §3.4.3-style configuration into the Method chapter — ReLU activations, softmax on last layer, output dimensionality $K=9$ for Sudoku, weight-decay regularization on all backbone parameters (Adam's `weight_decay`), naming of the resulting algorithm.
- 🔁 Softmax tail on M3N LeNet-5: partial answer. The HMC symbolic tightness experiment (`linear_chain.yaml`, no Softmax, `lr_phi` set) converges to the exact structured-hinge oracle — confirms that `lr_phi` alone is enough to control φ-saturation on chains, and validates LP-M3N implementation correctness end-to-end. **Open on the Sudoku side**: does LeNet-5 need the Softmax safety margin on the cyclic graph? Run a single Sudoku-visual ablation: same architecture, same `lr_phi`, two runs (Softmax vs no-Softmax). If both converge to similar test 0/1, drop the Softmax and rewrite §4.2.1's "Tail softmax" paragraph as "we tried both and chose X". If only the Softmax variant converges, the current paragraph stands but should explicitly contrast against the chain finding.
- ⏳ Drop MLP from §2.5.2. **Confirmed during the §4.2.1 codebase scan**: `configs/architectures/` contains only `linear_chain.yaml`, `hmc_lenet5.yaml`, `ocr_lenet5.yaml`, `sudoku_lenet5.yaml` — no MLP YAML. §4.2.1 (just-written) lists only the **two** concrete architectures actually exercised (linear, LeNet-5); §2.5.2 still claims **three families** (linear / MLP / LeNet-5). To restore consistency:
  - §2.5.2 opening: change "three concrete backbone families … a linear baseline, a multilayer perceptron, and LeNet-5" → "two concrete backbone families … a linear baseline and LeNet-5".
  - §2.5.2 paragraph stack: drop the `\paragraph{Multilayer perceptron (MLP).}` paragraph entirely.
- ❓ Decide $(w, \theta, W)$ notation across Chapter 2. Three places currently disagree: §2.2.2 and §2.4.5 imply $w$, $\theta$, $W$ are three separate entities ("the linear weights $w$, the neural backbone parameters $\theta$, and the pairwise matrix $W$"); §2.5.1 says $w = (\theta, W)$. Pick one convention and harmonize:
  - **Option I** (recommended) — $w$ is the abstract symbol used in §2.4 derivations, *realized* as $(\theta, W)$ in the implementation. Update §2.2.2 to drop "linear weights $w$" framing; update §2.4.5 to talk about $(\theta, W)$ explicitly when discussing convexity/optimization.
  - **Option II** — drop the abstract $w$ after §2.4 derivations. Use $(\theta, W)$ explicitly from §2.4.5 onwards; update §2.5.1 to drop the $w = (\theta, W)$ statement.
- ⏳ §3.1 (Datasets overview) — add a forward-reference to the library section: "All data construction is reproducible from the `mnlearn.data` library, whose layout is described in Section~\ref{sec:lib-implementation} (Chapter~\ref{chap:method})." Apply once the §4.6 label is locked.
- ❓ §4.1 (Method overview) — produce a system-diagram figure (`figures/method_pipeline.png`) showing the full pipeline (input → backbone → unaries → MN → MAP → output). Decision needed: TikZ block diagram inside LaTeX vs. external draw.io / equivalent exported as PNG.
- ⏳ §4.3.4 (hyperparameter selection) — pin final $\lambda$, weight_decay, and learning-rate values from the in-flight LeNet5 + λ sweep at $N=100$ before §4.3.4 can be written. Coupled with the "Experiments" entries above.
- ⏳ §4.4.2 (cyclic-graph inference) — describe BP in detail; describe ADAG as alternative once it is re-enabled (see "Inference" section above). Do not commit to a single decoder until decided.
- ⏳ §4.2.2 (MN decision layer) — pin pairwise-matrix initialization (hard-mask + learned residual vs. learned-from-zero) before §4.2.2 can be written. Closes the deferred §2.2.3 TODO entry above.

## References

- Franc & Yermakov, ACML 2021 — *Learning Maximum Margin Markov Networks from Examples with Missing Labels*.
- Franc, Průša & Yermakov, ECML 2022 — *Consistent and Tractable Algorithm for Markov Network Learning*.
- Wang, Donti, Wilder & Kolter, ICML 2019 — *SATNet*.
- Kadlec, FEL CVUT bachelor thesis 2022 — *Visual Sudoku Solver*.



fix 01 loss in triner // curr hamming

⏳ §3.3.4 Figure 3.7 (`hmc_transition_runs.png`): drop the right panel (run-length histogram) — looks unattractive — and keep just the empirical transition matrix. Update `mnlearn/data/figures.py::plot_hmc_transition_and_runs` accordingly (rename to e.g. `plot_hmc_transition` and remove the run-length panel). Update §3.3.4 caption + figure width and remove the run-length sentence in the surrounding text.


- sudoku solver citations