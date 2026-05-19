# `configs/`

Experiment and architecture YAMLs.

| Subdirectory | What | Referenced from |
|---|---|---|
| `architectures/` | Backbone + (for M3N) graph + pairwise specs | `experiments/*.yaml` via relative path |
| `experiments/` | Concrete runs — data + training + architecture reference | `mnlearn.learning.train` (M3N) or `mnlearn.baselines.train_baseline` (OCR) |

**Full schema reference:** [`docs/yaml_reference.md`](../docs/yaml_reference.md)
— every field, allowed values, defaults, validator constraints, and
worked examples for all three YAML kinds.
