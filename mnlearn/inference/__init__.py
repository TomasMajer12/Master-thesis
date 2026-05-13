"""mnlearn.inference — decoders for MAP inference at evaluation time.

Two decoders are implemented:

* :func:`viterbi_decode` (and :func:`loss_augmented_viterbi`) — exact MAP
  on chains via dynamic programming.
* :func:`bp_decode` — max-sum belief propagation for general graphs,
  exact on trees and a strong approximation on cyclic graphs (Sudoku).

The decoder is selected at runtime by the YAML field
``inference.eval`` and dispatched in
:func:`mnlearn.learning.builders.build_inference`.
"""

from .belief_prop import bp_decode
from .viterbi     import loss_augmented_viterbi, viterbi_decode

__all__ = [
    "viterbi_decode",
    "loss_augmented_viterbi",
    "bp_decode",
]
