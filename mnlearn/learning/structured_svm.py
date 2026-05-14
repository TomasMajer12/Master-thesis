"""
Structured hinge loss for Maximum Margin Markov Networks.

The structured SVM objective (for a single sample) is:

    L(x, y_true) = max_y [ F(y|x) + Delta(y, y_true) ]  -  F(y_true|x)

where:
    F(y|x)           = score of labeling y (unary + pairwise potentials)
    Delta(y, y_true)  = task loss (e.g. Hamming distance)

The inner maximization ("loss-augmented inference") finds the most-violating
labeling y*. In this codebase the inner max is solved by Viterbi (chain
graphs only); the config validator restricts this loss to
``loss=m3n_hinge`` paired with ``inference.train=viterbi``. The
LP-relaxed variant for general graphs is a separate loss function
(:func:`mnlearn.learning.lp_m3n_loss`) with its own training contract.
"""

import torch

from .evaluation import hamming_loss


def structured_hinge_loss(model, unary, y_true, edges, inference_fn):
    """Compute the structured hinge loss for a batch.

    Args:
        model:        M3N model (needed for score computation and pairwise weights)
        unary:        [batch, T, K] — unary potentials (with gradient)
        y_true:       [batch, T]    — ground truth labels
        edges:        [num_edges, 2] — graph structure
        inference_fn: callable(unary, pairwise, y_true) -> y_star
                      Loss-augmented inference function (e.g. loss_augmented_viterbi)

    Returns:
        loss: scalar — mean structured hinge loss over the batch
    """
    assert unary.dim() == 3, (
        f"structured_hinge_loss expects unary of shape [B, T, K]; "
        f"got {tuple(unary.shape)}"
    )

    # Step 1: Loss-augmented inference (no gradients needed)
    #   Find y* = argmax_y [ F(y|x) + Delta(y, y_true) ]
    y_star = inference_fn(unary, model.pairwise, y_true)

    # Step 2: Compute scores with gradients
    #   F(y*|x) and F(y_true|x) must be differentiable for backprop.
    score_star = model.score(unary, y_star, edges)    # [batch]
    score_true = model.score(unary, y_true, edges)    # [batch]

    # Step 3: Compute normalized Hamming loss Delta(y*, y_true) per example.
    hamming = hamming_loss(y_star, y_true, reduction="none")   # [batch]

    # Step 4: Structured hinge loss
    #   L = F(y*|x) + Delta(y*, y_true) - F(y_true|x)
    margin = score_star + hamming - score_true         # [batch]
    loss = torch.clamp(margin, min=0.0)                # [batch]

    return loss.mean()
