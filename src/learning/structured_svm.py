"""
Structured hinge loss for Maximum Margin Markov Networks.

The structured SVM objective (for a single sample) is:

    L(x, y_true) = max_y [ F(y|x) + Delta(y, y_true) ]  -  F(y_true|x)

where:
    F(y|x)           = score of labeling y (unary + pairwise potentials)
    Delta(y, y_true)  = task loss (e.g. Hamming distance)

The inner maximization ("loss-augmented inference") finds the most-violating
labeling y*. This is solved by the inference module (Viterbi for chains,
LP relaxation for general graphs).

Key property: since y* maximizes F + Delta, we always have L >= 0, so
the max(0, ...) clamp is technically redundant but included for safety.

Gradient flow:
    - The inference (finding y*) is NOT differentiable — it's an argmax.
      We run it with detached potentials.
    - The score computation IS differentiable — F(y*|x) and F(y_true|x)
      depend on unary potentials (from the backbone) and pairwise weights.
    - So gradients flow: loss -> score(y*) - score(y_true) -> unary/pairwise -> backbone params.
"""

import torch


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
    batch, T, K = unary.shape

    # Step 1: Loss-augmented inference (no gradients needed)
    #   Find y* = argmax_y [ F(y|x) + Delta(y, y_true) ]
    #   This runs on detached potentials internally.
    y_star = inference_fn(unary, model.pairwise, y_true)

    # Step 2: Compute scores WITH gradients
    #   F(y*|x) and F(y_true|x) must be differentiable for backprop.
    score_star = model.score(unary, y_star, edges)    # [batch]
    score_true = model.score(unary, y_true, edges)    # [batch]

    # Step 3: Compute Hamming loss Delta(y*, y_true)
    #   Number of positions where y* differs from y_true, per sample.
    hamming = (y_star != y_true).float().sum(dim=1)   # [batch]

    # Step 4: Structured hinge loss
    #   L = F(y*|x) + Delta(y*, y_true) - F(y_true|x)
    #
    #   Note: hamming has no gradient (it's a count), but score_star and
    #   score_true do. The gradient of L w.r.t. parameters is:
    #     dL/dθ = dF(y*|x)/dθ - dF(y_true|x)/dθ
    #   which pushes the model to score y_true higher than y*.
    margin = score_star + hamming - score_true         # [batch]
    loss = torch.clamp(margin, min=0.0)                # [batch]

    return loss.mean()
