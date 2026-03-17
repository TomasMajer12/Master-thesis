"""
Evaluation metrics for structured prediction.

Two metrics as required by the thesis:

    1. Hamming loss  — fraction of individual nodes predicted incorrectly
    2. Zero-one loss — fraction of samples where ANY node is wrong
                       (entire labeling must match)
"""

import torch


def hamming_loss(y_pred, y_true):
    """Fraction of incorrectly predicted nodes (averaged over all nodes).

    Args:
        y_pred: [batch, num_nodes] — predicted labels
        y_true: [batch, num_nodes] — ground truth labels

    Returns:
        float — error rate in [0, 1]

    Example:
        y_pred = [[0, 1, 2],     y_true = [[0, 1, 0],
                  [1, 1, 1]]               [1, 0, 1]]
        Hamming = 2 wrong out of 6 = 0.333
    """
    incorrect = (y_pred != y_true).float().sum()
    total = y_true.numel()
    return (incorrect / total).item()


def zero_one_loss(y_pred, y_true):
    """Fraction of samples where the ENTIRE labeling is wrong.

    A sample counts as correct only if ALL nodes match.

    Args:
        y_pred: [batch, num_nodes] — predicted labels
        y_true: [batch, num_nodes] — ground truth labels

    Returns:
        float — error rate in [0, 1]

    Example:
        y_pred = [[0, 1, 2],     y_true = [[0, 1, 2],    <- correct (all match)
                  [1, 1, 1]]               [1, 0, 1]]    <- wrong (node 1 differs)
        0/1 loss = 1 wrong out of 2 = 0.5
    """
    # A sample is correct only if no node differs
    sample_correct = (y_pred == y_true).all(dim=1).float()  # [batch]
    return (1.0 - sample_correct.mean()).item()
