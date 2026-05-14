"""
Evaluation metrics for structured prediction.

Two metrics as required by the thesis:

    1. Hamming loss      — fraction of individual nodes predicted
    2. Hamming distance  — raw count of disagreements per example
    3. Zero-one loss     — fraction of samples where ANY node is wrong

Both ``hamming_loss`` and ``hamming_distance`` accept a ``reduction``

    reduction='mean' → scalar Python float (default; metric for logging)
    reduction='none' → per-example tensor of shape [batch]
"""


def hamming_loss(y_pred, y_true, *, reduction: str = "mean"):
    """Normalized Hamming loss — fraction of incorrectly predicted nodes.

    ℓ(y, y') = (1/N) · Σ_v 𝟙[y_v ≠ y'_v].  Values in [0, 1].

    Args:
        y_pred:    [batch, num_nodes] — predicted labels
        y_true:    [batch, num_nodes] — ground truth labels
        reduction: 'mean' (default) → scalar float, mean over the batch.
                   'none'           → per-example tensor of shape [batch],
                                      values in [0, 1].

    Example:
        y_pred = [[0, 1, 2],     y_true = [[0, 1, 0],
                  [1, 1, 1]]               [1, 0, 1]]
        per-example fractions = [1/3, 2/3]
        reduction='mean' → 0.5
        reduction='none' → tensor([0.333, 0.667])
    """
    per_example = (y_pred != y_true).float().mean(dim=1)  # [batch], in [0,1]
    if reduction == "mean":
        return per_example.mean().item()
    if reduction == "none":
        return per_example
    raise ValueError(
        f"reduction must be 'mean' or 'none', got {reduction!r}"
    )


def hamming_distance(y_pred, y_true, *, reduction: str = "mean"):
    """Raw Hamming distance — integer count of disagreements per example.

    Args:
        y_pred:    [batch, num_nodes] — predicted labels
        y_true:    [batch, num_nodes] — ground truth labels
        reduction: 'mean' (default) → scalar float, mean count over batch.
                   'none'           → per-example tensor of shape [batch],
                                      values in {0, 1, ..., num_nodes}.

    Example:
        y_pred = [[0, 1, 2],     y_true = [[0, 1, 0],
                  [1, 1, 1]]               [1, 0, 1]]
        per-example counts = [1, 2]
        reduction='mean' → 1.5
        reduction='none' → tensor([1.0, 2.0])
    """
    per_example = (y_pred != y_true).float().sum(dim=1)  # [batch], integer-valued
    if reduction == "mean":
        return per_example.mean().item()
    if reduction == "none":
        return per_example
    raise ValueError(
        f"reduction must be 'mean' or 'none', got {reduction!r}"
    )


def zero_one_loss(y_pred, y_true, *, reduction: str = "mean"):
    """Zero-one loss — fraction of samples where the entire labeling is wrong.

    A sample counts as correct only if all nodes match.  Not
    node-decomposable

    Args:
        y_pred:    [batch, num_nodes] — predicted labels
        y_true:    [batch, num_nodes] — ground truth labels
        reduction: 'mean' (default) → scalar float, mean over the batch.
                   'none'           → per-example tensor of shape [batch],
                                      values in {0.0, 1.0}.

    Example:
        y_pred = [[0, 1, 2],     y_true = [[0, 1, 2],    <- correct (all match)
                  [1, 1, 1]]               [1, 0, 1]]    <- wrong (node 1 differs)
        per-example errors = [0.0, 1.0]
        reduction='mean' → 0.5
        reduction='none' → tensor([0.0, 1.0])
    """
    # A sample is wrong if ANY node differs.
    per_example = (~ (y_pred == y_true).all(dim=1)).float()  # [batch], in {0,1}
    if reduction == "mean":
        return per_example.mean().item()
    if reduction == "none":
        return per_example
    raise ValueError(
        f"reduction must be 'mean' or 'none', got {reduction!r}"
    )


