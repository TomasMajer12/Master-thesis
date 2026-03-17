"""
Viterbi algorithm for MAP inference on chain-structured graphs.

Given a chain graph  0 - 1 - 2 - ... - (T-1)  with K classes per node:

    y* = argmax_y  sum_t unary[t, y_t]  +  sum_t pairwise[y_{t-1}, y_t]

Viterbi solves this exactly in O(T * K^2) via dynamic programming.

Two variants:
    viterbi_decode          — standard MAP inference
    loss_augmented_viterbi  — adds a per-node loss term (for structured SVM training)

Both are fully vectorized over the batch dimension and over classes
(no Python loops over K). The only loop is over the sequence length T,
which is unavoidable because each step depends on the previous one.
"""

import torch


def viterbi_decode(unary, pairwise):
    """Standard Viterbi decoding: find the highest-scoring labeling.

    Args:
        unary:    [batch, T, K]  — unary potentials per node
        pairwise: [K, K]        — pairwise[i, j] = score for transitioning
                                   from class i to class j

    Returns:
        y: [batch, T] (LongTensor) — optimal labeling per sample

    Algorithm:
        Forward pass builds a DP table:
            dp[t, j] = max score of any partial labeling ending with y_t = j

        At each step t:
            dp[t, j] = max_i ( dp[t-1, i] + pairwise[i, j] ) + unary[t, j]

        Backward pass traces back through the argmax pointers to recover y*.
    """
    batch, T, K = unary.shape
    device = unary.device

    # DP tables: dp[t] = [batch, K], backpointer[t] = [batch, K]
    # We store all T steps for the backward pass.
    dp = torch.zeros(T, batch, K, device=device)
    bp = torch.zeros(T, batch, K, dtype=torch.long, device=device)

    # --- Forward pass ---

    # t = 0: no transition, just unary
    dp[0] = unary[:, 0, :]   # [batch, K]

    for t in range(1, T):
        # dp[t-1]:          [batch, K]
        # pairwise:         [K, K]     — pairwise[i, j]
        #
        # We want for each sample b and each current class j:
        #   max_i ( dp[t-1, b, i] + pairwise[i, j] )
        #
        # Expand:
        #   dp[t-1].unsqueeze(2)  -> [batch, K, 1]   (prev classes on dim=1)
        #   pairwise.unsqueeze(0) -> [1, K, K]        (prev=dim1, curr=dim2)
        #   sum                   -> [batch, K, K]     element [b, i, j]
        #   max over dim=1 (prev) -> [batch, K]        best prev for each curr

        scores = dp[t - 1].unsqueeze(2) + pairwise.unsqueeze(0)  # [batch, K_prev, K_curr]
        dp[t], bp[t] = scores.max(dim=1)        # [batch, K_curr] each
        dp[t] = dp[t] + unary[:, t, :]          # add unary for current step

    # --- Backward pass ---

    y = torch.zeros(batch, T, dtype=torch.long, device=device)

    # Start from the best class at the last position
    y[:, T - 1] = dp[T - 1].argmax(dim=1)       # [batch]

    # Trace back
    for t in range(T - 2, -1, -1):
        # bp[t+1] has shape [batch, K]; we index with y[:, t+1] to get [batch]
        y[:, t] = bp[t + 1].gather(1, y[:, t + 1].unsqueeze(1)).squeeze(1)

    return y


def loss_augmented_viterbi(unary, pairwise, y_true):
    """Loss-augmented Viterbi: find the most-violating labeling.

    Used during structured SVM training. Solves:

        y* = argmax_y [ F(y|x) + Delta(y, y_true) ]

    where Delta is the Hamming loss: number of positions where y != y_true.

    This is identical to standard Viterbi, except that we add +1 to the
    unary potential of every class that differs from y_true at that position.
    This "augmented unary" trick works because Hamming loss decomposes over
    nodes — each position contributes independently.

    Args:
        unary:    [batch, T, K]  — unary potentials
        pairwise: [K, K]        — pairwise potentials
        y_true:   [batch, T]    — ground truth labeling

    Returns:
        y_star: [batch, T] (LongTensor) — most-violating labeling
    """
    batch, T, K = unary.shape

    # Build the Hamming loss term: +1 for every class != y_true at each position.
    # loss_term[b, t, k] = 1.0 if k != y_true[b, t], else 0.0
    #
    # Equivalently: start with all ones, then set the true class to zero.
    loss_term = torch.ones(batch, T, K, device=unary.device)
    loss_term.scatter_(2, y_true.unsqueeze(-1), 0.0)

    # Run standard Viterbi on augmented unaries
    augmented_unary = unary.detach() + loss_term
    return viterbi_decode(augmented_unary, pairwise.detach())
