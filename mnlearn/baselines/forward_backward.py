"""
Forward-Backward marginal classifier for symbolic HMC.

Per-cell MAP predictor of P(y_t | x_1..x_T). Serves as the
Bayes-optimal Hamming-loss decoder for symbolic HMC and the
oracle floor for the HMC tightness experiments.
"""

import numpy as np
import torch


def _transition_matrix(num_states, p_self):
    """Build transition matrix A[i, j] = P(y_t = j | y_{t-1} = i)."""
    A = np.full((num_states, num_states), (1 - p_self) / (num_states - 1))
    np.fill_diagonal(A, p_self)
    return A


def _emission_matrix(num_states, p_emit):
    """Build emission matrix B[i, j] = P(x_t = j | y_t = i)."""
    B = np.full((num_states, num_states), (1 - p_emit) / (num_states - 1))
    np.fill_diagonal(B, p_emit)
    return B


class ForwardBackwardClassifier:
    """Optimal Bayes classifier using known HMC parameters.

    Args:
        num_states: number of hidden states (K)
        p_self:     self-transition probability
        p_emit:     correct emission probability
    """

    def __init__(self, num_states=10, p_self=0.7, p_emit=0.7):
        self.num_states = num_states
        self.A = _transition_matrix(num_states, p_self)
        self.B = _emission_matrix(num_states, p_emit)
        self.prior = np.full(num_states, 1.0 / num_states)

    def _forward_backward(self, observations):
        """Compute marginal posteriors P(y_t | X) for one sequence.

        Args:
            observations: [T] array of integer observation indices

        Returns:
            gamma: [T, K] — marginal posterior probabilities
        """
        T = len(observations)
        K = self.num_states
        A, B, pi = self.A, self.B, self.prior

        # --- Forward pass (scaled) ---
        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        alpha[0] = pi * B[:, observations[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        for t in range(1, T):
            for j in range(K):
                alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, observations[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        # --- Backward pass (scaled) ---
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                beta[t, i] = np.sum(A[i, :] * B[:, observations[t + 1]] * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        # --- Marginal posteriors ---
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1
        gamma = gamma / gamma_sum

        return gamma

    def predict(self, X):
        """Predict hidden states from symbolic observations.

        Args:
            X: [batch, seq_len, num_states] (one-hot) or [batch, seq_len] (indices)

        Returns:
            y_pred: [batch, seq_len] LongTensor
        """
        if isinstance(X, torch.Tensor):
            if X.dim() == 3:
                obs = torch.argmax(X, dim=-1).cpu().numpy()
            else:
                obs = X.cpu().numpy()
        else:
            obs = np.asarray(X)

        batch, seq_len = obs.shape
        preds = np.zeros((batch, seq_len), dtype=np.int64)

        for b in range(batch):
            gamma = self._forward_backward(obs[b])
            preds[b] = np.argmax(gamma, axis=1)

        return torch.from_numpy(preds)
