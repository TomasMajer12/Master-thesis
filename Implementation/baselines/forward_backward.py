"""Optimal Bayes classifier for symbolic HMC data using the Forward-Backward algorithm."""

import numpy as np
import torch


def get_transition_matrix(num_states, p_yt_ytm1):
    A = np.full((num_states, num_states), (1 - p_yt_ytm1) / (num_states - 1))
    np.fill_diagonal(A, p_yt_ytm1)
    return A


def get_emission_matrix(num_states, p_xt_yt):
    B = np.full((num_states, num_states), (1 - p_xt_yt) / (num_states - 1))
    np.fill_diagonal(B, p_xt_yt)
    return B


class ForwardBackwardClassifier:
    """Optimal classifier using Forward-Backward algorithm.

    Computes marginal posteriors P(y_t | X) and predicts:
        y_t = argmax_y P(y_t = y | X)

    This minimizes expected Hamming loss (pointwise optimal predictions).

    Note: this baseline requires symbolic (discrete) observations.
    It cannot be applied directly to visual (MNIST image) inputs.
    """

    def __init__(self, num_states=10, p_yt_ytm1=0.7, p_xt_yt=0.7):
        self.num_states = num_states
        self.p_yt_ytm1 = p_yt_ytm1
        self.p_xt_yt = p_xt_yt

        self.transition_prob = get_transition_matrix(num_states, p_yt_ytm1)
        self.emission_prob = get_emission_matrix(num_states, p_xt_yt)
        self.prior = np.full(num_states, 1.0 / num_states)

    def forward_backward(self, observations):
        seq_len = len(observations)
        A = self.transition_prob
        B = self.emission_prob
        pi = self.prior

        alpha = np.zeros((seq_len, self.num_states))
        scale = np.zeros(seq_len)

        # Initialization
        alpha[0] = pi * B[:, observations[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        # Forward recursion
        for t in range(1, seq_len):
            for j in range(self.num_states):
                alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, observations[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        # Backward pass
        beta = np.zeros((seq_len, self.num_states))
        beta[-1] = 1.0

        for t in range(seq_len - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.sum(A[i, :] * B[:, observations[t + 1]] * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1
        gamma = gamma / gamma_sum

        return gamma

    def predict_sequence(self, observations):
        gamma = self.forward_backward(observations)
        return np.argmax(gamma, axis=1)

    def predict(self, X):
        """Predict hidden states from symbolic observations.

        Args:
            X: torch.Tensor [batch, seq_len, num_states] (one-hot encoded)
               or [batch, seq_len] (integer indices)
        """
        if X.dim() == 3:
            observations = torch.argmax(X, dim=-1).cpu().numpy()
        else:
            observations = X.cpu().numpy()

        batch_size, seq_len = observations.shape
        preds = np.zeros((batch_size, seq_len), dtype=np.int64)

        for b in range(batch_size):
            preds[b] = self.predict_sequence(observations[b])

        return torch.LongTensor(preds)
