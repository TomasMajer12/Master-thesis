"""
Hidden Markov Chain (HMC) data generation.

Model:
    - Hidden states Y follow a Markov chain:
        P(y_t = y_{t-1})                 = p_self    (self-transition)
        P(y_t = j | y_{t-1} = i, j != i) = (1 - p_self) / (K - 1)

    - Observations X are generated from hidden states:
        P(x_t = y_t)                     = p_emit    (correct emission)
        P(x_t = j | y_t = i, j != i)     = (1 - p_emit) / (K - 1)

    - Initial state: uniform over K classes.

The task is to predict the hidden state sequence Y from observations X.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_hmc_sequences(num_samples, seq_len, num_states=10,
                           p_self=0.7, p_emit=0.7, seed=42):
    """Generate HMC sequences.

    Uses an explicit RNG instance (not global np.random) for reproducibility.

    Args:
        num_samples: number of sequences to generate
        seq_len:     length of each sequence
        num_states:  number of hidden states / observation symbols (K)
        p_self:      self-transition probability (higher = more persistent states)
        p_emit:      correct emission probability (higher = less noisy observations)
        seed:        random seed

    Returns:
        obs_indices: np.ndarray [num_samples, seq_len] — observation indices (integers)
        labels:      np.ndarray [num_samples, seq_len] — hidden state labels (integers)
    """
    rng = np.random.default_rng(seed)

    labels = np.zeros((num_samples, seq_len), dtype=np.int64)
    obs_indices = np.zeros((num_samples, seq_len), dtype=np.int64)

    for n in range(num_samples):
        # Initial state: uniform
        labels[n, 0] = rng.integers(num_states)

        # Hidden state Markov chain
        for t in range(1, seq_len):
            if rng.random() < p_self:
                labels[n, t] = labels[n, t - 1]
            else:
                choices = [s for s in range(num_states) if s != labels[n, t - 1]]
                labels[n, t] = rng.choice(choices)

        # Observations
        for t in range(seq_len):
            if rng.random() < p_emit:
                obs_indices[n, t] = labels[n, t]
            else:
                choices = [s for s in range(num_states) if s != labels[n, t]]
                obs_indices[n, t] = rng.choice(choices)

    return obs_indices, labels


class SymbolicHMCDataset(Dataset):
    """HMC dataset with one-hot encoded observations.

    Each sample is a pair (x, y):
        x: [seq_len, num_states]  — float32 one-hot encoded observations
        y: [seq_len]              — int64 hidden state labels
    """

    def __init__(self, obs_indices, labels, num_states=10):
        self.obs_indices = torch.as_tensor(obs_indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.num_states = num_states

    def __len__(self):
        return len(self.obs_indices)

    def __getitem__(self, idx):
        x = F.one_hot(self.obs_indices[idx], self.num_states).float()
        y = self.labels[idx]
        return x, y

    def get_all_tensors(self):
        """Return the full dataset as (X, Y) tensors (for small datasets).

        Returns:
            X: [N, seq_len, num_states] float32
            Y: [N, seq_len] int64
        """
        X = F.one_hot(self.obs_indices, self.num_states).float()
        Y = self.labels
        return X, Y
