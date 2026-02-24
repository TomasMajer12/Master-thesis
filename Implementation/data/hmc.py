import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_hmc_sequences(num_samples, seq_len, num_states=10,
                           p_yt_ytm1=0.7, p_xt_yt=0.7, seed=42):
    """Generate Hidden Markov Chain sequences.

    Hidden states Y follow a Markov chain with self-transition probability
    p_yt_ytm1. Observations X are generated from Y with correct-emission
    probability p_xt_yt.

    Uses an explicit RNG (not global seed) for reproducibility.

    Returns:
        obs_indices: np.ndarray [num_samples, seq_len] - integer observation indices
        labels:      np.ndarray [num_samples, seq_len] - integer hidden state labels
    """
    rng = np.random.default_rng(seed)

    labels = np.zeros((num_samples, seq_len), dtype=np.int64)
    obs_indices = np.zeros((num_samples, seq_len), dtype=np.int64)

    for n in range(num_samples):
        # Initial state: uniform
        labels[n, 0] = rng.integers(num_states)

        # Hidden state Markov chain
        for t in range(1, seq_len):
            if rng.random() < p_yt_ytm1:
                labels[n, t] = labels[n, t - 1]
            else:
                choices = [s for s in range(num_states) if s != labels[n, t - 1]]
                labels[n, t] = rng.choice(choices)

        # Observations
        for t in range(seq_len):
            if rng.random() < p_xt_yt:
                obs_indices[n, t] = labels[n, t]
            else:
                choices = [s for s in range(num_states) if s != labels[n, t]]
                obs_indices[n, t] = rng.choice(choices)

    return obs_indices, labels


class SymbolicHMCDataset(Dataset):
    """HMC dataset with one-hot encoded observations.

    Each sample is a sequence of one-hot vectors paired with hidden state labels.

    Returns per sample:
        x: [seq_len, num_states]  (float32 one-hot)
        y: [seq_len]              (int64 labels)
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


class VisualHMCDataset(Dataset):
    """HMC dataset with MNIST image observations.

    Each observation index (0-9) is mapped to a randomly chosen MNIST image
    of that digit. Image indices are pre-assigned in __init__ for
    reproducibility.

    Returns per sample:
        x: [seq_len, 1, 28, 28]  (float32 images)
        y: [seq_len]              (int64 labels)
    """

    def __init__(self, obs_indices, labels, mnist_pool, seed=0,
                 image_indices=None):
        self.obs_indices = torch.as_tensor(obs_indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.mnist_pool = mnist_pool
        self.num_states = 10

        if image_indices is not None:
            # Use pre-computed indices (loaded from disk)
            self.image_indices = torch.as_tensor(image_indices, dtype=torch.long)
        else:
            # Compute random but reproducible MNIST image assignments
            obs_np = self.obs_indices.numpy() if isinstance(obs_indices, torch.Tensor) else obs_indices
            rng = np.random.default_rng(seed)
            img_idx = np.zeros(obs_np.shape, dtype=np.int64)
            for digit in range(10):
                mask = (obs_np == digit)
                count = mask.sum()
                pool_size = mnist_pool.pool_size(digit)
                img_idx[mask] = rng.integers(pool_size, size=count)
            self.image_indices = torch.LongTensor(img_idx)

    def __len__(self):
        return len(self.obs_indices)

    def __getitem__(self, idx):
        seq_obs = self.obs_indices[idx]        # [seq_len]
        seq_img_idx = self.image_indices[idx]   # [seq_len]
        y = self.labels[idx]                    # [seq_len]

        images = []
        for t in range(len(seq_obs)):
            digit = seq_obs[t].item()
            img_idx = seq_img_idx[t].item()
            images.append(self.mnist_pool.images_by_digit[digit][img_idx])
        x = torch.stack(images)  # [seq_len, 1, 28, 28]
        return x, y
