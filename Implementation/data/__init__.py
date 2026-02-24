import os
import json
import torch
import numpy as np

from .hmc import generate_hmc_sequences, SymbolicHMCDataset, VisualHMCDataset
from .mnist_pool import MNISTPool


def save_hmc_benchmark(output_dir='./benchmarks/hmc', num_samples=50000,
                       seq_len=30, num_states=10, p_yt_ytm1=0.7, p_xt_yt=0.7,
                       hmc_seed=42, mnist_seed=123, mnist_root='./mnist_data',
                       train_size=30000, val_size=10000, test_size=10000):
    """Generate and save HMC benchmark to disk.

    Creates the following structure:
        output_dir/
            config.json          # All generation parameters
            train/
                obs_indices.pt   # [30000, 30] int64
                labels.pt        # [30000, 30] int64
                visual_image_indices.pt  # [30000, 30] int64 (MNIST pool indices)
            val/
                ...
            test/
                ...

    The symbolic representation (one-hot) is computed on-the-fly from
    obs_indices. The visual representation uses visual_image_indices to
    look up MNIST images from the pool, ensuring reproducibility.
    """
    assert train_size + val_size + test_size == num_samples

    # Generate all HMC sequences
    obs_indices, labels = generate_hmc_sequences(
        num_samples, seq_len, num_states, p_yt_ytm1, p_xt_yt, hmc_seed
    )

    # Split by index
    split_ranges = {
        'train': (0, train_size),
        'val':   (train_size, train_size + val_size),
        'test':  (train_size + val_size, num_samples),
    }

    # Pre-compute MNIST image index assignments for visual dataset
    train_pool = MNISTPool(train=True, root=mnist_root)
    test_pool = MNISTPool(train=False, root=mnist_root)
    seed_offsets = {'train': 0, 'val': 1, 'test': 2}

    # Save each split
    for split_name, (start, end) in split_ranges.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_obs = obs_indices[start:end]
        split_labels = labels[start:end]

        torch.save(torch.LongTensor(split_obs),
                    os.path.join(split_dir, 'obs_indices.pt'))
        torch.save(torch.LongTensor(split_labels),
                    os.path.join(split_dir, 'labels.pt'))

        # Compute visual image indices for this split
        pool = train_pool if split_name == 'train' else test_pool
        rng = np.random.default_rng(mnist_seed + seed_offsets[split_name])
        image_indices = np.zeros_like(split_obs, dtype=np.int64)
        for digit in range(num_states):
            mask = (split_obs == digit)
            count = mask.sum()
            pool_size = pool.pool_size(digit)
            image_indices[mask] = rng.integers(pool_size, size=count)

        torch.save(torch.LongTensor(image_indices),
                    os.path.join(split_dir, 'visual_image_indices.pt'))

    # Save config
    config = {
        'num_samples': num_samples,
        'seq_len': seq_len,
        'num_states': num_states,
        'p_yt_ytm1': p_yt_ytm1,
        'p_xt_yt': p_xt_yt,
        'hmc_seed': hmc_seed,
        'mnist_seed': mnist_seed,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved HMC benchmark to {output_dir}/")
    for split_name, (start, end) in split_ranges.items():
        print(f"  {split_name}: {end - start} sequences")


def load_hmc_datasets(benchmark_dir='./benchmarks/hmc', mode='both',
                      mnist_root='./mnist_data'):
    """Load saved HMC benchmark from disk and wrap as PyTorch Datasets.

    Args:
        benchmark_dir: path to saved benchmark (created by save_hmc_benchmark)
        mode: 'symbolic', 'visual', or 'both'
        mnist_root: path to MNIST data (only needed for mode='visual' or 'both')

    Returns:
        dict with keys like 'symbolic_train', 'visual_test', etc.
    """
    with open(os.path.join(benchmark_dir, 'config.json')) as f:
        config = json.load(f)

    num_states = config['num_states']
    datasets = {}

    # Load MNIST pools if needed for visual mode
    train_pool = None
    test_pool = None
    if mode in ('visual', 'both'):
        train_pool = MNISTPool(train=True, root=mnist_root)
        test_pool = MNISTPool(train=False, root=mnist_root)

    for split_name in ('train', 'val', 'test'):
        split_dir = os.path.join(benchmark_dir, split_name)
        obs_indices = torch.load(os.path.join(split_dir, 'obs_indices.pt'),
                                 weights_only=True)
        labels = torch.load(os.path.join(split_dir, 'labels.pt'),
                            weights_only=True)

        if mode in ('symbolic', 'both'):
            datasets[f'symbolic_{split_name}'] = SymbolicHMCDataset(
                obs_indices, labels, num_states
            )

        if mode in ('visual', 'both'):
            image_indices = torch.load(
                os.path.join(split_dir, 'visual_image_indices.pt'),
                weights_only=True
            )
            pool = train_pool if split_name == 'train' else test_pool
            datasets[f'visual_{split_name}'] = VisualHMCDataset(
                obs_indices, labels, pool, image_indices=image_indices
            )

    datasets['config'] = config
    return datasets
