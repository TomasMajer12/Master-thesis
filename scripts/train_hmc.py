"""
End-to-end training experiment on the HMC benchmark (symbolic input).

Trains M3N models with different training set sizes, compares against
the optimal Bayes classifier, and generates learning curve plots.

Usage:
    python scripts/train_hmc.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_hmc_sequences, SymbolicHMCDataset
from src.models import LinearBackbone, MLPBackbone, M3N, chain_edges
from src.inference import viterbi_decode, loss_augmented_viterbi
from src.learning import Trainer, hamming_loss, zero_one_loss
from src.baselines import ForwardBackwardClassifier


# =====================================================================
# Configuration
# =====================================================================

CONFIG = {
    # HMC data parameters
    'num_states':  20,
    'seq_len':     50,
    'p_self':      0.7,
    'p_emit':      0.7,

    # Dataset sizes
    'train_sizes': [10, 50, 100, 250, 500, 1000, 2000, 5000],
    'test_size':   2000,

    # Model
    'backbone':    'linear',         # 'linear' or 'mlp'
    'hidden_dims': [64],          # only used if backbone == 'mlp'

    # Training
    'num_epochs':  200,
    'batch_size':  32,
    'lr':          0.01,
    'weight_decay': 0.01,
    'eval_every':  1,
    'patience':    15,
    'min_delta':   0.001,

    # Reproducibility
    'seed':        42,
}


# =====================================================================
# Helpers
# =====================================================================

def make_model(config):
    """Create a fresh M3N model based on config."""
    K = config['num_states']
    if config['backbone'] == 'linear':
        backbone = LinearBackbone(K, K)
    elif config['backbone'] == 'mlp':
        backbone = MLPBackbone(K, K, hidden_dims=tuple(config['hidden_dims']))
    else:
        raise ValueError(f"Unknown backbone: {config['backbone']}")
    return M3N(backbone, K)


def plot_results(all_results, config, bayes_hamming, bayes_01, output_dir):
    """Generate and save learning curve plots."""
    train_sizes = config['train_sizes']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(train_sizes)))

    # --- Plot 1: Test Hamming loss over epochs ---
    ax = axes[0]
    for i, n in enumerate(train_sizes):
        h = all_results[n]
        ax.plot(h['epoch'], h['test_hamming'], color=colors[i],
                marker='o', markersize=2, label=f'n={n}')
    ax.axhline(bayes_hamming, color='red', linestyle='--', lw=2,
               label=f'Bayes ({bayes_hamming:.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Hamming Loss')
    ax.set_title('Test Hamming Loss During Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Learning curve (final error vs training size) ---
    ax = axes[1]
    final_hamming = [all_results[n]['best_test_hamming'] for n in train_sizes]
    ax.plot(train_sizes, final_hamming, 'bo-', markersize=8, lw=2, label='M3N')
    ax.axhline(bayes_hamming, color='red', linestyle='--', lw=2,
               label=f'Bayes ({bayes_hamming:.4f})')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Best Test Hamming Loss')
    ax.set_title('Learning Curve')
    ax.set_xscale('log')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Gap to Bayes optimal ---
    ax = axes[2]
    gaps = [h - bayes_hamming for h in final_hamming]
    bars = ax.bar(range(len(train_sizes)), gaps, color=colors, edgecolor='black')
    ax.set_xticks(range(len(train_sizes)))
    ax.set_xticklabels([str(n) for n in train_sizes])
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Gap to Bayes Optimal')
    ax.set_title('Distance from Optimal')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, gap in zip(bars, gaps):
        ax.annotate(f'{gap:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'hmc_learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def save_summary(all_results, config, bayes_hamming, bayes_01, output_dir):
    """Save text summary of results."""
    path = os.path.join(output_dir, 'hmc_results.txt')
    with open(path, 'w') as f:
        f.write("HMC Symbolic Benchmark — Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBayes Optimal — Hamming: {bayes_hamming:.4f}, 0/1: {bayes_01:.4f}\n\n")

        f.write(f"{'N_train':<10} {'Hamming':<12} {'0/1 loss':<12} "
                f"{'Gap':<12} {'Epoch':<8} {'Stopped':<8}\n")
        f.write("-" * 62 + "\n")

        for n in config['train_sizes']:
            h = all_results[n]
            # Get 0/1 loss at best epoch
            best_idx = h['epoch'].index(h['best_epoch'])
            z1 = h['test_zero_one'][best_idx]
            gap = h['best_test_hamming'] - bayes_hamming
            stopped = 'Yes' if h['early_stopped'] else 'No'
            f.write(f"{n:<10} {h['best_test_hamming']:<12.4f} {z1:<12.4f} "
                    f"{gap:<12.4f} {h['best_epoch']:<8} {stopped:<8}\n")

        f.write("-" * 62 + "\n")
        f.write(f"{'Bayes':<10} {bayes_hamming:<12.4f} {bayes_01:<12.4f}\n")
    return path


# =====================================================================
# Main
# =====================================================================

def main():
    config = CONFIG
    torch.manual_seed(config['seed'])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'results', 'hmc_symbolic')
    os.makedirs(output_dir, exist_ok=True)

    K = config['num_states']
    T = config['seq_len']
    edges = chain_edges(T)

    # ------------------------------------------------------------------
    # Generate test data (fixed for all experiments)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("HMC Symbolic Benchmark")
    print("=" * 60)
    print(f"States={K}, SeqLen={T}, p_self={config['p_self']}, p_emit={config['p_emit']}")
    print(f"Backbone: {config['backbone']}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    print("\nGenerating test data...")
    test_obs, test_labels = generate_hmc_sequences(
        config['test_size'], T, K,
        p_self=config['p_self'], p_emit=config['p_emit'],
        seed=config['seed'],
    )
    test_ds = SymbolicHMCDataset(test_obs, test_labels, K)
    test_X, test_Y = test_ds.get_all_tensors()

    # ------------------------------------------------------------------
    # Bayes optimal baseline
    # ------------------------------------------------------------------
    print("Computing Bayes optimal error...")
    bayes = ForwardBackwardClassifier(K, config['p_self'], config['p_emit'])
    bayes_pred = bayes.predict(test_X)
    bayes_hamming = hamming_loss(bayes_pred, test_Y)
    bayes_01 = zero_one_loss(bayes_pred, test_Y)
    print(f"  Bayes Hamming: {bayes_hamming:.4f}  |  0/1: {bayes_01:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Train M3N for each training set size
    # ------------------------------------------------------------------
    all_results = {}

    for n_train in config['train_sizes']:
        print(f"\n--- Training with {n_train} samples ---")

        # Generate training data (different seed per size for independence)
        train_obs, train_labels = generate_hmc_sequences(
            n_train, T, K,
            p_self=config['p_self'], p_emit=config['p_emit'],
            seed=config['seed'] + n_train,
        )
        train_ds = SymbolicHMCDataset(train_obs, train_labels, K)
        train_X, train_Y = train_ds.get_all_tensors()

        # Fresh model for each size
        torch.manual_seed(config['seed'])
        model = make_model(config)

        trainer = Trainer(
            model, loss_augmented_viterbi, viterbi_decode, edges,
            lr=config['lr'], weight_decay=config['weight_decay'],
        )

        history = trainer.fit(train_X, train_Y, test_X, test_Y, config={
            'num_epochs':  config['num_epochs'],
            'batch_size':  config['batch_size'],
            'eval_every':  config['eval_every'],
            'patience':    config['patience'],
            'min_delta':   config['min_delta'],
            'verbose':     True,
        })

        all_results[n_train] = history

        # Save model checkpoint
        ckpt_path = os.path.join(output_dir, f'model_n{n_train}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'n_train': n_train,
            'history': history,
        }, ckpt_path)
        print(f"  Best: Hamming={history['best_test_hamming']:.4f} at epoch {history['best_epoch']}")

    # ------------------------------------------------------------------
    # Generate outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating outputs...")

    plot_path = plot_results(all_results, config, bayes_hamming, bayes_01, output_dir)
    print(f"  Plot: {plot_path}")

    summary_path = save_summary(all_results, config, bayes_hamming, bayes_01, output_dir)
    print(f"  Summary: {summary_path}")

    # Final summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'N_train':<10} {'Hamming':<12} {'0/1 loss':<12} {'Gap':<12} {'Epoch':<8}")
    print("-" * 54)
    for n in config['train_sizes']:
        h = all_results[n]
        best_idx = h['epoch'].index(h['best_epoch'])
        z1 = h['test_zero_one'][best_idx]
        gap = h['best_test_hamming'] - bayes_hamming
        early = " *" if h['early_stopped'] else ""
        print(f"{n:<10} {h['best_test_hamming']:<12.4f} {z1:<12.4f} {gap:<12.4f} {h['best_epoch']:<8}{early}")
    print("-" * 54)
    print(f"{'Bayes':<10} {bayes_hamming:<12.4f} {bayes_01:<12.4f}")
    print("=" * 60)
    print("\n* = early stopped")

    return all_results


if __name__ == "__main__":
    main()
