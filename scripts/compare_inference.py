"""
Compare Viterbi vs LP relaxation inference for M3N training on HMC.

Trains two identical M3N models — one using Viterbi, one using LP relaxation
(max-sum diffusion) — and compares:
    1. Agreement of decoded labelings (should be 100% on chains)
    2. Learning curves (Hamming loss over epochs)
    3. Final test performance
    4. Training time per epoch

Usage:
    python scripts/compare_inference.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_hmc_sequences, SymbolicHMCDataset
from src.models import LinearBackbone, M3N, chain_edges
from src.inference import viterbi_decode, loss_augmented_viterbi
from src.inference import lp_decode, loss_augmented_lp
from src.learning import Trainer, hamming_loss, zero_one_loss
from src.baselines import ForwardBackwardClassifier


# =====================================================================
# Configuration
# =====================================================================

CONFIG = {
    # HMC data
    'num_states':  20,
    'seq_len':     50,
    'p_self':      0.7,
    'p_emit':      0.7,

    # Dataset sizes to compare
    'train_sizes': [50, 250, 1000],
    'test_size':   2000,

    # Model & training
    'num_epochs':  150,
    'batch_size':  32,
    'lr':          0.01,
    'weight_decay': 0.01,
    'eval_every':  1,
    'patience':    15,
    'min_delta':   0.001,

    'seed':        42,
}


# =====================================================================
# Helpers
# =====================================================================

def make_model(K, seed):
    """Create a fresh M3N with linear backbone."""
    torch.manual_seed(seed)
    backbone = LinearBackbone(K, K)
    return M3N(backbone, K)


def train_with_inference(method_name, model, inference_fn, predict_fn, edges,
                         train_X, train_Y, test_X, test_Y, config):
    """Train a model and return history + timing info."""
    trainer = Trainer(
        model, inference_fn, predict_fn, edges,
        lr=config['lr'], weight_decay=config['weight_decay'],
    )

    t_start = time.time()
    history = trainer.fit(train_X, train_Y, test_X, test_Y, config={
        'num_epochs':  config['num_epochs'],
        'batch_size':  config['batch_size'],
        'eval_every':  config['eval_every'],
        'patience':    config['patience'],
        'min_delta':   config['min_delta'],
        'verbose':     True,
    })
    t_elapsed = time.time() - t_start

    history['method'] = method_name
    history['total_time'] = t_elapsed
    history['time_per_epoch'] = t_elapsed / len(history['epoch'])

    return history, trainer


def check_inference_agreement(test_X, model, edges):
    """Check that Viterbi and LP relaxation produce the same labels."""
    model.eval()
    with torch.no_grad():
        unary = model.unary(test_X)
        pw = model.pairwise

        y_viterbi = viterbi_decode(unary, pw)
        y_lp = lp_decode(unary, pw, edges)

    agreement = (y_viterbi == y_lp).float().mean().item()
    return agreement, y_viterbi, y_lp


# =====================================================================
# Plotting
# =====================================================================

def plot_comparison(results, config, bayes_hamming, output_dir):
    """Generate comparison plots."""
    train_sizes = config['train_sizes']
    n_sizes = len(train_sizes)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Learning curves (Hamming loss over epochs) per train size ---
    for col, n_train in enumerate(train_sizes):
        ax = axes[0, col]
        for method in ['Viterbi', 'LP Relaxation']:
            h = results[n_train][method]
            ax.plot(h['epoch'], h['test_hamming'],
                    marker='o', markersize=2, label=method)
        ax.axhline(bayes_hamming, color='red', linestyle='--', lw=1.5,
                    label=f'Bayes ({bayes_hamming:.4f})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Hamming Loss')
        ax.set_title(f'N_train = {n_train}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Row 2, Plot 1: Final Hamming vs train size ---
    ax = axes[1, 0]
    for method in ['Viterbi', 'LP Relaxation']:
        finals = [results[n][method]['best_test_hamming'] for n in train_sizes]
        ax.plot(train_sizes, finals, 'o-', markersize=8, lw=2, label=method)
    ax.axhline(bayes_hamming, color='red', linestyle='--', lw=1.5,
               label=f'Bayes ({bayes_hamming:.4f})')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Best Test Hamming Loss')
    ax.set_title('Learning Curve')
    ax.set_xscale('log')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 2, Plot 2: Training time comparison ---
    ax = axes[1, 1]
    x_pos = np.arange(n_sizes)
    width = 0.35
    times_viterbi = [results[n]['Viterbi']['total_time'] for n in train_sizes]
    times_lp = [results[n]['LP Relaxation']['total_time'] for n in train_sizes]
    ax.bar(x_pos - width / 2, times_viterbi, width, label='Viterbi', color='tab:blue')
    ax.bar(x_pos + width / 2, times_lp, width, label='LP Relaxation', color='tab:orange')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Total Training Time (s)')
    ax.set_title('Training Time Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(train_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- Row 2, Plot 3: Difference in final Hamming ---
    ax = axes[1, 2]
    diffs = []
    for n in train_sizes:
        h_vit = results[n]['Viterbi']['best_test_hamming']
        h_lp = results[n]['LP Relaxation']['best_test_hamming']
        diffs.append(h_lp - h_vit)
    colors = ['green' if d <= 0.001 else 'orange' for d in diffs]
    bars = ax.bar(range(n_sizes), diffs, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Hamming(LP) - Hamming(Viterbi)')
    ax.set_title('Performance Difference\n(≈0 expected on chains)')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels(train_sizes)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, d in zip(bars, diffs):
        ax.annotate(f'{d:+.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    plt.suptitle('Viterbi vs LP Relaxation (Max-Sum Diffusion) on Chain HMC',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'viterbi_vs_lp_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def save_summary(results, config, bayes_hamming, output_dir):
    """Save text summary."""
    path = os.path.join(output_dir, 'comparison_results.txt')
    with open(path, 'w') as f:
        f.write("Viterbi vs LP Relaxation — Comparison Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBayes Optimal Hamming: {bayes_hamming:.4f}\n\n")

        f.write(f"{'N_train':<10} {'Method':<18} {'Hamming':<12} {'Time(s)':<12} "
                f"{'Epochs':<10} {'Stopped':<8}\n")
        f.write("-" * 70 + "\n")

        for n in config['train_sizes']:
            for method in ['Viterbi', 'LP Relaxation']:
                h = results[n][method]
                stopped = 'Yes' if h['early_stopped'] else 'No'
                f.write(f"{n:<10} {method:<18} {h['best_test_hamming']:<12.4f} "
                        f"{h['total_time']:<12.1f} {h['best_epoch']:<10} {stopped:<8}\n")
            f.write("\n")

        # Agreement check
        f.write("\nInference Agreement (after training):\n")
        for n in config['train_sizes']:
            agr = results[n].get('agreement', 'N/A')
            f.write(f"  N={n}: {agr}\n")

    return path


# =====================================================================
# Main
# =====================================================================

def main():
    config = CONFIG
    torch.manual_seed(config['seed'])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'results', 'inference_comparison')
    os.makedirs(output_dir, exist_ok=True)

    K = config['num_states']
    T = config['seq_len']
    edges = chain_edges(T)

    # ------------------------------------------------------------------
    # Generate test data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Viterbi vs LP Relaxation — Comparison")
    print("=" * 60)
    print(f"States={K}, SeqLen={T}, p_self={config['p_self']}, p_emit={config['p_emit']}")
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

    # Bayes baseline
    print("Computing Bayes optimal error...")
    bayes = ForwardBackwardClassifier(K, config['p_self'], config['p_emit'])
    bayes_pred = bayes.predict(test_X)
    bayes_hamming = hamming_loss(bayes_pred, test_Y)
    print(f"  Bayes Hamming: {bayes_hamming:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Train with both inference methods
    # ------------------------------------------------------------------
    results = {}

    for n_train in config['train_sizes']:
        print(f"\n{'='*60}")
        print(f"  Training set size: {n_train}")
        print(f"{'='*60}")

        # Generate training data
        train_obs, train_labels = generate_hmc_sequences(
            n_train, T, K,
            p_self=config['p_self'], p_emit=config['p_emit'],
            seed=config['seed'] + n_train,
        )
        train_ds = SymbolicHMCDataset(train_obs, train_labels, K)
        train_X, train_Y = train_ds.get_all_tensors()

        results[n_train] = {}

        # --- Viterbi ---
        print(f"\n  [Viterbi] Training...")
        model_vit = make_model(K, config['seed'])
        history_vit, trainer_vit = train_with_inference(
            'Viterbi', model_vit, loss_augmented_viterbi, viterbi_decode,
            edges, train_X, train_Y, test_X, test_Y, config,
        )
        results[n_train]['Viterbi'] = history_vit
        print(f"  [Viterbi] Best Hamming={history_vit['best_test_hamming']:.4f} "
              f"at epoch {history_vit['best_epoch']} "
              f"({history_vit['total_time']:.1f}s)")

        # --- LP Relaxation ---
        print(f"\n  [LP Relaxation] Training...")
        model_lp = make_model(K, config['seed'])

        # Wrap LP functions to pass edges
        def lp_inference_fn(unary, pairwise, y_true):
            return loss_augmented_lp(unary, pairwise, y_true, edges=edges)

        def lp_predict_fn(unary, pairwise):
            return lp_decode(unary, pairwise, edges=edges)

        history_lp, trainer_lp = train_with_inference(
            'LP Relaxation', model_lp, lp_inference_fn, lp_predict_fn,
            edges, train_X, train_Y, test_X, test_Y, config,
        )
        results[n_train]['LP Relaxation'] = history_lp
        print(f"  [LP Relaxation] Best Hamming={history_lp['best_test_hamming']:.4f} "
              f"at epoch {history_lp['best_epoch']} "
              f"({history_lp['total_time']:.1f}s)")

        # --- Check agreement ---
        # Use the Viterbi-trained model to decode with both methods
        agreement, _, _ = check_inference_agreement(test_X, model_vit, edges)
        results[n_train]['agreement'] = f"{agreement * 100:.2f}%"
        print(f"\n  Inference agreement (Viterbi model): {agreement * 100:.2f}%")

    # ------------------------------------------------------------------
    # Generate outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Generating outputs...")

    plot_path = plot_comparison(results, config, bayes_hamming, output_dir)
    print(f"  Plot: {plot_path}")

    summary_path = save_summary(results, config, bayes_hamming, output_dir)
    print(f"  Summary: {summary_path}")

    # Final table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'N_train':<10} {'Method':<18} {'Hamming':<12} {'Time(s)':<10}")
    print("-" * 50)
    for n in config['train_sizes']:
        for method in ['Viterbi', 'LP Relaxation']:
            h = results[n][method]
            print(f"{n:<10} {method:<18} {h['best_test_hamming']:<12.4f} "
                  f"{h['total_time']:<10.1f}")
        agr = results[n]['agreement']
        print(f"{'':>10} Agreement: {agr}")
        print()
    print(f"{'Bayes':<10} {'':18} {bayes_hamming:<12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()