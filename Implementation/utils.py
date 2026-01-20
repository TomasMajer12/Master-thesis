"""
Utility functions for M3N experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_error(y_pred, y_true):
    """Compute error rate (percentage of incorrect predictions)"""
    incorrect = (y_pred != y_true).float().sum()
    total = y_true.numel()
    return (incorrect / total).item() * 100


def save_model(model, filepath, config, num_train_samples, final_test_error, epoch):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'num_train_samples': num_train_samples,
        'final_test_error': final_test_error,
        'epoch': epoch
    }, filepath)


def load_model(filepath, model_class, input_dim, num_classes):
    """Load model from checkpoint"""
    checkpoint = torch.load(filepath)
    model = model_class(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def plot_training_results(results, train_sizes, bayes_error, config, output_dir):
    """
    Generate and save training result plots.
    
    Creates 3 subplots:
    1. Test error over epochs for each training size
    2. Final error vs training set size
    3. Gap to Bayes optimal
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(train_sizes)))
    
    # Plot 1: Test error over epochs
    ax1 = axes[0]
    for idx, n_train in enumerate(train_sizes):
        epochs = results[n_train]['epochs']
        ax1.plot(epochs, results[n_train]['test_error'], 
                 marker='o', markersize=3, color=colors[idx],
                 label=f'n={n_train}')
    
    ax1.axhline(y=bayes_error, color='red', linestyle='--', linewidth=2, 
                label=f'Bayes Optimal ({bayes_error:.1f}%)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Error (%)')
    ax1.set_title('Test Error During Training')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Final test error vs training size
    ax2 = axes[1]
    final_errors = [results[n]['best_test_error'] for n in train_sizes]
    
    ax2.plot(train_sizes, final_errors, 'bo-', markersize=10, linewidth=2, label='M3N')
    ax2.axhline(y=bayes_error, color='red', linestyle='--', linewidth=2,
                label=f'Bayes Optimal ({bayes_error:.1f}%)')
    ax2.set_xlabel('Number of Training Sequences')
    ax2.set_ylabel('Best Test Error (%)')
    ax2.set_title('Best Error vs Training Set Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks(train_sizes)
    ax2.set_xticklabels(train_sizes)
    ax2.set_ylim([0, 100])
    
    # Plot 3: Gap to Bayes optimal
    ax3 = axes[2]
    gaps = [err - bayes_error for err in final_errors]
    
    bars = ax3.bar(range(len(train_sizes)), gaps, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(train_sizes)))
    ax3.set_xticklabels([str(n) for n in train_sizes])
    ax3.set_xlabel('Number of Training Sequences')
    ax3.set_ylabel('Gap to Bayes Optimal (%)')
    ax3.set_title('Distance from Optimal Classifier')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax3.annotate(f'{gap:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def save_results_summary(results, train_sizes, bayes_error, config, output_dir):
    """Save text summary of results"""
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("M3N Training Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBayes Optimal Error: {bayes_error:.2f}%\n\n")
        f.write("Results:\n")
        f.write(f"{'Training Size':<15} {'Best Error (%)':<15} {'Gap to Bayes (%)':<18} {'Best Epoch':<12} {'Stopped':<10}\n")
        f.write("-" * 70 + "\n")
        for n_train in train_sizes:
            err = results[n_train]['best_test_error']
            gap = err - bayes_error
            best_epoch = results[n_train]['best_epoch']
            stopped = results[n_train].get('early_stopped', False)
            stopped_str = 'Yes' if stopped else 'No'
            f.write(f"{n_train:<15} {err:<15.2f} {gap:<18.2f} {best_epoch:<12} {stopped_str:<10}\n")
    
    return summary_path


def print_summary(results, train_sizes, bayes_error):
    """Print summary table to console"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Training Size':<15} {'Best Error (%)':<15} {'Gap to Bayes (%)':<18} {'Best Epoch':<12}")
    print("-" * 60)
    for n_train in train_sizes:
        err = results[n_train]['best_test_error']
        gap = err - bayes_error
        best_epoch = results[n_train]['best_epoch']
        early = " (early stop)" if results[n_train].get('early_stopped', False) else ""
        print(f"{n_train:<15} {err:<15.2f} {gap:<18.2f} {best_epoch:<12}{early}")
    print("-" * 60)
    print(f"{'Bayes Optimal':<15} {bayes_error:<15.2f} {0.0:<18.2f}")
    print("=" * 70)