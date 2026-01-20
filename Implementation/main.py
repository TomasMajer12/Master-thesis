"""
Main script for training and evaluating M3N on HMC data.
Compares performance across different training set sizes against optimal Bayes classifier.

Run from the same directory where dataset_HMC.py and SimpleM3N.py are located.
All outputs will be saved in the same directory.
"""

import torch
import numpy as np
import os
from NeuralM3N import NeuralM3N
from dataset_HMC import generate_hmc_data, ForwardBackwardClassifier
from SimpleM3N import SimpleM3N_NN, M3NTrainer
from utils import (
    compute_error, 
    save_model, 
    plot_training_results, 
    save_results_summary,
    print_summary
)


class EarlyStopping:
    """
    Early stopping to stop training when test error stops improving.
    """
    def __init__(self, patience=10, min_delta=0.1):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change in error to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_error = float('inf')
        self.should_stop = False
    
    def __call__(self, current_error):
        if current_error < self.best_error - self.min_delta:
            self.best_error = current_error
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_and_evaluate(num_train_samples, config, test_X, test_Y, output_dir, verbose=True):
    """
    Train M3N model with early stopping, save best model.
    
    Returns:
        history: Dictionary with training history and best results
    """
    # Generate training data
    train_X, train_Y = generate_hmc_data(
        num_samples=num_train_samples,
        seq_len=config['seq_len'],
        num_states=config['num_states'],
        p_yt_ytm1=config['p_yt_ytm1'],
        p_xt_yt=config['p_xt_yt'],
        seed=config['seed'] + num_train_samples  # Different seed per training size
    )
    
    # Initialize model
    model = NeuralM3N(
        input_dim=config['num_states'],
        num_classes=config['num_states']
    )
    
    trainer = M3NTrainer(
        model,
        learning_rate=config['learning_rate'],
        C=config['C'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stop_patience'],
        min_delta=config['early_stop_min_delta']
    )
    
    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_error': [],
        'test_error': [],
        'best_test_error': float('inf'),
        'best_epoch': 0,
        'early_stopped': False
    }
    
    # Best model state
    best_model_state = None
    
    batch_size = min(config['batch_size'], num_train_samples)
    num_batches = max(1, num_train_samples // batch_size)
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        
        # Shuffle training data
        perm = torch.randperm(num_train_samples)
        train_X_shuffled = train_X[perm]
        train_Y_shuffled = train_Y[perm]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train_samples)
            
            batch_X = train_X_shuffled[start_idx:end_idx]
            batch_Y = train_Y_shuffled[start_idx:end_idx]
            
            loss = trainer.train_step(batch_X, batch_Y)
            epoch_loss += loss
        
        epoch_loss /= num_batches
        
        # Evaluate periodically
        if (epoch + 1) % config['eval_every'] == 0 or epoch == 0:
            with torch.no_grad():
                train_pred = trainer.predict(train_X)
                train_error = compute_error(train_pred, train_Y)
                
                test_pred = trainer.predict(test_X)
                test_error = compute_error(test_pred, test_Y)
            
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(epoch_loss)
            history['train_error'].append(train_error)
            history['test_error'].append(test_error)
            
            # Track best model
            if test_error < history['best_test_error']:
                history['best_test_error'] = test_error
                history['best_epoch'] = epoch + 1
                best_model_state = model.state_dict().copy()
            
            if verbose:
                print(f"  Epoch {epoch+1:3d}: Loss={epoch_loss:.4f}, "
                      f"Train Err={train_error:.2f}%, Test Err={test_error:.2f}%")
            
            # Check early stopping
            if early_stopping(test_error):
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {config['early_stop_patience']} evaluations)")
                history['early_stopped'] = True
                break
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model_path = os.path.join(output_dir, f'best_model_n{num_train_samples}.pt')
        save_model(
            model, 
            model_path, 
            config, 
            num_train_samples, 
            history['best_test_error'],
            history['best_epoch']
        )
        if verbose:
            print(f"  Saved best model (epoch {history['best_epoch']}, error {history['best_test_error']:.2f}%)")
    
    return history


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir  # Save everything in the same directory
    
    # Configuration
    config = {
        'num_states': 25,
        'seq_len': 30,
        'p_yt_ytm1': 0.7,
        'p_xt_yt': 0.7,
        'seed': 42,
        'num_epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.01,
        'C': 1.0,
        'weight_decay': 0.01,
        'eval_every': 1,
        'num_test_samples': 200,
        'early_stop_patience': 5,      # Stop after 5 evaluations without improvement
        'early_stop_min_delta': 0.1     # Minimum improvement threshold
    }
    
    # Training set sizes to compare
    train_sizes = [10, 50, 100, 250 ,500]
    
    print("=" * 60)
    print("M3N Training Experiment")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Config: {config['num_states']} states, seq_len={config['seq_len']}")
    print(f"HMC params: p(y_t|y_t-1)={config['p_yt_ytm1']}, p(x_t|y_t)={config['p_xt_yt']}")
    print(f"Early stopping: patience={config['early_stop_patience']}, min_delta={config['early_stop_min_delta']}")
    print("=" * 60)
    
    # Generate test data (fixed for all experiments)
    print("\nGenerating test data...")
    test_X, test_Y = generate_hmc_data(
        num_samples=config['num_test_samples'],
        seq_len=config['seq_len'],
        num_states=config['num_states'],
        p_yt_ytm1=config['p_yt_ytm1'],
        p_xt_yt=config['p_xt_yt'],
        seed=config['seed'] + 1 # Different seed for test
    )
    
    # Compute optimal Bayes error
    print("Computing optimal Bayes classifier error...")
    bayes_classifier = ForwardBackwardClassifier(
        num_states=config['num_states'],
        p_yt_ytm1=config['p_yt_ytm1'],
        p_xt_yt=config['p_xt_yt']
    )
    bayes_pred = bayes_classifier.predict(test_X)
    bayes_error = compute_error(bayes_pred, test_Y)
    print(f"Optimal Bayes error: {bayes_error:.2f}%")
    print("=" * 60)
    
    # Train models for each training set size
    results = {}
    
    for n_train in train_sizes:
        print(f"\n--- Training with {n_train} samples ---")
        history = train_and_evaluate(
            num_train_samples=n_train,
            config=config,
            test_X=test_X,
            test_Y=test_Y,
            output_dir=output_dir,
            verbose=True
        )
        results[n_train] = history
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating outputs...")
    
    plot_path = plot_training_results(results, train_sizes, bayes_error, config, output_dir)
    print(f"  Saved: {os.path.basename(plot_path)}")
    
    summary_path = save_results_summary(results, train_sizes, bayes_error, config, output_dir)
    print(f"  Saved: {os.path.basename(summary_path)}")
    
    # Print summary
    print_summary(results, train_sizes, bayes_error)
    
    return results, bayes_error


if __name__ == "__main__":
    results, bayes_error = main()




#grap testovaci chyba - pocet testovacich dat -10,100,1000 .... + chyba bayes optimal
#popsat prechodove funkce 
#pracovat na psani dp - baysovsky predictor s hamming loss, generovani dat HMM, Structured output SVM + DP
#baumwells algorithm for optimal classifier - e stepmarginal posterious prob - backward forward, minimalizuje hamming loss
#LP - relaxaci
#ukladat si modely - pro porovnani