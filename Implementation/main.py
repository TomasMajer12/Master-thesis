import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from SimpleM3N import SimpleM3N_NN, M3NTrainer
from dataset_HMC import generate_hmc_data, OptimalBayesClassifier
import seaborn as sns


def evaluate(model, trainer, X, Y):
    predictions = trainer.predict(X)
    accuracy = (predictions == Y).float().mean().item()
    return accuracy


def run_experiment(num_states=30, seq_len=15, num_train=200, num_test=40, num_epochs=10, batch_size=32, seed=123, run_name="Run",p_yt_ytm1=0.7, p_xt_yt=0.7):
    print(f"\n=== {run_name} ===")
    print("Generating data...")
    num_samples = num_train + num_test
    X, Y = generate_hmc_data(num_samples, seq_len, num_states, seed=seed,p_yt_ytm1=p_yt_ytm1, p_xt_yt=p_xt_yt)
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]

    # Bayes optimal classifier
    print("\nEvaluating Optimal Bayes Classifier...")
    bayes_classifier = OptimalBayesClassifier(num_states=num_states, p_yt_ytm1=p_yt_ytm1, p_xt_yt=p_xt_yt)
    bayes_train_acc = (bayes_classifier.predict(X_train) == Y_train).float().mean().item()
    bayes_test_acc = (bayes_classifier.predict(X_test) == Y_test).float().mean().item()
    print(f"Bayes Optimal Train Accuracy: {bayes_train_acc:.4f}")
    print(f"Bayes Optimal Test Accuracy: {bayes_test_acc:.4f}")

    # Neural network
    print("\nTraining Neural Network...")
    model = SimpleM3N_NN(input_dim=num_states, num_classes=num_states)
    trainer = M3NTrainer(model, learning_rate=0.01, C=1.0)

    train_accs, test_accs, losses = [], [], []

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for i in range(0, num_train, batch_size):
            end_idx = min(i + batch_size, num_train)
            X_batch = X_train[i:end_idx]
            Y_batch = Y_train[i:end_idx]

            loss = trainer.train_step(X_batch, Y_batch)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # Evaluate
        train_acc = evaluate(model, trainer, X_train, Y_train)
        test_acc = evaluate(model, trainer, X_test, Y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Gap to Bayes: {bayes_test_acc - test_acc:.4f}")

    # Final summary
    print("\n" + "="*60)
    print(f"FINAL RESULTS ({run_name}):")
    print("="*60)
    print(f"Bayes Optimal Test Accuracy: {bayes_test_acc:.4f}")
    print(f"Neural Network Test Accuracy: {test_accs[-1]:.4f}")
    print(f"Gap to Optimal: {bayes_test_acc - test_accs[-1]:.4f}")
    print(f"Percentage of Optimal: {100 * test_accs[-1] / bayes_test_acc:.2f}%")

    return model, bayes_test_acc, test_accs[-1]


def plot_pairwise_weights(weights, title="Pairwise Weights", filename=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(weights.detach().cpu().numpy(), annot=False, cmap="coolwarm")
    plt.title(title)
    plt.xlabel("y_t")
    plt.ylabel("y_{t-1}")
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_unary_weights(model, title="Unary Weights", filename=None):
    W = model.unary_net[0].weight.detach().cpu().numpy()
    plt.figure(figsize=(8,6))
    sns.heatmap(W, annot=False, cmap="viridis")
    plt.title(title)
    plt.xlabel("Input Features")
    plt.ylabel("Classes")
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_pairwise_difference(weights1, weights2, title="Pairwise Weights Difference", filename=None):
    diff = (weights1 - weights2).detach().cpu().numpy()
    plt.figure(figsize=(6,5))
    sns.heatmap(diff, annot=False, cmap="bwr", center=0)
    plt.title(title)
    plt.xlabel("y_t")
    plt.ylabel("y_{t-1}")
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_unary_difference(model1, model2, title="Unary Weights Difference", filename=None):
    W1 = model1.unary_net[0].weight.detach().cpu()
    W2 = model2.unary_net[0].weight.detach().cpu()
    diff = (W1 - W2).numpy()
    plt.figure(figsize=(8,6))
    sns.heatmap(diff, annot=False, cmap="bwr", center=0)
    plt.title(title)
    plt.xlabel("Input Features")
    plt.ylabel("Classes")
    if filename:
        plt.savefig(filename)
    plt.show()


# --- Run the comparison ---
if __name__ == "__main__":

    num_states = 15
    seq_len = 30
    num_train = 400
    num_test = 80
    num_epochs = 20
    batch_size = 32
    seed = 123
    p_yt_ytm1=0.7
    p_xt_yt=0.7

    # Call run_experiment with parameters
    model1, bayes_acc, test_acc1 = run_experiment(
        run_name="Run 1",
        num_states=num_states,
        seq_len=seq_len,
        num_train=num_train,
        num_test=num_test,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        p_yt_ytm1=p_yt_ytm1,
        p_xt_yt=p_xt_yt
    )

    model2, _, test_acc2 = run_experiment(
        run_name="Run 2",
        num_states=num_states,
        seq_len=seq_len,
        num_train=num_train,
        num_test=num_test,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        p_yt_ytm1=p_yt_ytm1,
        p_xt_yt=p_xt_yt
    )

    print(f"\nTest Accuracy Run 1: {test_acc1:.4f}")
    print(f"Test Accuracy Run 2: {test_acc2:.4f}")

    # --- Graphs ---
    plot_pairwise_weights(model1.pairwise_weights, title="Pairwise Weights - Run 1", filename="graphs/pairwise_run1.png")
    plot_pairwise_weights(model2.pairwise_weights, title="Pairwise Weights - Run 2", filename="graphs/pairwise_run2.png")
    plot_pairwise_difference(model1.pairwise_weights, model2.pairwise_weights, title="Pairwise Weights Difference", filename="graphs/pairwise_diff.png")

    plot_unary_weights(model1, title="Unary Weights - Run 1", filename="graphs/unary_run1.png")
    plot_unary_weights(model2, title="Unary Weights - Run 2", filename="graphs/unary_run2.png")
    plot_unary_difference(model1, model2, title="Unary Weights Difference", filename="graphs/unary_diff.png")
