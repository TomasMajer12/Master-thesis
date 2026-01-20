
import numpy as np
import torch
import torch.nn.functional as F

def generate_hmc_data(num_samples=1000, seq_len=100, num_states=30, 
                      p_yt_ytm1=0.7, p_xt_yt=0.7, seed=None):
    """
    Generate sequences from a Hidden Markov Chain (HMC) model.
    
    Model Description:
    -----------------
    - Hidden states Y follow a Markov chain with transition probability:
        P(y_t = y_{t-1}) = p_yt_ytm1  (self-transition)
        P(y_t = j | y_{t-1} = i, j != i) = (1 - p_yt_ytm1) / (num_states - 1)
    
    - Observations X are generated from hidden states with emission probability:
        P(x_t = y_t) = p_xt_yt  (correct emission)
        P(x_t = j | y_t = i, j != i) = (1 - p_xt_yt) / (num_states - 1)
    
    Parameters:
    -----------
    num_samples : int - Number of sequences to generate
    seq_len : int - Length of each sequence
    num_states : int - Number of possible states/classes
    p_yt_ytm1 : float - Self-transition probability (higher = more persistent states)
    p_xt_yt : float - Correct emission probability (higher = clearer observations)
    seed : int - Random seed for reproducibility
    
    Returns:
    --------
    X : torch.Tensor - One-hot encoded observations [num_samples, seq_len, num_states]
    Y : torch.Tensor - Hidden state labels [num_samples, seq_len]
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.zeros((num_samples, seq_len), dtype=np.int64)
    Y = np.zeros((num_samples, seq_len), dtype=np.int64)
    
    for n in range(num_samples):
        # Initial state: uniform random
        Y[n, 0] = np.random.randint(num_states)
        
        # Generate hidden state sequence (Markov chain)
        for t in range(1, seq_len):
            if np.random.rand() < p_yt_ytm1:
                Y[n, t] = Y[n, t-1]  # Stay in same state
            else:
                choices = [s for s in range(num_states) if s != Y[n, t-1]]
                Y[n, t] = np.random.choice(choices)
        
        # Generate observations from hidden states
        for t in range(seq_len):
            if np.random.rand() < p_xt_yt:
                X[n, t] = Y[n, t]  # Correct emission
            else:
                choices = [s for s in range(num_states) if s != Y[n, t]]
                X[n, t] = np.random.choice(choices)
    
    X = torch.LongTensor(X)
    Y = torch.LongTensor(Y)
    X = F.one_hot(X, num_classes=num_states).float()
    
    return X, Y


def get_transition_matrix(num_states, p_yt_ytm1):
    A = np.full((num_states, num_states), (1 - p_yt_ytm1) / (num_states - 1))
    np.fill_diagonal(A, p_yt_ytm1)
    return A


def get_emission_matrix(num_states, p_xt_yt):
    B = np.full((num_states, num_states), (1 - p_xt_yt) / (num_states - 1))
    np.fill_diagonal(B, p_xt_yt)
    return B


class ForwardBackwardClassifier:
    """
    Optimal classifier using Forward-Backward algorithm.
    
    This computes marginal posteriors P(y_t | X) and predicts:
        ŷ_t = argmax_y P(y_t = y | X)
    
    This MINIMIZES expected Hamming loss (pointwise optimal predictions).
    
    The Forward-Backward algorithm:
    - Forward pass: α_t(j) = P(x_1,...,x_t, y_t=j)
    - Backward pass: β_t(j) = P(x_{t+1},...,x_T | y_t=j)
    - Marginal posterior: P(y_t=j | X) ∝ α_t(j) * β_t(j)
    """
    
    def __init__(self, num_states=30, p_yt_ytm1=0.7, p_xt_yt=0.7):
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
                alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
        
        # Backward pass
        beta = np.zeros((seq_len, self.num_states))
        beta[-1] = 1.0
        
        for t in range(seq_len - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.sum(A[i, :] * B[:, observations[t+1]] * beta[t+1])
            if scale[t+1] > 0:
                beta[t] /= scale[t+1]
        
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1  # Avoid division by zero
        gamma = gamma / gamma_sum
        
        return gamma
    
    def predict_sequence(self, observations):
        gamma = self.forward_backward(observations)
        return np.argmax(gamma, axis=1)
    
    def predict(self, X):
        observations = torch.argmax(X, dim=-1).cpu().numpy()
        batch_size, seq_len = observations.shape
        preds = np.zeros((batch_size, seq_len), dtype=np.int64)
        
        for b in range(batch_size):
            preds[b] = self.predict_sequence(observations[b])
        
        return torch.LongTensor(preds)
    

