
import numpy as np
import torch
import torch.nn.functional as F


# Your data generation function
def generate_hmc_data(num_samples=1000, seq_len=100, num_states=30, p_yt_ytm1=0.7, p_xt_yt=0.7, seed = None):
    """Generate sequences from a Hidden Markov Chain (HMC) model."""
    if seed is not None:
        np.random.seed(seed)
    X = np.zeros((num_samples, seq_len), dtype=np.int64)
    Y = np.zeros((num_samples, seq_len), dtype=np.int64)
    
    for n in range(num_samples):
        Y[n, 0] = np.random.randint(num_states)
        for t in range(1, seq_len):
            if np.random.rand() < p_yt_ytm1:
                Y[n, t] = Y[n, t-1]
            else:
                choices = [s for s in range(num_states) if s != Y[n, t-1]]
                Y[n, t] = np.random.choice(choices)
        
        for t in range(seq_len):
            if np.random.rand() < p_xt_yt:
                X[n, t] = Y[n, t]
            else:
                choices = [s for s in range(num_states) if s != Y[n, t]]
                X[n, t] = np.random.choice(choices)
    
    X = torch.LongTensor(X)
    Y = torch.LongTensor(Y)
    X = F.one_hot(X, num_classes=num_states).float()
    return X, Y

class OptimalBayesClassifier:
    """
    Fully generative optimal Bayes classifier for your HMC data.
    Uses ONLY true transition and emission probabilities.
    Performs classical HMM Viterbi decoding.
    """
    def __init__(self, num_states=30, p_yt_ytm1=0.7, p_xt_yt=0.7):
        self.num_states = num_states
        self.p_yt_ytm1 = p_yt_ytm1
        self.p_xt_yt = p_xt_yt

        # Transition matrix A[i,j] = P(y_t=j | y_{t-1}=i)
        self.transition_prob = np.full((num_states, num_states),
                                       (1 - p_yt_ytm1) / (num_states - 1))
        np.fill_diagonal(self.transition_prob, p_yt_ytm1)

        # Emission matrix B[y,x] = P(x_t=x | y_t=y)
        self.emission_prob = np.full((num_states, num_states),
                                     (1 - p_xt_yt) / (num_states - 1))
        np.fill_diagonal(self.emission_prob, p_xt_yt)

        # Uniform prior
        self.prior = np.full(num_states, 1.0 / num_states)


    def viterbi_decode(self, observations):
        """
        observations: int array [batch_size, seq_len]
        Returns: predicted labels (torch.LongTensor)
        """
        batch_size, seq_len = observations.shape
        preds = np.zeros((batch_size, seq_len), dtype=np.int64)

        logA = np.log(self.transition_prob)
        logB = np.log(self.emission_prob)
        logpi = np.log(self.prior)

        for b in range(batch_size):
            obs = observations[b]
            log_delta = np.zeros((seq_len, self.num_states))
            psi = np.zeros((seq_len, self.num_states), dtype=np.int64)

            # Initialization
            log_delta[0] = logpi + logB[:, obs[0]]

            # Forward
            for t in range(1, seq_len):
                for j in range(self.num_states):
                    trans_scores = log_delta[t-1] + logA[:, j]
                    psi[t, j] = np.argmax(trans_scores)
                    log_delta[t, j] = trans_scores[psi[t,j]] + logB[j, obs[t]]

            # Backtracking
            preds[b, -1] = np.argmax(log_delta[-1])
            for t in reversed(range(1, seq_len)):
                preds[b, t-1] = psi[t, preds[b, t]]

        return torch.LongTensor(preds)

    def predict(self, X):
        """
        X: one-hot observations [batch, seq_len, num_states]
        """
        observations = torch.argmax(X, dim=-1).cpu().numpy()
        return self.viterbi_decode(observations)




