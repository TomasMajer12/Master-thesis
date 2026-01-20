
import torch
import torch.nn as nn
from SimpleM3N import M3NTrainer  # Reuse the trainer


class NeuralM3N(nn.Module):
    
    def __init__(self, input_dim, num_classes, hidden_dim=32):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = [hidden_dim]

        self.unary_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

        self.pairwise_weights = nn.Parameter(0.1 * torch.randn(num_classes, num_classes))
    
    def get_model_description(self):
        """Return string description of the model architecture"""
        dims = [self.input_dim] + self.hidden_dims + [self.num_classes]
        return f"Neural M3N: {' -> '.join(map(str, dims))}"
    
    def compute_unary_potentials(self, x):
        """Compute unary potentials for all positions in sequence"""
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        unary = self.unary_net(x_flat)
        return unary.reshape(batch_size, seq_len, self.num_classes)
    
    def compute_pairwise_potentials(self, y_i, y_j):
        """Get pairwise potential between two class labels"""
        return self.pairwise_weights[y_i, y_j]
    
    def compute_sequence_score(self, x, y):
        """
        Compute total score for a sequence.
        
        score(y|x) = Σ_t unary[t, y_t] + Σ_t pairwise[y_{t-1}, y_t]
        """
        unary = self.compute_unary_potentials(x)
        batch_size, seq_len = y.shape
        
        score = 0.0
        for b in range(batch_size):
            for t in range(seq_len):
                score += unary[b, t, y[b, t]]
                if t > 0:
                    score += self.pairwise_weights[y[b, t-1], y[b, t]]
        return score
    
    def forward(self, x):
        """Forward pass returns unary potentials"""
        return self.compute_unary_potentials(x)
