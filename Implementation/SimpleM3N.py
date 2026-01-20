import torch
import torch.nn as nn
import torch.optim as optim


class SimpleM3N_NN(nn.Module):
    """
    Simple Neural Network + Markov Network classifier (Linear version)
    
    Architecture:
    - Unary potentials: Linear(input_dim, num_classes)
    - Pairwise potentials: Learnable weight matrix W[i,j]
    
    F(y|x) = Σ_t ψ_unary(x_t, y_t) + Σ_t ψ_pairwise(y_{t-1}, y_t)
    """
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Single linear layer
        self.unary_net = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )
        
        # W[i,j] represents affinity between class i and class j
        self.pairwise_weights = nn.Parameter(torch.randn(num_classes, num_classes) * 0.1)
    
    def compute_unary_potentials(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        unary = self.unary_net(x_flat)
        return unary.reshape(batch_size, seq_len, self.num_classes)
    
    def compute_pairwise_potentials(self, y_i, y_j):
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
        return self.compute_unary_potentials(x)


class M3NTrainer:
    """
    Structured SVM Trainer
    """
    
    def __init__(self, model, learning_rate=0.01, C=1.0, weight_decay=0.01):
        self.model = model
        self.C = C  # Regularization parameter
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def hamming_loss(self, y_pred, y_true):
        return (y_pred != y_true).float().sum()
    
    def loss_augmented_inference_viterbi(self, unary, y_true):
        """
        Loss-augmented Viterbi inference (exact for chain structure).
        
        Finds: argmax_y [score(y|x) + Δ(y, y_true)]
        """
        batch_size, seq_len, num_classes = unary.shape
        y_pred = torch.zeros(batch_size, seq_len, dtype=torch.long, device=unary.device)
        
        pairwise = self.model.pairwise_weights.detach()
        
        for b in range(batch_size):
            # DP tables
            dp = torch.zeros(seq_len, num_classes, device=unary.device)
            backpointer = torch.zeros(seq_len, num_classes, dtype=torch.long, device=unary.device)
            
            # Initialize with unary + loss augmentation
            for c in range(num_classes):
                dp[0, c] = unary[b, 0, c].detach()
                if c != y_true[b, 0]:
                    dp[0, c] += 1.0  # Hamming loss augmentation
            
            # Forward pass
            for t in range(1, seq_len):
                for curr_class in range(num_classes):
                    # Score = previous best + transition + unary + loss
                    scores = dp[t-1] + pairwise[:, curr_class] + unary[b, t, curr_class].detach()
                    if curr_class != y_true[b, t]:
                        scores = scores + 1.0  # Hamming loss augmentation
                    
                    dp[t, curr_class] = torch.max(scores)
                    backpointer[t, curr_class] = torch.argmax(scores)
            
            # Backward pass
            y_pred[b, -1] = torch.argmax(dp[-1])
            for t in range(seq_len - 2, -1, -1):
                y_pred[b, t] = backpointer[t + 1, y_pred[b, t + 1]]
        
        return y_pred
    
    def viterbi_decode(self, unary):
        """
        Finds: argmax_y score(y|x)
        """
        batch_size, seq_len, num_classes = unary.shape
        y_pred = torch.zeros(batch_size, seq_len, dtype=torch.long, device=unary.device)
        
        pairwise = self.model.pairwise_weights.detach()
        
        for b in range(batch_size):
            dp = torch.zeros(seq_len, num_classes, device=unary.device)
            backpointer = torch.zeros(seq_len, num_classes, dtype=torch.long, device=unary.device)
            
            # Initialize
            dp[0] = unary[b, 0].detach()
            
            # Forward pass
            for t in range(1, seq_len):
                for curr_class in range(num_classes):
                    scores = dp[t-1] + pairwise[:, curr_class] + unary[b, t, curr_class].detach()
                    dp[t, curr_class] = torch.max(scores)
                    backpointer[t, curr_class] = torch.argmax(scores)
            
            # Backward pass
            y_pred[b, -1] = torch.argmax(dp[-1])
            for t in range(seq_len - 2, -1, -1):
                y_pred[b, t] = backpointer[t + 1, y_pred[b, t + 1]]
        
        return y_pred
    
    def structured_hinge_loss(self, unary, y_true):
        """
        loss = max_y [score(y) + Δ(y,y_true)] - score(y_true)
        """
        batch_size, seq_len, num_classes = unary.shape
        
        # Loss-augmented inference 
        y_star = self.loss_augmented_inference_viterbi(unary, y_true)
        
        # Compute score(y*)
        score_star = torch.zeros(1, device=unary.device)
        for b in range(batch_size):
            for t in range(seq_len):
                score_star = score_star + unary[b, t, y_star[b, t]]
                if t > 0:
                    score_star = score_star + self.model.pairwise_weights[y_star[b, t-1], y_star[b, t]]
        
        # Compute score(y_true)
        score_true = torch.zeros(1, device=unary.device)
        for b in range(batch_size):
            for t in range(seq_len):
                score_true = score_true + unary[b, t, y_true[b, t]]
                if t > 0:
                    score_true = score_true + self.model.pairwise_weights[y_true[b, t-1], y_true[b, t]]
        
        
        # Structured hinge loss
        loss = score_star - score_true
        loss = torch.clamp(loss, min=0.0)
        
        return loss / (batch_size * seq_len)
    
    def train_step(self, x, y_true):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        unary = self.model(x)
        
        # Compute structured loss
        loss = self.structured_hinge_loss(unary, y_true)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, x):
        with torch.no_grad():
            unary = self.model(x)
            return self.viterbi_decode(unary)


