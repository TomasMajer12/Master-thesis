import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import torch.nn.functional as F


class SimpleM3N_NN(nn.Module):
    """
    Simple Neural Network + Markov Network classifier
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        #Basic NN
        self.unary_net = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )
        
        #MN parameters
        #W[i,j] represents affinity between class i and class j
        self.pairwise_weights = nn.Parameter(torch.randn(num_classes, num_classes) * 0.1)
    
    def compute_unary_potentials(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        unary = self.unary_net(x_flat)
        return unary.reshape(batch_size, seq_len, self.num_classes)
    
    def compute_pairwise_potentials(self, y_i, y_j):
        return self.pairwise_weights[y_i, y_j]
    
    def forward(self, x):
        return self.compute_unary_potentials(x)


class M3NTrainer:
    """Structured SVM with LP relaxation"""
    def __init__(self, model, learning_rate=0.01, C=1.0):
        self.model = model
        self.C = C  #Regularization parameter
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
    def hamming_loss(self, y_pred, y_true):
        return (y_pred != y_true).float().sum()
    
    def loss_augmented_inference_greedy(self, unary, y_true):
        """
        Greedy loss-augmented inference for sequences
        For small problems, this approximates: argmax [score(y) + loss(y, y_true)]
        """
        batch_size, seq_len, num_classes = unary.shape
        y_pred = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        for b in range(batch_size):
            # For each position
            for t in range(seq_len):
                scores = unary[b, t].clone()
                
                #Add loss augmentation (favor incorrect labels)
                for c in range(num_classes):
                    if c != y_true[b, t]:
                        scores[c] += 1.0  #Hamming loss = 1 for wrong label
                
                #Add pairwise potential with previous label
                if t > 0:
                    prev_label = y_pred[b, t-1]
                    for c in range(num_classes):
                        scores[c] += self.model.pairwise_weights[prev_label, c]
                
                y_pred[b, t] = torch.argmax(scores)
        
        return y_pred
    

    #Viterbi - for small instances
    def viterbi_decode(self, unary):
        batch_size, seq_len, num_classes = unary.shape
        y_pred = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        for b in range(batch_size):
            dp = torch.zeros(seq_len, num_classes)
            backpointer = torch.zeros(seq_len, num_classes, dtype=torch.long)
            
            # Initialize
            dp[0] = unary[b, 0]
            
            #Forward pass
            for t in range(1, seq_len):
                for curr_class in range(num_classes):
                    scores = dp[t-1] + self.model.pairwise_weights[:, curr_class] + unary[b, t, curr_class]
                    dp[t, curr_class] = torch.max(scores)
                    backpointer[t, curr_class] = torch.argmax(scores)
            
            #Backward pass
            y_pred[b, -1] = torch.argmax(dp[-1])
            for t in range(seq_len-2, -1, -1):
                y_pred[b, t] = backpointer[t+1, y_pred[b, t+1]]
        
        return y_pred
    
    def structured_hinge_loss(self, unary, y_true):
        """
        loss = max_y [score(y) + Δ(y,y_true)] - score(y_true)
        """
        batch_size, seq_len, num_classes = unary.shape
        
        # Loss-augmented inference
        y_star = self.loss_augmented_inference_greedy(unary, y_true)
        
        score_star = 0.0
        for b in range(batch_size):
            for t in range(seq_len):
                score_star += unary[b, t, y_star[b, t]]
                if t > 0:
                    score_star += self.model.pairwise_weights[y_star[b, t-1], y_star[b, t]]
        
        # Compute score(y_true)
        score_true = 0.0
        for b in range(batch_size):
            for t in range(seq_len):
                score_true += unary[b, t, y_true[b, t]]
                if t > 0:
                    score_true += self.model.pairwise_weights[y_true[b, t-1], y_true[b, t]]
        
        #Hamming loss
        hamming = self.hamming_loss(y_star, y_true)
        
        #Structured hinge loss
        loss = score_star + hamming - score_true
        loss = torch.clamp(loss, min=0.0)  
        
        return loss / (batch_size * seq_len) 
    
    def train_step(self, x, y_true):
        """Single training step"""
        self.optimizer.zero_grad()
        
        #Forward pass
        unary = self.model(x)
        
        #Compute structured loss
        loss = self.structured_hinge_loss(unary, y_true)
        
        #Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, x):
        with torch.no_grad():
            unary = self.model(x)
            return self.viterbi_decode(unary)




#popsat kod do sablony
#todo parametrizovat
#Spocitat baesovsky klasifikator chybu - implementovat a overit presnost - boumwelsch algoritmus, minimalizujici hamming loss
#udelat vice modelu - porovnat a vice dat - aspon 10k
#v pripade linearni klasifikatoru --> da to stejne reseni? ->> stejne vahy
#porovnat chybu s baesovskym klasifikatorem - udelat benchmark 
#udelat jen jednu linearni vrstvu - zkusit jak funguje i pro vice vrstev a porovnat
#markovuv retezec - ktery s nejakou pravdepodobnosti bude rust - generator rostoucich sekvenci - z mnistu mohu vybrat nahodny label
