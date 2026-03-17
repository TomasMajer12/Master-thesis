"""
Training loop for M3N models.

Handles:
    - Mini-batch SGD with structured hinge loss
    - Periodic evaluation (Hamming + 0/1 loss on train and test)
    - Early stopping based on test Hamming loss
    - Checkpointing the best model
    - Logging training history for plotting
"""

import torch
import torch.optim as optim
import copy

from .structured_svm import structured_hinge_loss
from .evaluation import hamming_loss, zero_one_loss


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Args:
        patience:  how many evaluations to wait after last improvement
        min_delta: minimum decrease in metric to count as improvement
    """

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float('inf')

    def step(self, metric):
        """Returns True if training should stop."""
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Training loop for M3N.

    Args:
        model:        M3N model
        inference_fn: loss-augmented inference function
                      signature: (unary, pairwise, y_true) -> y_star
        predict_fn:   standard inference function for evaluation
                      signature: (unary, pairwise) -> y_pred
        edges:        [num_edges, 2] — graph structure (LongTensor)
        lr:           learning rate
        weight_decay: L2 regularization
        optimizer:    optimizer name ('adam' or 'sgd')
    """

    def __init__(self, model, inference_fn, predict_fn, edges,
                 lr=0.01, weight_decay=0.01, optimizer='adam'):
        self.model = model
        self.inference_fn = inference_fn
        self.predict_fn = predict_fn
        self.edges = edges

        if optimizer == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def train_step(self, x, y_true):
        """Single gradient step on one mini-batch.

        Returns:
            loss value (float)
        """
        self.model.train()
        self.optimizer.zero_grad()

        unary = self.model.unary(x)
        loss = structured_hinge_loss(
            self.model, unary, y_true, self.edges, self.inference_fn
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, x):
        """Run inference to get predicted labels.

        Returns:
            y_pred: [batch, num_nodes] (LongTensor)
        """
        self.model.eval()
        unary = self.model.unary(x)
        return self.predict_fn(unary, self.model.pairwise)

    @torch.no_grad()
    def evaluate(self, x, y_true):
        """Compute Hamming and 0/1 loss on a dataset.

        Returns:
            dict with 'hamming' and 'zero_one' keys (values in [0, 1])
        """
        y_pred = self.predict(x)
        return {
            'hamming': hamming_loss(y_pred, y_true),
            'zero_one': zero_one_loss(y_pred, y_true),
        }

    def fit(self, train_x, train_y, test_x, test_y, config):
        """Full training loop.

        Args:
            train_x: [N_train, T, input_dim]
            train_y: [N_train, T]
            test_x:  [N_test, T, input_dim]
            test_y:  [N_test, T]
            config:  dict with keys:
                num_epochs:   max epochs
                batch_size:   mini-batch size
                eval_every:   evaluate every N epochs
                patience:     early stopping patience
                min_delta:    early stopping threshold
                verbose:      print progress (default True)

        Returns:
            history: dict with training curves and best results
        """
        N = train_x.shape[0]
        batch_size = min(config['batch_size'], N)
        num_batches = max(1, N // batch_size)
        verbose = config.get('verbose', True)

        early_stop = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001),
        )

        history = {
            'epoch':        [],
            'train_loss':   [],
            'train_hamming': [],
            'train_zero_one': [],
            'test_hamming':  [],
            'test_zero_one': [],
            'best_test_hamming': float('inf'),
            'best_epoch':   0,
            'early_stopped': False,
        }

        best_state = None

        for epoch in range(config['num_epochs']):
            # --- Train one epoch ---
            perm = torch.randperm(N)
            epoch_loss = 0.0

            for i in range(num_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                loss = self.train_step(train_x[idx], train_y[idx])
                epoch_loss += loss

            epoch_loss /= num_batches

            # --- Evaluate periodically ---
            if (epoch + 1) % config.get('eval_every', 1) == 0 or epoch == 0:
                train_metrics = self.evaluate(train_x, train_y)
                test_metrics = self.evaluate(test_x, test_y)

                history['epoch'].append(epoch + 1)
                history['train_loss'].append(epoch_loss)
                history['train_hamming'].append(train_metrics['hamming'])
                history['train_zero_one'].append(train_metrics['zero_one'])
                history['test_hamming'].append(test_metrics['hamming'])
                history['test_zero_one'].append(test_metrics['zero_one'])

                # Track best model
                if test_metrics['hamming'] < history['best_test_hamming']:
                    history['best_test_hamming'] = test_metrics['hamming']
                    history['best_epoch'] = epoch + 1
                    best_state = copy.deepcopy(self.model.state_dict())

                if verbose:
                    print(
                        f"  Epoch {epoch+1:3d}: "
                        f"loss={epoch_loss:.4f}  "
                        f"trn_ham={train_metrics['hamming']:.4f}  "
                        f"tst_ham={test_metrics['hamming']:.4f}  "
                        f"tst_01={test_metrics['zero_one']:.4f}"
                    )

                # Early stopping
                if early_stop.step(test_metrics['hamming']):
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    history['early_stopped'] = True
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
