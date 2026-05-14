"""
Classifier trainer for non-structured prediction.

Used by the OCR digit classifier in the two-stage Sudoku baseline. The
trainer expects flat (image, label) pairs (per-cell, not per-puzzle) and
trains a single-label classifier with ``nn.CrossEntropyLoss``.

Mirrors the API of :class:`mnlearn.learning.Trainer`:

  * ``fit(train_X, train_Y, val_X, val_Y, config) -> history`` — same shape.
  * ``EarlyStopping`` from :mod:`mnlearn.learning.trainer` — same helper.
  * Scheduler is stepped once per epoch — same contract.
  * Best model state (lowest ``val_error``) restored on ``fit()`` exit — same.

Model contract: an ``nn.Module`` mapping ``[batch, *input_shape]`` to
``[batch, num_classes]`` of **raw logits**.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from .trainer import EarlyStopping


class ClassifierTrainer:
    """Single-label classification trainer (cross-entropy).

    Args:
        model:        ``nn.Module`` mapping ``[batch, *input_shape] ->
                      [batch, num_classes]`` of raw logits.
        lr:           learning rate.
        weight_decay: L2 regularisation on model parameters.
        optimizer:    ``"adam"`` or ``"sgd"``.
        scheduler:    optional torch LR scheduler. ``None`` = constant lr.
        device:       defaults to whatever device the model is on.
    """

    def __init__(self, model: nn.Module, lr: float = 0.01,
                 weight_decay: float = 0.0, optimizer: str = "adam",
                 scheduler=None, device: torch.device | None = None):
        self.model = model
        self.scheduler = scheduler
        self.device = device if device is not None else next(model.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss()

        if optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # ------------------------------------------------------------------
    # Single-step / inference
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, y_true: torch.Tensor) -> float:
        """Single gradient step on one mini-batch. Returns loss as a Python float."""
        self.model.train()
        self.optimizer.zero_grad()
        x = x.to(self.device, non_blocking=True)
        y_true = y_true.to(self.device, non_blocking=True)

        logits = self.model(x)
        loss = self.loss_fn(logits, y_true)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        """Return ``[N]`` LongTensor of predicted classes (argmax over logits).

        Chunked over the leading dimension so a 30k-sample validation tensor
        does not materialise its full activation map at once. Each chunk is
        moved to the trainer's device just before the forward pass; the
        returned tensor lives on the same device as the model.
        """
        self.model.eval()
        preds = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i + chunk_size].to(self.device, non_blocking=True)
            preds.append(self.model(chunk).argmax(dim=-1))
        return torch.cat(preds, dim=0)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        """Return ``[N, K]`` softmaxed probabilities. Chunked, same contract."""
        self.model.eval()
        outputs = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i + chunk_size].to(self.device, non_blocking=True)
            outputs.append(self.model(chunk).softmax(dim=-1))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y_true: torch.Tensor,
                 chunk_size: int = 1024) -> dict:
        """Return ``{'error': float, 'loss': float}`` (both scalars).

        ``error`` is the fraction of samples whose argmax differs from
        ``y_true``. ``loss`` is the mean cross-entropy. Computed in one
        chunked sweep so peak memory matches :func:`predict`.
        """
        self.model.eval()
        total = 0
        total_correct = 0
        total_loss = 0.0
        for i in range(0, x.shape[0], chunk_size):
            chunk_x = x[i:i + chunk_size].to(self.device, non_blocking=True)
            chunk_y = y_true[i:i + chunk_size].to(self.device, non_blocking=True)
            logits = self.model(chunk_x)
            n = chunk_x.shape[0]
            total_loss += self.loss_fn(logits, chunk_y).item() * n
            total_correct += (logits.argmax(dim=-1) == chunk_y).sum().item()
            total += n
        if total == 0:
            return {"error": float("nan"), "loss": float("nan")}
        return {
            "error": 1.0 - total_correct / total,
            "loss":  total_loss / total,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor,
            val_x:   torch.Tensor, val_y:   torch.Tensor,
            config:  dict) -> dict:
        """Train with mini-batch SGD; early-stop on ``val_error``.

        Args:
            train_x: ``[N_train, *input_shape]`` (CPU is fine; per-batch transfer).
            train_y: ``[N_train]`` LongTensor.
            val_x:   ``[N_val,   *input_shape]``.
            val_y:   ``[N_val]``  LongTensor.
            config:  dict with keys ``num_epochs``, ``batch_size``,
                     ``eval_every``, ``patience``, ``min_delta``, ``verbose``.

        Returns:
            History dict with per-epoch trajectories and the best-epoch
            summary (``best_val_error``, ``best_epoch``, ``early_stopped``).
        """
        N = train_x.shape[0]
        batch_size = min(config["batch_size"], N)
        num_batches = max(1, (N + batch_size - 1) // batch_size)
        verbose = config.get("verbose", True)

        early_stop = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001),
        )

        history: dict = {
            "epoch":          [],
            "train_loss":     [],
            "train_error":    [],
            "val_error":      [],
            "val_loss":       [],
            "lr":             [],
            "best_val_error": float("inf"),
            "best_epoch":     0,
            "early_stopped":  False,
        }

        best_state = None

        for epoch in range(config["num_epochs"]):
            # --- Train one epoch ---
            perm = torch.randperm(N)
            epoch_loss = 0.0
            total_seen = 0

            for i in range(num_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                batch_n = idx.shape[0]
                epoch_loss += self.train_step(train_x[idx], train_y[idx]) * batch_n
                total_seen += batch_n

            epoch_loss /= total_seen

            # --- Periodic evaluation ---
            if (epoch + 1) % config.get("eval_every", 1) == 0 or epoch == 0:
                train_metrics = self.evaluate(train_x, train_y)
                val_metrics   = self.evaluate(val_x,   val_y)

                history["epoch"].append(epoch + 1)
                history["train_loss"].append(epoch_loss)
                history["train_error"].append(train_metrics["error"])
                history["val_error"].append(val_metrics["error"])
                history["val_loss"].append(val_metrics["loss"])
                history["lr"].append(self.optimizer.param_groups[0]["lr"])

                if val_metrics["error"] < history["best_val_error"]:
                    history["best_val_error"] = val_metrics["error"]
                    history["best_epoch"] = epoch + 1
                    best_state = copy.deepcopy(self.model.state_dict())

                if verbose:
                    print(
                        f"  Epoch {epoch + 1:3d}: "
                        f"loss={epoch_loss:.4f}  "
                        f"trn_err={train_metrics['error']:.4f}  "
                        f"val_err={val_metrics['error']:.4f}  "
                        f"val_loss={val_metrics['loss']:.4f}"
                    )

                if early_stop.step(val_metrics["error"]):
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    history["early_stopped"] = True
                    break

            # Step the optional LR schedule once per epoch (no-op if scheduler
            # was not provided). Stepped after the eval block so the logged
            # ``lr`` for this epoch matches the lr that was used during training.
            if self.scheduler is not None:
                self.scheduler.step()

        # Restore the model state with the lowest val_error seen.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
