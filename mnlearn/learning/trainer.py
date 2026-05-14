"""
Unified trainer for structured M3N methods.

One ``Trainer`` class covers both training paths the thesis uses:

  * ``loss = "m3n_hinge"`` — structured hinge loss with Viterbi
    loss-augmented inference (chain graphs only at training time).
  * ``loss = "lp_m3n"`` — LP-relaxed M3N with per-example dual variables
    ``phi_i`` jointly optimised with the model parameters

History dict (returned by :meth:`Trainer.fit`):

    {
        "loss":               "lp_m3n",                       # one of "m3n_hinge" | "lp_m3n"
        "epoch":              [1, 2, 3, ...],                 # parallel arrays
        "epoch_seconds":      [...],
        "lr":                 [...],                          # model param group
        "lr_phi":             [...],                          # phi group; None for m3n_hinge
        "train_loss":         [...],
        "train_metrics":      [{"hamming": 0.3, "zero_one": 0.9}, ...],
        "val_metrics":        [{"hamming": 0.4, "zero_one": 0.95}, ...],
        "diagnostics":        [{"phi_norm": 0.0, "pairwise_diag_mean": 0.1, ...}, ...],
        "monitor":            "val_metrics.zero_one",
        "best_epoch":         12,
        "best_monitor_value": 0.05,
        "early_stopped":      False,
    }

The metric set is task-driven: whatever ``self.metrics(x, y)`` and
``self.diagnostics()`` return shows up in the per-epoch dicts. Notebooks
can pull any series for plotting, e.g.::

    xs = history["epoch"]
    ys = [m["zero_one"] for m in history["val_metrics"]]
    plt.plot(xs, ys)
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.optim as optim

from mnlearn.config.schema import TrainingCfg

from .evaluation import hamming_loss, zero_one_loss
from .lp_m3n import lp_m3n_loss
from .structured_svm import structured_hinge_loss


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop when the monitored scalar stops improving.

    The monitored value must be a "lower is better" scalar (Hamming loss,
    zero-one loss, validation cross-entropy, etc.). For metrics that are
    "higher is better" (accuracy), pass ``1 - acc``.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Unified trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer for structured M3N methods.

    Args:
        model:    M3N model (provides ``unary``, ``score``, ``pairwise``,
                  ``num_classes``).
        edges:    ``[E, 2]`` LongTensor — graph structure.
        cfg:      :class:`TrainingCfg` from the experiment YAML.
                  ``cfg.loss`` selects the training mode.
        n_train:  number of training examples. Used to pre-allocate the
                  per-example phi bank when ``cfg.loss == "lp_m3n"``;
                  ignored otherwise.
        device:   defaults to whatever device the model parameters are on.

    Attributes set in ``__init__``:
        loss_kind:        the string from ``cfg.loss`` ("m3n_hinge" or "lp_m3n").
        train_inference:  loss-augmented inference function (None for LP-M3N).
        eval_inference:   decoder used by ``predict``/``metrics``.
        phi_bank:         list of per-example phi tensors (LP-M3N only; else None).
        optimizer:        torch.optim.{Adam,SGD} with the right param groups.
        scheduler:        optional torch LR scheduler.
    """

    def __init__(self, model, edges: torch.Tensor, cfg: TrainingCfg,
                 n_train: int, device: torch.device | None = None):
        self.model     = model
        self.edges     = edges
        self.cfg       = cfg
        self.loss_kind = cfg.loss
        self.device    = device if device is not None else next(model.parameters()).device

        if self.loss_kind not in {"m3n_hinge", "lp_m3n"}:
            raise ValueError(
                f"Trainer: unsupported cfg.loss={self.loss_kind!r}; "
                f"expected one of 'm3n_hinge' | 'lp_m3n'"
            )

        # Inference callables (loss-augmented for training, plain for eval).
        from .builders import build_inference, build_scheduler
        self.train_inference, self.eval_inference = build_inference(cfg.inference, edges)

        # Per-example phi bank only for LP-M3N.
        self.phi_bank = self._build_phi_bank(n_train) if self.loss_kind == "lp_m3n" else None

        self.optimizer = self._build_optimizer()
        self.scheduler = build_scheduler(self.optimizer, cfg.scheduler)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_phi_bank(self, n_train: int) -> list[torch.Tensor]:
        """
        Allocate one ``[2, E, K]`` phi tensor per training example.
        """
        K = self.model.num_classes
        E = self.edges.shape[0]
        std = self.cfg.optimizer.phi_init_std
        bank: list[torch.Tensor] = []
        for _ in range(n_train):
            phi = torch.zeros(
                2, E, K,
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )
            if std > 0.0:
                phi.add_(torch.randn_like(phi) * std)
            phi.requires_grad_(True)
            bank.append(phi)
        return bank

    def _build_optimizer(self) -> optim.Optimizer:
        """Build the optimizer with one param group per loss kind.

        For ``m3n_hinge``: a single group containing the model
        parameters (``lr``, ``weight_decay``).

        For ``lp_m3n``: two groups — model (``lr``, ``weight_decay``)
        and the phi bank (``lr_phi``, ``weight_decay_phi``). Each
        group's ``lr`` is set explicitly so the LR scheduler picks up
        the right per-group ``base_lr`` at construction time; the two
        learning rates then decay in lockstep.
        """
        opt_cfg = self.cfg.optimizer
        groups: list[dict[str, Any]] = [{
            "params":       list(self.model.parameters()),
            "weight_decay": opt_cfg.weight_decay,
            "lr":           opt_cfg.lr,
        }]
        if self.loss_kind == "lp_m3n":
            # Sentinel: lr_phi=0.0 means "same as lr_model".
            phi_lr = opt_cfg.lr_phi if opt_cfg.lr_phi > 0 else opt_cfg.lr
            groups.append({
                "params":       self.phi_bank,
                "weight_decay": opt_cfg.weight_decay_phi,
                "lr":           phi_lr,
            })

        if opt_cfg.type == "adam":
            return optim.Adam(groups, lr=opt_cfg.lr)
        if opt_cfg.type == "sgd":
            return optim.SGD(groups, lr=opt_cfg.lr)
        raise ValueError(f"Unknown optimizer.type={opt_cfg.type!r}")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                   idx: torch.Tensor) -> float:
        """Single gradient step on one mini-batch. Returns scalar loss as float.

        ``idx`` carries the global indices of the examples in this batch
        (LongTensor of shape ``[B]``). It is needed by LP-M3N to look up
        each example's phi tensor in the bank; the M3N hinge branch
        ignores it.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        unary = self.model.unary(x)

        if self.loss_kind == "m3n_hinge":
            loss = structured_hinge_loss(
                self.model, unary, y, self.edges, self.train_inference,
            )
        else:  # lp_m3n
            # Stack the per-example phi tensors into a single [B, 2, E, K]
            # tensor for the batched lp_m3n_loss. torch.stack preserves
            # the leaf identity of each phi_bank entry: gradients computed
            # on the stacked tensor route back to the underlying leaves
            # per slice, so optimizer.step() updates each phi^i in place.
            idx_list = idx.tolist()  # one CPU sync instead of B .item() calls
            phi_batched = torch.stack(
                [self.phi_bank[i] for i in idx_list]
            )  # [B, 2, E, K]
            per_example_losses = lp_m3n_loss(
                unary, self.model.pairwise, y, self.edges, phi_batched,
            )  # [B]
            loss = per_example_losses.mean()

        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Inference / metrics / diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        """Decode predicted labels for ``x``. Chunked over the leading dim."""
        self.model.eval()
        preds: list[torch.Tensor] = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i + chunk_size].to(self.device, non_blocking=True)
            unary = self.model.unary(chunk)
            preds.append(self.eval_inference(unary, self.model.pairwise).cpu())
        return torch.cat(preds, dim=0)

    @torch.no_grad()
    def metrics(self, x: torch.Tensor, y: torch.Tensor,
                chunk_size: int = 64) -> dict[str, float]:
        """Compute Hamming + 0/1 losses on ``(x, y)``. Both are scalars in [0, 1].

        Chunked predict to bound peak memory; the 0/1 reduction is on CPU
        (predictions are gathered to CPU before the comparison).
        """
        preds = self.predict(x, chunk_size=chunk_size)
        y_cpu = y.cpu()
        return {
            "hamming":  hamming_loss(preds, y_cpu),
            "zero_one": zero_one_loss(preds, y_cpu),
        }

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        """Per-epoch task-internal diagnostics.

        Returns
        -------
        dict
            Always present:
              ``pairwise_diag_mean``     — mean of the K diagonal entries of W.
              ``pairwise_off_diag_mean`` — mean of the K(K−1) off-diagonal entries.
            Present only when ``self.phi_bank`` is not None (LP-M3N):
              ``phi_norm`` — mean of per-example L2 norms,
                             ``mean_i ‖phi^i‖_2``.  N-invariant on purpose:
                             useful for spotting per-example phi-saturation
                             without a confounding ``sqrt(N)`` factor.
                             Not the L2 norm of the concatenated bank.
        """
        diag: dict[str, float] = {}

        pw = self.model.pairwise.detach()
        K  = pw.shape[0]
        diag["pairwise_diag_mean"] = pw.diagonal().mean().item()
        off_count = K * K - K
        if off_count > 0:
            off_sum = (pw.sum() - pw.diagonal().sum()).item()
            diag["pairwise_off_diag_mean"] = off_sum / off_count
        else:
            diag["pairwise_off_diag_mean"] = 0.0

        if self.phi_bank is not None and self.phi_bank:
            norms = torch.stack([phi.detach().norm() for phi in self.phi_bank])
            diag["phi_norm"] = norms.mean().item()

        return diag

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self,
            train_data: tuple[torch.Tensor, torch.Tensor],
            val_data:   tuple[torch.Tensor, torch.Tensor],
            *,
            num_epochs: int,
            batch_size: int,
            eval_every: int = 1,
            monitor:    str = "val_metrics.hamming",
            patience:   int = 10,
            min_delta:  float = 0.001,
            verbose:    bool = True,
            print_metrics: list[str] | None = None) -> dict[str, Any]:
        """Run the generic fit loop. Returns the rich history dict.

        ``monitor`` is a dotted path resolved against the per-epoch record
        described in the module docstring (``val_metrics.hamming``,
        ``train_loss``, ``diagnostics.phi_norm``, ...). The Trainer raises
        a clear KeyError if the path doesn't resolve, listing what was
        available — so a typo or task mismatch surfaces fast.
        """
        train_x, train_y = train_data
        val_x,   val_y   = val_data

        N = train_x.shape[0]
        bs = min(batch_size, N)
        num_batches = max(1, (N + bs - 1) // bs)

        early_stop = EarlyStopping(patience=patience, min_delta=min_delta)

        history: dict[str, Any] = {
            "loss":               self.loss_kind,
            "epoch":              [],
            "epoch_seconds":      [],
            "lr":                 [],
            "lr_phi":             [],
            "train_loss":         [],
            "train_metrics":      [],
            "val_metrics":        [],
            "diagnostics":        [],
            "monitor":            monitor,
            "best_epoch":         0,
            "best_monitor_value": float("inf"),
            "early_stopped":      False,
        }

        best_state: dict | None = None

        for epoch in range(num_epochs):
            # ---- Training pass ----
            t_start = time.time()
            perm = torch.randperm(N)
            epoch_loss = 0.0
            total_seen = 0
            for b in range(num_batches):
                idx = perm[b * bs : (b + 1) * bs]
                batch_n = idx.shape[0]
                epoch_loss += self.train_step(train_x[idx], train_y[idx], idx) * batch_n
                total_seen += batch_n
            epoch_loss /= total_seen
            epoch_seconds = time.time() - t_start

            should_eval = (epoch + 1) % eval_every == 0 or epoch == 0
            if should_eval:
                train_m = self.metrics(train_x, train_y)
                val_m   = self.metrics(val_x,   val_y)
                diag    = self.diagnostics()
                # Log both param groups' lr's. Group 0 is always the model;
                # group 1 (if present) is the phi bank for lp_m3n. For
                # m3n_hinge there is no phi group, so lr_phi is None 
                groups = self.optimizer.param_groups
                cur_lr     = groups[0]["lr"]
                cur_lr_phi = groups[1]["lr"] if len(groups) > 1 else None

                history["epoch"].append(epoch + 1)
                history["epoch_seconds"].append(epoch_seconds)
                history["lr"].append(cur_lr)
                history["lr_phi"].append(cur_lr_phi)
                history["train_loss"].append(epoch_loss)
                history["train_metrics"].append(train_m)
                history["val_metrics"].append(val_m)
                history["diagnostics"].append(diag)

                record = _epoch_record(
                    train_loss=epoch_loss, epoch_seconds=epoch_seconds, lr=cur_lr,
                    train_metrics=train_m, val_metrics=val_m, diagnostics=diag,
                )
                monitor_val = _resolve_path(monitor, record)

                if monitor_val < history["best_monitor_value"]:
                    history["best_monitor_value"] = monitor_val
                    history["best_epoch"]         = epoch + 1
                    best_state = copy.deepcopy(self.model.state_dict())

                if verbose:
                    _print_eval_row(epoch + 1, record, print_metrics)

                if early_stop.step(monitor_val):
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    history["early_stopped"] = True
                    break

            # Step LR schedule once per epoch (no-op when scheduler is None).
            # After the eval block so the logged ``lr`` for this epoch
            # reflects the lr that was actually used during training.
            if self.scheduler is not None:
                self.scheduler.step()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history


# ---------------------------------------------------------------------------
# Per-epoch record + dotted-path resolution + printer
# ---------------------------------------------------------------------------

def _epoch_record(*, train_loss: float, epoch_seconds: float, lr: float,
                  train_metrics: dict, val_metrics: dict,
                  diagnostics: dict) -> dict:
    """Build the dict that ``monitor`` paths resolve against."""
    return {
        "train_loss":    float(train_loss),
        "epoch_seconds": float(epoch_seconds),
        "lr":            float(lr),
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "diagnostics":   diagnostics,
    }


def _resolve_path(path: str, record: dict) -> float:
    """Resolve a dotted path ``a.b.c`` against ``record``. Returns a float.

    Raises KeyError with the available keys at the failure point if the
    path doesn't fully resolve, or TypeError if it lands on a non-scalar.
    """
    parts = path.split(".")
    cur: Any = record
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(
                    f"monitor path {path!r}: key {p!r} missing. "
                    f"Available at this level: {sorted(cur.keys())}"
                )
            cur = cur[p]
        else:
            raise KeyError(
                f"monitor path {path!r}: cannot descend past a non-dict value "
                f"({type(cur).__name__}) at part {p!r}."
            )
    if not isinstance(cur, (int, float)):
        raise TypeError(
            f"monitor path {path!r} resolved to {type(cur).__name__}; "
            f"expected a scalar (int / float)."
        )
    return float(cur)


def _print_eval_row(epoch: int, record: dict,
                    print_metrics: list[str] | None) -> None:
    """Emit one console line summarising an evaluation tick.

    When ``print_metrics`` is None, the default is ``train_loss`` plus
    every key in ``val_metrics`` (sorted). The label printed for each
    path is its trailing component (``val_metrics.hamming`` -> ``hamming``)
    so the line reads cleanly without restating the namespace each time.
    """
    if print_metrics is None:
        print_metrics = ["train_loss"]
        for k in sorted(record["val_metrics"].keys()):
            print_metrics.append(f"val_metrics.{k}")

    parts = [f"Epoch {epoch:3d}:"]
    for path in print_metrics:
        try:
            val = _resolve_path(path, record)
            label = path.rsplit(".", 1)[-1]
            parts.append(f"{label}={val:.4f}")
        except (KeyError, TypeError):
            parts.append(f"{path}=?")
    print("  " + "  ".join(parts))
