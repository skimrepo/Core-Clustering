import json
import os
import time
from typing import List, Optional

import torch

from core_clustering.losses_contrastive import DEFAULT_WEIGHTS, MultiHeadContrastiveLoss

ATTRS = ("shape", "location", "extent", "intensity")


class ContrastiveTrainer:
    """Mirrors TCNTrainer's checkpointing/early-stop/history-JSON
    conventions. Batches come from a BalancedBatchSampler-backed
    DataLoader (see dataset_contrastive.py); loss is MultiHeadContrastiveLoss
    (see losses_contrastive.py) -- one dedicated embedding head per
    attribute, each read by its own loss term, no fixed margins.

    One optimizer PER ATTRIBUTE, each covering the shared trunk (encoder +
    pool_query + pool_attn) plus only that attribute's own head (and, for
    shape, its own contrastive-temperature parameter). A
    single shared AdamW would blend all four attributes' gradient
    statistics into ONE moving-average per trunk parameter -- if location/
    extent/intensity's regression gradients are noisier or larger than
    shape's, that noise distorts Adam's adaptive step size for shape too,
    even though shape has no direct competitor on its own head. Separate
    optimizers keep each attribute's adaptive scaling on the trunk
    independent; heads never overlap between optimizers regardless.

    The trunk's LR is divided by len(ATTRS): every batch applies FOUR full
    Adam steps to the same shared trunk parameters in sequence (one per
    attribute), which is effectively a 4x amplified trunk learning rate
    compared to a single combined-gradient step -- confirmed empirically
    (trunk parameter norm grew steadily batch over batch, and extent/
    intensity's loss diverged to the tens within ~30 epochs on a toy
    dataset). Dividing by len(ATTRS) brings the trunk's total per-batch
    movement back to roughly one combined step's worth; head parameters
    keep the full lr since they're never touched by more than one
    optimizer."""

    def __init__(self, model, device: str = "cpu", lr: float = 0.001, max_grad_norm: float = 1.0,
                 patience: int = 10, weights=DEFAULT_WEIGHTS,
                 output_dir: Optional[str] = None):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = MultiHeadContrastiveLoss(weights=weights).to(device)

        trunk_params = (
            list(self.model.encoder.parameters())
            + list(self.model.pool_attn.parameters())
            + [self.model.pool_query]
        )
        trunk_lr = lr / len(ATTRS)
        self.param_groups = {}
        self.optimizers = {}
        for attr in ATTRS:
            loss_module = getattr(self.loss_fn, f"{attr}_loss")
            own_params = list(self.model.heads[attr].parameters()) + list(loss_module.parameters())
            self.param_groups[attr] = trunk_params + own_params
            self.optimizers[attr] = torch.optim.AdamW([
                {"params": trunk_params, "lr": trunk_lr},
                {"params": own_params, "lr": lr},
            ])

        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.output_dir = output_dir
        self.best_val_loss = float("inf")
        self.best_epoch: Optional[int] = None

    def _run_epoch(self, dataloader, train: bool):
        self.model.train(train)
        self.loss_fn.train(train)
        total_loss = 0.0
        total_by_attr = {a: 0.0 for a in ATTRS}
        step_count = 0
        for batch in dataloader:
            Y = batch["Y"].to(self.device)
            pad_mask = batch["pad_mask"].to(self.device)
            shape = batch["shape_label"].to(self.device)
            loc = batch["location_value"].to(self.device)
            ext = batch["extent_value"].to(self.device)
            inten = batch["intensity_value"].to(self.device)

            with torch.set_grad_enabled(train):
                emb = self.model(Y, pad_mask=pad_mask)
                comp = self.loss_fn.compute_components(emb, shape, loc, ext, inten)
            if any(torch.isnan(comp[a]) for a in ATTRS):
                continue

            if train:
                # Compute all four attributes' gradients FIRST (via the
                # functional autograd.grad, not .backward()+.step()
                # interleaved) -- stepping an optimizer mutates shared trunk
                # parameters in place, which corrupts the retained graph
                # any later backward() call still needs. Only mutate
                # parameters after every gradient has been extracted.
                grads = {}
                for i, attr in enumerate(ATTRS):
                    weighted = self.loss_fn.weights[i] * comp[attr]
                    retain = i < len(ATTRS) - 1
                    grads[attr] = torch.autograd.grad(
                        weighted, self.param_groups[attr], retain_graph=retain, allow_unused=True
                    )
                for attr in ATTRS:
                    self.optimizers[attr].zero_grad()
                    for p, g in zip(self.param_groups[attr], grads[attr]):
                        p.grad = g
                    torch.nn.utils.clip_grad_norm_(self.param_groups[attr], self.max_grad_norm)
                    self.optimizers[attr].step()

            loss_value = sum(self.loss_fn.weights[i] * comp[a].item() for i, a in enumerate(ATTRS))
            total_loss += loss_value
            for a in ATTRS:
                total_by_attr[a] += comp[a].item()
            step_count += 1

        if step_count == 0:
            return float("nan"), {a: float("nan") for a in ATTRS}
        avg_by_attr = {a: total_by_attr[a] / step_count for a in ATTRS}
        return total_loss / step_count, avg_by_attr

    def train(self, train_dataloader, val_dataloader=None, epochs: int = 100) -> List[dict]:
        history: List[dict] = []
        stop_counter = 0

        for epoch in range(epochs):
            start = time.time()
            train_loss, train_by_attr = self._run_epoch(train_dataloader, train=True)
            epoch_seconds = time.time() - start

            val_loss = None
            val_by_attr = {a: None for a in ATTRS}
            is_best = False
            if val_dataloader is not None:
                val_loss, val_by_attr = self._run_epoch(val_dataloader, train=False)
                if val_loss != val_loss:  # NaN
                    stop_counter += 10
                elif val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    stop_counter = 0
                    is_best = True
                    if self.output_dir:
                        os.makedirs(self.output_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "bestmodel.pkl"))
                else:
                    stop_counter += 1

            record = {"epoch": epoch, "train_loss": train_loss}
            for a in ATTRS:
                record[f"loss_{a}"] = train_by_attr[a]
            record["val_loss"] = val_loss
            for a in ATTRS:
                record[f"val_loss_{a}"] = val_by_attr[a]
            record["epoch_seconds"] = epoch_seconds
            record["is_best"] = is_best
            record["early_stop_counter"] = stop_counter
            history.append(record)

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(os.path.join(self.output_dir, "epoch_history.json"), "w") as f:
                    json.dump(history, f, indent=2)

            line = (f"epoch {epoch:4d}  train_loss {train_loss:.4f}  "
                    + "  ".join(f"{a} {train_by_attr[a]:.4f}" for a in ATTRS))
            if val_loss is not None:
                line += f"  val_loss {val_loss:.4f}  " + "  ".join(f"val_{a} {val_by_attr[a]:.4f}" for a in ATTRS)
            line += f"  time {epoch_seconds:.1f}s"
            if is_best:
                line += "  *best*"
            print(line)

            if val_dataloader is not None and stop_counter > self.patience - 1:
                break

        if self.best_epoch is None and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "bestmodel.pkl"))

        return history
