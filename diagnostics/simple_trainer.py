import json
import os
import time
from typing import Optional

import torch

from core_clustering.losses_contrastive import DEFAULT_WEIGHTS, MultiHeadContrastiveLoss

ATTRS = ("shape", "location", "extent", "intensity")


class SimpleTrainer:
    """Diagnostic-only trainer: ONE plain AdamW over model+loss params, a
    single combined backward() per batch -- no per-attribute optimizer
    split, no trunk-lr scaling (that lives in
    core_clustering.trainer_contrastive.ContrastiveTrainer and is
    deliberately NOT used here, per the MTL diagnostic's principle of
    starting from the simplest standard setup). This also sidesteps a real
    confound: multiple per-attribute AdamW optimizers each independently
    apply AdamW's decoupled weight decay to the SAME shared trunk params
    every step, so a naive weights=(1,0,0,0) "single-task" run on top of
    that trainer would still get 4x the weight-decay shrinkage a true
    single-task setup would.
    """

    def __init__(self, model, device: str = "cpu", lr: float = 0.001, max_grad_norm: float = 1.0,
                 patience: int = 10, weights=DEFAULT_WEIGHTS, output_dir: Optional[str] = None):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = MultiHeadContrastiveLoss(weights=weights).to(device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()), lr=lr
        )
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

            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                emb = self.model(Y, pad_mask=pad_mask)
                loss, comp = self.loss_fn(emb, shape, loc, ext, inten)
            if torch.isnan(loss):
                continue
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.loss_fn.parameters()), self.max_grad_norm
                )
                self.optimizer.step()

            total_loss += loss.item()
            for a in ATTRS:
                total_by_attr[a] += comp[f"loss_{a}"].item()
            step_count += 1

        if step_count == 0:
            return float("nan"), {a: float("nan") for a in ATTRS}
        avg_by_attr = {a: total_by_attr[a] / step_count for a in ATTRS}
        return total_loss / step_count, avg_by_attr

    def train(self, train_dataloader, val_dataloader=None, epochs: int = 100):
        history = []
        stop_counter = 0
        early_stop_reason = "completed all epochs"

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
            history.append(record)

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(os.path.join(self.output_dir, "epoch_history.json"), "w") as f:
                    json.dump(history, f, indent=2)

            if train_loss == train_loss and train_loss > 1e4:  # explicit divergence guard
                early_stop_reason = f"divergence: train_loss={train_loss:.2f} at epoch {epoch}"
                break
            if val_dataloader is not None and stop_counter > self.patience - 1:
                early_stop_reason = f"no val improvement for {self.patience} epochs (patience exhausted)"
                break

        if self.best_epoch is None and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "bestmodel.pkl"))

        return history, early_stop_reason
