import json
import os
import time
from dataclasses import asdict
from typing import List, Optional

import torch

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.trainer import EpochRecord


def default_tcn_hyperparameters(n_features: int, n_time_max: int, **overrides) -> ConvBottleneckConfig:
    return ConvBottleneckConfig(n_features=n_features, n_time_max=n_time_max, **overrides)


class TCNTrainer:
    """Mirrors Trainer's checkpointing/early-stop/history-JSON conventions
    (core_clustering.trainer.Trainer) exactly -- reuses EpochRecord from
    there rather than duplicating it. Only batch handling (dict batches
    from pad_collate) and the calculate_loss call differ.

    Unlike Trainer, batches are never skipped for batch_size==1: GroupNorm
    (ConvBottleneckAEC's default normalization) computes well-defined per-sample
    statistics even for a lone sample, unlike BatchNorm1d which requires
    batch size > 1.
    """

    def __init__(self, model, device: str = "cpu", lr: float = 0.001, max_grad_norm: float = 1.0,
                 patience: int = 10, output_dir: Optional[str] = None):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.output_dir = output_dir
        self.best_val_loss = float("inf")
        self.best_epoch: Optional[int] = None

    def _run_epoch(self, dataloader, train: bool):
        self.model.train(train)
        total_loss = total_loss_ae = total_loss_c = 0.0
        step_count = 0
        for batch in dataloader:
            Y = batch["Y"].to(self.device)
            is_anomaly = batch["is_anomaly"].to(self.device)
            anomaly_mask = batch["anomaly_mask"].to(self.device)
            pad_mask = batch["pad_mask"].to(self.device)

            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                recon, anomaly_logits, feat = self.model(Y, pad_mask=pad_mask)
                loss, loss_ae, loss_c = self.model.calculate_loss(
                    Y, recon, anomaly_logits, is_anomaly, anomaly_mask, pad_mask
                )

            if torch.isnan(loss):
                continue
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            total_loss_ae += loss_ae.item()
            total_loss_c += loss_c.item()
            step_count += 1

        if step_count == 0:
            return float("nan"), float("nan"), float("nan")
        return total_loss / step_count, total_loss_ae / step_count, total_loss_c / step_count

    def train(self, train_dataloader, val_dataloader=None, epochs: int = 100) -> List[EpochRecord]:
        history: List[EpochRecord] = []
        stop_counter = 0

        for epoch in range(epochs):
            start = time.time()
            train_loss, train_loss_ae, train_loss_c = self._run_epoch(train_dataloader, train=True)
            epoch_seconds = time.time() - start

            val_loss = val_loss_ae = val_loss_c = None
            is_best = False
            if val_dataloader is not None:
                val_loss, val_loss_ae, val_loss_c = self._run_epoch(val_dataloader, train=False)
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

            record = EpochRecord(
                epoch=epoch, train_loss=train_loss, train_loss_ae=train_loss_ae, train_loss_c=train_loss_c,
                val_loss=val_loss, val_loss_ae=val_loss_ae, val_loss_c=val_loss_c,
                epoch_seconds=epoch_seconds, is_best=is_best, early_stop_counter=stop_counter,
            )
            history.append(record)

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(os.path.join(self.output_dir, "epoch_history.json"), "w") as f:
                    json.dump([asdict(r) for r in history], f, indent=2)

            line = (
                f"epoch {epoch:4d}  train_loss {train_loss:.4f}  "
                f"train_loss_ae {train_loss_ae:.4f}  train_loss_c {train_loss_c:.4f}"
            )
            if val_loss is not None:
                line += (
                    f"  val_loss {val_loss:.4f}  "
                    f"val_loss_ae {val_loss_ae:.4f}  val_loss_c {val_loss_c:.4f}"
                )
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
