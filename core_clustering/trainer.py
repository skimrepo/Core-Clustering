import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset

from core_clustering.dataset import LoadedDataset
from core_clustering.models import ModelConfig


def default_model_hyperparameters(n_features: int, n_time: int, classes: int, **overrides) -> ModelConfig:
    return ModelConfig(n_features=n_features, n_time=n_time, classes=classes, **overrides)


def make_torch_dataset(dataset: LoadedDataset, indices: np.ndarray) -> TensorDataset:
    """Y/labels come out of LoadedDataset as (n, n_features, window_size);
    ConvEncoder expects (batch, window, n_features), matching RedLamp's own
    `batch['Y'].transpose(2, 1)` convention."""
    Y = torch.from_numpy(dataset.Y[indices]).float().transpose(2, 1).contiguous()
    mask = torch.from_numpy(dataset.labels[indices]).float().transpose(2, 1).contiguous()
    one_hot = torch.from_numpy(dataset.one_hot_labels()[indices]).float()
    return TensorDataset(Y, mask, one_hot)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_loss_ae: float
    train_loss_c: float
    val_loss: Optional[float]
    val_loss_ae: Optional[float]
    val_loss_c: Optional[float]
    epoch_seconds: float
    is_best: bool
    early_stop_counter: int


class Trainer:
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
        for Y, mask, label in dataloader:
            if Y.shape[0] == 1:
                continue  # BatchNorm requires batch size > 1
            Y = Y.to(self.device)
            mask = mask.to(self.device)
            label = label.to(self.device)

            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                x_hat, x_out, x_enc = self.model(Y)
                loss, loss_ae, loss_c = self.model.calculate_loss(Y, x_hat, label, x_out, mask, epoch=0)

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
            train_dataset = getattr(train_dataloader, "dataset", None)
            if hasattr(train_dataset, "set_epoch"):
                # OnlineWindowedDataset injects fresh per epoch (matching
                # RedLamp's own on-the-fly augmentation); TensorDataset (the
                # pre-baked path) has no such method, so this is a no-op there.
                train_dataset.set_epoch(epoch)

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

            line = (
                f"epoch {epoch:4d}  train_loss {train_loss:.4f}  "
                f"train_loss_ae {train_loss_ae:.4f}  train_loss_c {train_loss_c:.4f}"
            )
            if val_loss is not None:
                line += f"  val_loss {val_loss:.4f}"
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


def write_run_summary(
    path: str,
    *,
    run_id: str,
    dataset_dir: str,
    seed: int,
    device: str,
    included_domains: List[str],
    held_out_domains: List[str],
    val_fraction_requested: float,
    val_fraction_actual: float,
    n_entities_attempted: int,
    n_entities_loaded: int,
    n_entities_failed: int,
    domain_window_counts: List[dict],
    epochs: List[EpochRecord],
    epochs_requested: int,
    early_stop_patience: int,
    model_hyperparameters: dict,
    held_out_accuracy: List[dict],
) -> None:
    n_windows_train = sum(row.get("n_windows_train") or 0 for row in domain_window_counts)
    n_windows_val = sum(row.get("n_windows_val") or 0 for row in domain_window_counts)

    # is_best flags every epoch that improved on the running best at the time,
    # so the LAST such epoch is the true best (monotonically improving), not
    # the first.
    best_flagged = [e for e in epochs if e.is_best]
    if best_flagged:
        best_epoch_record = best_flagged[-1]
    elif epochs:
        with_val = [e for e in epochs if e.val_loss is not None]
        best_epoch_record = min(with_val, key=lambda e: e.val_loss) if with_val else epochs[-1]
    else:
        best_epoch_record = None

    epochs_ran = len(epochs)
    early_stopped = bool(epochs) and epochs_ran < epochs_requested
    total_train_seconds = sum(e.epoch_seconds for e in epochs)

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_dir": dataset_dir,
        "seed": seed,
        "device": device,
        "included_domains": included_domains,
        "held_out_domains": held_out_domains,
        "val_fraction_requested": val_fraction_requested,
        "val_fraction_actual": val_fraction_actual,
        "n_entities_attempted": n_entities_attempted,
        "n_entities_loaded": n_entities_loaded,
        "n_entities_failed": n_entities_failed,
        "n_windows_train": n_windows_train,
        "n_windows_val": n_windows_val,
        "n_windows_total": n_windows_train + n_windows_val,
        "domain_window_counts": domain_window_counts,
        "epochs_requested": epochs_requested,
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "early_stop_epoch": epochs[-1].epoch if early_stopped and epochs else None,
        "best_epoch": best_epoch_record.epoch if best_epoch_record else None,
        "best_val_loss": best_epoch_record.val_loss if best_epoch_record else None,
        "best_val_loss_ae": best_epoch_record.val_loss_ae if best_epoch_record else None,
        "best_val_loss_c": best_epoch_record.val_loss_c if best_epoch_record else None,
        "total_train_seconds": total_train_seconds,
        "mean_epoch_seconds": (total_train_seconds / epochs_ran) if epochs_ran else 0.0,
        "model_hyperparameters": model_hyperparameters,
        "held_out_accuracy": held_out_accuracy,
        "epochs": [asdict(e) for e in epochs],
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
