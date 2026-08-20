import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class WholeSeriesRecord:
    Y: np.ndarray
    Z: np.ndarray
    label: np.ndarray
    entity_dir: str
    split: str
    n_time: int


@dataclass
class LoadStats:
    manifest_path: str
    n_manifest_lines: int
    n_attempted: int
    n_loaded: int
    n_failed: int
    failures: List[Dict[str, object]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Loaded {self.n_loaded}/{self.n_attempted} instances "
            f"({self.n_failed} failed) from {self.manifest_path}"
        )


def load_whole_series_pool(
    dataset_dir: str, split: Optional[str] = None
) -> Tuple[List[WholeSeriesRecord], LoadStats]:
    """Read an AnomSim_v3-style directory: `_manifest.jsonl` + per-instance
    `Y.npy`/`Z.npy`/`label.npy`. Pure file-based -- no anomsim package
    import, mirroring dataset.py's legacy-path convention of consuming
    AnomSim's on-disk contract with zero runtime coupling. Per-instance
    failures (missing/corrupt files) are skipped and counted rather than
    crashing the whole load, matching dataset.py's LoadStats pattern."""
    manifest_path = os.path.join(dataset_dir, "_manifest.jsonl")
    with open(manifest_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    records = []
    failures: List[Dict[str, object]] = []
    n_attempted = 0
    for line in lines:
        row = json.loads(line)
        if split is not None and row.get("split") != split:
            continue
        n_attempted += 1
        entity_dir = row["entity_dir"]
        try:
            d = os.path.join(dataset_dir, entity_dir)
            Y = np.load(os.path.join(d, "Y.npy"))
            Z = np.load(os.path.join(d, "Z.npy"))
            label = np.load(os.path.join(d, "label.npy"))
            records.append(WholeSeriesRecord(
                Y=Y, Z=Z, label=label, entity_dir=entity_dir,
                split=row.get("split"), n_time=row.get("n_time", Y.shape[1]),
            ))
        except OSError as e:
            failures.append({"entity_dir": entity_dir, "reason": str(e)})

    stats = LoadStats(
        manifest_path=manifest_path,
        n_manifest_lines=len(lines),
        n_attempted=n_attempted,
        n_loaded=len(records),
        n_failed=len(failures),
        failures=failures,
    )
    return records, stats


class WholeSeriesDataset(torch.utils.data.Dataset):
    """One item = one whole series. Normalizes per-instance via z-score
    using the CLEAN (pre-injection) Z stats -- never Y's own stats, since
    an injected shift would otherwise inflate its own normalization
    denominator and hide itself. Structurally identical to RedLamp's
    loaders/dataset_whole_series.py::WholeSeriesDataset."""

    def __init__(self, records: List[WholeSeriesRecord]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        clean_mean = r.Z.mean()
        clean_std = r.Z.std()
        Y_norm = (r.Y - clean_mean) / (clean_std + 1e-8)
        is_anomaly = r.label.astype(np.float32)
        anomaly_mask = 1.0 - is_anomaly
        return {
            "Y": torch.from_numpy(Y_norm).float(),
            "is_anomaly": torch.from_numpy(is_anomaly).float(),
            "anomaly_mask": torch.from_numpy(anomaly_mask).float(),
            "n_time": Y_norm.shape[1],
        }


def pad_collate(batch, max_len=550):
    """Right-pads every series in the batch up to a FIXED max_len. Raises
    loudly if any sample exceeds max_len rather than silently truncating."""
    lengths = [item["n_time"] for item in batch]
    too_long = [n for n in lengths if n > max_len]
    if too_long:
        raise ValueError(
            f"sample length(s) {too_long} exceed max_len={max_len}"
        )

    B, T = len(batch), max_len
    Y_padded = torch.zeros(B, 1, T)
    is_anomaly_padded = torch.zeros(B, 1, T)
    anomaly_mask_padded = torch.ones(B, 1, T)
    pad_mask = torch.zeros(B, 1, T)

    for i, item in enumerate(batch):
        n = item["n_time"]
        Y_padded[i, :, :n] = item["Y"]
        is_anomaly_padded[i, :, :n] = item["is_anomaly"]
        anomaly_mask_padded[i, :, :n] = item["anomaly_mask"]
        pad_mask[i, :, :n] = 1.0

    return {
        "Y": Y_padded, "is_anomaly": is_anomaly_padded,
        "anomaly_mask": anomaly_mask_padded, "pad_mask": pad_mask,
        "lengths": torch.tensor(lengths),
    }
