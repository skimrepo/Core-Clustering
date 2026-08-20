import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

NORMAL_SENTINEL = -1.0
SHAPE_LABELS = {"normal": 0, "shift": 1}


@dataclass
class ContrastiveRecord:
    Y: np.ndarray
    Z: np.ndarray
    shape_label: int
    # Still literal bucket indices (0.0/1.0/2.0...) under the hood for now --
    # the dataset generation side hasn't moved to genuinely continuous,
    # dynamically-injected values yet. Named/typed as floats already so the
    # loss side (regression against real gaps, log-scale for intensity) is
    # correct today and needs no further change once continuous values land.
    location_value: float
    extent_value: float
    intensity_value: float
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


def load_contrastive_pool(
    dataset_dir: str, split: Optional[str] = None
) -> Tuple[List[ContrastiveRecord], LoadStats]:
    """Read an AnomSim_v3-style directory, pulling out the shape label
    (normal vs anomaly type) and the stratified location/extent/intensity
    buckets each anomalous instance was generated with -- the ground truth
    the hierarchical contrastive loss is built from. Normal instances get
    NORMAL_SENTINEL for location/extent/intensity (no injected region to
    bucket)."""
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
            if row.get("is_anomalous"):
                anomaly = row["anomaly"]
                shape_label = SHAPE_LABELS[anomaly["type"]]
                strata = anomaly["strata"]
                loc = float(strata["location_bucket"])
                ext = float(strata["extent_bucket"])
                inten = float(strata["intensity_bucket"])
            else:
                shape_label = SHAPE_LABELS["normal"]
                loc = ext = inten = NORMAL_SENTINEL
            records.append(ContrastiveRecord(
                Y=Y, Z=Z, shape_label=shape_label,
                location_value=loc, extent_value=ext, intensity_value=inten,
                entity_dir=entity_dir, split=row.get("split"), n_time=row.get("n_time", Y.shape[1]),
            ))
        except (OSError, KeyError) as e:
            failures.append({"entity_dir": entity_dir, "reason": str(e)})

    stats = LoadStats(
        manifest_path=manifest_path, n_manifest_lines=len(lines), n_attempted=n_attempted,
        n_loaded=len(records), n_failed=len(failures), failures=failures,
    )
    return records, stats


class ContrastiveDataset(torch.utils.data.Dataset):
    """One item = one whole series, z-scored using its CLEAN (pre-injection)
    Z stats (same convention as WholeSeriesDataset), plus the attribute
    labels the hierarchical loss needs."""

    def __init__(self, records: List[ContrastiveRecord]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        clean_mean, clean_std = r.Z.mean(), r.Z.std()
        Y_norm = (r.Y - clean_mean) / (clean_std + 1e-8)
        return {
            "Y": torch.from_numpy(Y_norm).float(),
            "shape_label": int(r.shape_label),
            "location_value": float(r.location_value),
            "extent_value": float(r.extent_value),
            "intensity_value": float(r.intensity_value),
            "n_time": Y_norm.shape[1],
        }


def contrastive_pad_collate(batch, max_len=550):
    """Right-pads Y up to max_len (same convention as dataset_tcn.pad_collate)
    and stacks the per-instance scalar attribute labels -- those need no
    padding, one value per instance regardless of series length."""
    lengths = [item["n_time"] for item in batch]
    too_long = [n for n in lengths if n > max_len]
    if too_long:
        raise ValueError(f"sample length(s) {too_long} exceed max_len={max_len}")

    B, T = len(batch), max_len
    Y_padded = torch.zeros(B, 1, T)
    pad_mask = torch.zeros(B, 1, T)
    for i, item in enumerate(batch):
        n = item["n_time"]
        Y_padded[i, :, :n] = item["Y"]
        pad_mask[i, :, :n] = 1.0

    return {
        "Y": Y_padded,
        "pad_mask": pad_mask,
        "shape_label": torch.tensor([item["shape_label"] for item in batch], dtype=torch.long),
        "location_value": torch.tensor([item["location_value"] for item in batch], dtype=torch.float32),
        "extent_value": torch.tensor([item["extent_value"] for item in batch], dtype=torch.float32),
        "intensity_value": torch.tensor([item["intensity_value"] for item in batch], dtype=torch.float32),
        "lengths": torch.tensor(lengths),
    }


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Yields batches with a FIXED, equal count per class (here: normal vs
    anomalous shape_label) regardless of the underlying dataset's class
    proportions -- naive random batching would under-represent whichever
    class is rarer, skewing what the contrastive loss learns to prioritize.
    Number of batches per epoch is capped by the smallest class."""

    def __init__(self, labels, batch_size: int, seed: int = 0):
        self.labels = list(labels)
        self.batch_size = batch_size
        self.seed = seed
        self.classes = sorted(set(self.labels))
        self.indices_by_class = {
            c: [i for i, l in enumerate(self.labels) if l == c] for c in self.classes
        }
        self.per_class = batch_size // len(self.classes)
        self.n_batches = min(len(idxs) for idxs in self.indices_by_class.values()) // self.per_class

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        shuffled = {c: rng.permutation(idxs).tolist() for c, idxs in self.indices_by_class.items()}
        for b in range(self.n_batches):
            batch = []
            for c in self.classes:
                start = b * self.per_class
                batch.extend(shuffled[c][start:start + self.per_class])
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
