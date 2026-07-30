"""
Single-entity temporal train/val split for AnomSim base-pool entities,
matching RedLamp's own per-entity "Self" model convention instead of
splits.py's make_cross_domain_split() (a cross-entity GROUP split that needs
multiple entities per domain to produce a val split at all -- with only one
entity, `if len(groups) <= 1` puts the whole entity in train and leaves val
empty).

RedLamp's loaders/load.py (load_anomaly_archive/load_iops, validation=True)
splits ONE entity's own series temporally: train_length = int(Y.shape[1] *
0.9), first 90% = train, last 10% = val. This module reproduces that exactly
for one AnomSim entity, wrapping the result as a 2-row BasePool (row 0 =
train portion, row 1 = val portion, same domain/base_instance_id on both)
so every downstream piece of Core-Clustering (OnlineWindowedDataset,
Trainer, evaluate_classification, online_cli.py's CSV/run_summary.json
writing) works completely unchanged -- only the split CONSTRUCTION differs
from the pool path, nothing that consumes it needs to know the difference.
"""
import json
import os
from typing import List, Tuple

import numpy as np

from core_clustering.dataset import LoadStats
from core_clustering.online_dataset import BasePool
from core_clustering.splits import SplitResult


def list_entities(dataset_dir: str, manifest_name: str = "_manifest.jsonl") -> List[str]:
    """All entity_dir names in an AnomSim base-pool dataset, in manifest order."""
    manifest_path = os.path.join(dataset_dir, manifest_name)
    entities = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            entities.append(meta["entity_dir"])
    return entities


def load_single_entity_split(dataset_dir: str, entity_dir: str, val_fraction: float = 0.1) -> Tuple[BasePool, SplitResult]:
    entity_path = os.path.join(dataset_dir, entity_dir)
    with open(os.path.join(entity_path, "meta.json")) as f:
        meta = json.load(f)

    Y = np.load(os.path.join(entity_path, "Y.npy")).astype(np.float64)
    n_time = Y.shape[1]
    train_len = int(n_time * (1 - val_fraction))
    if train_len <= 0 or train_len >= n_time:
        raise ValueError(
            f"entity {entity_dir!r} (n_time={n_time}) too short for a temporal "
            f"90/10-style split at val_fraction={val_fraction}"
        )

    Y_train = Y[:, :train_len]
    Y_val = Y[:, train_len:]
    domain = meta["type"]
    base_instance_id = meta["base_instance_id"]

    load_stats = LoadStats(
        manifest_path=os.path.join(dataset_dir, "_manifest.jsonl"),
        n_manifest_lines=1,
        n_attempted=1,
        n_loaded=1,
        n_failed=0,
        failures=[],
        failures_by_reason={},
        domains=[domain],
        anomaly_types=[],  # not applicable: injection happens on the fly at training time
    )
    pool = BasePool(
        Y=[Y_train, Y_val],
        domain=np.array([domain, domain]),
        base_instance_id=np.array([base_instance_id, base_instance_id]),
        base_seed=np.array([meta.get("base_seed"), meta.get("base_seed")]),
        n_time=np.array([Y_train.shape[1], Y_val.shape[1]]),
        entity_dir=np.array([entity_dir, entity_dir]),
        load_stats=load_stats,
    )

    val_fraction_actual = Y_val.shape[1] / n_time
    split = SplitResult(
        train_idx=np.array([0]),
        val_idx=np.array([1]),
        holdout_idx=np.array([], dtype=int),
        train_groups=[(domain, base_instance_id)],
        val_groups=[(domain, base_instance_id)],
        included_domains=[domain],
        holdout_domains=[],
        group_counts={domain: 1},
        warnings=[
            f"single-entity split for {entity_dir!r}: temporal "
            f"{1 - val_fraction:.0%}/{val_fraction:.0%} split of its own timeline "
            f"(RedLamp's per-entity Self convention), not a cross-entity group split"
        ],
        val_fraction_requested=val_fraction,
        val_fraction_actual=val_fraction_actual,
    )
    return pool, split
