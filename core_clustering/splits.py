from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core_clustering.dataset import LoadedDataset


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    holdout_idx: np.ndarray
    train_groups: List[Tuple[str, int]]
    val_groups: List[Tuple[str, int]]
    included_domains: List[str]
    holdout_domains: List[str]
    group_counts: Dict[str, int]
    warnings: List[str]
    val_fraction_requested: float
    val_fraction_actual: float


def make_cross_domain_split(
    dataset: LoadedDataset,
    holdout_domains: Sequence[str],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> SplitResult:
    present_domains = sorted(set(dataset.domain.tolist()))
    unknown = [d for d in holdout_domains if d not in present_domains]
    if unknown:
        raise ValueError(f"unknown holdout domain(s): {unknown}; present domains are {present_domains}")

    included_domains = sorted(set(present_domains) - set(holdout_domains))
    warnings: List[str] = []
    train_groups: List[Tuple[str, int]] = []
    val_groups: List[Tuple[str, int]] = []
    group_counts: Dict[str, int] = {}

    for domain_rank, domain in enumerate(included_domains):
        domain_mask = dataset.domain == domain
        groups = sorted(set(dataset.base_instance_id[domain_mask].tolist()))
        group_counts[domain] = len(groups)
        rng = np.random.default_rng(seed + domain_rank)

        if len(groups) <= 1:
            train_groups.extend((domain, g) for g in groups)
            warnings.append(
                f"domain '{domain}' has only {len(groups)} base-instance group(s); "
                f"no val group allocated"
            )
            continue

        n_val = max(1, round(val_fraction * len(groups)))
        n_val = min(n_val, len(groups) - 1)
        shuffled = list(groups)
        rng.shuffle(shuffled)
        val_groups.extend((domain, g) for g in shuffled[:n_val])
        train_groups.extend((domain, g) for g in shuffled[n_val:])

    group_key = dataset.group_key()
    train_group_keys = {f"{d}::{g}" for d, g in train_groups}
    val_group_keys = {f"{d}::{g}" for d, g in val_groups}

    train_idx = np.where(np.isin(group_key, list(train_group_keys)))[0]
    val_idx = np.where(np.isin(group_key, list(val_group_keys)))[0]
    holdout_idx = np.where(np.isin(dataset.domain, list(holdout_domains)))[0]

    n_train_val = len(train_idx) + len(val_idx)
    val_fraction_actual = (len(val_idx) / n_train_val) if n_train_val > 0 else 0.0

    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        holdout_idx=holdout_idx,
        train_groups=train_groups,
        val_groups=val_groups,
        included_domains=included_domains,
        holdout_domains=sorted(holdout_domains),
        group_counts=group_counts,
        warnings=warnings,
        val_fraction_requested=val_fraction,
        val_fraction_actual=val_fraction_actual,
    )
