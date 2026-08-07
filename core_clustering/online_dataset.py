import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from core_clustering.dataset import LoadStats

try:
    from anomsim.anomalies.base import get_anomaly
    from anomsim.windowing import window_positions
except ImportError:
    # online_dataset.py reuses AnomSim's actual anomaly-injection classes
    # directly (not a vendored copy) so the two repos' injection logic can
    # never silently drift apart. Fall back to AnomSim as a sibling
    # directory (this project's own layout convention) before giving up.
    _sibling_anomsim = os.path.join(os.path.dirname(__file__), "..", "..", "AnomSim")
    if os.path.isdir(_sibling_anomsim):
        import sys

        sys.path.insert(0, _sibling_anomsim)
    try:
        from anomsim.anomalies.base import get_anomaly
        from anomsim.windowing import window_positions
    except ImportError as e:
        raise ImportError(
            "core_clustering.online_dataset requires the AnomSim package to be importable. "
            "Clone AnomSim as a sibling directory next to Core-Clustering, or add its "
            "repo root to PYTHONPATH."
        ) from e


@dataclass
class BasePool:
    Y: List[np.ndarray]
    domain: np.ndarray
    base_instance_id: np.ndarray
    base_seed: np.ndarray
    n_time: np.ndarray
    entity_dir: np.ndarray
    load_stats: LoadStats

    def group_key(self) -> np.ndarray:
        """Compound leakage-safe key, one per row: '<domain>::<base_instance_id>'.
        Duck-type identical to LoadedDataset.group_key(), so
        core_clustering.splits.make_cross_domain_split works unchanged."""
        return np.array([f"{d}::{b}" for d, b in zip(self.domain, self.base_instance_id)])


def load_base_pool(
    out_dir: str,
    manifest_name: str = "_manifest.jsonl",
    max_failure_details: int = 200,
    exclude_entity_dirs: List[str] = None,
) -> BasePool:
    """`exclude_entity_dirs`: entity_dir names (e.g. 'square_b3') to leave out
    of the loaded pool entirely -- e.g. a domain's fixed held-back test
    instances that must never be visible to any training run (see
    AnomSim's scripts/carve_experiment3_fixed_test_ids.py). Excluded entities
    are treated as intentionally not part of this load, not as failures:
    they're removed before n_attempted is computed, so LoadStats' own
    n_failed = n_attempted - n_loaded invariant still holds and they never
    show up in failures/failures_by_reason. Default None preserves prior
    behavior exactly (no filtering)."""
    manifest_path = os.path.join(out_dir, manifest_name)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    with open(manifest_path) as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    exclude_set = set(exclude_entity_dirs) if exclude_entity_dirs else set()
    n_excluded = 0

    Y_list, domain_list, base_instance_id_list = [], [], []
    base_seed_list, n_time_list, entity_dir_list = [], [], []
    failures: List[dict] = []
    failures_by_reason: dict = {}
    n_loaded = 0

    def record_failure(line_no, entity_dir_name, reason):
        failures_by_reason[reason] = failures_by_reason.get(reason, 0) + 1
        if len(failures) < max_failure_details:
            failures.append({"line_no": line_no, "entity_dir": entity_dir_name, "reason": reason})

    for line_no, raw_line in enumerate(raw_lines):
        try:
            meta = json.loads(raw_line)
        except json.JSONDecodeError:
            record_failure(line_no, None, "bad_json")
            continue

        try:
            _ = meta["type"]
            _ = meta["base_instance_id"]
            entity_dir_name = meta["entity_dir"]
        except KeyError:
            record_failure(line_no, meta.get("entity_dir"), "missing_manifest_field")
            continue

        if entity_dir_name in exclude_set:
            n_excluded += 1
            continue

        entity_dir = os.path.join(out_dir, entity_dir_name)
        try:
            Y = np.load(os.path.join(entity_dir, "Y.npy"))
        except FileNotFoundError:
            record_failure(line_no, entity_dir_name, "missing_file")
            continue
        except (ValueError, OSError, EOFError):
            record_failure(line_no, entity_dir_name, "corrupt_npy")
            continue

        Y_list.append(Y.astype(np.float64))
        domain_list.append(meta["type"])
        base_instance_id_list.append(meta["base_instance_id"])
        base_seed_list.append(meta.get("base_seed"))
        n_time_list.append(Y.shape[1])
        entity_dir_list.append(entity_dir_name)
        n_loaded += 1

    n_attempted = len(raw_lines) - n_excluded
    n_failed = n_attempted - n_loaded
    if n_loaded == 0:
        raise ValueError(f"Zero base instances loaded successfully from {manifest_path} ({n_failed} failed)")

    domains = sorted(set(domain_list))
    load_stats = LoadStats(
        manifest_path=manifest_path,
        n_manifest_lines=len(raw_lines),
        n_attempted=n_attempted,
        n_loaded=n_loaded,
        n_failed=n_failed,
        failures=failures,
        failures_by_reason=failures_by_reason,
        domains=domains,
        anomaly_types=[],  # not applicable: injection happens on the fly at training time
    )

    return BasePool(
        Y=Y_list,
        domain=np.array(domain_list),
        base_instance_id=np.array(base_instance_id_list),
        base_seed=np.array(base_seed_list),
        n_time=np.array(n_time_list),
        entity_dir=np.array(entity_dir_list),
        load_stats=load_stats,
    )


class OnlineWindowedDataset(Dataset):
    """Slides a window across each included base series and injects every
    anomaly type at every window position, exactly like AnomSim's own
    inject_windows() / RedLamp's Loader_aug. Nothing is precomputed eagerly
    (only the lightweight (row, window position, class) index tuples are
    built in __init__, not the actual injected arrays) -- but the injection
    itself IS fixed once, matching RedLamp's Loader_aug semantics exactly:
    Loader_aug injects every (entity, window, anomaly-type) combination
    once at construction and only reshuffles order across epochs, it never
    re-injects. This class reproduces that by seeding purely off
    (base_seed, row, window position, class) -- deliberately NOT epoch --
    so a given item is bitwise identical every time it's fetched, regardless
    of which epoch's pass over the DataLoader produced the fetch.

    set_epoch() is kept only so Trainer.train()'s unconditional
    hasattr(dataset, 'set_epoch') call keeps working; the stored epoch is no
    longer read anywhere, so calling it is a harmless no-op in practice."""

    def __init__(
        self,
        base_pool: BasePool,
        indices: np.ndarray,
        window_size: int,
        window_step: int,
        class_list: List[str],
        anomaly_params: dict = None,
        base_seed: int = 0,
    ):
        self.base_pool = base_pool
        self.window_size = window_size
        self.window_step = window_step
        self.class_list = list(class_list)
        self.anomaly_params = anomaly_params or {}
        self.base_seed = base_seed
        self.epoch = 0

        self.index = []
        for row_idx in indices:
            n_time = int(base_pool.n_time[row_idx])
            positions = window_positions(n_time, window_size, window_step)
            for window_idx, (start, end) in enumerate(positions):
                for type_idx in range(len(self.class_list)):
                    self.index.append((int(row_idx), window_idx, start, end, type_idx))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row_idx, window_idx, start, end, type_idx = self.index[idx]
        anomaly_type = self.class_list[type_idx]
        window = self.base_pool.Y[row_idx][:, start:end]

        rng = np.random.default_rng([self.base_seed, row_idx, window_idx, type_idx])
        params = self.anomaly_params.get(anomaly_type, {})
        y, _z, mask = get_anomaly(anomaly_type)(**params).apply(window, rng)

        one_hot = np.zeros(len(self.class_list), dtype=np.float32)
        one_hot[type_idx] = 1.0

        Y_t = torch.from_numpy(y).float().transpose(1, 0).contiguous()
        mask_t = torch.from_numpy(mask).float().transpose(1, 0).contiguous()
        label_t = torch.from_numpy(one_hot).float()
        return Y_t, mask_t, label_t


def materialize_windows(pool, idx_array, window_size, window_step, class_list, max_samples=None, seed=0):
    """One-off materialization of up to max_samples (Y, one_hot_label) pairs
    for the given pool rows -- used only for reporting (accuracy/embeddings/
    sample plots), never for training itself. Domain-agnostic: callers that
    want a single domain's rows (online_cli.py's own reporting loop) filter
    idx_array by pool.domain themselves before calling this; callers scoring
    an external model against a single AnomSim entity (no domain filtering
    needed, since a single-entity split is already domain-homogeneous) pass
    idx_array straight through. Returns None if idx_array (or the resulting
    windowed dataset) is empty."""
    if len(idx_array) == 0:
        return None
    online_ds = OnlineWindowedDataset(pool, idx_array, window_size, window_step, class_list)
    n = len(online_ds)
    if n == 0:
        return None
    sample_idx = np.arange(n)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n, size=max_samples, replace=False)

    Y_list, label_list = [], []
    for i in sample_idx:
        Y_t, _mask_t, label_t = online_ds[int(i)]
        Y_list.append(Y_t.numpy().T)  # (window_size, 1) -> (1, window_size)
        label_list.append(label_t.numpy())
    return np.stack(Y_list), np.stack(label_list), online_ds, sample_idx
