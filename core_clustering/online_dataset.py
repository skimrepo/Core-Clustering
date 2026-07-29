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
) -> BasePool:
    manifest_path = os.path.join(out_dir, manifest_name)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    with open(manifest_path) as f:
        raw_lines = [line.strip() for line in f if line.strip()]

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

    n_attempted = len(raw_lines)
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
    inject_windows() / RedLamp's Loader_aug -- except nothing is
    precomputed: every __getitem__ call injects fresh, using a
    (base_seed, epoch, row, window position, class) seed. Call set_epoch()
    before each training epoch so injections vary epoch to epoch (matching
    RedLamp's own on-the-fly augmentation) while staying fully reproducible
    for a given (base_seed, epoch) pair (unlike RedLamp's Loader_aug, which
    uses unseeded global randomness)."""

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

        rng = np.random.default_rng([self.base_seed, self.epoch, row_idx, window_idx, type_idx])
        params = self.anomaly_params.get(anomaly_type, {})
        y, _z, mask = get_anomaly(anomaly_type)(**params).apply(window, rng)

        one_hot = np.zeros(len(self.class_list), dtype=np.float32)
        one_hot[type_idx] = 1.0

        Y_t = torch.from_numpy(y).float().transpose(1, 0).contiguous()
        mask_t = torch.from_numpy(mask).float().transpose(1, 0).contiguous()
        label_t = torch.from_numpy(one_hot).float()
        return Y_t, mask_t, label_t
