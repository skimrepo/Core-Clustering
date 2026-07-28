import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class LoadStats:
    manifest_path: str
    n_manifest_lines: int
    n_attempted: int
    n_loaded: int
    n_failed: int
    failures: List[Dict[str, object]]
    failures_by_reason: Dict[str, int]
    domains: List[str]
    anomaly_types: List[str]

    def summary(self) -> str:
        return (
            f"Loaded {self.n_loaded}/{self.n_attempted} entities "
            f"({self.n_failed} failed) from {self.manifest_path} "
            f"— domains={self.domains} anomaly_types={self.anomaly_types}"
        )


@dataclass
class LoadedDataset:
    Y: np.ndarray
    labels: np.ndarray
    Z: np.ndarray
    domain: np.ndarray
    anomaly_type: np.ndarray
    base_instance_id: np.ndarray
    window_index: np.ndarray
    entity_dir: np.ndarray
    window_size: int
    class_list: List[str]
    load_stats: LoadStats

    def group_key(self) -> np.ndarray:
        """Compound leakage-safe key, one per row: '<domain>::<base_instance_id>'."""
        return np.array([f"{d}::{b}" for d, b in zip(self.domain, self.base_instance_id)])

    def one_hot_labels(self) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.class_list)}
        out = np.zeros((len(self.anomaly_type), len(self.class_list)), dtype=np.float32)
        out[np.arange(len(self.anomaly_type)), [idx[a] for a in self.anomaly_type]] = 1.0
        return out


def _required_fields_present(meta: dict) -> bool:
    try:
        _ = meta["waveform"]["type"]
        _ = meta["anomaly"]["type"]
        _ = meta["base_instance_id"]
        _ = meta["window"]["index"]
        _ = meta["entity_dir"]
    except (KeyError, TypeError):
        return False
    return True


def load_windowed_dataset(
    out_dir: str,
    manifest_name: str = "_manifest.jsonl",
    dtype=np.float32,
    max_failure_details: int = 200,
) -> LoadedDataset:
    manifest_path = os.path.join(out_dir, manifest_name)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    with open(manifest_path) as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    # Determine the expected window size as the statistical mode across all
    # manifest lines, so one corrupt/mismatched entity can't redefine the
    # whole dataset's shape.
    window_sizes = []
    parsed_lines = []
    for line_no, raw_line in enumerate(raw_lines):
        try:
            meta = json.loads(raw_line)
        except json.JSONDecodeError:
            parsed_lines.append((line_no, None))
            continue
        parsed_lines.append((line_no, meta))
        if _required_fields_present(meta):
            size = meta.get("window", {}).get("size")
            if isinstance(size, int):
                window_sizes.append(size)
    expected_window_size = Counter(window_sizes).most_common(1)[0][0] if window_sizes else None

    Y_list, labels_list, Z_list = [], [], []
    domain_list, anomaly_type_list, base_instance_id_list = [], [], []
    window_index_list, entity_dir_list = [], []
    failures: List[Dict[str, object]] = []
    failures_by_reason: Dict[str, int] = {}
    n_loaded = 0

    def record_failure(line_no, entity_dir_name, reason):
        failures_by_reason[reason] = failures_by_reason.get(reason, 0) + 1
        if len(failures) < max_failure_details:
            failures.append({"line_no": line_no, "entity_dir": entity_dir_name, "reason": reason})

    for line_no, meta in parsed_lines:
        if meta is None:
            record_failure(line_no, None, "bad_json")
            continue
        if not _required_fields_present(meta):
            record_failure(line_no, meta.get("entity_dir"), "missing_manifest_field")
            continue

        entity_dir_name = meta["entity_dir"]
        entity_dir = os.path.join(out_dir, entity_dir_name)

        try:
            Y = np.load(os.path.join(entity_dir, "Y.npy"))
            labels = np.load(os.path.join(entity_dir, "labels.npy"))
            Z = np.load(os.path.join(entity_dir, "Z.npy"))
        except FileNotFoundError:
            record_failure(line_no, entity_dir_name, "missing_file")
            continue
        except (ValueError, OSError, EOFError):
            record_failure(line_no, entity_dir_name, "corrupt_npy")
            continue

        if expected_window_size is not None and Y.shape != (1, expected_window_size):
            record_failure(line_no, entity_dir_name, "window_size_mismatch")
            continue

        Y_list.append(Y.astype(dtype))
        labels_list.append(labels.astype(dtype))
        Z_list.append(Z.astype(dtype))
        domain_list.append(meta["waveform"]["type"])
        anomaly_type_list.append(meta["anomaly"]["type"])
        base_instance_id_list.append(meta["base_instance_id"])
        window_index_list.append(meta["window"]["index"])
        entity_dir_list.append(entity_dir_name)
        n_loaded += 1

    n_attempted = len(raw_lines)
    n_failed = n_attempted - n_loaded
    if n_loaded == 0:
        raise ValueError(f"Zero entities loaded successfully from {manifest_path} ({n_failed} failed)")

    domains = sorted(set(domain_list))
    anomaly_types = sorted(set(anomaly_type_list))

    load_stats = LoadStats(
        manifest_path=manifest_path,
        n_manifest_lines=len(raw_lines),
        n_attempted=n_attempted,
        n_loaded=n_loaded,
        n_failed=n_failed,
        failures=failures,
        failures_by_reason=failures_by_reason,
        domains=domains,
        anomaly_types=anomaly_types,
    )

    return LoadedDataset(
        Y=np.stack(Y_list),
        labels=np.stack(labels_list),
        Z=np.stack(Z_list),
        domain=np.array(domain_list),
        anomaly_type=np.array(anomaly_type_list),
        base_instance_id=np.array(base_instance_id_list),
        window_index=np.array(window_index_list),
        entity_dir=np.array(entity_dir_list),
        window_size=expected_window_size,
        class_list=anomaly_types,
        load_stats=load_stats,
    )
