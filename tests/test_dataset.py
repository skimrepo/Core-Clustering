import json
import os

import numpy as np
import pytest

from core_clustering.dataset import load_windowed_dataset

WINDOW_SIZE = 20


def _write_entity(base_dir, name, meta, window_size=WINDOW_SIZE, corrupt=False, skip_Y=False):
    entity_dir = os.path.join(base_dir, name)
    os.makedirs(entity_dir, exist_ok=True)
    Y = np.random.default_rng(0).normal(size=(1, window_size)).astype(np.float32)
    labels = np.ones((1, window_size), dtype=np.float32)
    Z = Y.copy()
    if not skip_Y:
        if corrupt:
            with open(os.path.join(entity_dir, "Y.npy"), "wb") as f:
                f.write(b"not a valid npy file")
        else:
            np.save(os.path.join(entity_dir, "Y.npy"), Y)
    np.save(os.path.join(entity_dir, "labels.npy"), labels)
    np.save(os.path.join(entity_dir, "Z.npy"), Z)
    np.save(os.path.join(entity_dir, "mask.npy"), np.ones_like(Y))
    with open(os.path.join(entity_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(entity_dir, "_name.txt"), "w") as f:
        f.write(name)
    return entity_dir


def _build_small_dataset(tmp_path, domains=("sine", "trend"), n_base=2, atypes=("normal", "spike"), n_windows=2):
    out_dir = str(tmp_path / "dataset")
    os.makedirs(out_dir, exist_ok=True)
    manifest_lines = []
    for domain in domains:
        for base_id in range(n_base):
            for atype in atypes:
                for w in range(n_windows):
                    name = f"{domain}_b{base_id}_w{w}_{atype}"
                    meta = {
                        "waveform": {"type": domain, "params": {}},
                        "anomaly": {"type": atype, "params": {}},
                        "window": {"index": w, "start": w * 10, "end": w * 10 + WINDOW_SIZE, "size": WINDOW_SIZE, "step": 10},
                        "base_instance_id": base_id,
                        "base_seed": base_id,
                        "base_n_time": 100,
                        "seed": base_id,
                    }
                    _write_entity(out_dir, name, meta)
                    manifest_lines.append({**meta, "entity_dir": name})
    with open(os.path.join(out_dir, "_manifest.jsonl"), "w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")
    return out_dir, manifest_lines


def test_loads_all_well_formed_entities(tmp_path):
    out_dir, manifest_lines = _build_small_dataset(tmp_path)
    dataset = load_windowed_dataset(out_dir)
    assert dataset.load_stats.n_attempted == len(manifest_lines)
    assert dataset.load_stats.n_loaded == len(manifest_lines)
    assert dataset.load_stats.n_failed == 0
    assert dataset.Y.shape == (len(manifest_lines), 1, WINDOW_SIZE)
    assert dataset.window_size == WINDOW_SIZE


def test_missing_Y_npy_is_recorded_as_failure_not_crash(tmp_path):
    out_dir, manifest_lines = _build_small_dataset(tmp_path)
    bad_entity_dir = os.path.join(out_dir, manifest_lines[0]["entity_dir"])
    os.remove(os.path.join(bad_entity_dir, "Y.npy"))

    dataset = load_windowed_dataset(out_dir)
    assert dataset.load_stats.n_loaded == len(manifest_lines) - 1
    assert dataset.load_stats.n_failed == 1
    assert dataset.load_stats.failures_by_reason.get("missing_file") == 1


def test_corrupted_npy_is_recorded_as_failure(tmp_path):
    domains = ("sine", "trend")
    out_dir, manifest_lines = _build_small_dataset(tmp_path)
    corrupt_name = manifest_lines[0]["entity_dir"]
    corrupt_dir = os.path.join(out_dir, corrupt_name)
    with open(os.path.join(corrupt_dir, "Y.npy"), "wb") as f:
        f.write(b"garbage-not-npy")

    dataset = load_windowed_dataset(out_dir)
    assert dataset.load_stats.n_failed == 1
    assert dataset.load_stats.failures_by_reason.get("corrupt_npy") == 1


def test_window_size_mismatch_is_recorded_as_failure(tmp_path):
    out_dir, manifest_lines = _build_small_dataset(tmp_path)
    mismatched_name = manifest_lines[0]["entity_dir"]
    mismatched_dir = os.path.join(out_dir, mismatched_name)
    np.save(os.path.join(mismatched_dir, "Y.npy"), np.zeros((1, WINDOW_SIZE + 5), dtype=np.float32))

    dataset = load_windowed_dataset(out_dir)
    assert dataset.load_stats.n_failed == 1
    assert dataset.load_stats.failures_by_reason.get("window_size_mismatch") == 1


def test_class_list_and_domains_derived_not_hardcoded(tmp_path):
    out_dir, _ = _build_small_dataset(tmp_path, atypes=("normal", "spike"))
    dataset = load_windowed_dataset(out_dir)
    assert dataset.class_list == ["normal", "spike"]
    assert set(dataset.load_stats.domains) == {"sine", "trend"}


def test_group_key_is_compound_domain_and_base_instance_id(tmp_path):
    out_dir, _ = _build_small_dataset(tmp_path, domains=("sine", "trend"), n_base=2)
    dataset = load_windowed_dataset(out_dir)
    keys = dataset.group_key()
    sine_b0 = set(keys[(dataset.domain == "sine") & (dataset.base_instance_id == 0)])
    trend_b0 = set(keys[(dataset.domain == "trend") & (dataset.base_instance_id == 0)])
    assert sine_b0.isdisjoint(trend_b0)
    assert len(sine_b0) == 1
    assert len(trend_b0) == 1


def test_missing_manifest_raises_file_not_found(tmp_path):
    empty_dir = str(tmp_path / "empty")
    os.makedirs(empty_dir)
    with pytest.raises(FileNotFoundError):
        load_windowed_dataset(empty_dir)


def test_one_hot_labels_shape_and_values(tmp_path):
    out_dir, manifest_lines = _build_small_dataset(tmp_path, atypes=("normal", "spike"))
    dataset = load_windowed_dataset(out_dir)
    one_hot = dataset.one_hot_labels()
    assert one_hot.shape == (len(manifest_lines), 2)
    assert np.all(one_hot.sum(axis=1) == 1.0)
    normal_idx = dataset.class_list.index("normal")
    rows_marked_normal = np.where(dataset.anomaly_type == "normal")[0]
    assert np.all(one_hot[rows_marked_normal, normal_idx] == 1.0)


def test_class_list_override_pins_given_order(tmp_path):
    # Dataset only contains "normal"/"spike", but a cross-repo-compatible
    # training run needs the full RedLamp order (extra classes just end up
    # with zero training examples, which is fine -- the classifier head still
    # needs the right width and index-0-is-normal semantics).
    out_dir, manifest_lines = _build_small_dataset(tmp_path, atypes=("normal", "spike"))
    fixed_order = ["normal", "spike", "flip", "speedup", "noise", "cutoff",
                   "average", "scale", "wander", "contextual", "upsidedown", "mixture"]
    dataset = load_windowed_dataset(out_dir, class_list=fixed_order)
    assert dataset.class_list == fixed_order
    one_hot = dataset.one_hot_labels()
    assert one_hot.shape == (len(manifest_lines), 12)
    assert np.all(one_hot.sum(axis=1) == 1.0)


def test_class_list_override_raises_on_anomaly_type_not_in_list(tmp_path):
    out_dir, _ = _build_small_dataset(tmp_path, atypes=("normal", "spike"))
    with pytest.raises(ValueError, match="spike"):
        load_windowed_dataset(out_dir, class_list=["normal", "noise"])
