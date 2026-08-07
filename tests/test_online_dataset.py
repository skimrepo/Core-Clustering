import json
import os

import numpy as np
import pytest

from core_clustering.online_dataset import BasePool, OnlineWindowedDataset, load_base_pool
from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
from core_clustering.splits import make_cross_domain_split


def _write_base_instance(base_dir, name, meta, n_time=150, seed=0):
    entity_dir = os.path.join(base_dir, name)
    os.makedirs(entity_dir, exist_ok=True)
    Y = np.random.default_rng(seed).normal(size=(1, n_time)).astype(np.float64)
    np.save(os.path.join(entity_dir, "Y.npy"), Y)
    with open(os.path.join(entity_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return entity_dir


def _build_small_base_pool(tmp_path, domains=("sine", "trend"), n_base=2, n_time=150):
    out_dir = str(tmp_path / "base_pool")
    os.makedirs(out_dir, exist_ok=True)
    manifest_lines = []
    for domain in domains:
        for base_id in range(n_base):
            name = f"{domain}_b{base_id}"
            meta = {"type": domain, "params": {}, "n_time": n_time,
                     "base_instance_id": base_id, "base_seed": base_id}
            _write_base_instance(out_dir, name, meta, n_time=n_time, seed=base_id)
            manifest_lines.append({**meta, "entity_dir": name})
    with open(os.path.join(out_dir, "_manifest.jsonl"), "w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")
    return out_dir


def test_load_base_pool_reads_all_well_formed_entities(tmp_path):
    out_dir = _build_small_base_pool(tmp_path)
    pool = load_base_pool(out_dir)
    assert pool.load_stats.n_loaded == 4
    assert pool.load_stats.n_failed == 0
    assert len(pool.Y) == 4
    assert set(pool.domain.tolist()) == {"sine", "trend"}


def test_load_base_pool_excludes_specified_entity_dirs(tmp_path):
    out_dir = _build_small_base_pool(tmp_path)
    pool = load_base_pool(out_dir, exclude_entity_dirs=["sine_b0", "trend_b1"])
    assert pool.load_stats.n_loaded == 2
    assert pool.load_stats.n_failed == 0
    assert pool.load_stats.n_attempted == 2
    assert set(pool.entity_dir.tolist()) == {"sine_b1", "trend_b0"}


def test_load_base_pool_exclude_entity_dirs_default_is_unchanged_behavior(tmp_path):
    out_dir = _build_small_base_pool(tmp_path)
    pool_default = load_base_pool(out_dir)
    pool_explicit_none = load_base_pool(out_dir, exclude_entity_dirs=None)
    assert pool_default.load_stats.n_loaded == pool_explicit_none.load_stats.n_loaded == 4
    assert sorted(pool_default.entity_dir.tolist()) == sorted(pool_explicit_none.entity_dir.tolist())


def test_load_base_pool_missing_file_recorded_as_failure(tmp_path):
    out_dir = _build_small_base_pool(tmp_path)
    with open(os.path.join(out_dir, "_manifest.jsonl")) as f:
        first = json.loads(f.readline())
    os.remove(os.path.join(out_dir, first["entity_dir"], "Y.npy"))

    pool = load_base_pool(out_dir)
    assert pool.load_stats.n_loaded == 3
    assert pool.load_stats.n_failed == 1
    assert pool.load_stats.failures_by_reason.get("missing_file") == 1


def test_load_base_pool_corrupt_npy_recorded_as_failure(tmp_path):
    out_dir = _build_small_base_pool(tmp_path)
    with open(os.path.join(out_dir, "_manifest.jsonl")) as f:
        first = json.loads(f.readline())
    with open(os.path.join(out_dir, first["entity_dir"], "Y.npy"), "wb") as f:
        f.write(b"not a valid npy file")

    pool = load_base_pool(out_dir)
    assert pool.load_stats.n_failed == 1
    assert pool.load_stats.failures_by_reason.get("corrupt_npy") == 1


def test_load_base_pool_missing_manifest_raises(tmp_path):
    empty_dir = str(tmp_path / "empty")
    os.makedirs(empty_dir)
    with pytest.raises(FileNotFoundError):
        load_base_pool(empty_dir)


def test_base_pool_group_key_matches_domain_and_base_instance_id(tmp_path):
    out_dir = _build_small_base_pool(tmp_path, domains=("sine", "trend"), n_base=2)
    pool = load_base_pool(out_dir)
    keys = pool.group_key()
    assert len(set(keys)) == 4  # every row is its own group (one base instance per row)


def test_base_pool_is_compatible_with_make_cross_domain_split(tmp_path):
    # One row per base instance now (not per window), but domain/group_key duck-type
    # identically to LoadedDataset, so the existing split logic works unchanged.
    out_dir = _build_small_base_pool(tmp_path, domains=("a", "b"), n_base=5)
    pool = load_base_pool(out_dir)
    result = make_cross_domain_split(pool, holdout_domains=["b"], val_fraction=0.2, seed=0)
    assert set(pool.domain[result.train_idx]) == {"a"}
    assert set(pool.domain[result.holdout_idx]) == {"b"}


WINDOW_SIZE = 100
WINDOW_STEP = 10


def _make_dataset(tmp_path, n_time=150, class_list=None):
    out_dir = _build_small_base_pool(tmp_path, domains=("sine",), n_base=1, n_time=n_time)
    pool = load_base_pool(out_dir)
    class_list = class_list or ["normal", "spike"]
    ds = OnlineWindowedDataset(
        pool, indices=np.array([0]), window_size=WINDOW_SIZE, window_step=WINDOW_STEP,
        class_list=class_list, base_seed=0,
    )
    return pool, ds


def test_online_dataset_length_matches_positions_times_classes(tmp_path):
    pool, ds = _make_dataset(tmp_path, n_time=150)
    # (150-100)//10 + 1 = 6 window positions x 2 classes
    assert len(ds) == 6 * 2


def test_online_dataset_item_shapes(tmp_path):
    pool, ds = _make_dataset(tmp_path)
    Y, mask, label = ds[0]
    assert Y.shape == (WINDOW_SIZE, 1)
    assert mask.shape == (WINDOW_SIZE, 1)
    assert label.shape == (2,)
    assert label.sum().item() == 1.0


def test_online_dataset_same_epoch_is_deterministic(tmp_path):
    pool, ds = _make_dataset(tmp_path, class_list=["normal", "spike"])
    ds.set_epoch(0)
    Y1, mask1, label1 = ds[7]
    Y2, mask2, label2 = ds[7]
    np.testing.assert_array_equal(Y1.numpy(), Y2.numpy())
    np.testing.assert_array_equal(mask1.numpy(), mask2.numpy())


def test_online_dataset_injection_is_fixed_across_epochs(tmp_path):
    # Matches RedLamp's Loader_aug semantics: anomalies are injected once and
    # reused every epoch (only iteration order is reshuffled by the
    # DataLoader), not re-injected fresh each epoch. set_epoch() is kept for
    # backward-compatible hasattr() calls in Trainer.train() but must no
    # longer change what __getitem__ returns for a given index.
    pool, ds = _make_dataset(tmp_path, class_list=["normal", "spike"])
    # index space is (window_idx, type_idx) with type_idx minor, so index 1 is
    # row 0, window 0, type_idx=1 ("spike").
    spike_idx = 1
    ds.set_epoch(0)
    Y_epoch0, _, label0 = ds[spike_idx]
    ds.set_epoch(1)
    Y_epoch1, _, label1 = ds[spike_idx]
    assert label0.argmax().item() == label1.argmax().item() == 1  # both "spike"
    assert np.array_equal(Y_epoch0.numpy(), Y_epoch1.numpy())


def test_online_dataset_normal_class_leaves_window_unmodified(tmp_path):
    pool, ds = _make_dataset(tmp_path, class_list=["normal", "spike"])
    Y_t, mask_t, label_t = ds[0]  # row 0, window 0, type_idx=0 -> "normal"
    assert label_t.argmax().item() == 0
    Y_base = pool.Y[0]
    expected = Y_base[:, 0:WINDOW_SIZE].T
    np.testing.assert_allclose(Y_t.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert np.all(mask_t.numpy() == 1.0)


def test_online_dataset_respects_redlamp_class_list_order(tmp_path):
    pool, ds = _make_dataset(tmp_path, class_list=REDLAMP_ANOMALY_TYPES)
    assert len(ds) == 6 * len(REDLAMP_ANOMALY_TYPES)
    Y_t, mask_t, label_t = ds[0]
    assert label_t.shape == (12,)
    assert label_t.argmax().item() == 0  # "normal" is class 0
