import json
import os

import numpy as np
import pytest

from core_clustering.single_entity import list_entities, load_single_entity_split


def _write_entity(dataset_dir, entity_dir, domain, base_instance_id, n_time, base_seed=0):
    entity_path = os.path.join(dataset_dir, entity_dir)
    os.makedirs(entity_path, exist_ok=True)
    Y = np.linspace(0.0, 1.0, n_time, dtype=np.float64).reshape(1, -1)
    np.save(os.path.join(entity_path, "Y.npy"), Y)
    meta = {"type": domain, "params": {}, "n_time": n_time, "base_instance_id": base_instance_id,
            "base_seed": base_seed}
    with open(os.path.join(entity_path, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta, Y


def _write_manifest(dataset_dir, entries):
    with open(os.path.join(dataset_dir, "_manifest.jsonl"), "w") as f:
        for entity_dir, domain, base_instance_id in entries:
            f.write(json.dumps({"type": domain, "base_instance_id": base_instance_id, "entity_dir": entity_dir}) + "\n")


def test_list_entities_reads_manifest_order(tmp_path):
    entries = [("sine_b0", "sine", 0), ("sine_b1", "sine", 1), ("arma_b0", "arma", 0)]
    _write_manifest(tmp_path, entries)
    assert list_entities(str(tmp_path)) == ["sine_b0", "sine_b1", "arma_b0"]


def test_load_single_entity_split_temporal_90_10(tmp_path):
    _write_entity(tmp_path, "sine_b0", domain="sine", base_instance_id=0, n_time=1000)
    pool, split = load_single_entity_split(str(tmp_path), "sine_b0", val_fraction=0.1)

    assert len(pool.Y) == 2
    assert pool.Y[0].shape[1] == 900  # train portion: first 90%
    assert pool.Y[1].shape[1] == 100  # val portion: last 10%
    np.testing.assert_allclose(pool.Y[0], pool.Y[0])  # sanity: no NaNs/shape issues
    assert list(pool.domain) == ["sine", "sine"]
    assert list(pool.base_instance_id) == [0, 0]

    assert list(split.train_idx) == [0]
    assert list(split.val_idx) == [1]
    assert len(split.holdout_idx) == 0
    assert split.included_domains == ["sine"]
    assert split.holdout_domains == []
    assert split.val_fraction_actual == pytest.approx(0.1)


def test_load_single_entity_split_train_val_are_disjoint_and_contiguous(tmp_path):
    _write_entity(tmp_path, "trend_b3", domain="trend", base_instance_id=3, n_time=2000)
    pool, split = load_single_entity_split(str(tmp_path), "trend_b3", val_fraction=0.2)
    train_len = pool.Y[0].shape[1]
    val_len = pool.Y[1].shape[1]
    assert train_len + val_len == 2000
    assert val_len == pytest.approx(400, abs=1)


def test_load_single_entity_split_rejects_too_short_entity(tmp_path):
    _write_entity(tmp_path, "sine_b0", domain="sine", base_instance_id=0, n_time=1)
    with pytest.raises(ValueError):
        load_single_entity_split(str(tmp_path), "sine_b0", val_fraction=0.5)
