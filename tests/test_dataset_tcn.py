import json
import os

import numpy as np
import pytest
import torch

from core_clustering.dataset_tcn import WholeSeriesDataset, load_whole_series_pool, pad_collate


def _write_instance(out_dir, name, n_time, split, is_anomalous, Y, Z, label):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "Y.npy"), Y)
    np.save(os.path.join(d, "Z.npy"), Z)
    np.save(os.path.join(d, "label.npy"), label)
    meta = {"n_time": n_time, "split": split, "is_anomalous": is_anomalous, "entity_dir": name}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _build_fixture_pool(out_dir, include_broken=False):
    manifest_rows = []
    n_time = 20
    Y = np.zeros((1, n_time))
    Z = Y.copy()
    label = np.zeros((1, n_time))
    manifest_rows.append(_write_instance(out_dir, "a_normal", n_time, "train", False, Y, Z, label))

    Y2 = np.zeros((1, n_time))
    Z2 = np.zeros((1, n_time))
    Y2[0, 5:10] += 4.0
    label2 = np.zeros((1, n_time))
    label2[0, 5:10] = 1.0
    manifest_rows.append(_write_instance(out_dir, "b_anom", n_time, "val", True, Y2, Z2, label2))

    if include_broken:
        manifest_rows.append({"n_time": 20, "split": "train", "is_anomalous": False, "entity_dir": "c_missing"})

    manifest_path = os.path.join(out_dir, "_manifest.jsonl")
    with open(manifest_path, "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    return manifest_rows


def test_load_whole_series_pool_reads_all_instances(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir)
    records, stats = load_whole_series_pool(out_dir)
    assert len(records) == 2
    assert stats.n_loaded == 2
    assert stats.n_failed == 0
    names = {r.entity_dir for r in records}
    assert names == {"a_normal", "b_anom"}


def test_load_whole_series_pool_filters_by_split(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir)
    records, stats = load_whole_series_pool(out_dir, split="train")
    assert len(records) == 1
    assert records[0].entity_dir == "a_normal"


def test_load_whole_series_pool_skips_missing_files_and_records_failure(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir, include_broken=True)
    records, stats = load_whole_series_pool(out_dir)
    assert len(records) == 2
    assert stats.n_failed == 1
    assert stats.failures[0]["entity_dir"] == "c_missing"


def test_dataset_normalizes_using_clean_z_stats_not_injected_y(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir)
    records, _ = load_whole_series_pool(out_dir)
    rec = next(r for r in records if r.entity_dir == "b_anom")
    item = WholeSeriesDataset([rec])[0]
    assert torch.isfinite(item["Y"]).all()
    assert item["Y"][0, 5] != 0.0


def test_dataset_is_anomaly_and_anomaly_mask_are_complementary(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir)
    records, _ = load_whole_series_pool(out_dir)
    rec = next(r for r in records if r.entity_dir == "b_anom")
    item = WholeSeriesDataset([rec])[0]
    np.testing.assert_array_equal(
        item["anomaly_mask"].numpy(), 1.0 - item["is_anomaly"].numpy()
    )
    assert item["is_anomaly"][0, 5:10].sum() == 5


def test_pad_collate_pads_and_builds_pad_mask():
    batch = [
        {"Y": torch.ones(1, 8), "is_anomaly": torch.zeros(1, 8),
         "anomaly_mask": torch.ones(1, 8), "n_time": 8},
        {"Y": torch.ones(1, 5), "is_anomaly": torch.zeros(1, 5),
         "anomaly_mask": torch.ones(1, 5), "n_time": 5},
    ]
    out = pad_collate(batch, max_len=10)
    assert out["Y"].shape == (2, 1, 10)
    assert torch.all(out["pad_mask"][0, :, :8] == 1.0)
    assert torch.all(out["pad_mask"][0, :, 8:] == 0.0)
    assert torch.all(out["pad_mask"][1, :, :5] == 1.0)
    assert torch.all(out["pad_mask"][1, :, 5:] == 0.0)
    assert torch.all(out["Y"][1, :, 5:] == 0.0)


def test_pad_collate_raises_when_sample_exceeds_max_len():
    batch = [{"Y": torch.ones(1, 20), "is_anomaly": torch.zeros(1, 20),
              "anomaly_mask": torch.ones(1, 20), "n_time": 20}]
    with pytest.raises(ValueError):
        pad_collate(batch, max_len=10)
