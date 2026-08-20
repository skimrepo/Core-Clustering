import json
import os

import numpy as np
import torch

from core_clustering.dataset_contrastive import (
    NORMAL_SENTINEL,
    BalancedBatchSampler,
    ContrastiveDataset,
    contrastive_pad_collate,
    load_contrastive_pool,
)


def _write_instance(out_dir, name, n_time, split, is_anomalous, Y, Z, anomaly_meta=None):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "Y.npy"), Y)
    np.save(os.path.join(d, "Z.npy"), Z)
    np.save(os.path.join(d, "label.npy"), np.zeros((1, n_time)))
    meta = {"n_time": n_time, "split": split, "is_anomalous": is_anomalous, "entity_dir": name}
    if anomaly_meta is not None:
        meta["anomaly"] = anomaly_meta
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _build_fixture_pool(out_dir):
    rows = []
    n_time = 20
    Z = np.zeros((1, n_time))
    rows.append(_write_instance(out_dir, "a_normal", n_time, "train", False, Z.copy(), Z.copy()))

    Y2 = np.zeros((1, n_time)); Y2[0, 5:10] += 4.0
    rows.append(_write_instance(
        out_dir, "b_shift", n_time, "train", True, Y2, Z.copy(),
        anomaly_meta={
            "type": "shift", "region": {"start": 5, "end": 10}, "offset": 4.0, "clean_std": 1.0,
            "strata": {"location_bucket": 1, "extent_bucket": 0, "intensity_bucket": 2},
        },
    ))
    with open(os.path.join(out_dir, "_manifest.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_contrastive_pool_reads_shape_and_strata(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir)
    records, stats = load_contrastive_pool(out_dir)
    assert len(records) == 2
    normal_rec = next(r for r in records if r.entity_dir == "a_normal")
    shift_rec = next(r for r in records if r.entity_dir == "b_shift")

    assert normal_rec.shape_label == 0
    assert normal_rec.location_value == NORMAL_SENTINEL
    assert normal_rec.extent_value == NORMAL_SENTINEL
    assert normal_rec.intensity_value == NORMAL_SENTINEL

    assert shift_rec.shape_label == 1
    assert shift_rec.location_value == 1.0
    assert shift_rec.extent_value == 0.0
    assert shift_rec.intensity_value == 2.0


def test_contrastive_dataset_returns_expected_fields(tmp_path):
    out_dir = str(tmp_path / "pool")
    _build_fixture_pool(out_dir)
    records, _ = load_contrastive_pool(out_dir)
    ds = ContrastiveDataset(records)
    item = ds[0]
    assert item["Y"].shape == (1, 20)
    assert isinstance(item["shape_label"], int)
    assert isinstance(item["n_time"], int)
    assert isinstance(item["location_value"], float)


def test_contrastive_pad_collate_pads_Y_and_stacks_scalars():
    batch = [
        {"Y": torch.ones(1, 8), "shape_label": 0, "location_value": -1.0,
         "extent_value": -1.0, "intensity_value": -1.0, "n_time": 8},
        {"Y": torch.ones(1, 5), "shape_label": 1, "location_value": 2.0,
         "extent_value": 1.0, "intensity_value": 0.0, "n_time": 5},
    ]
    out = contrastive_pad_collate(batch, max_len=10)
    assert out["Y"].shape == (2, 1, 10)
    assert out["pad_mask"].shape == (2, 1, 10)
    assert torch.equal(out["shape_label"], torch.tensor([0, 1]))
    assert torch.allclose(out["location_value"], torch.tensor([-1.0, 2.0]))
    assert out["location_value"].dtype == torch.float32


def test_balanced_batch_sampler_yields_fixed_ratio_per_batch():
    labels = [1] * 100 + [0] * 10  # heavily imbalanced: 100 shift, 10 normal
    sampler = BalancedBatchSampler(labels, batch_size=4, seed=0)
    batches = list(sampler)
    assert len(batches) == 5  # limited by minority class: 10 normal // 2 per batch
    for batch in batches:
        batch_labels = [labels[i] for i in batch]
        assert batch_labels.count(0) == 2
        assert batch_labels.count(1) == 2
