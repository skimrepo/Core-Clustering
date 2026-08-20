import numpy as np
import torch

from core_clustering.dataset_contrastive import NORMAL_SENTINEL
from core_clustering.dataset_dynamic_contrastive import (
    DynamicContrastiveDataset,
    dynamic_worker_init_fn,
    generate_entity_manifest,
)


def test_generate_entity_manifest_balances_roles_and_splits():
    entities = generate_entity_manifest(n_instances=100, anomaly_ratio=0.5, base_seed=0)
    assert len(entities) == 100
    n_anomalous = sum(e.is_anomalous for e in entities)
    assert n_anomalous == 50

    splits = [e.split for e in entities]
    assert set(splits) == {"train", "val", "test"}
    # each ROLE is split independently -- train/val/test should each have a
    # roughly even normal/anomalous mix, not e.g. all anomalous instances
    # dumped into train.
    train_anomalous = sum(e.is_anomalous for e in entities if e.split == "train")
    train_normal = sum(not e.is_anomalous for e in entities if e.split == "train")
    assert abs(train_anomalous - train_normal) <= 1


def test_generate_entity_manifest_seeds_are_unique_and_deterministic():
    a = generate_entity_manifest(n_instances=20, base_seed=42)
    b = generate_entity_manifest(n_instances=20, base_seed=42)
    assert [e.seed for e in a] == [e.seed for e in b]
    assert [e.split for e in a] == [e.split for e in b]
    assert len({e.seed for e in a}) == 20


def test_dynamic_dataset_normal_entity_returns_sentinel_and_unmodified_series():
    entities = generate_entity_manifest(n_instances=20, anomaly_ratio=0.5, base_seed=0)
    ds = DynamicContrastiveDataset(entities, split="train", train=True, base_seed=0,
                                    length_range=(50, 50))
    normal_idx = next(i for i, e in enumerate(ds.entities) if not e.is_anomalous)
    item = ds[normal_idx]
    assert item["shape_label"] == 0
    assert item["location_value"] == NORMAL_SENTINEL
    assert item["extent_value"] == NORMAL_SENTINEL
    assert item["intensity_value"] == NORMAL_SENTINEL


def test_dynamic_dataset_anomalous_entity_gets_injected_values_in_range():
    entities = generate_entity_manifest(n_instances=20, anomaly_ratio=0.5, base_seed=0)
    ds = DynamicContrastiveDataset(entities, split="train", train=True, base_seed=0,
                                    length_range=(200, 200),
                                    min_range_ratio=0.05, max_range_ratio=0.5,
                                    min_magnitude_std_multiplier=0.2, max_magnitude_std_multiplier=4.0)
    anom_idx = next(i for i, e in enumerate(ds.entities) if e.is_anomalous)
    item = ds[anom_idx]
    assert item["shape_label"] == 1
    assert 0.0 <= item["location_value"] <= 1.0
    assert 0.05 <= item["extent_value"] <= 0.5
    assert 0.2 <= item["intensity_value"] <= 4.0
    assert not torch.equal(item["Y"], torch.zeros_like(item["Y"]))


def test_dynamic_dataset_train_mode_resamples_every_call():
    entities = generate_entity_manifest(n_instances=20, anomaly_ratio=0.5, base_seed=0)
    ds = DynamicContrastiveDataset(entities, split="train", train=True, base_seed=0,
                                    length_range=(200, 200))
    anom_idx = next(i for i, e in enumerate(ds.entities) if e.is_anomalous)
    values = {(ds[anom_idx]["location_value"], ds[anom_idx]["extent_value"], ds[anom_idx]["intensity_value"])
              for _ in range(5)}
    assert len(values) > 1  # continuous resampling -- collisions would be astronomically unlikely


def test_dynamic_dataset_eval_mode_caches_identical_values_across_calls():
    entities = generate_entity_manifest(n_instances=20, anomaly_ratio=0.5, base_seed=0)
    ds = DynamicContrastiveDataset(entities, split="val", train=False, base_seed=0,
                                    length_range=(200, 200))
    anom_idx = next(i for i, e in enumerate(ds.entities) if e.is_anomalous)
    first = ds[anom_idx]
    second = ds[anom_idx]
    assert first["location_value"] == second["location_value"]
    assert first["extent_value"] == second["extent_value"]
    assert first["intensity_value"] == second["intensity_value"]
    assert torch.equal(first["Y"], second["Y"])


def test_dynamic_dataset_eval_mode_is_reproducible_across_separate_datasets():
    entities = generate_entity_manifest(n_instances=20, anomaly_ratio=0.5, base_seed=0)
    ds_a = DynamicContrastiveDataset(entities, split="val", train=False, base_seed=0, length_range=(200, 200))
    ds_b = DynamicContrastiveDataset(entities, split="val", train=False, base_seed=0, length_range=(200, 200))
    anom_idx = next(i for i, e in enumerate(ds_a.entities) if e.is_anomalous)
    assert torch.equal(ds_a[anom_idx]["Y"], ds_b[anom_idx]["Y"])


def test_dynamic_worker_init_fn_gives_each_worker_a_different_stream():
    entities = generate_entity_manifest(n_instances=20, anomaly_ratio=0.5, base_seed=0)

    def make_dataset():
        return DynamicContrastiveDataset(entities, split="train", train=True, base_seed=0,
                                          length_range=(200, 200))

    ds_worker0 = make_dataset()
    ds_worker1 = make_dataset()

    class _FakeWorkerInfo:
        def __init__(self, dataset):
            self.dataset = dataset

    import core_clustering.dataset_dynamic_contrastive as mod
    mod.torch.utils.data.get_worker_info = lambda: _FakeWorkerInfo(ds_worker0)
    dynamic_worker_init_fn(0)
    mod.torch.utils.data.get_worker_info = lambda: _FakeWorkerInfo(ds_worker1)
    dynamic_worker_init_fn(1)

    anom_idx = next(i for i, e in enumerate(ds_worker0.entities) if e.is_anomalous)
    assert ds_worker0[anom_idx]["location_value"] != ds_worker1[anom_idx]["location_value"]
