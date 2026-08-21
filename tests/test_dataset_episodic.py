import numpy as np
import pytest
import torch

from core_clustering.dataset_dynamic_contrastive import generate_entity_manifest
from core_clustering.dataset_episodic import EpisodicContrastiveDataset, episodic_pad_collate


def make_ds(train=True, n_instances=30, seed=0, **kwargs):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)
    return EpisodicContrastiveDataset(
        entities, split="train" if train else "val", train=train, base_seed=seed,
        length_range=(200, 200),
        intensity_mode="universal_deviation_intensity", intensity_min=0.2, intensity_max=4.0,
        intensity_metric_transform="identity",
        **kwargs,
    )


def test_episodic_dataset_normal_query_has_zero_d_target():
    ds = make_ds()
    normal_idx = next(i for i, e in enumerate(ds.entities) if not e.is_anomalous)
    item = ds[normal_idx]
    assert item["D"] == 0.0


def test_episodic_dataset_anomalous_query_d_matches_raw_rms_times_sigma_ref():
    ds = make_ds()
    anom_idx = next(i for i, e in enumerate(ds.entities) if e.is_anomalous)
    item = ds[anom_idx]
    assert item["D"] >= 0.0
    assert item["D"] == pytest.approx(item["intensity_value_raw"] * item["sigma_ref"], rel=1e-6)


def test_episodic_dataset_d_is_not_divided_by_sigma_ref():
    # D must be the UNDIVIDED RMS(delta) -- i.e. scale with sigma_ref, not
    # be invariant to it. Sanity: for two different sigma_ref values with
    # the same intensity_value_raw, D differs proportionally. We can't
    # force sigma_ref directly, but we can check D/intensity_value_raw ==
    # sigma_ref (not 1, not intensity_value_raw itself) across several samples.
    ds = make_ds(n_instances=60)
    checked = 0
    for i, e in enumerate(ds.entities):
        if not e.is_anomalous:
            continue
        item = ds[i]
        assert item["D"] == pytest.approx(item["sigma_ref"] * item["intensity_value_raw"], rel=1e-6)
        checked += 1
    assert checked > 0


def test_episodic_dataset_k_always_within_configured_regime():
    k_regimes = (0, 3, 10)
    ds = make_ds(n_instances=40, k_regimes=k_regimes)
    seen_k = {ds[i]["K"] for i in range(len(ds))}
    assert seen_k.issubset(set(k_regimes))


def test_episodic_dataset_reference_list_length_matches_k():
    ds = make_ds(n_instances=20, k_regimes=(3,))
    item = ds[0]
    assert item["K"] == 3
    assert len(item["references"]) == 3


def test_episodic_dataset_k0_has_empty_reference_list():
    ds = make_ds(n_instances=20, k_regimes=(0,))
    item = ds[0]
    assert item["K"] == 0
    assert item["references"] == []


def test_episodic_dataset_eval_mode_reproducible_k_and_references():
    ds = make_ds(train=False, n_instances=20, k_regimes=(3, 10))
    first = ds[0]
    second = ds[0]
    assert first["K"] == second["K"]
    assert len(first["references"]) == len(second["references"])
    for (y1, _), (y2, _) in zip(first["references"], second["references"]):
        assert np.allclose(y1, y2)


def test_episodic_dataset_contamination_rate_is_roughly_configured():
    rng_check = np.random.default_rng(0)
    ds = make_ds(n_instances=200, k_regimes=(10,), contamination_prob=0.5)
    flags = []
    for i in range(30):
        item = ds[i]
        flags.extend(item["reference_contaminated"])
    frac = float(np.mean(flags))
    assert 0.3 < frac < 0.7  # loose statistical bound around the configured 0.5


def test_episodic_dataset_sample_alternate_references_differs_from_original():
    ds = make_ds(n_instances=20, k_regimes=(5,))
    item = ds[0]
    alt_refs, alt_contam = ds.sample_alternate_references(0, K=5)
    assert len(alt_refs) == 5
    # Not bitwise-guaranteed different, but with continuous waveforms the
    # chance of an exact match is astronomically small.
    assert not np.allclose(item["references"][0][0], alt_refs[0][0])


# --- collate -------------------------------------------------------------

def test_episodic_dataset_include_alternate_references_attaches_second_set():
    ds = make_ds(n_instances=20, k_regimes=(4,), include_alternate_references=True)
    item = ds[0]
    assert "references_b" in item
    assert len(item["references_b"]) == item["K"]


def test_episodic_collate_includes_alternate_reference_tensors_when_present():
    ds = make_ds(n_instances=20, k_regimes=(4,), include_alternate_references=True)
    batch = [ds[i] for i in range(4)]
    out = episodic_pad_collate(batch, max_len=200)
    assert "ref_x_b" in out
    assert out["ref_x_b"].shape == out["ref_x"].shape
    assert not torch.allclose(out["ref_x"], out["ref_x_b"])


def test_episodic_collate_produces_expected_shapes():
    ds = make_ds(n_instances=20, k_regimes=(0, 3))
    batch = [ds[i] for i in range(6)]
    out = episodic_pad_collate(batch, max_len=200)

    B = len(batch)
    max_k = max(item["K"] for item in batch)
    assert out["Y"].shape == (B, 1, 200)
    assert out["D"].shape == (B,)
    assert out["ref_x"].shape == (B, max_k, 1, 200)
    assert out["ref_pad_mask"].shape == (B, max_k, 1, 200)
    assert out["ref_k_valid_mask"].shape == (B, max_k)
    for i, item in enumerate(batch):
        assert out["ref_k_valid_mask"][i, :item["K"]].sum().item() == item["K"]
        assert out["ref_k_valid_mask"][i, item["K"]:].sum().item() == 0
