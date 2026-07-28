import numpy as np
import pytest

from core_clustering.dataset import LoadedDataset, LoadStats
from core_clustering.splits import make_cross_domain_split


def _make_dataset(rows):
    """rows: list of (domain, base_instance_id) tuples, one per window row."""
    n = len(rows)
    domain = np.array([r[0] for r in rows])
    base_instance_id = np.array([r[1] for r in rows])
    load_stats = LoadStats(
        manifest_path="fake",
        n_manifest_lines=n,
        n_attempted=n,
        n_loaded=n,
        n_failed=0,
        failures=[],
        failures_by_reason={},
        domains=sorted(set(domain.tolist())),
        anomaly_types=["normal"],
    )
    return LoadedDataset(
        Y=np.zeros((n, 1, 10), dtype=np.float32),
        labels=np.ones((n, 1, 10), dtype=np.float32),
        Z=np.zeros((n, 1, 10), dtype=np.float32),
        domain=domain,
        anomaly_type=np.array(["normal"] * n),
        base_instance_id=base_instance_id,
        window_index=np.arange(n),
        entity_dir=np.array([f"e{i}" for i in range(n)]),
        window_size=10,
        class_list=["normal"],
        load_stats=load_stats,
    )


def _rows_for_domain(domain, group_sizes):
    """group_sizes: list of window-counts per base_instance_id group."""
    rows = []
    for base_id, count in enumerate(group_sizes):
        rows += [(domain, base_id)] * count
    return rows


def test_holdout_domains_excluded_from_train_and_val():
    rows = _rows_for_domain("a", [5, 5, 5]) + _rows_for_domain("b", [5, 5, 5])
    dataset = _make_dataset(rows)
    result = make_cross_domain_split(dataset, holdout_domains=["b"], val_fraction=0.2, seed=0)
    assert set(dataset.domain[result.train_idx]) == {"a"}
    assert set(dataset.domain[result.val_idx]) == {"a"}
    assert set(dataset.domain[result.holdout_idx]) == {"b"}


def test_group_never_split_across_train_and_val():
    rows = _rows_for_domain("a", [3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    dataset = _make_dataset(rows)
    result = make_cross_domain_split(dataset, holdout_domains=[], val_fraction=0.3, seed=1)
    assert set(result.train_groups).isdisjoint(set(result.val_groups))


def test_single_group_domain_goes_entirely_to_train_with_warning():
    rows = _rows_for_domain("a", [4])
    dataset = _make_dataset(rows)
    result = make_cross_domain_split(dataset, holdout_domains=[], val_fraction=0.2, seed=0)
    assert len(result.val_idx) == 0
    assert len(result.train_idx) == 4
    assert any("a" in w for w in result.warnings)


def test_val_fraction_respected_for_domain_with_many_groups():
    rows = _rows_for_domain("a", [1] * 10)
    dataset = _make_dataset(rows)
    result = make_cross_domain_split(dataset, holdout_domains=[], val_fraction=0.2, seed=0)
    assert len(result.val_groups) == 2
    assert len(result.train_groups) == 8


def test_unknown_holdout_domain_raises_value_error():
    rows = _rows_for_domain("a", [3, 3])
    dataset = _make_dataset(rows)
    with pytest.raises(ValueError):
        make_cross_domain_split(dataset, holdout_domains=["does_not_exist"], val_fraction=0.2, seed=0)


def test_split_is_reproducible_given_same_seed():
    rows = _rows_for_domain("a", [1] * 10) + _rows_for_domain("b", [1] * 10)
    dataset = _make_dataset(rows)
    result1 = make_cross_domain_split(dataset, holdout_domains=[], val_fraction=0.3, seed=7)
    result2 = make_cross_domain_split(dataset, holdout_domains=[], val_fraction=0.3, seed=7)
    assert sorted(result1.train_groups) == sorted(result2.train_groups)
    assert sorted(result1.val_groups) == sorted(result2.val_groups)


def test_val_fraction_actual_reflects_real_split():
    rows = _rows_for_domain("a", [1] * 10)
    dataset = _make_dataset(rows)
    result = make_cross_domain_split(dataset, holdout_domains=[], val_fraction=0.2, seed=0)
    assert result.val_fraction_actual == pytest.approx(len(result.val_idx) / (len(result.train_idx) + len(result.val_idx)))
