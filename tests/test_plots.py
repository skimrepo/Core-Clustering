import os

import numpy as np

from core_clustering.dataset import LoadedDataset, LoadStats
from core_clustering.plots import (
    plot_example_window,
    plot_representative_samples,
    plot_tsne_by_class,
    plot_tsne_by_domain,
)


def _make_tiny_dataset(n=12, window_size=16):
    domain = np.array(["a", "b"] * (n // 2))
    anomaly_type = np.array(["normal", "spike"] * (n // 2))
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(n, 1, window_size)).astype(np.float32)
    Z = Y.copy()
    labels = np.ones((n, 1, window_size), dtype=np.float32)
    labels[:, 0, 5] = 0  # pretend a spike anomaly at index 5 for all rows
    load_stats = LoadStats(
        manifest_path="fake", n_manifest_lines=n, n_attempted=n, n_loaded=n, n_failed=0,
        failures=[], failures_by_reason={}, domains=["a", "b"], anomaly_types=["normal", "spike"],
    )
    return LoadedDataset(
        Y=Y, labels=labels, Z=Z, domain=domain, anomaly_type=anomaly_type,
        base_instance_id=np.arange(n) % 3, window_index=np.arange(n),
        entity_dir=np.array([f"e{i}" for i in range(n)]), window_size=window_size,
        class_list=["normal", "spike"], load_stats=load_stats,
    )


def test_plot_example_window_normal_no_error(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_example_window(ax, np.zeros(16), np.zeros(16), np.ones(16), "normal")
    plt.close(fig)


def test_plot_example_window_spike_shades_single_point(tmp_path):
    import matplotlib.pyplot as plt

    mask = np.ones(16)
    mask[5] = 0
    fig, ax = plt.subplots()
    plot_example_window(ax, np.zeros(16), np.zeros(16), mask, "spike")
    assert len(ax.lines) >= 2  # Z + Y lines at minimum, plus axvline
    plt.close(fig)


def test_plot_tsne_by_class_creates_file(tmp_path):
    embeddings = np.random.default_rng(0).normal(size=(10, 4))
    class_idx = np.array([0, 1] * 5)
    save_path = str(tmp_path / "tsne_class.png")
    plot_tsne_by_class(embeddings, class_idx, ["normal", "spike"], save_path, perplexity=2, seed=0)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 500


def test_plot_tsne_by_domain_creates_file(tmp_path):
    embeddings = np.random.default_rng(0).normal(size=(10, 4))
    class_idx = np.array([0, 1] * 5)
    domain_idx = np.array([0, 0, 1, 1] * 2 + [0, 1])
    save_path = str(tmp_path / "tsne_domain.png")
    plot_tsne_by_domain(embeddings, class_idx, domain_idx, ["normal", "spike"], ["a", "b"], save_path, perplexity=2, seed=0)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 500


def test_plot_representative_samples_creates_files_per_domain(tmp_path):
    dataset = _make_tiny_dataset(n=12)
    out_dir = str(tmp_path / "samples")
    indices = np.arange(12)
    plot_representative_samples(dataset, indices, out_dir, n_per_domain=2, seed=0)
    files = os.listdir(out_dir)
    assert any(f.startswith("a_") for f in files)
    assert any(f.startswith("b_") for f in files)
    assert len([f for f in files if f.startswith("a_")]) == 2
