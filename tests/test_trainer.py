import json
import os

import numpy as np
import torch

from core_clustering.dataset import LoadedDataset, LoadStats
from core_clustering.models import ConvAEC, ModelConfig
from core_clustering.trainer import EpochRecord, Trainer, make_torch_dataset, write_run_summary


def _make_tiny_dataset(n=40, window_size=16, classes=3, domains=("a", "b")):
    domain = np.array([domains[i % len(domains)] for i in range(n)])
    base_instance_id = np.array([i % 4 for i in range(n)])
    anomaly_type = np.array([f"c{i % classes}" for i in range(n)])
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(n, 1, window_size)).astype(np.float32)
    labels = np.ones((n, 1, window_size), dtype=np.float32)
    Z = Y.copy()
    load_stats = LoadStats(
        manifest_path="fake", n_manifest_lines=n, n_attempted=n, n_loaded=n, n_failed=0,
        failures=[], failures_by_reason={}, domains=sorted(set(domain.tolist())),
        anomaly_types=sorted(set(anomaly_type.tolist())),
    )
    return LoadedDataset(
        Y=Y, labels=labels, Z=Z, domain=domain, anomaly_type=anomaly_type,
        base_instance_id=base_instance_id, window_index=np.arange(n),
        entity_dir=np.array([f"e{i}" for i in range(n)]), window_size=window_size,
        class_list=sorted(set(anomaly_type.tolist())), load_stats=load_stats,
    )


def _tiny_model_config(window_size, classes):
    return ModelConfig(
        n_features=1, n_time=window_size, classes=classes,
        num_filters=[8, 8], embedding_dim=4, kernel_size=4, dropout=0.0,
        normalization="batch", stride=2, padding=2, classifier_dim=4,
    )


def test_make_torch_dataset_shapes():
    dataset = _make_tiny_dataset(n=20, window_size=16)
    idx = np.arange(20)
    torch_ds = make_torch_dataset(dataset, idx)
    Y, mask, label = torch_ds[0]
    assert Y.shape == (16, 1)
    assert mask.shape == (16, 1)
    assert label.shape == (len(dataset.class_list),)


class _SetEpochRecordingDataset(torch.utils.data.Dataset):
    """Minimal stand-in for OnlineWindowedDataset -- just enough shape to run
    through Trainer, plus a set_epoch() that records every call so the test
    can assert Trainer actually calls it once per epoch with the right value."""

    def __init__(self, n=16, window_size=8, classes=2):
        self.n = n
        self.window_size = window_size
        self.classes = classes
        self.epoch_calls = []

    def set_epoch(self, epoch):
        self.epoch_calls.append(epoch)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        Y = torch.zeros(self.window_size, 1)
        mask = torch.ones(self.window_size, 1)
        label = torch.zeros(self.classes)
        label[idx % self.classes] = 1.0
        return Y, mask, label


def test_train_calls_set_epoch_on_train_dataset_each_epoch_if_present(tmp_path):
    train_dataset = _SetEpochRecordingDataset(n=16, window_size=8, classes=2)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    config = _tiny_model_config(8, 2)
    model = ConvAEC(config)
    trainer = Trainer(model, device="cpu", output_dir=str(tmp_path), patience=10)
    trainer.train(train_dl, val_dataloader=None, epochs=3)

    assert train_dataset.epoch_calls == [0, 1, 2]


def test_train_writes_epoch_history_incrementally(tmp_path):
    # So a crash right after training (e.g. during a separate reporting/
    # plotting step) never loses the epoch history needed for run_summary.json
    # -- a fresh invocation can reload it instead of retraining from scratch.
    dataset = _make_tiny_dataset(n=40, window_size=16, classes=3)
    idx = np.arange(40)
    train_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, idx[:30]), batch_size=8, shuffle=True)
    val_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, idx[30:]), batch_size=8, shuffle=False)

    trainer = Trainer(ConvAEC(_tiny_model_config(16, 3)), device="cpu", output_dir=str(tmp_path), patience=10)
    history = trainer.train(train_dl, val_dl, epochs=3)

    history_path = os.path.join(str(tmp_path), "epoch_history.json")
    assert os.path.exists(history_path)
    with open(history_path) as f:
        saved = json.load(f)
    assert len(saved) == 3
    assert saved == [asdict_record(r) for r in history]


def asdict_record(record):
    from dataclasses import asdict
    return asdict(record)


def test_train_does_not_crash_when_train_dataset_lacks_set_epoch(tmp_path):
    dataset = _make_tiny_dataset(n=20, window_size=16, classes=3)
    train_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, np.arange(20)), batch_size=4, shuffle=True)
    config = _tiny_model_config(16, 3)
    model = ConvAEC(config)
    trainer = Trainer(model, device="cpu", output_dir=str(tmp_path), patience=10)
    history = trainer.train(train_dl, val_dataloader=None, epochs=2)
    assert len(history) == 2


def test_train_returns_one_epoch_record_per_epoch(tmp_path):
    dataset = _make_tiny_dataset(n=40, window_size=16, classes=3)
    idx = np.arange(40)
    train_ds = make_torch_dataset(dataset, idx[:30])
    val_ds = make_torch_dataset(dataset, idx[30:])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

    config = _tiny_model_config(16, 3)
    model = ConvAEC(config)
    trainer = Trainer(model, device="cpu", output_dir=str(tmp_path), patience=10)
    history = trainer.train(train_dl, val_dl, epochs=3)

    assert len(history) == 3
    for record in history:
        assert isinstance(record, EpochRecord)
        assert record.val_loss is not None


def test_first_epoch_is_best_with_zero_early_stop_counter(tmp_path):
    dataset = _make_tiny_dataset(n=40, window_size=16, classes=3)
    idx = np.arange(40)
    train_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, idx[:30]), batch_size=8, shuffle=True)
    val_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, idx[30:]), batch_size=8, shuffle=False)

    trainer = Trainer(ConvAEC(_tiny_model_config(16, 3)), device="cpu", output_dir=str(tmp_path), patience=10)
    history = trainer.train(train_dl, val_dl, epochs=1)

    assert history[0].is_best is True
    assert history[0].early_stop_counter == 0


def test_bestmodel_saved_on_improvement(tmp_path):
    dataset = _make_tiny_dataset(n=40, window_size=16, classes=3)
    idx = np.arange(40)
    train_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, idx[:30]), batch_size=8, shuffle=True)
    val_dl = torch.utils.data.DataLoader(make_torch_dataset(dataset, idx[30:]), batch_size=8, shuffle=False)

    trainer = Trainer(ConvAEC(_tiny_model_config(16, 3)), device="cpu", output_dir=str(tmp_path), patience=10)
    trainer.train(train_dl, val_dl, epochs=2)

    assert os.path.exists(os.path.join(str(tmp_path), "bestmodel.pkl"))


def test_write_run_summary_schema(tmp_path):
    epochs = [
        EpochRecord(epoch=0, train_loss=0.9, train_loss_ae=0.8, train_loss_c=0.1,
                    val_loss=0.85, val_loss_ae=0.8, val_loss_c=0.05,
                    epoch_seconds=1.2, is_best=True, early_stop_counter=0),
        EpochRecord(epoch=1, train_loss=0.7, train_loss_ae=0.6, train_loss_c=0.1,
                    val_loss=0.65, val_loss_ae=0.6, val_loss_c=0.05,
                    epoch_seconds=1.1, is_best=True, early_stop_counter=0),
    ]
    out_path = str(tmp_path / "run_summary.json")
    write_run_summary(
        out_path,
        run_id="run-test",
        dataset_dir="/some/dataset",
        seed=0,
        device="cpu",
        included_domains=["a", "b"],
        held_out_domains=["c"],
        val_fraction_requested=0.2,
        val_fraction_actual=0.187,
        n_entities_attempted=100,
        n_entities_loaded=96,
        n_entities_failed=4,
        domain_window_counts=[
            {"domain": "a", "role": "included", "n_windows_train": 10, "n_windows_val": 2,
             "n_windows_eval": None, "n_entities_loaded": 48, "n_entities_failed": 2},
            {"domain": "c", "role": "held_out", "n_windows_train": None, "n_windows_val": None,
             "n_windows_eval": 5, "n_entities_loaded": 48, "n_entities_failed": 2},
        ],
        epochs=epochs,
        epochs_requested=100,
        early_stop_patience=10,
        model_hyperparameters={"model": "ConvAEC", "n_features": 1},
        held_out_accuracy=[{"domain": "c", "n_total": 5, "n_correct": 3, "n_incorrect": 2, "accuracy": 0.6}],
    )

    with open(out_path) as f:
        summary = json.load(f)

    for key in [
        "schema_version", "run_id", "created_at", "dataset_dir", "seed", "device",
        "included_domains", "held_out_domains", "val_fraction_requested", "val_fraction_actual",
        "n_entities_attempted", "n_entities_loaded", "n_entities_failed",
        "n_windows_train", "n_windows_val", "n_windows_total", "domain_window_counts",
        "epochs_requested", "epochs_ran", "early_stopped", "early_stop_patience",
        "early_stop_epoch", "best_epoch", "best_val_loss", "best_val_loss_ae", "best_val_loss_c",
        "total_train_seconds", "mean_epoch_seconds", "model_hyperparameters",
        "held_out_accuracy", "epochs",
    ]:
        assert key in summary, f"missing key {key}"

    assert summary["n_windows_train"] + summary["n_windows_val"] == summary["n_windows_total"]
    assert summary["epochs_ran"] == 2
    assert summary["best_epoch"] == 1
    assert len(summary["epochs"]) == 2
