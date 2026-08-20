import functools
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_clustering.dataset_tcn import WholeSeriesDataset, WholeSeriesRecord, pad_collate
from core_clustering.models_conv_bottleneck import ConvBottleneckAEC, ConvBottleneckConfig
from core_clustering.trainer_tcn import TCNTrainer, default_tcn_hyperparameters


def _make_records(n=4, n_time=20):
    records = []
    rng = np.random.default_rng(0)
    for i in range(n):
        Z = rng.normal(size=(1, n_time))
        Y = Z.copy()
        label = np.zeros((1, n_time))
        if i % 2 == 0:
            Y[0, 5:10] += 4.0
            label[0, 5:10] = 1.0
        records.append(WholeSeriesRecord(Y=Y, Z=Z, label=label, entity_dir=f"e{i}", split="train", n_time=n_time))
    return records


def _make_loader(records, batch_size, max_len=20):
    ds = WholeSeriesDataset(records)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                       collate_fn=functools.partial(pad_collate, max_len=max_len))


def _tiny_config(**overrides):
    defaults = dict(n_time_max=20, num_filters=[4, 4], bottleneck_channels=2, num_groups=2, dropout=0.0)
    defaults.update(overrides)
    return ConvBottleneckConfig(**defaults)


def test_default_tcn_hyperparameters_builds_config():
    config = default_tcn_hyperparameters(n_features=1, n_time_max=550, num_filters=[8, 8])
    assert isinstance(config, ConvBottleneckConfig)
    assert config.n_time_max == 550
    assert config.num_filters == [8, 8]


def test_trainer_runs_epochs_and_saves_checkpoint(tmp_path):
    records = _make_records(n=4)
    train_loader = _make_loader(records, batch_size=2)
    val_loader = _make_loader(records, batch_size=2)

    model = ConvBottleneckAEC(_tiny_config())
    trainer = TCNTrainer(model, device="cpu", patience=5, output_dir=str(tmp_path))

    history = trainer.train(train_loader, val_loader, epochs=2)

    assert len(history) == 2
    assert os.path.exists(os.path.join(tmp_path, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(tmp_path, "epoch_history.json"))


def test_trainer_prints_val_loss_ae_and_val_loss_c_breakdown(capsys):
    records = _make_records(n=4)
    train_loader = _make_loader(records, batch_size=2)
    val_loader = _make_loader(records, batch_size=2)

    model = ConvBottleneckAEC(_tiny_config())
    trainer = TCNTrainer(model, device="cpu", patience=5)

    trainer.train(train_loader, val_loader, epochs=1)
    out = capsys.readouterr().out
    assert "val_loss_ae" in out
    assert "val_loss_c" in out


def test_trainer_does_not_skip_batch_size_one():
    records = _make_records(n=2)
    loader = _make_loader(records, batch_size=1)

    model = ConvBottleneckAEC(_tiny_config())
    trainer = TCNTrainer(model, device="cpu")

    train_loss, _, _ = trainer._run_epoch(loader, train=True)
    assert train_loss == train_loss  # not NaN -- both batch_size=1 batches contributed
