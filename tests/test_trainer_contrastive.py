import functools
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_clustering.dataset_contrastive import (
    BalancedBatchSampler,
    ContrastiveDataset,
    ContrastiveRecord,
    contrastive_pad_collate,
)
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive import ContrastiveEncoder
from core_clustering.trainer_contrastive import ContrastiveTrainer


def _make_records(n_normal=6, n_shift=6, n_time=20):
    records = []
    rng = np.random.default_rng(0)
    for i in range(n_normal):
        Z = rng.normal(size=(1, n_time))
        records.append(ContrastiveRecord(
            Y=Z.copy(), Z=Z.copy(), shape_label=0,
            location_value=-1.0, extent_value=-1.0, intensity_value=-1.0,
            entity_dir=f"n{i}", split="train", n_time=n_time,
        ))
    for i in range(n_shift):
        Z = rng.normal(size=(1, n_time))
        Y = Z.copy()
        Y[0, 5:10] += 4.0
        records.append(ContrastiveRecord(
            Y=Y, Z=Z, shape_label=1,
            location_value=float(i % 2), extent_value=0.0, intensity_value=0.0,
            entity_dir=f"s{i}", split="train", n_time=n_time,
        ))
    return records


def _make_loader(records, batch_size=4, max_len=20, seed=0):
    labels = [r.shape_label for r in records]
    sampler = BalancedBatchSampler(labels, batch_size=batch_size, seed=seed)
    return DataLoader(ContrastiveDataset(records), batch_sampler=sampler,
                       collate_fn=functools.partial(contrastive_pad_collate, max_len=max_len))


def _tiny_config():
    return ConvBottleneckConfig(
        n_time_max=20, num_filters=[4, 4], bottleneck_channels=2,
        num_groups=2, dropout=0.0, attention_max_resolution=0,
    )


def test_trainer_runs_epochs_and_saves_checkpoint(tmp_path):
    records = _make_records()
    train_loader = _make_loader(records)
    val_loader = _make_loader(records, seed=1)

    model = ContrastiveEncoder(_tiny_config(), embedding_dim=8)
    trainer = ContrastiveTrainer(model, device="cpu", patience=5, output_dir=str(tmp_path))

    history = trainer.train(train_loader, val_loader, epochs=2)

    assert len(history) == 2
    assert os.path.exists(os.path.join(tmp_path, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(tmp_path, "epoch_history.json"))
    for key in ("loss_shape", "loss_location", "loss_extent", "loss_intensity"):
        assert key in history[0]


def test_trainer_loss_decreases_over_a_few_epochs():
    records = _make_records(n_normal=10, n_shift=10)
    train_loader = _make_loader(records, batch_size=8)

    model = ContrastiveEncoder(_tiny_config(), embedding_dim=8)
    trainer = ContrastiveTrainer(model, device="cpu", lr=0.01)

    history = trainer.train(train_loader, val_dataloader=None, epochs=15)
    losses = [r["train_loss"] for r in history]
    assert losses[-1] < losses[0]


def test_trainer_has_one_optimizer_per_attribute_sharing_the_trunk():
    # One shared AdamW blending all four attributes' gradient statistics on
    # the trunk let a noisy attribute's Adam moment estimates distort the
    # step size for a clean one (e.g. shape) sharing the same trunk
    # parameters. Separate optimizers -- each covering the trunk plus only
    # its own head -- keep each attribute's adaptive scaling independent.
    model = ContrastiveEncoder(_tiny_config(), embedding_dim=8)
    trainer = ContrastiveTrainer(model, device="cpu")

    assert set(trainer.optimizers.keys()) == {"shape", "location", "extent", "intensity"}
    trunk_param_ids = (
        {id(p) for p in model.encoder.parameters()}
        | {id(p) for p in model.pool_attn.parameters()}
        | {id(model.pool_query)}
    )
    for attr, opt in trainer.optimizers.items():
        opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
        assert trunk_param_ids <= opt_param_ids
        head_param_ids = {id(p) for p in model.heads[attr].parameters()}
        assert head_param_ids <= opt_param_ids
        other_head_ids = {
            id(p) for name, head in model.heads.items() if name != attr for p in head.parameters()
        }
        assert opt_param_ids.isdisjoint(other_head_ids)


def test_trainer_optimizes_loss_modules_learnable_parameters_too():
    # shape's temperature lives in the loss module, not the encoder -- if
    # the optimizer only tracked model.parameters(), it would never move
    # from its initial value.
    records = _make_records()
    train_loader = _make_loader(records)

    model = ContrastiveEncoder(_tiny_config(), embedding_dim=8)
    trainer = ContrastiveTrainer(model, device="cpu", lr=0.1)
    initial_temperature = trainer.loss_fn.shape_loss.log_temperature.item()

    trainer.train(train_loader, val_dataloader=None, epochs=3)

    assert trainer.loss_fn.shape_loss.log_temperature.item() != initial_temperature
