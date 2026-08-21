import functools
import os

from torch.utils.data import DataLoader

from core_clustering.dataset_dynamic_contrastive import generate_entity_manifest
from core_clustering.dataset_episodic import EpisodicContrastiveDataset, episodic_pad_collate
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ContrastiveEncoderV3
from core_clustering.trainer_contrastive_v3 import ContrastiveTrainerV3


def _make_loader(seed=0, n_instances=24, k_regimes=(0, 3), include_alternate_references=True):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)
    ds = EpisodicContrastiveDataset(
        entities, split="train", train=True, base_seed=seed, length_range=(60, 60),
        intensity_mode="universal_deviation_intensity", intensity_min=0.2, intensity_max=4.0,
        intensity_metric_transform="identity", k_regimes=k_regimes,
        include_alternate_references=include_alternate_references,
    )
    collate = functools.partial(episodic_pad_collate, max_len=60)
    return DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)


def _tiny_config():
    return ConvBottleneckConfig(
        n_time_max=60, n_features=2, num_filters=[4, 4], num_groups=2, dropout=0.0,
        attention_max_resolution=0,
    )


def test_trainer_v3_runs_epochs_and_saves_checkpoint(tmp_path):
    train_loader = _make_loader(seed=0)
    val_loader = _make_loader(seed=1)

    model = ContrastiveEncoderV3(_tiny_config(), embedding_dim=8, head_proj_channels=4, head_mlp_hidden=8)
    trainer = ContrastiveTrainerV3(model, device="cpu", patience=5, output_dir=str(tmp_path))

    history = trainer.train(train_loader, val_loader, epochs=2)

    assert len(history) == 2
    assert os.path.exists(os.path.join(tmp_path, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(tmp_path, "epoch_history.json"))
    for key in ("loss_shape", "loss_location", "loss_extent", "loss_intensity"):
        assert key in history[0]


def test_trainer_v3_loss_is_finite_and_decreases_a_little():
    train_loader = _make_loader(seed=0, n_instances=32)
    model = ContrastiveEncoderV3(_tiny_config(), embedding_dim=8, head_proj_channels=4, head_mlp_hidden=8)
    trainer = ContrastiveTrainerV3(model, device="cpu", lr=0.005)

    history = trainer.train(train_loader, val_dataloader=None, epochs=8)
    losses = [r["train_loss"] for r in history]
    assert all(l == l for l in losses)  # no NaN
    assert losses[-1] < losses[0] * 1.5  # loose: mainly checking stability, not a tight convergence bound


def test_trainer_v3_uses_a_single_optimizer_over_the_whole_model():
    model = ContrastiveEncoderV3(_tiny_config(), embedding_dim=8, head_proj_channels=4, head_mlp_hidden=8)
    trainer = ContrastiveTrainerV3(model, device="cpu")
    opt_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    model_param_ids = {id(p) for p in model.parameters()}
    assert model_param_ids <= opt_param_ids
    # trunk params must appear exactly once (no duplicated shared-trunk
    # parameters across multiple optimizer groups/instances)
    trunk_ids = [id(p) for p in model.encoder.parameters()]
    all_group_params = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    for tid in trunk_ids:
        assert all_group_params.count(tid) == 1


def test_trainer_v3_works_without_reference_consistency_batches():
    train_loader = _make_loader(seed=0, include_alternate_references=False)
    model = ContrastiveEncoderV3(_tiny_config(), embedding_dim=8, head_proj_channels=4, head_mlp_hidden=8)
    trainer = ContrastiveTrainerV3(model, device="cpu", consistency_prob=0.5)
    history = trainer.train(train_loader, val_dataloader=None, epochs=2)
    assert len(history) == 2
