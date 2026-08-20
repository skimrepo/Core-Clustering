import pytest
import torch
import torch.nn.functional as F

from core_clustering.models_conv_bottleneck import (
    ConvBottleneckAEC,
    ConvBottleneckConfig,
    ConvBottleneckDecoder,
    ConvBottleneckEncoder,
    SelfAttentionBlock,
    compute_num_filters,
    compute_num_stages,
)


def make_tiny_config(**overrides):
    defaults = dict(
        n_time_max=550,
        n_features=1,
        num_filters=[8, 16, 32],
        bottleneck_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        dropout=0.0,
        normalization="group",
        num_groups=4,
        bce_loss_ratio=0.1,
        # Attention placement also depends on n_time_max (via nominal
        # per-stage length) -- disabled by default here so the many
        # generic tests below (shape/pad-mask/loss checks) aren't coupled
        # to attention behavior. Dedicated attention tests override this.
        attention_max_resolution=0,
    )
    defaults.update(overrides)
    return ConvBottleneckConfig(**defaults)


@pytest.mark.parametrize("T", [61, 137, 300, 550])
def test_encoder_downsamples_length_and_squeezes_to_bottleneck_channels(T):
    encoder = ConvBottleneckEncoder(num_inputs=1, num_filters=[8, 16, 32], bottleneck_channels=4,
                                     kernel_size=3, stride=2, padding=1)
    x = torch.randn(2, 1, T)
    feat, lengths, masks = encoder(x)
    assert feat.shape[0] == 2 and feat.shape[1] == 4
    assert lengths[0] == T
    assert feat.shape[2] == lengths[-1]
    assert feat.shape[2] < T  # a real bottleneck: strictly shorter than input


def test_encoder_include_squeeze_false_skips_squeeze_layer_entirely():
    # V2 (Core-Clustering models_contrastive_v2.py) needs the raw last-stage
    # feature map (128ch) as its shared representation, not the channel-
    # squeezed (B,4,T') bottleneck -- include_squeeze=False must not even
    # construct the squeeze conv, so it contributes zero dead parameters.
    encoder = ConvBottleneckEncoder(num_inputs=1, num_filters=[8, 16, 32], bottleneck_channels=4,
                                     kernel_size=3, stride=2, padding=1, include_squeeze=False)
    assert encoder.squeeze is None
    x = torch.randn(2, 1, 137)
    feat, lengths, masks = encoder(x)
    assert feat.shape[1] == 32  # last stage's own channel width, NOT bottleneck_channels
    assert feat.shape[2] == lengths[-1]


def test_encoder_include_squeeze_true_is_unchanged_default():
    encoder = ConvBottleneckEncoder(num_inputs=1, num_filters=[8, 16, 32], bottleneck_channels=4,
                                     kernel_size=3, stride=2, padding=1)
    assert encoder.squeeze is not None
    assert isinstance(encoder.squeeze, torch.nn.Conv1d)


@pytest.mark.parametrize("T", [61, 137, 300, 550])
def test_full_model_reconstructs_exact_input_length(T):
    model = ConvBottleneckAEC(make_tiny_config())
    x = torch.randn(2, 1, T)
    recon, anomaly_logits, feat = model(x)
    assert recon.shape == (2, 1, T)
    assert anomaly_logits.shape == (2, 1, T)


def test_bottleneck_feature_is_much_smaller_than_input():
    # Use the real default num_filters/bottleneck_channels (4 stride-2
    # stages), not the lightweight 3-stage test config -- the 3-stage
    # config doesn't compress enough at T=550 to satisfy "much smaller."
    model = ConvBottleneckAEC(ConvBottleneckConfig(n_time_max=550))
    T = 550
    x = torch.randn(1, 1, T)
    with torch.no_grad():
        feat = model.encoder(x)[0]
    bottleneck_elems = feat.shape[1] * feat.shape[2]
    input_elems = x.shape[1] * x.shape[2]
    assert bottleneck_elems < input_elems / 2


def test_n_time_max_is_not_baked_into_any_layer_weight():
    model_a = ConvBottleneckAEC(make_tiny_config(n_time_max=300))
    model_b = ConvBottleneckAEC(make_tiny_config(n_time_max=9999))
    shapes_a = {k: v.shape for k, v in model_a.state_dict().items()}
    shapes_b = {k: v.shape for k, v in model_b.state_dict().items()}
    assert shapes_a == shapes_b


def test_pad_mask_is_respected_through_downsample_and_upsample():
    model = ConvBottleneckAEC(make_tiny_config())
    T, real_len = 200, 150
    x = torch.randn(1, 1, T)
    pad_mask = torch.ones(1, 1, T)
    pad_mask[:, :, real_len:] = 0.0
    recon, anomaly_logits, feat = model(x, pad_mask=pad_mask)
    assert feat.shape[2] == T
    # padded tail of the restored feature map must be exactly zero
    assert torch.all(feat[:, :, real_len:] == 0.0)


def test_compute_num_stages_matches_known_examples():
    assert compute_num_stages(550, target_bottleneck_len=32, stride=2) == 4
    assert compute_num_stages(100000, target_bottleneck_len=32, stride=2) == 12


def test_compute_num_filters_doubles_from_base():
    filters = compute_num_filters(550, target_bottleneck_len=32, stride=2, channel_base=16, channel_max=128)
    assert filters == [16, 32, 64, 128]


def test_compute_num_filters_caps_at_channel_max_for_deep_networks():
    filters = compute_num_filters(100000, target_bottleneck_len=32, stride=2, channel_base=16, channel_max=128)
    assert len(filters) == 12
    assert max(filters) <= 128
    assert filters[-1] == 128


def test_config_auto_computes_num_filters_when_not_given():
    config = ConvBottleneckConfig(n_time_max=550)
    assert config.num_filters == [16, 32, 64, 128]


def test_config_respects_explicit_num_filters_override():
    config = ConvBottleneckConfig(n_time_max=550, num_filters=[8, 8])
    assert config.num_filters == [8, 8]


def test_stem_layer_preserves_length_before_downsampling_begins():
    encoder = ConvBottleneckEncoder(num_inputs=1, num_filters=[8, 16, 32], bottleneck_channels=4,
                                     kernel_size=3, stride=2, padding=1, num_stem_layers=1)
    x = torch.randn(2, 1, 137)
    feat, lengths, masks = encoder(x)
    assert lengths[0] == 137
    assert lengths[1] == 137  # stem: stride=1, no length change
    assert lengths[2] < lengths[1]  # first real downsampling stage


def test_encoder_exact_halving_with_kernel3_stride2_pad1():
    encoder = ConvBottleneckEncoder(num_inputs=1, num_filters=[16, 32, 64, 128], bottleneck_channels=4,
                                     kernel_size=3, stride=2, padding=1, num_stem_layers=1)
    x = torch.randn(1, 1, 550)
    feat, lengths, masks = encoder(x)
    assert lengths == [550, 550, 275, 138, 69, 35]


def test_encoder_uses_reflect_padding_mode_by_default():
    encoder = ConvBottleneckEncoder(num_inputs=1, num_filters=[8, 16], bottleneck_channels=4,
                                     kernel_size=3, stride=2, padding=1, num_stem_layers=1)
    assert encoder.stem[0].conv.padding_mode == "reflect"
    assert encoder.blocks[0].conv.padding_mode == "reflect"


def test_self_attention_block_preserves_shape():
    block = SelfAttentionBlock(channels=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 37)
    out = block(x)
    assert out.shape == x.shape


def test_self_attention_block_handles_key_padding_mask_without_nan():
    block = SelfAttentionBlock(channels=8, num_heads=2, dropout=0.0)
    x = torch.randn(1, 8, 10)
    key_padding_mask = torch.zeros(1, 10, dtype=torch.bool)
    key_padding_mask[:, 6:] = True  # True = ignore (padded), our pad_mask convention is inverted
    out = block(x, key_padding_mask=key_padding_mask)
    assert torch.isfinite(out).all()


def test_encoder_attaches_attention_only_at_stages_within_resolution_threshold():
    # num_filters has 4 stages; with n_time_max=550, kernel=3,stride=2,padding=1,
    # num_stem_layers=1, nominal lengths after each stage are 275,138,69,35
    # (matches test_encoder_exact_halving_with_kernel3_stride2_pad1). With
    # attention_max_resolution=256, only stages 1,2,3 (138,69,35) qualify --
    # stage 0 (275) does not.
    encoder = ConvBottleneckEncoder(
        num_inputs=1, num_filters=[16, 32, 64, 128], bottleneck_channels=4,
        kernel_size=3, stride=2, padding=1, num_stem_layers=1,
        n_time_max=550, attention_max_resolution=256,
    )
    assert set(encoder.attn_by_stage.keys()) == {"1", "2", "3"}


def test_encoder_attaches_no_attention_when_threshold_is_zero():
    encoder = ConvBottleneckEncoder(
        num_inputs=1, num_filters=[16, 32, 64, 128], bottleneck_channels=4,
        kernel_size=3, stride=2, padding=1, num_stem_layers=1,
        n_time_max=550, attention_max_resolution=0,
    )
    assert len(encoder.attn_by_stage) == 0


@pytest.mark.parametrize("T", [61, 137, 300, 550])
def test_full_model_with_attention_still_reconstructs_exact_length(T):
    config = make_tiny_config(num_filters=[16, 32, 64, 128], attention_max_resolution=256)
    model = ConvBottleneckAEC(config)
    x = torch.randn(2, 1, T)
    recon, anomaly_logits, feat = model(x)
    assert recon.shape == (2, 1, T)
    assert anomaly_logits.shape == (2, 1, T)


def test_full_model_with_attention_respects_pad_mask():
    config = make_tiny_config(num_filters=[16, 32, 64, 128], attention_max_resolution=256)
    model = ConvBottleneckAEC(config)
    T, real_len = 550, 400
    x = torch.randn(1, 1, T)
    pad_mask = torch.ones(1, 1, T)
    pad_mask[:, :, real_len:] = 0.0
    recon, anomaly_logits, feat = model(x, pad_mask=pad_mask)
    assert torch.isfinite(feat).all()
    assert torch.all(feat[:, :, real_len:] == 0.0)


def test_calculate_loss_matches_manual_masked_mean():
    model = ConvBottleneckAEC(make_tiny_config(bce_loss_ratio=0.3))
    torch.manual_seed(0)
    Y = torch.randn(2, 1, 60)
    recon = torch.randn(2, 1, 60)
    anomaly_logits = torch.randn(2, 1, 60)
    is_anomaly = (torch.rand(2, 1, 60) > 0.5).float()
    anomaly_mask = torch.ones(2, 1, 60)
    anomaly_mask[:, :, 3:6] = 0.0
    pad_mask = torch.ones(2, 1, 60)
    pad_mask[:, :, 50:] = 0.0

    loss, loss_ae, loss_c = model.calculate_loss(
        Y, recon, anomaly_logits, is_anomaly, anomaly_mask, pad_mask
    )

    mse_gate = anomaly_mask * pad_mask
    expected_ae = ((Y - recon) ** 2 * mse_gate).sum() / mse_gate.sum().clamp_min(1.0)
    bce_raw = F.binary_cross_entropy_with_logits(anomaly_logits, is_anomaly, reduction="none")
    expected_c = (bce_raw * pad_mask).sum() / pad_mask.sum().clamp_min(1.0)
    expected = (1 - 0.3) * expected_ae + 0.3 * expected_c

    assert torch.allclose(loss_ae, expected_ae)
    assert torch.allclose(loss_c, expected_c)
    assert torch.allclose(loss, expected)
