import pytest
import torch

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import (
    ATTRS,
    AttributeHead,
    ContrastiveEncoderV2,
    build_position_channel,
    count_parameters,
)


def make_tiny_config(**overrides):
    defaults = dict(
        n_time_max=200,
        n_features=2,
        num_filters=[8, 16, 32],
        kernel_size=3,
        stride=2,
        padding=1,
        dropout=0.0,
        normalization="group",
        num_groups=4,
        attention_max_resolution=0,
    )
    defaults.update(overrides)
    return ConvBottleneckConfig(**defaults)


# --- build_position_channel ---------------------------------------------

def test_build_position_channel_linear_ramp_within_valid_length():
    # L=5 valid steps out of T=8: position should be 0, .25, .5, .75, 1.0
    # at the valid steps, then exactly 0 in the padded tail.
    x = torch.randn(1, 1, 8)
    pad_mask = torch.zeros(1, 1, 8)
    pad_mask[:, :, :5] = 1.0

    pos = build_position_channel(x, pad_mask)

    assert pos.shape == (1, 1, 8)
    expected_valid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.allclose(pos[0, 0, :5], expected_valid, atol=1e-6)
    assert torch.all(pos[0, 0, 5:] == 0.0)


def test_build_position_channel_first_and_last_valid_step_are_0_and_1():
    x = torch.randn(2, 1, 20)
    pad_mask = torch.zeros(2, 1, 20)
    lengths = [20, 10]
    for i, length in enumerate(lengths):
        pad_mask[i, :, :length] = 1.0

    pos = build_position_channel(x, pad_mask)

    for i, length in enumerate(lengths):
        assert pos[i, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert pos[i, 0, length - 1].item() == pytest.approx(1.0, abs=1e-6)


def test_build_position_channel_single_valid_step_is_zero_not_nan():
    # L=1: no well-defined "first vs last" -- defined as position 0.0,
    # and must not divide by zero.
    x = torch.randn(1, 1, 5)
    pad_mask = torch.zeros(1, 1, 5)
    pad_mask[:, :, :1] = 1.0
    pos = build_position_channel(x, pad_mask)
    assert torch.isfinite(pos).all()
    assert pos[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_build_position_channel_defaults_to_full_length_when_no_mask():
    x = torch.randn(1, 1, 6)
    pos = build_position_channel(x, pad_mask=None)
    assert pos[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert pos[0, 0, 5].item() == pytest.approx(1.0, abs=1e-6)


# --- AttributeHead --------------------------------------------------------

def test_attribute_head_output_shape():
    head = AttributeHead(in_channels=16, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    feat = torch.randn(3, 16, 25)
    out = head(feat)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_attribute_head_ignores_padded_positions():
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    head.eval()
    real_len = 15
    feat = torch.randn(1, 8, real_len)
    feat_padded = torch.cat([feat, torch.zeros(1, 8, 10)], dim=2)
    pad_mask = torch.ones(1, 1, real_len + 10)
    pad_mask[:, :, real_len:] = 0.0

    feat_padded_garbage = feat_padded.clone()
    feat_padded_garbage[:, :, real_len:] = 5.0

    with torch.no_grad():
        out_zero = head(feat_padded, pad_mask=pad_mask)
        out_garbage = head(feat_padded_garbage, pad_mask=pad_mask)

    assert torch.allclose(out_zero, out_garbage, atol=1e-4)


def test_attribute_head_two_instances_have_independent_parameters():
    head_a = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    head_b = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    assert not torch.allclose(head_a.queries, head_b.queries)


# --- ContrastiveEncoderV2 --------------------------------------------------

def test_encoder_v2_produces_one_embedding_per_attribute():
    model = ContrastiveEncoderV2(make_tiny_config(), embedding_dim=8)
    x = torch.randn(4, 1, 137)
    emb = model(x)
    assert set(emb.keys()) == set(ATTRS)
    for name in ATTRS:
        assert emb[name].shape == (4, 8)
        assert torch.isfinite(emb[name]).all()


def test_encoder_v2_variable_length_batch_via_pad_mask():
    model = ContrastiveEncoderV2(make_tiny_config(), embedding_dim=8)
    T, real_len = 137, 90
    x = torch.randn(2, 1, T)
    pad_mask = torch.ones(2, 1, T)
    pad_mask[0, :, real_len:] = 0.0
    emb = model(x, pad_mask=pad_mask)
    for name in ATTRS:
        assert emb[name].shape == (2, 8)
        assert torch.isfinite(emb[name]).all()


def test_encoder_v2_heads_are_independent_projections():
    torch.manual_seed(0)
    model = ContrastiveEncoderV2(make_tiny_config(), embedding_dim=8)
    x = torch.randn(2, 1, 137)
    emb = model(x)
    assert not torch.allclose(emb["shape"], emb["location"])


def test_encoder_v2_trunk_has_no_squeeze_or_pool_query():
    # The whole point of V2: no shared Conv1d(128->4) squeeze, no shared
    # single-query attention pool/z=4 bottleneck.
    model = ContrastiveEncoderV2(make_tiny_config(), embedding_dim=8)
    assert model.encoder.squeeze is None
    assert not hasattr(model, "pool_query")
    assert not hasattr(model, "pool_attn")


def test_encoder_v2_input_stem_takes_two_channels():
    model = ContrastiveEncoderV2(make_tiny_config(), embedding_dim=8)
    assert model.encoder.stem[0].conv.in_channels == 2


# --- parameter counting ----------------------------------------------------

# --- gradient-flow sanity check (MTL_V2_REPORT.md Section 15) -------------

def test_gradient_flow_isolated_per_attribute_loss():
    # Backward from ONE attribute's own (scalar) loss must reach the shared
    # trunk and that attribute's own head, but must NOT reach any other
    # head's parameters -- the four AttributeHeads never share parameters
    # and are only connected through the shared trunk feature map, so a
    # single-attribute loss's computational graph never touches other
    # heads' weights at all (not just "zero gradient" -- literally absent
    # from the graph, hence allow_unused=True + assert is None).
    torch.manual_seed(0)
    model = ContrastiveEncoderV2(make_tiny_config(), embedding_dim=8)
    x = torch.randn(3, 1, 137)
    emb = model(x)
    trunk_params = list(model.encoder.parameters())

    for attr in ATTRS:
        own_params = list(model.attribute_heads[attr].parameters())
        other_params = [p for name in ATTRS if name != attr for p in model.attribute_heads[name].parameters()]
        loss = emb[attr].sum()

        grads = torch.autograd.grad(
            loss, trunk_params + own_params + other_params, retain_graph=True, allow_unused=True
        )
        trunk_grads = grads[:len(trunk_params)]
        own_grads = grads[len(trunk_params):len(trunk_params) + len(own_params)]
        other_grads = grads[len(trunk_params) + len(own_params):]

        assert any(g is not None and torch.any(g != 0) for g in trunk_grads), \
            f"{attr}: shared trunk received no gradient"
        assert all(g is not None for g in own_grads), f"{attr}: its own head has an untouched parameter"
        assert all(g is None for g in other_grads), f"{attr}: gradient leaked into another attribute's head"


def test_count_parameters_shared_trunk_dominates():
    # Production-scale trunk (not the deliberately tiny unit-test config
    # used elsewhere) -- the "shared trunk should dominate" expectation is
    # about the real V2 default, not an artificially shrunk trunk paired
    # with default-sized heads.
    config = ConvBottleneckConfig(n_time_max=550, n_features=2, attention_max_resolution=256)
    model = ContrastiveEncoderV2(config, embedding_dim=32)
    counts = count_parameters(model)
    assert counts["shared_trunk"] > 0
    assert counts["single_attribute_head"] > 0
    assert counts["all_attribute_heads"] == pytest.approx(counts["single_attribute_head"] * len(ATTRS), rel=0.05)
    assert counts["total"] == counts["shared_trunk"] + counts["all_attribute_heads"]
    assert counts["shared_ratio"] + counts["task_specific_ratio"] == pytest.approx(1.0)
    assert counts["shared_trunk"] > counts["all_attribute_heads"]
