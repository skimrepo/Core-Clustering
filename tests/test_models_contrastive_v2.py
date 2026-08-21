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


# --- V2.1: final embedding L2 normalization (config flag, default off) ----

def test_attribute_head_normalize_embedding_defaults_to_v2_behavior():
    # normalize_embedding=False (default) must be bit-identical to plain V2
    # -- no accidental behavior change for anyone not opting in.
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    assert head.normalize_embedding is False
    feat = torch.randn(3, 8, 20)
    out = head(feat)
    assert not torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-3)  # NOT unit-norm by default


def test_attribute_head_normalize_embedding_true_produces_unit_norm():
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          normalize_embedding=True)
    feat = torch.randn(5, 8, 20)
    out = head(feat)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_attribute_head_normalize_embedding_handles_exact_zero_raw_embedding():
    # Force the MLP's last layer to output exactly zero regardless of input
    # -- F.normalize's eps must keep this finite (0/eps = 0), never NaN/Inf.
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          normalize_embedding=True)
    with torch.no_grad():
        head.mlp[-1].weight.zero_()
        head.mlp[-1].bias.zero_()
    feat = torch.randn(2, 8, 20)
    out = head(feat)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_attribute_head_stashes_raw_pre_normalization_embedding_for_inspection():
    # Diagnostic-only introspection point (MTL_V21_REPORT.md Section 5/9's
    # raw-vs-normalized embedding norm tracking) -- must not affect the
    # public forward() contract or gradient flow, just expose the
    # pre-normalization value for a diagnostic script to read.
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          normalize_embedding=True)
    feat = torch.randn(4, 8, 20)
    out = head(feat)
    assert hasattr(head, "last_raw_embedding")
    assert head.last_raw_embedding.shape == out.shape
    assert not torch.allclose(head.last_raw_embedding.norm(dim=-1), torch.ones(4), atol=1e-3)


def test_encoder_v2_normalize_embedding_flag_applies_to_all_attributes_uniformly():
    config = ConvBottleneckConfig(n_time_max=200, n_features=2, num_filters=[8, 16, 32],
                                   attention_max_resolution=0)
    model = ContrastiveEncoderV2(config, embedding_dim=8, normalize_embedding=True)
    for attr in ATTRS:
        assert model.attribute_heads[attr].normalize_embedding is True
    x = torch.randn(3, 1, 137)
    emb = model(x)
    for attr in ATTRS:
        assert torch.allclose(emb[attr].norm(dim=-1), torch.ones(3), atol=1e-5)


def test_encoder_v2_gradient_flow_isolated_still_holds_with_normalize_embedding():
    torch.manual_seed(0)
    config = ConvBottleneckConfig(n_time_max=200, n_features=2, num_filters=[8, 16, 32],
                                   attention_max_resolution=0)
    model = ContrastiveEncoderV2(config, embedding_dim=8, normalize_embedding=True)
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
        assert any(g is not None and torch.any(g != 0) for g in trunk_grads)
        assert all(g is not None for g in own_grads)
        assert all(g is None for g in other_grads)


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


def test_attribute_head_pooling_defaults_to_multi_query_attention():
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    assert head.pooling == "multi_query_attention"
    assert hasattr(head, "pool_attn")
    assert not hasattr(head, "position_pool")


def test_attribute_head_unknown_pooling_raises():
    with pytest.raises(ValueError):
        AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                      pooling="bogus")


def test_attribute_head_position_aware_pooling_output_shape_and_finite():
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          pooling="position_aware")
    feat = torch.randn(3, 8, 25)
    out = head(feat)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_attribute_head_position_aware_ignores_padded_positions():
    # Same contract as the multi_query_attention pool's own test: appending
    # garbage into the padded tail (instead of zeros) must not change the
    # output at all -- padded positions must receive exactly zero attention
    # mass, not just a small amount.
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          pooling="position_aware")
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


def test_attribute_head_position_aware_no_nan_for_fully_padded_row():
    # Defensive edge case: a batch row whose pad_mask is entirely zero must
    # not produce NaN (an all -inf row before softmax) even though this
    # shouldn't occur for a real query in practice.
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          pooling="position_aware")
    feat = torch.randn(2, 8, 20)
    pad_mask = torch.ones(2, 1, 20)
    pad_mask[0] = 0.0  # row 0 is fully padded
    out = head(feat, pad_mask=pad_mask)
    assert torch.isfinite(out).all()


def test_attribute_head_position_aware_records_attention_weights_summing_to_one():
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          pooling="position_aware")
    real_len = 12
    feat = torch.randn(2, 8, 20)
    pad_mask = torch.ones(2, 1, 20)
    pad_mask[:, :, real_len:] = 0.0
    head(feat, pad_mask=pad_mask)
    assert head.last_attention_weights.shape == (2, 20)
    assert torch.allclose(head.last_attention_weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(head.last_attention_weights[:, real_len:], torch.zeros(2, 20 - real_len), atol=1e-6)


def test_position_aware_pool_position_summary_moves_with_attention_mass():
    # Test PositionAwarePool directly (bypassing AttributeHead's own 1x1
    # conv projection, whose random weights would otherwise scramble a
    # hand-crafted input channel) -- drives WHERE the learned score
    # concentrates by hand-setting score.weight/bias so a specific input
    # channel dominates the score, then checks attention (and therefore
    # position_summary) genuinely follows it to different timesteps.
    from core_clustering.models_contrastive_v2 import PositionAwarePool

    torch.manual_seed(0)
    pool = PositionAwarePool(channels=4, out_dim=8)
    T = 10
    h_t = torch.zeros(1, T, 4)
    h_t[0, :, 0] = torch.arange(T, dtype=torch.float32)  # channel 0 carries the timestep index
    with torch.no_grad():
        pool.score.weight.zero_()
        pool.score.weight[0, 0] = 10.0  # score = 10 * channel_0_value -> softmax favors the largest index
        pool.score.bias.zero_()

    pool(h_t)
    a_t = pool.last_attention_weights[0]
    assert a_t.argmax().item() == T - 1  # attention concentrates on the last timestep

    h_t2 = h_t.flip(dims=[1])  # now index 0 carries the largest channel-0 value
    pool(h_t2)
    a_t2 = pool.last_attention_weights[0]
    assert a_t2.argmax().item() == 0
    # attention mass genuinely moved from one end of the sequence to the other
    assert a_t.argmax().item() != a_t2.argmax().item()


def test_attribute_head_position_aware_gradients_propagate_to_score_and_project():
    torch.manual_seed(0)
    head = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                          pooling="position_aware")
    feat = torch.randn(3, 8, 20, requires_grad=True)
    out = head(feat)
    out.sum().backward()
    assert feat.grad is not None and torch.any(feat.grad != 0)
    assert head.position_pool.score.weight.grad is not None and torch.any(head.position_pool.score.weight.grad != 0)
    assert head.position_pool.project.weight.grad is not None and torch.any(head.position_pool.project.weight.grad != 0)


def test_attribute_head_position_aware_downstream_mlp_dimension_unchanged():
    # The whole point of the "project" step: the existing MLP must accept
    # EXACTLY the same input dimension (num_queries * proj_channels) it
    # always has, so nothing downstream of pooling needed to change.
    head_mqa = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8)
    head_pos = AttributeHead(in_channels=8, proj_channels=8, num_queries=4, mlp_hidden=16, embedding_dim=8,
                              pooling="position_aware")
    assert head_mqa.mlp[0].in_features == head_pos.mlp[0].in_features


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
