import torch

from core_clustering.models_contrastive_v2 import AttributeHead
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ATTRS, ContrastiveEncoderV3


def make_tiny_config(**overrides):
    defaults = dict(
        n_time_max=200, n_features=2, num_filters=[8, 16, 32], attention_max_resolution=0,
    )
    defaults.update(overrides)
    return ConvBottleneckConfig(**defaults)


def test_v3_reuses_attribute_head_class_unchanged():
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    for name in ATTRS:
        assert isinstance(model.attribute_heads[name], AttributeHead)


def test_v3_forward_k0_produces_zero_gate_and_valid_shapes():
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    x = torch.randn(3, 1, 137)
    out = model(x)

    assert torch.all(out["gate"] == 0.0)
    for name in ATTRS:
        assert out["embeddings"][name].shape == (3, 8)
        assert torch.isfinite(out["embeddings"][name]).all()
    for name in ("location", "extent"):
        assert out[f"{name}_mu"].shape == (3,)
        assert torch.all(out[f"{name}_mu"] >= 0) and torch.all(out[f"{name}_mu"] <= 1)
        assert torch.all(out[f"{name}_scale"] > 0)
    assert torch.all(out["intensity_mu"] >= 0)
    assert torch.all(out["intensity_scale"] > 0)
    assert out["shape_scale"].shape == (3,)
    assert torch.all(out["shape_scale"] > 0)


def test_v3_forward_with_references_changes_embeddings_vs_k0():
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    model.eval()
    x = torch.randn(2, 1, 137)
    K = 5
    ref_x = torch.randn(2, K, 1, 137)

    with torch.no_grad():
        out_k0 = model(x)
        out_kref = model(x, ref_x=ref_x)

    assert torch.all(out_kref["gate"] > 0.0)
    assert not torch.allclose(out_k0["embeddings"]["intensity"], out_kref["embeddings"]["intensity"])
    assert out_kref["reference_weights"].shape == (2, K)
    assert torch.allclose(out_kref["reference_weights"].sum(dim=1), torch.ones(2), atol=1e-5)


def test_v3_forward_with_reference_padding_mask_respected():
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    B, K, T = 2, 3, 137
    real_len = 100
    x = torch.randn(B, 1, T)
    ref_x = torch.randn(B, K, 1, T)
    ref_pad_mask = torch.ones(B, K, 1, T)
    ref_pad_mask[:, :, :, real_len:] = 0.0

    out = model(x, ref_x=ref_x, ref_pad_mask=ref_pad_mask)
    for name in ATTRS:
        assert torch.isfinite(out["embeddings"][name]).all()


def test_v3_intensity_prediction_unbounded_unlike_v1_normalized_distance():
    # The intensity EMBEDDING stays L2-normalized (bounded to the unit
    # sphere) regardless of input scale -- so varying the raw input alone
    # can't demonstrate unboundedness (that's a property of learned weight
    # magnitude, not input scale). What must be true structurally is that
    # NOTHING in the adapter path clamps/bounds mu the way raw normalized-
    # embedding distance was mathematically capped at ~2 -- scaling up the
    # adapter's own (learnable) weights must be enough by itself to exceed
    # that old ceiling from a perfectly ordinary unit-norm embedding.
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    with torch.no_grad():
        model.scalar_adapters["intensity"].linear.weight.mul_(50.0)
        model.scalar_adapters["intensity"].linear.bias.fill_(50.0)
    x = torch.randn(4, 1, 137)
    out = model(x)
    assert out["intensity_mu"].max().item() > 2.0


def test_v3_forward_with_self_attention_and_mixed_k0_padding_does_not_nan():
    # Regression test: with self-attention ENABLED (attention_max_resolution
    # > 0), a batch mixing K=0 items with K>0 items pads the K=0 rows' K
    # slots with pure-zero placeholders (ref_k_valid_mask=0 there). Those
    # placeholder sequences are 100% padding -- if fed through the trunk's
    # self-attention, EVERY key is masked, and softmax over an all-masked
    # row produces NaN internally, silently corrupting the whole batch
    # (shared cross-sample losses like Shape's cdist/logsumexp couple every
    # row together). The model must never run genuinely all-padding
    # reference slots through the trunk.
    torch.manual_seed(0)
    config = make_tiny_config(num_filters=[8, 16, 32, 64], attention_max_resolution=64)
    model = ContrastiveEncoderV3(config, embedding_dim=8)
    B, max_k, T = 4, 5, 137
    x = torch.randn(B, 1, T)
    ref_x = torch.randn(B, max_k, 1, T)
    ref_pad_mask = torch.ones(B, max_k, 1, T)
    # item 0: K=0 (all max_k slots are pure padding); items 1-3: K=5 (all real)
    ref_k_valid_mask = torch.tensor([[0.0] * max_k, [1.0] * max_k, [1.0] * max_k, [1.0] * max_k])
    ref_pad_mask[0] = 0.0  # K=0 item's placeholder slots really are all-zero/all-padding

    out = model(x, ref_x=ref_x, ref_pad_mask=ref_pad_mask, ref_k_valid_mask=ref_k_valid_mask)
    for name in ATTRS:
        assert torch.isfinite(out["embeddings"][name]).all()
    assert out["gate"][0].item() == 0.0


def test_v3_forward_no_grad_with_fully_padded_reference_slots_does_not_nan():
    # Regression test for a real bug found via full-scale training: under
    # torch.no_grad() specifically (not under normal grad-tracked forward),
    # PyTorch's fused attention kernel dispatch can differ for a FULLY
    # masked row (every key padded) and produce NaN, unlike the grad-
    # tracked path. This silently corrupted an entire validation epoch
    # (every loss NaN, immediate early stop) even though the identical
    # forward pass under grad-tracking was completely finite. Reproduces
    # at production scale (n_time_max=550, attention enabled, K=100 with
    # many items having fewer real references than the batch's max K, so
    # most of their reference slots are pure padding).
    torch.manual_seed(0)
    config = ConvBottleneckConfig(n_time_max=550, n_features=2, attention_max_resolution=256)
    model = ContrastiveEncoderV3(config, embedding_dim=32)
    model.eval()

    B, max_k, T = 5, 100, 550
    ks = [0, 3, 10, 30, 100]
    x = torch.randn(B, 1, T)
    ref_x = torch.randn(B, max_k, 1, T)
    ref_pad_mask = torch.ones(B, max_k, 1, T)
    ref_k_valid_mask = torch.zeros(B, max_k)
    for i, k in enumerate(ks):
        ref_k_valid_mask[i, :k] = 1.0
        ref_pad_mask[i, k:] = 0.0

    with torch.no_grad():
        out = model(x, ref_x=ref_x, ref_pad_mask=ref_pad_mask, ref_k_valid_mask=ref_k_valid_mask)
    for name in ATTRS:
        assert torch.isfinite(out["embeddings"][name]).all(), f"{name} embedding has NaN/Inf under no_grad"
    assert torch.isfinite(out["gate"]).all()


def test_v3_forward_with_variable_k_per_item_via_k_valid_mask():
    # Batch mixing K=0, K=1, K=3 items, padded up to max_k=3 -- the model
    # must handle this without NaN and must zero-gate the K=0 row exactly.
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    B, max_k, T = 3, 3, 137
    x = torch.randn(B, 1, T)
    ref_x = torch.randn(B, max_k, 1, T)
    ref_k_valid_mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    out = model(x, ref_x=ref_x, ref_k_valid_mask=ref_k_valid_mask)
    for name in ATTRS:
        assert torch.isfinite(out["embeddings"][name]).all()
    assert out["gate"][0].item() == 0.0
    assert out["gate"][1].item() >= 0.0
    assert out["gate"][2].item() >= 0.0


def test_v3_default_detach_scale_attrs_is_empty_and_matches_v31_behavior():
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    assert model.detach_scale_attrs == ()


def test_v3_default_location_pooling_is_multi_query_attention():
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    assert model.attribute_heads["location"].pooling == "multi_query_attention"


def test_v3_location_position_aware_pooling_flag_applies_only_to_location():
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8, location_position_aware_pooling=True)
    assert model.attribute_heads["location"].pooling == "position_aware"
    for name in ("shape", "extent", "intensity"):
        assert model.attribute_heads[name].pooling == "multi_query_attention"


def test_v3_location_position_aware_pooling_forward_works_at_k0_and_with_references():
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8, location_position_aware_pooling=True)
    x = torch.randn(3, 1, 137)
    out_k0 = model(x)
    assert torch.isfinite(out_k0["embeddings"]["location"]).all()

    ref_x = torch.randn(3, 5, 1, 137)
    out_kref = model(x, ref_x=ref_x)
    assert torch.isfinite(out_kref["embeddings"]["location"]).all()


def test_v3_intensity_scale_stop_gradient_blocks_z_but_not_scale_adapter_params():
    # V3.2 change #1: with detach_scale_attrs=("intensity",), the scale
    # branch must NOT backpropagate into the Intensity embedding (or
    # anything upstream of it -- the head, the trunk), while (a) mu's
    # gradient into the embedding must still be nonzero and (b) the scale
    # adapter's OWN parameters must still receive gradient (it must keep
    # training, just not shape the shared representation).
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8, detach_scale_attrs=("intensity",))
    x = torch.randn(3, 1, 137)
    out = model(x)

    z = out["embeddings"]["intensity"]
    z.retain_grad()
    mu, scale = out["intensity_mu"], out["intensity_scale"]

    trunk_params = list(model.encoder.parameters())
    scale_adapter_params = list(model.scalar_adapters["intensity"].parameters())

    # A. mu path -> z: nonzero gradient.
    (g_z_mu,) = torch.autograd.grad(mu.sum(), z, retain_graph=True)
    assert torch.any(g_z_mu != 0)

    # B. scale path -> z: exactly zero (scale's graph never touches z at
    # all once detached, so autograd.grad must report it as unconnected).
    g_z_scale = torch.autograd.grad(scale.sum(), z, retain_graph=True, allow_unused=True)[0]
    assert g_z_scale is None or torch.all(g_z_scale == 0)

    # scale adapter's own parameters still receive gradient from the scale path.
    g_scale_params = torch.autograd.grad(scale.sum(), scale_adapter_params, retain_graph=True, allow_unused=True)
    assert all(g is not None and torch.any(g != 0) for g in g_scale_params)

    # C. shared trunk receives Intensity gradient only via the mu path.
    g_trunk_from_scale = torch.autograd.grad(scale.sum(), trunk_params, retain_graph=True, allow_unused=True)
    assert all(g is None or torch.all(g == 0) for g in g_trunk_from_scale)
    g_trunk_from_mu = torch.autograd.grad(mu.sum(), trunk_params, retain_graph=True, allow_unused=True)
    assert any(g is not None and torch.any(g != 0) for g in g_trunk_from_mu)


def test_v3_location_and_extent_scale_unaffected_by_detach_scale_attrs():
    # detach_scale_attrs must be scoped to exactly the named attributes --
    # Location/Extent's scale must still backprop into their own embeddings
    # exactly as in V3.1, confirming this is not a global behavior change.
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8, detach_scale_attrs=("intensity",))
    x = torch.randn(3, 1, 137)
    out = model(x)
    for name in ("location", "extent"):
        z = out["embeddings"][name]
        scale = out[f"{name}_scale"]
        (g_z_scale,) = torch.autograd.grad(scale.sum(), z, retain_graph=True)
        assert torch.any(g_z_scale != 0)


def test_v3_gradient_flow_isolated_per_attribute_with_and_without_references():
    torch.manual_seed(0)
    model = ContrastiveEncoderV3(make_tiny_config(), embedding_dim=8)
    x = torch.randn(3, 1, 137)
    K = 4
    ref_x = torch.randn(3, K, 1, 137)

    for ref_kwargs in ({}, {"ref_x": ref_x}):
        out = model(x, **ref_kwargs)
        emb = out["embeddings"]
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
