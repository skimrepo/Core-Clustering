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
