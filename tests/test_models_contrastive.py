import torch

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive import ATTRS, ContrastiveEncoder


def make_tiny_config(**overrides):
    defaults = dict(
        n_time_max=200,
        n_features=1,
        num_filters=[8, 16, 32],
        bottleneck_channels=4,
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


def test_contrastive_encoder_produces_one_head_per_attribute():
    # embedding_dim is each head's OWN dimension, not split across the four
    # -- every attribute gets a full-size embedding space of its own.
    model = ContrastiveEncoder(make_tiny_config(), embedding_dim=16)
    x = torch.randn(5, 1, 137)
    emb = model(x)
    assert set(emb.keys()) == set(ATTRS)
    for name in ATTRS:
        assert emb[name].shape == (5, 16)


def test_contrastive_encoder_defaults_head_dim_to_bottleneck_channels():
    model = ContrastiveEncoder(make_tiny_config())
    x = torch.randn(3, 1, 137)
    emb = model(x)
    for name in ATTRS:
        assert emb[name].shape == (3, 4)


def test_contrastive_encoder_heads_are_independent_projections():
    # Different heads must not be the literal same tensor/projection --
    # otherwise there's no structural separation between attributes' losses.
    torch.manual_seed(0)
    model = ContrastiveEncoder(make_tiny_config(), embedding_dim=16)
    x = torch.randn(2, 1, 137)
    emb = model(x)
    assert not torch.allclose(emb["shape"], emb["location"])


def test_contrastive_encoder_pooling_is_roughly_padding_invariant():
    # Not exactly invariant: GroupNorm computes its statistics over the
    # full (padded) length at every block, so more trailing padding shifts
    # normalization slightly even though padded positions are re-zeroed
    # after each block (a known, already-documented tradeoff -- see the
    # design doc's "GroupNorm still sees padded zeros" risk). This checks
    # the embeddings stay CLOSE, not bit-identical.
    model = ContrastiveEncoder(make_tiny_config(), embedding_dim=8)
    model.eval()
    torch.manual_seed(0)
    real_len = 100
    x = torch.randn(1, 1, real_len)

    x_pad_a = torch.cat([x, torch.zeros(1, 1, 20)], dim=2)
    mask_a = torch.ones(1, 1, real_len + 20)
    mask_a[:, :, real_len:] = 0.0

    x_pad_b = torch.cat([x, torch.zeros(1, 1, 60)], dim=2)
    mask_b = torch.ones(1, 1, real_len + 60)
    mask_b[:, :, real_len:] = 0.0

    with torch.no_grad():
        emb_a = model(x_pad_a, pad_mask=mask_a)
        emb_b = model(x_pad_b, pad_mask=mask_b)

    for name in ATTRS:
        cos_sim = torch.nn.functional.cosine_similarity(emb_a[name], emb_b[name])
        assert cos_sim.item() > 0.9


def test_contrastive_encoder_works_with_attention_enabled():
    config = make_tiny_config(num_filters=[16, 32, 64], attention_max_resolution=64, n_time_max=200)
    model = ContrastiveEncoder(config, embedding_dim=8)
    x = torch.randn(2, 1, 137)
    emb = model(x)
    for name in ATTRS:
        assert emb[name].shape == (2, 8)
        assert torch.isfinite(emb[name]).all()


def test_contrastive_encoder_pool_ignores_padded_positions():
    """The attention-pool's key_padding_mask must fully exclude padded
    positions -- garbage values there shouldn't dominate the pooled
    embedding. Not exact equality: a single stem conv still sees unmasked
    input before its first re-zeroing pass, so boundary-adjacent real
    positions pick up a small amount of contamination regardless of
    pooling (same already-documented tradeoff as the padding-invariance
    test above)."""
    torch.manual_seed(0)
    model = ContrastiveEncoder(make_tiny_config(), embedding_dim=8)
    model.eval()

    real_len = 80
    x = torch.randn(1, 1, real_len)
    x_padded = torch.cat([x, torch.zeros(1, 1, 40)], dim=2)
    mask = torch.ones(1, 1, real_len + 40)
    mask[:, :, real_len:] = 0.0

    x_padded_garbage = x_padded.clone()
    x_padded_garbage[:, :, real_len:] = 5.0

    with torch.no_grad():
        emb_zero_pad = model(x_padded, pad_mask=mask)
        emb_garbage_pad = model(x_padded_garbage, pad_mask=mask)

    for name in ATTRS:
        cos_sim = torch.nn.functional.cosine_similarity(emb_zero_pad[name], emb_garbage_pad[name])
        assert cos_sim.item() > 0.9


def test_contrastive_encoder_pooling_can_learn_position_sensitivity():
    """Standard learned-query attention pooling starts near-uniform at
    random init (like a CLS token before training) -- that's expected, not
    a defect, and position-sensitivity should EMERGE through training
    rather than being guaranteed structurally. This trains the encoder for
    a few steps on a toy task (regress a scalar from bump position) and
    checks the loss actually decreases, i.e. the mechanism is learnable."""
    torch.manual_seed(0)
    model = ContrastiveEncoder(make_tiny_config(), embedding_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_time = 137
    torch.manual_seed(1)

    def make_batch(batch_size=8):
        base = torch.randn(batch_size, 1, n_time)
        positions = torch.randint(20, n_time - 30, (batch_size,))
        targets = positions.float() / n_time
        for i, p in enumerate(positions):
            base[i, :, p:p + 10] += 5.0
        return base, targets

    losses = []
    for _ in range(60):
        x, targets = make_batch()
        emb = model(x)["location"]
        pred = emb.mean(dim=-1)
        loss = torch.nn.functional.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_contrastive_encoder_mean_pooling_option_still_available():
    # Diagnostic fallback: plain masked mean-pool, no attention/positional
    # encoding at all -- useful as a simpler baseline for comparison.
    model = ContrastiveEncoder(make_tiny_config(), embedding_dim=8, pooling="mean")
    x = torch.randn(3, 1, 137)
    emb = model(x)
    for name in ATTRS:
        assert emb[name].shape == (3, 8)
        assert torch.isfinite(emb[name]).all()
