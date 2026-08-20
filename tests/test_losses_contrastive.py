import torch

from core_clustering.losses_contrastive import (
    MultiHeadContrastiveLoss,
    NormalRelativeRegressionLoss,
    PairwiseGapRegressionLoss,
    RadialOrdinalLoss,
    ShapeContrastiveLoss,
)


def test_shape_loss_is_lower_when_same_shape_pairs_are_closer():
    shape = torch.tensor([0, 0, 1, 1])
    loss_fn = ShapeContrastiveLoss()

    close_same = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])
    far_same = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])

    loss_close = loss_fn(close_same, shape)
    loss_far = loss_fn(far_same, shape)
    assert loss_close.item() < loss_far.item()


def test_shape_loss_has_a_learnable_temperature():
    loss_fn = ShapeContrastiveLoss()
    assert any(p.requires_grad for p in loss_fn.parameters())


def test_pairwise_gap_regression_pulls_small_gap_pairs_closer_than_large_gap_pairs():
    loss_fn = PairwiseGapRegressionLoss()
    # instance 0&1 have a tiny location gap, 0&2 have a huge one; but right
    # now all three sit at the SAME embedding distance from each other --
    # loss should push toward making D_01 << D_02, not just D_01 < D_02.
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    location = torch.tensor([0.50, 0.51, 0.99])  # gap(0,1)=0.01, gap(0,2)=0.49
    valid_mask = torch.ones(3, 3, dtype=torch.bool)

    loss = loss_fn(embeddings, location, valid_mask)
    loss.backward()
    # embedding 1 (tiny gap partner) should feel a much stronger pull toward
    # embedding 0 than embedding 2 (huge gap partner) does, since the
    # regression target for pair (0,1) is far below the current distance=1
    # while pair (0,2)'s target is much closer to 1.
    assert embeddings.grad[1].abs().sum() > embeddings.grad[2].abs().sum()


def test_pairwise_gap_regression_penalizes_full_collapse():
    # Collapsing every embedding to the same point must NOT be a free way
    # to minimize this loss -- with no learnable scale to shrink alongside
    # the embeddings, a fixed nonzero gap always costs (0 - gap)**2 > 0.
    loss_fn = PairwiseGapRegressionLoss()
    collapsed = torch.zeros(3, 2)
    location = torch.tensor([0.1, 0.5, 0.9])
    valid_mask = torch.ones(3, 3, dtype=torch.bool)
    loss = loss_fn(collapsed, location, valid_mask)
    assert loss.item() > 0.01


def test_pairwise_gap_regression_respects_valid_mask():
    loss_fn = PairwiseGapRegressionLoss()
    embeddings = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    location = torch.tensor([0.0, 1.0])
    no_valid = torch.zeros(2, 2, dtype=torch.bool)
    loss = loss_fn(embeddings, location, no_valid)
    assert loss.item() == 0.0


def test_normal_relative_regression_pulls_normal_cluster_together():
    loss_fn = NormalRelativeRegressionLoss()
    is_anomalous = torch.tensor([False, False, True])
    value = torch.tensor([-1.0, -1.0, 0.3])

    # Both cases share the SAME centroid (0,0) and the SAME anomaly position,
    # so the regression term is identical in both -- only normal_pull differs,
    # cleanly isolating what this test checks.
    tight_normals = torch.tensor([[-0.05, 0.0], [0.05, 0.0], [5.0, 0.0]], requires_grad=True)
    loose_normals = torch.tensor([[-3.0, 0.0], [3.0, 0.0], [5.0, 0.0]], requires_grad=True)

    loss_tight = loss_fn(tight_normals, is_anomalous, value)
    loss_loose = loss_fn(loose_normals, is_anomalous, value)
    assert loss_tight.item() < loss_loose.item()


def test_normal_relative_regression_wants_larger_value_farther_from_centroid():
    loss_fn = NormalRelativeRegressionLoss()
    is_anomalous = torch.tensor([False, False, True, True])
    value = torch.tensor([-1.0, -1.0, 0.1, 0.9])  # small vs large extent

    # centroid sits at (0,0). idx 2 (small value=0.1, wants to be CLOSE) is
    # placed far away (10) -- overshooting, gradient descent should pull it
    # in (a positive gradient here, since param -= lr*grad decreases a
    # positive coordinate). idx 3 (large value=0.9, wants to be FAR) is
    # placed close (0.2) -- undershooting, gradient should push it out
    # (a negative gradient, since param -= lr*grad increases it).
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.2, 0.0]], requires_grad=True
    )
    loss = loss_fn(embeddings, is_anomalous, value)
    loss.backward()
    assert embeddings.grad[2, 0].item() > 0
    assert embeddings.grad[3, 0].item() < 0


def test_normal_relative_regression_penalizes_anomalies_collapsing_onto_normal():
    # An anomaly sitting exactly at the normal centroid must not be free --
    # with no learnable scale, its distance target is pinned to its own
    # (nonzero) value, so collapsing onto normal costs (0 - value)**2 > 0.
    loss_fn = NormalRelativeRegressionLoss()
    is_anomalous = torch.tensor([False, False, True])
    value = torch.tensor([-1.0, -1.0, 0.5])
    collapsed = torch.zeros(3, 2)
    loss = loss_fn(collapsed, is_anomalous, value)
    assert loss.item() > 0.01


def test_normal_relative_regression_handles_intensity_values_below_one():
    # Intensity's real range (~0.2 to 4.0 std-multiplier) straddles 1.0.
    # An earlier log-transformed version of this loss used log(value) as
    # the distance target directly -- log(0.5) is NEGATIVE, and a norm can
    # never match a negative target, silently breaking the "smaller value
    # = closer to normal" ordering for every value below 1. Using the raw
    # value (no log) sidesteps this entirely: targets stay positive across
    # the whole real range.
    loss_fn = NormalRelativeRegressionLoss()
    is_anomalous = torch.tensor([False, False, True, True])
    value = torch.tensor([-1.0, -1.0, 0.3, 2.5])  # both realistic intensities, 0.3 < 1.0 < 2.5
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [0.1, 0.0]], requires_grad=True
    )
    loss = loss_fn(embeddings, is_anomalous, value)
    assert torch.isfinite(loss).all()
    loss.backward()
    # idx 2 (small value=0.3, overshooting at distance 5) pulled in (positive grad)
    assert embeddings.grad[2, 0].item() > 0
    # idx 3 (large value=2.5, undershooting at distance 0.1) pushed out (negative grad)
    assert embeddings.grad[3, 0].item() < 0


# --- RadialOrdinalLoss (V2.3) ----------------------------------------------

def _make_embeddings_1d(normal_positions, anomaly_positions):
    # 1D embeddings so distance from centroid is just |x - c| -- makes the
    # desired severities exactly controllable in a test without needing to
    # reason about multi-dim geometry.
    positions = normal_positions + anomaly_positions
    return torch.tensor([[p] for p in positions], dtype=torch.float32)


def test_radial_ordinal_correct_ordering_has_lower_loss_than_reversed():
    loss_fn = RadialOrdinalLoss()
    is_anomalous = torch.tensor([False, False, True, True])
    value = torch.tensor([0.0, 0.0, 1.0, 3.0])  # anomaly[0] value=1 < anomaly[1] value=3

    # normals at 0.0/0.2 -> centroid=0.1. CORRECT: severity(1.0)=0.5 < severity(3.0)=1.2
    correct = _make_embeddings_1d([0.0, 0.2], [0.6, 1.3])  # |0.6-0.1|=0.5, |1.3-0.1|=1.2
    # REVERSED: severity(1.0)=1.2 > severity(3.0)=0.5 -- wrong order
    reversed_ = _make_embeddings_1d([0.0, 0.2], [1.3, 0.6])

    loss_correct = loss_fn(correct, is_anomalous, value)
    loss_reversed = loss_fn(reversed_, is_anomalous, value)
    assert loss_correct.item() < loss_reversed.item()


def test_radial_ordinal_is_invariant_to_raw_target_scale():
    # Proves the ranking component uses ORDER only, not raw magnitude --
    # proportionally rescaling the anomalous targets must not change the
    # loss at all (same embeddings, same relative order, no new ties).
    loss_fn = RadialOrdinalLoss()
    is_anomalous = torch.tensor([False, True, True])
    embeddings = _make_embeddings_1d([0.0], [0.5, 1.2])

    small_scale = torch.tensor([0.0, 1.0, 2.0])
    large_scale = torch.tensor([0.0, 100.0, 200.0])

    loss_small = loss_fn(embeddings, is_anomalous, small_scale)
    loss_large = loss_fn(embeddings, is_anomalous, large_scale)
    assert torch.isclose(loss_small, loss_large, atol=1e-6)


def test_radial_ordinal_excludes_equal_target_pairs():
    # Two anomalies with the SAME raw value must contribute nothing to the
    # ranking term regardless of how far apart they actually sit -- moving
    # the tied pair around must not change the loss, as long as each one's
    # OWN severity (hence its pairing with the normal, whose target differs)
    # stays fixed.
    loss_fn = RadialOrdinalLoss()
    is_anomalous = torch.tensor([False, True, True])
    value = torch.tensor([0.0, 2.0, 2.0])  # tied anomaly targets

    normal = [[0.0, 0.0]]
    close = torch.tensor(normal + [[0.5, 0.0], [0.5, 0.0]])  # anomalies coincide (severity 0.5 each)
    far = torch.tensor(normal + [[0.5, 0.0], [-0.5, 0.0]])  # anomalies far apart, same severity (0.5 each)

    loss_close = loss_fn(close, is_anomalous, value)
    loss_far = loss_fn(far, is_anomalous, value)
    assert torch.isclose(loss_close, loss_far, atol=1e-6)


def test_radial_ordinal_normal_pull_matches_existing_convention():
    # Same normal-clustering term as NormalRelativeRegressionLoss: tighter
    # normal cluster -> lower loss, holding anomaly placement fixed.
    loss_fn = RadialOrdinalLoss()
    is_anomalous = torch.tensor([False, False, True])
    value = torch.tensor([0.0, 0.0, 1.0])

    tight_normals = _make_embeddings_1d([-0.05, 0.05], [5.0])
    loose_normals = _make_embeddings_1d([-3.0, 3.0], [5.0])

    loss_tight = loss_fn(tight_normals, is_anomalous, value)
    loss_loose = loss_fn(loose_normals, is_anomalous, value)
    assert loss_tight.item() < loss_loose.item()


def test_radial_ordinal_no_nan_or_inf():
    loss_fn = RadialOrdinalLoss()
    torch.manual_seed(0)
    embeddings = torch.randn(8, 4)
    is_anomalous = torch.tensor([False, False, False, True, True, True, True, True])
    value = torch.tensor([0.0, 0.0, 0.0, 0.1, 1.0, 5.0, 50.0, 0.1])  # includes an exact tie
    loss = loss_fn(embeddings, is_anomalous, value)
    assert torch.isfinite(loss)


def test_radial_ordinal_gradient_reaches_embeddings():
    loss_fn = RadialOrdinalLoss()
    torch.manual_seed(0)
    embeddings = torch.randn(6, 4, requires_grad=True)
    is_anomalous = torch.tensor([False, False, True, True, True, True])
    value = torch.tensor([0.0, 0.0, 0.5, 1.0, 2.0, 4.0])
    loss = loss_fn(embeddings, is_anomalous, value)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.any(embeddings.grad != 0)


def test_multi_head_loss_intensity_objective_defaults_to_radial_regression():
    loss_fn = MultiHeadContrastiveLoss()
    assert isinstance(loss_fn.intensity_loss, NormalRelativeRegressionLoss)


def test_multi_head_loss_intensity_objective_can_select_radial_ordinal():
    loss_fn = MultiHeadContrastiveLoss(intensity_objective="radial_ordinal")
    assert isinstance(loss_fn.intensity_loss, RadialOrdinalLoss)


def test_multi_head_loss_combines_all_four_terms_with_weights():
    loss_fn = MultiHeadContrastiveLoss(weights=(2.0, 1.0, 1.0, 1.0))
    n = 6
    embeddings = {
        "shape": torch.randn(n, 3),
        "location": torch.randn(n, 3),
        "extent": torch.randn(n, 3),
        "intensity": torch.randn(n, 3),
    }
    shape = torch.tensor([0, 0, 0, 1, 1, 1])
    location = torch.tensor([-1.0, -1.0, -1.0, 0.2, 0.4, 0.8])
    extent = torch.tensor([-1.0, -1.0, -1.0, 0.1, 0.2, 0.4])
    intensity = torch.tensor([-1.0, -1.0, -1.0, 0.3, 1.0, 3.0])

    total, comp = loss_fn(embeddings, shape, location, extent, intensity)
    expected = (
        2.0 * comp["loss_shape"] + 1.0 * comp["loss_location"]
        + 1.0 * comp["loss_extent"] + 1.0 * comp["loss_intensity"]
    )
    assert torch.isclose(total, expected)
    for key in ("loss_shape", "loss_location", "loss_extent", "loss_intensity"):
        assert key in comp


def test_multi_head_loss_compute_components_returns_raw_differentiable_losses():
    # Needed so the trainer can backward() each attribute's loss
    # separately (for per-attribute optimizer state on the shared trunk)
    # instead of only ever seeing the combined, weighted, detached total.
    loss_fn = MultiHeadContrastiveLoss(weights=(2.0, 1.0, 1.0, 1.0))
    n = 6
    embeddings = {
        "shape": torch.randn(n, 3, requires_grad=True),
        "location": torch.randn(n, 3, requires_grad=True),
        "extent": torch.randn(n, 3, requires_grad=True),
        "intensity": torch.randn(n, 3, requires_grad=True),
    }
    shape = torch.tensor([0, 0, 0, 1, 1, 1])
    location = torch.tensor([-1.0, -1.0, -1.0, 0.2, 0.4, 0.8])
    extent = torch.tensor([-1.0, -1.0, -1.0, 0.1, 0.2, 0.4])
    intensity = torch.tensor([-1.0, -1.0, -1.0, 0.3, 1.0, 3.0])

    comp = loss_fn.compute_components(embeddings, shape, location, extent, intensity)
    assert set(comp.keys()) == {"shape", "location", "extent", "intensity"}
    for key in comp:
        assert comp[key].requires_grad
    comp["shape"].backward()
    assert embeddings["shape"].grad is not None
    assert embeddings["location"].grad is None  # backward on shape alone shouldn't touch other heads


def test_multi_head_loss_exposes_shape_temperature_for_the_optimizer():
    # location/extent/intensity no longer carry a learnable scale (that was
    # redundant with each head's own Linear weights, and let embeddings
    # collapse alongside a shrinking scale) -- only shape's temperature
    # remains a loss-owned learnable parameter.
    loss_fn = MultiHeadContrastiveLoss()
    params = list(loss_fn.parameters())
    assert len(params) == 1
    assert params[0] is loss_fn.shape_loss.log_temperature
