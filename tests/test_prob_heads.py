import math

import pytest
import torch

from core_clustering.prob_heads import (
    ScalarPredictionAdapter,
    ShapeUncertaintyAdapter,
    heteroscedastic_weight,
    laplace_nll,
)


def test_scalar_adapter_sigmoid_link_bounds_mu_to_unit_interval():
    torch.manual_seed(0)
    adapter = ScalarPredictionAdapter(embedding_dim=8, link="sigmoid")
    emb = torch.randn(5, 8) * 10  # deliberately large magnitude
    mu, scale = adapter(emb)
    assert torch.all(mu >= 0.0) and torch.all(mu <= 1.0)
    assert torch.all(scale > 0.0)


def test_scalar_adapter_softplus_link_is_unbounded_nonnegative():
    torch.manual_seed(0)
    adapter = ScalarPredictionAdapter(embedding_dim=8, link="softplus")
    emb = torch.randn(200, 8) * 50  # push toward the tail
    mu, scale = adapter(emb)
    assert torch.all(mu >= 0.0)
    assert torch.all(scale > 0.0)
    # unbounded: with enough spread in the input, mu should exceed 2 somewhere
    # (V1-era intensity's normalized-distance ceiling was ~2 -- this proves
    # the adapter itself imposes no such ceiling).
    assert mu.max().item() > 2.0


def test_scalar_adapter_invalid_link_raises():
    with pytest.raises(ValueError):
        ScalarPredictionAdapter(embedding_dim=8, link="not_a_real_link")


def test_scalar_adapter_two_instances_have_independent_parameters():
    a = ScalarPredictionAdapter(embedding_dim=8, link="sigmoid")
    b = ScalarPredictionAdapter(embedding_dim=8, link="sigmoid")
    assert not torch.allclose(a.linear.weight, b.linear.weight)


def test_scalar_adapter_gradient_reaches_embedding():
    adapter = ScalarPredictionAdapter(embedding_dim=8, link="softplus")
    emb = torch.randn(4, 8, requires_grad=True)
    mu, scale = adapter(emb)
    (mu.sum() + scale.sum()).backward()
    assert emb.grad is not None and torch.any(emb.grad != 0)


def test_shape_uncertainty_adapter_outputs_positive_scale():
    torch.manual_seed(0)
    adapter = ShapeUncertaintyAdapter(embedding_dim=8)
    emb = torch.randn(6, 8) * 20
    b = adapter(emb)
    assert b.shape == (6,)
    assert torch.all(b > 0.0)


# --- laplace_nll -------------------------------------------------------

def test_laplace_nll_matches_manual_formula():
    y = torch.tensor([1.0, 2.0, 5.0])
    mu = torch.tensor([1.5, 2.0, 3.0])
    scale = torch.tensor([0.5, 1.0, 2.0])
    nll = laplace_nll(y, mu, scale, reduction="none")
    expected = torch.abs(y - mu) / scale + torch.log(2 * scale)
    assert torch.allclose(nll, expected)


def test_laplace_nll_is_minimized_at_correct_mean():
    # For fixed scale, moving mu toward y must strictly decrease the loss.
    y = torch.tensor([3.0])
    scale = torch.tensor([1.0])
    near = laplace_nll(y, torch.tensor([2.9]), scale, reduction="none")
    far = laplace_nll(y, torch.tensor([0.0]), scale, reduction="none")
    assert near.item() < far.item()


def test_heteroscedastic_weight_matches_manual_formula():
    loss_per_sample = torch.tensor([1.0, 4.0])
    scale = torch.tensor([1.0, 2.0])
    out = heteroscedastic_weight(loss_per_sample, scale, reduction="none")
    expected = loss_per_sample / scale + torch.log(scale)
    assert torch.allclose(out, expected)


def test_heteroscedastic_weight_lets_high_uncertainty_discount_a_hard_sample():
    # A sample with a large base loss can be down-weighted by predicting a
    # large scale, at the cost of a log(scale) penalty -- same trade-off
    # structure as laplace_nll's |y-mu|/scale + log(scale).
    hard_loss = torch.tensor([10.0])
    small_scale = heteroscedastic_weight(hard_loss, torch.tensor([1.0]))
    large_scale = heteroscedastic_weight(hard_loss, torch.tensor([5.0]))
    assert large_scale.item() < small_scale.item()


def test_laplace_nll_reduction_mean():
    y = torch.tensor([1.0, 2.0])
    mu = torch.tensor([1.0, 2.0])
    scale = torch.tensor([1.0, 1.0])
    nll = laplace_nll(y, mu, scale, reduction="mean")
    assert nll.item() == pytest.approx(math.log(2.0))
