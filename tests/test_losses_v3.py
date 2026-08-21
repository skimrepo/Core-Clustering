import torch

from core_clustering.losses_v3 import ReferenceConsistencyLoss


def _make_outputs(loc_mu, ext_mu, int_mu, shape_emb):
    return {
        "location_mu": torch.tensor(loc_mu), "extent_mu": torch.tensor(ext_mu),
        "intensity_mu": torch.tensor(int_mu), "embeddings": {"shape": torch.tensor(shape_emb)},
    }


def test_reference_consistency_zero_when_outputs_identical():
    out = _make_outputs([0.3, 0.5], [0.2, 0.4], [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
    loss_fn = ReferenceConsistencyLoss()
    loss = loss_fn(out, out)
    assert loss.item() == 0.0


def test_reference_consistency_increases_with_disagreement():
    loss_fn = ReferenceConsistencyLoss()
    out_a = _make_outputs([0.3, 0.5], [0.2, 0.4], [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
    out_b_close = _make_outputs([0.31, 0.5], [0.2, 0.4], [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
    out_b_far = _make_outputs([0.9, 0.5], [0.2, 0.4], [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])

    assert loss_fn(out_a, out_b_close).item() < loss_fn(out_a, out_b_far).item()


def test_reference_consistency_gradient_flows_to_both_sides():
    loss_fn = ReferenceConsistencyLoss()
    a_mu = torch.tensor([0.3], requires_grad=True)
    b_mu = torch.tensor([0.6], requires_grad=True)
    out_a = {"location_mu": a_mu, "embeddings": {}}
    out_b = {"location_mu": b_mu, "embeddings": {}}
    loss = loss_fn(out_a, out_b)
    loss.backward()
    assert a_mu.grad is not None and a_mu.grad.item() != 0
    assert b_mu.grad is not None and b_mu.grad.item() != 0
