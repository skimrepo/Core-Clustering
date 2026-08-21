import math

import torch

from core_clustering.reference_context import ContextFusion, ReferenceContextEncoder


def _zero_score_proj(encoder):
    # Forces softmax over all-zero scores -> exactly uniform weights (1/K)
    # regardless of ref_feat content, so mean/var have a directly
    # verifiable closed form (plain unweighted mean/variance).
    with torch.no_grad():
        encoder.score_proj.weight.zero_()
        encoder.score_proj.bias.zero_()


def test_reference_context_uniform_weights_give_plain_mean_and_variance():
    torch.manual_seed(0)
    encoder = ReferenceContextEncoder(channels=4)
    _zero_score_proj(encoder)

    B, K, C, T = 2, 5, 4, 6
    ref_feat = torch.randn(B, K, C, T)

    out = encoder(ref_feat)
    expected_mean = ref_feat.mean(dim=1)
    expected_var = ((ref_feat - expected_mean.unsqueeze(1)) ** 2).mean(dim=1)

    assert torch.allclose(out["mean_ref"], expected_mean, atol=1e-5)
    assert torch.allclose(out["log_var_ref"], torch.log(expected_var + encoder.eps), atol=1e-5)
    assert torch.allclose(out["weights"], torch.full((B, K), 1.0 / K), atol=1e-5)


def test_reference_context_weights_sum_to_one():
    torch.manual_seed(1)
    encoder = ReferenceContextEncoder(channels=8)
    ref_feat = torch.randn(3, 7, 8, 10)
    out = encoder(ref_feat)
    assert torch.allclose(out["weights"].sum(dim=1), torch.ones(3), atol=1e-5)


def test_reference_context_respects_ref_mask():
    # Garbage values in the padded (masked-out) region must not affect the
    # pooled score or the weighted mean/variance.
    torch.manual_seed(2)
    encoder = ReferenceContextEncoder(channels=4)
    _zero_score_proj(encoder)

    B, K, C, T = 1, 3, 4, 10
    real_len = 6
    ref_feat = torch.randn(B, K, C, T)
    ref_feat_garbage = ref_feat.clone()
    ref_feat_garbage[:, :, :, real_len:] = 999.0
    mask = torch.zeros(B, K, 1, T)
    mask[:, :, :, :real_len] = 1.0

    out_clean = encoder(ref_feat, ref_mask=mask)
    out_garbage = encoder(ref_feat_garbage, ref_mask=mask)
    assert torch.allclose(out_clean["mean_ref"][:, :, :real_len], out_garbage["mean_ref"][:, :, :real_len], atol=1e-4)


def test_reference_context_k_valid_mask_ignores_padding_slots():
    # A batch mixing real K=2 items with padding slots up to K=4 must give
    # the SAME mean/var as if those padding slots didn't exist at all.
    torch.manual_seed(0)
    encoder = ReferenceContextEncoder(channels=4)
    _zero_score_proj(encoder)

    real = torch.randn(1, 2, 4, 5)
    padding = torch.randn(1, 2, 4, 5) * 999  # garbage, must be fully excluded
    padded_feat = torch.cat([real, padding], dim=1)
    k_valid_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    out_real_only = encoder(real)
    out_padded = encoder(padded_feat, k_valid_mask=k_valid_mask)
    assert torch.allclose(out_real_only["mean_ref"], out_padded["mean_ref"], atol=1e-4)
    assert torch.allclose(out_padded["weights"][:, 2:], torch.zeros(1, 2), atol=1e-6)


def test_reference_context_k_valid_mask_all_invalid_row_does_not_nan():
    # A row with K=0 real references (all slots padding) must not produce
    # NaN -- ContextFusion's hard gate=0 makes the actual numeric content
    # irrelevant, but it must still be finite.
    encoder = ReferenceContextEncoder(channels=4)
    ref_feat = torch.randn(2, 3, 4, 5)
    k_valid_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    out = encoder(ref_feat, k_valid_mask=k_valid_mask)
    assert torch.isfinite(out["mean_ref"]).all()
    assert torch.isfinite(out["weights"]).all()


def test_reference_context_count_feature_matches_log1p_k():
    encoder = ReferenceContextEncoder(channels=4)
    ref_feat = torch.randn(2, 10, 4, 5)
    out = encoder(ref_feat)
    assert out["count_feature"] == math.log(1 + 10)


# --- ContextFusion -----------------------------------------------------

def test_context_fusion_k0_forces_gate_exactly_zero_and_leaves_h_unchanged():
    torch.manual_seed(0)
    fusion = ContextFusion(channels=4)
    B, C, T = 2, 4, 5
    Hq = torch.randn(B, C, T)
    mean_ref = torch.randn(B, C, T) * 100  # deliberately large, to prove it's fully suppressed
    log_var_ref = torch.randn(B, C, T) * 100
    has_reference = torch.zeros(B)

    H_fused, gate = fusion(Hq, mean_ref, log_var_ref, count_feature=0.0, has_reference=has_reference)
    assert torch.all(gate == 0.0)
    assert torch.allclose(H_fused, Hq)


def test_context_fusion_k_gt_0_gate_in_unit_interval_and_changes_h():
    torch.manual_seed(0)
    fusion = ContextFusion(channels=4)
    B, C, T = 2, 4, 5
    Hq = torch.randn(B, C, T)
    mean_ref = torch.randn(B, C, T)
    log_var_ref = torch.randn(B, C, T)
    has_reference = torch.ones(B)

    H_fused, gate = fusion(Hq, mean_ref, log_var_ref, count_feature=math.log(11), has_reference=has_reference)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    assert not torch.allclose(H_fused, Hq)


def test_context_fusion_gradient_reaches_query_and_reference_inputs():
    torch.manual_seed(0)
    fusion = ContextFusion(channels=4)
    B, C, T = 2, 4, 5
    Hq = torch.randn(B, C, T, requires_grad=True)
    mean_ref = torch.randn(B, C, T, requires_grad=True)
    log_var_ref = torch.randn(B, C, T, requires_grad=True)
    has_reference = torch.ones(B)

    H_fused, gate = fusion(Hq, mean_ref, log_var_ref, count_feature=math.log(11), has_reference=has_reference)
    H_fused.sum().backward()
    assert Hq.grad is not None and torch.any(Hq.grad != 0)
    assert mean_ref.grad is not None and torch.any(mean_ref.grad != 0)
