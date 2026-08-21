import math

import torch
import torch.nn as nn


class ReferenceContextEncoder(nn.Module):
    """Shared (not per-task) module that turns a variable-quality set of K
    reference feature maps -- each produced by the SAME trunk as the query,
    no separate normal/anomaly trunk -- into first/second-order sequence
    statistics (weighted mean and log-variance, channel/time structure
    preserved). References are treated as "the currently available
    reference population", not IID draws from one exact Gaussian -- the
    per-reference weight lets the model down-weight an inconsistent
    reference rather than assuming every reference is equally trustworthy.

    Deliberately does NOT claim the resulting distribution is Gaussian --
    mean/log-variance are just the two cheapest order statistics to expose
    to the fusion step, not a modeling assumption."""

    def __init__(self, channels: int = 128, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.score_proj = nn.Linear(channels, 1)

    def forward(self, ref_feat: torch.Tensor, ref_mask: torch.Tensor = None,
                k_valid_mask: torch.Tensor = None) -> dict:
        """ref_feat: (B, K, C, T). ref_mask: (B, K, 1, T) with 1=real/0=pad
        (per-timestep), or None (all valid). k_valid_mask: (B, K) with
        1=this reference SLOT is real/0=batch padding (a batch mixing
        different per-item K needs references padded up to the batch's own
        max K), or None (every slot real). K must be >= 1 -- the "no
        references at all" case is handled by the caller (ContextFusion's
        has_reference flag), not here; a row whose k_valid_mask is entirely
        0 is tolerated (returns finite, arbitrary values) precisely because
        the caller is expected to null it out via has_reference regardless."""
        B, K, C, T = ref_feat.shape
        if ref_mask is not None:
            denom = ref_mask.sum(dim=3).clamp_min(1.0)  # (B, K, 1)
            pooled = (ref_feat * ref_mask).sum(dim=3) / denom  # (B, K, C)
        else:
            pooled = ref_feat.mean(dim=3)  # (B, K, C)

        raw_score = self.score_proj(pooled).squeeze(-1)  # (B, K)
        if k_valid_mask is not None:
            # A row with ZERO valid slots would otherwise be all -inf ->
            # softmax NaN; give it a harmless uniform-score fallback since
            # ContextFusion's hard gate=0 makes the actual value irrelevant.
            all_invalid = k_valid_mask.sum(dim=1) == 0
            safe_mask = k_valid_mask.clone()
            safe_mask[all_invalid] = 1.0
            raw_score = raw_score.masked_fill(safe_mask < 0.5, float("-inf"))
        weights = torch.softmax(raw_score, dim=1)  # (B, K), sums to 1 over K

        w = weights.view(B, K, 1, 1)
        mean_ref = (w * ref_feat).sum(dim=1)  # (B, C, T)
        var_ref = (w * (ref_feat - mean_ref.unsqueeze(1)) ** 2).sum(dim=1)  # (B, C, T)
        log_var_ref = torch.log(var_ref + self.eps)

        return {
            "mean_ref": mean_ref,
            "log_var_ref": log_var_ref,
            "weights": weights,
            "count_feature": math.log(1 + K),
        }


class ContextFusion(nn.Module):
    """Shared (before all four AttributeHeads) gated-residual fusion of the
    query's own trunk features with the reference population's context.
    The gate is forced to exactly zero whenever has_reference=0 (K=0),
    regardless of what the learned gate network outputs -- so a query with
    no references falls back EXACTLY to the globally-learned, reference-
    free behavior, not to some learned-but-untrained-for null value."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.channels = channels
        in_channels = channels * 3 + 1  # Hq, mean_ref, log_var_ref, count channel
        self.context_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.gate_proj = nn.Linear(in_channels, 1)

    def forward(self, Hq, mean_ref, log_var_ref, count_feature, has_reference, query_mask=None):
        """Hq, mean_ref, log_var_ref: (B, C, T). count_feature: python float
        or (B,) tensor (log(1+K), constant across a batch in this project's
        fixed-K-per-batch episode design, but accepted either way).
        has_reference: (B,) tensor, 1.0 where K>0 else 0.0."""
        B, C, T = Hq.shape
        device, dtype = Hq.device, Hq.dtype
        if isinstance(count_feature, torch.Tensor):
            count_channel = count_feature.view(B, 1, 1).expand(B, 1, T).to(device=device, dtype=dtype)
        else:
            count_channel = torch.full((B, 1, T), float(count_feature), device=device, dtype=dtype)

        context_input = torch.cat([Hq, mean_ref, log_var_ref, count_channel], dim=1)  # (B, 3C+1, T)
        context_projection = self.context_proj(context_input)  # (B, C, T)

        if query_mask is not None:
            denom = query_mask.sum(dim=2).clamp_min(1.0)
            pooled = (context_input * query_mask).sum(dim=2) / denom
        else:
            pooled = context_input.mean(dim=2)
        gate_raw = self.gate_proj(pooled).squeeze(-1)  # (B,)
        has_reference = has_reference.to(device=device, dtype=dtype)
        gate = torch.sigmoid(gate_raw) * has_reference  # hard-zero when K=0

        gate_b = gate.view(B, 1, 1)
        H_fused = Hq + gate_b * context_projection
        return H_fused, gate
