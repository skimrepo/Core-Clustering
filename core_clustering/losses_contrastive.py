from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NORMAL_SENTINEL = -1.0
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


class ShapeContrastiveLoss(nn.Module):
    """Supervised-contrastive (SupCon-style) loss: no fixed margin -- same-
    shape pairs are pulled together and different-shape pairs pushed apart
    via a softmax over distances, with a learnable temperature standing in
    for what a margin used to control (how sharply pairs are compared)."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, embeddings: torch.Tensor, shape: torch.Tensor, return_per_sample: bool = False):
        n = shape.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=shape.device)
        d = torch.cdist(embeddings, embeddings)
        temperature = torch.exp(self.log_temperature)
        logits = (-d / temperature).masked_fill(eye, float("-inf"))

        same = (shape.unsqueeze(0) == shape.unsqueeze(1)) & ~eye
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        # log_prob is -inf at the masked diagonal; log_prob * same would be
        # 0 * -inf = nan there, so gate with `where` instead of multiplying.
        gated = torch.where(same, log_prob, torch.zeros_like(log_prob))
        pos_count = same.sum(dim=1).clamp_min(1)
        loss_per_anchor = -gated.sum(dim=1) / pos_count

        valid = same.sum(dim=1) > 0
        if not valid.any():
            mean_loss = embeddings.new_tensor(0.0)
        else:
            mean_loss = loss_per_anchor[valid].mean()

        # return_per_sample=False (default): unchanged plain-scalar return,
        # for every existing caller (V1-V2.3 trainers/tests).
        if not return_per_sample:
            return mean_loss
        # return_per_sample=True (V3): also exposes per-anchor loss + which
        # anchors were valid, so a heteroscedastic wrapper can weight each
        # sample by its own predicted uncertainty scale (see prob_heads.py).
        return mean_loss, loss_per_anchor, valid


class PairwiseGapRegressionLoss(nn.Module):
    """Regress pairwise embedding distance toward |value_i - value_j|
    directly (no learnable scale) for all valid pairs. Each pair is scored
    independently -- no batch-wide statistic is needed, unlike a
    correlation-based objective, so it stays well-defined regardless of how
    much spread of values a given batch happens to have.

    No scale parameter here: this head's own Linear projection (see
    ContrastiveEncoder) already has full freedom to rescale its output --
    scaling its weight matrix by c scales every resulting pairwise distance
    by exactly c. An extra learnable scale in the loss would just duplicate
    that same degree of freedom, and having two independent ways to shrink
    the effective scale (this loss's k, or the head's own weights) is
    exactly what let embeddings collapse alongside a shrinking k -- with a
    fixed target, collapsing to d=0 costs gap**2 for every mismatched pair
    instead of being free.

    Used for location: a symmetric axis with no 'normal' reference point,
    so the natural target is the gap between two anomalies directly."""

    def forward(self, embeddings: torch.Tensor, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        n = value.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=value.device)
        mask = valid_mask & ~eye
        if not mask.any():
            return embeddings.new_tensor(0.0)

        target = (value.unsqueeze(0) - value.unsqueeze(1)).abs()
        d = torch.cdist(embeddings, embeddings)
        return ((d - target) ** 2)[mask].mean()


class NormalRelativeRegressionLoss(nn.Module):
    """Used for extent/intensity: each anomaly's distance to the (stop-
    gradient) normal-cluster centroid should regress toward value directly
    (no learnable scale -- see PairwiseGapRegressionLoss's docstring for why
    that would just duplicate the head's own Linear weights and reopen a
    collapse route). Bigger value = farther from normal, per the
    framework's core premise (anomalies are injected deviations FROM
    normal). Also keeps the normal cluster itself tight: a dedicated
    per-attribute head gives normal instances no other loss pressure at
    all, so without this the centroid would be an arbitrary, unstable
    reference point.

    Uses the raw value directly as the target (not log(value)): intensity's
    real range (~0.2 to 4.0) straddles 1.0, and log(value) goes NEGATIVE
    below 1.0 -- a norm can never match a negative target, which silently
    broke the "smaller value = closer to normal" ordering for every value
    under 1. The raw value stays positive across the whole real range."""

    def forward(self, embeddings: torch.Tensor, is_anomalous: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        normal_emb = embeddings[~is_anomalous]
        anomaly_emb = embeddings[is_anomalous]
        if normal_emb.shape[0] == 0 or anomaly_emb.shape[0] == 0:
            return embeddings.new_tensor(0.0)

        centroid = normal_emb.mean(dim=0)
        normal_pull = ((normal_emb - centroid) ** 2).sum(dim=-1).mean()

        target = value[is_anomalous]
        d = (anomaly_emb - centroid.detach()).norm(dim=-1)
        reg = ((d - target) ** 2).mean()
        return normal_pull + reg


class RadialOrdinalLoss(nn.Module):
    """Generic (NOT intensity-specific) loss for representing a POSITIVE,
    UNBOUNDED scalar attribute (y in [0, infinity), y=0 for the 'normal'
    reference class) as a BOUNDED radial severity score in embedding space,
    without ever regressing that score toward y's raw numeric scale --
    only the ORDERING y_i < y_j => s_i < s_j is supervised (see
    MTL_V23_ORDINAL_INTENSITY_REPORT.md). Reusable for any future
    positive-unbounded attribute with a normal reference, not just
    intensity.

    Same normal-clustering term and centroid convention as
    NormalRelativeRegressionLoss (mean of normal embeddings; the
    severity score itself uses a stop-gradient centroid, exactly mirroring
    that loss's own `centroid.detach()` for its regression term) --
    reused, not reinvented.

    Ranking term: for every pair (i, j) with y_i != y_j (normal-normal ties
    excluded automatically, since every normal gets y=0), a smooth,
    parameter-free pairwise loss `softplus(-sign(y_i - y_j) * (s_i - s_j))`
    -- zero extra hyperparameters, and deliberately NOT weighted by
    |y_i - y_j| (see class docstring section on raw-magnitude weighting in
    the report): only the SIGN of the raw gap enters, never its magnitude,
    so the loss is exactly invariant to any order-preserving rescaling of
    y (see tests)."""

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, is_anomalous: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        normal_emb = embeddings[~is_anomalous]
        anomaly_emb = embeddings[is_anomalous]
        if normal_emb.shape[0] == 0 or anomaly_emb.shape[0] == 0:
            return embeddings.new_tensor(0.0)

        centroid = normal_emb.mean(dim=0)
        normal_pull = ((normal_emb - centroid) ** 2).sum(dim=-1).mean()

        y = torch.zeros(embeddings.shape[0], dtype=value.dtype, device=embeddings.device)
        y[is_anomalous] = value[is_anomalous]
        s = (embeddings - centroid.detach()).norm(dim=-1)

        n = s.shape[0]
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)  # (n, n): y_i - y_j
        s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # (n, n): s_i - s_j
        eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        valid = (y_diff.abs() > self.eps) & ~eye

        if not valid.any():
            rank_loss = embeddings.new_tensor(0.0)
        else:
            direction = torch.sign(y_diff)
            pair_losses = F.softplus(-direction * s_diff)
            rank_loss = pair_losses[valid].mean()

        return rank_loss + normal_pull


class MultiHeadContrastiveLoss(nn.Module):
    """Combines the four per-attribute losses above, one per dedicated
    embedding head (see ContrastiveEncoder) -- each term reads only its own
    sub-embedding, so gradients from different attributes never compete
    over the same numbers the way a single shared embedding's margin/pull
    terms used to."""

    def __init__(self, weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS,
                 intensity_objective: str = "radial_regression"):
        super().__init__()
        self.weights = weights
        self.shape_loss = ShapeContrastiveLoss()
        self.location_loss = PairwiseGapRegressionLoss()
        self.extent_loss = NormalRelativeRegressionLoss()
        # intensity_objective="radial_regression" (default): unchanged V1-
        # V2.2a behavior, regresses embedding distance toward the (possibly
        # metric-transformed) scalar target directly.
        # "radial_ordinal" (V2.3): only the ordering of the raw scalar
        # target is supervised -- see RadialOrdinalLoss.
        if intensity_objective == "radial_regression":
            self.intensity_loss = NormalRelativeRegressionLoss()
        elif intensity_objective == "radial_ordinal":
            self.intensity_loss = RadialOrdinalLoss()
        else:
            raise ValueError(
                f"intensity_objective must be 'radial_regression' or 'radial_ordinal', "
                f"got {intensity_objective!r}"
            )
        self.intensity_objective = intensity_objective

    def compute_components(self, embeddings, shape, location, extent, intensity):
        """Raw, still-differentiable per-attribute losses (not detached, not
        weighted/combined) -- lets the trainer backward() each one
        separately, e.g. to give each attribute its own optimizer state on
        the shared trunk instead of one Adam instance blending all four
        attributes' gradient statistics together."""
        is_anomalous = shape == 1
        anomaly_pair_mask = is_anomalous.unsqueeze(0) & is_anomalous.unsqueeze(1)
        return {
            "shape": self.shape_loss(embeddings["shape"], shape),
            "location": self.location_loss(embeddings["location"], location, anomaly_pair_mask),
            "extent": self.extent_loss(embeddings["extent"], is_anomalous, extent),
            "intensity": self.intensity_loss(embeddings["intensity"], is_anomalous, intensity),
        }

    def forward(self, embeddings, shape, location, extent, intensity):
        comp = self.compute_components(embeddings, shape, location, extent, intensity)
        total = (
            self.weights[0] * comp["shape"] + self.weights[1] * comp["location"]
            + self.weights[2] * comp["extent"] + self.weights[3] * comp["intensity"]
        )
        components = {f"loss_{a}": comp[a].detach() for a in comp}
        return total, components
