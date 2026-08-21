import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_LINKS = ("sigmoid", "softplus")


class ScalarPredictionAdapter(nn.Module):
    """Generic (not task-named) lightweight adapter from a normalized, unit-
    norm AttributeHead embedding to a scalar prediction (mu, scale). The
    input embedding stays L2-normalized (keeps the gradient-stability
    benefit that motivated normalization in the first place); the OUTPUT
    is deliberately allowed to be unbounded even though its input isn't --
    that decoupling is intentional, so e.g. Intensity's mu is never capped
    by the ~2 max-distance ceiling a normalized embedding space would
    otherwise impose if distance itself were used as the prediction.

    link="sigmoid": mu in [0,1] (Location, Extent -- already-bounded
    fractional attributes).
    link="softplus": mu in [0, infinity) (Intensity -- realized deviation
    has no upper bound by definition).

    Same class or "architecture" reused across attributes, but each
    instance owns its own independent parameters -- consistent with the
    Generic AttributeHead philosophy (same form, independent weights,
    task-specific only where mathematically necessary -- here, only the
    link function differs)."""

    def __init__(self, embedding_dim: int = 32, link: str = "sigmoid", eps: float = 1e-6):
        super().__init__()
        if link not in _VALID_LINKS:
            raise ValueError(f"link must be one of {_VALID_LINKS}, got {link!r}")
        self.link = link
        self.eps = eps
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, embedding: torch.Tensor):
        raw = self.linear(embedding)
        raw_mean, raw_scale = raw[..., 0], raw[..., 1]
        mu = torch.sigmoid(raw_mean) if self.link == "sigmoid" else F.softplus(raw_mean)
        scale = F.softplus(raw_scale) + self.eps
        return mu, scale


class ShapeUncertaintyAdapter(nn.Module):
    """Lightweight per-sample uncertainty/confidence scalar from the SAME
    Shape embedding ShapeContrastiveLoss already reads -- NOT a physical
    variance in a scalar quantity (Shape isn't a scalar), but an
    "expected ambiguity/residual" scale used to heteroscedastically weight
    the existing contrastive loss per anchor."""

    def __init__(self, embedding_dim: int = 32, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        raw = self.linear(embedding).squeeze(-1)
        return F.softplus(raw) + self.eps


def heteroscedastic_weight(loss_per_sample: torch.Tensor, scale: torch.Tensor,
                            reduction: str = "mean") -> torch.Tensor:
    """Generic heteroscedastic reweighting of an ALREADY-COMPUTED per-sample
    loss (e.g. ShapeContrastiveLoss's per-anchor residual) by a predicted
    positive uncertainty scale: loss_i/scale_i + log(scale_i). Not specific
    to Shape -- reusable for any per-sample base loss paired with a
    predicted scale, the same relationship laplace_nll encodes for a direct
    scalar residual |y-mu|."""
    weighted = loss_per_sample / scale + torch.log(scale)
    if reduction == "none":
        return weighted
    if reduction == "mean":
        return weighted.mean()
    if reduction == "sum":
        return weighted.sum()
    raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")


def laplace_nll(y: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Laplace negative log-likelihood: |y-mu|/scale + log(2*scale).
    Preferred over Gaussian NLL here for its robustness (linear, not
    quadratic, penalty in the residual) and safer gradient behavior."""
    nll = torch.abs(y - mu) / scale + torch.log(2 * scale)
    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")
