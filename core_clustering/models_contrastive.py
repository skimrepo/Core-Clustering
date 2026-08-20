import math

import torch
import torch.nn as nn

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig, ConvBottleneckEncoder

ATTRS = ("shape", "location", "extent", "intensity")


def _sinusoidal_position_encoding(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard Transformer sinusoidal positional encoding (Vaswani et al.),
    applied to a continuous fractional position in [0, 1) rather than a raw
    integer index -- stays meaningful across variable sequence lengths.
    pos: (..., 1). Returns (..., dim)."""
    device = pos.device
    i = torch.arange(dim, device=device, dtype=torch.float32)
    div_term = torch.exp(-(i // 2) * (2.0 * math.log(10000.0) / dim))
    angles = pos * div_term
    return torch.where(i % 2 == 0, torch.sin(angles), torch.cos(angles))


class ContrastiveEncoder(nn.Module):
    """Wraps ConvBottleneckEncoder (stem + downsample stages + optional
    self-attention + channel-squeeze) with a standard learned-query
    attention-pool (nn.MultiheadAttention) over the compressed time axis,
    producing ONE embedding vector per whole instance. No decoder here --
    that only exists for the separate per-timepoint detection fine-tuning
    stage (ConvBottleneckAEC).

    Standard positional encoding (fixed sinusoidal, Transformer-style, on a
    continuous fractional position so it stays valid across variable
    lengths) plus a standard attention-pool: a single learned query
    attending over all timesteps via nn.MultiheadAttention. At random init
    this starts close to uniform (like a CLS token before training) --
    that's the normal starting point for this kind of mechanism, not a
    defect; position/content sensitivity is expected to emerge THROUGH
    training, not be guaranteed structurally from the first forward pass.

    (Earlier iterations hand-engineered a feature-magnitude-based saliency
    weighting, plus explicit weighted mean-position/spread statistics, to
    force non-uniform attention from initialization and to recover
    extent/location information a plain weighted-average pool discards.
    Both were removed in favor of this standard mechanism -- hand-derived
    statistics tied to one anomaly shape (e.g. a rectangular plateau) don't
    generalize to other shapes (spike, trend, ...), and the standard
    attention-pool's normal training dynamics are the more general fix.)

    The shared trunk (conv encoder + attention-pool) feeds FOUR separate
    small Linear heads, one per attribute (see ATTRS) -- low-level feature
    extraction is genuinely useful for all four attributes and stays
    shared, but each attribute's own loss only ever reads its own head's
    output."""

    def __init__(self, config: ConvBottleneckConfig, embedding_dim: int = None, pooling: str = "attention"):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(f"pooling must be 'attention' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.encoder = ConvBottleneckEncoder(
            config.n_features, config.num_filters, config.bottleneck_channels,
            kernel_size=config.kernel_size, stride=config.stride, padding=config.padding,
            padding_mode=config.padding_mode, num_stem_layers=config.num_stem_layers,
            n_time_max=config.n_time_max, attention_max_resolution=config.attention_max_resolution,
            attention_heads=config.attention_heads,
            dropout=config.dropout, normalization=config.normalization, num_groups=config.num_groups,
        )
        channels = config.bottleneck_channels
        self.pool_query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)
        self.pool_attn = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)

        # embedding_dim is each HEAD's own dimension (not split across the
        # four) -- every attribute gets its own full-size embedding space.
        self.head_dim = embedding_dim or channels
        self.heads = nn.ModuleDict({name: nn.Linear(channels, self.head_dim) for name in ATTRS})

    def forward(self, x, pad_mask=None):
        feat, lengths, masks = self.encoder(x, pad_mask=pad_mask)  # feat: (batch, channels, time)
        bottleneck_mask = masks[-1]
        batch, channels, time = feat.shape
        feat_t = feat.transpose(1, 2)  # (batch, time, channels)

        if self.pooling == "mean":
            # Simple baseline: plain masked mean-pool, no positional
            # encoding or attention at all.
            if bottleneck_mask is not None:
                denom = bottleneck_mask.sum(dim=2).clamp_min(1.0)
                pooled = (feat * bottleneck_mask).sum(dim=2) / denom
            else:
                pooled = feat.mean(dim=2)
            return {name: head(pooled) for name, head in self.heads.items()}

        # Fractional position (not a raw index) so it stays meaningful for
        # variable-length series compressed to a variable bottleneck time.
        pos = torch.linspace(0.0, 1.0, time, device=feat.device).view(1, time, 1).expand(batch, time, 1)
        pos_enc = _sinusoidal_position_encoding(pos, channels)  # (batch, time, channels)
        feat_t = feat_t + pos_enc

        key_padding_mask = (bottleneck_mask[:, 0, :] < 0.5) if bottleneck_mask is not None else None
        query = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.pool_attn(query, feat_t, feat_t, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = pooled.squeeze(1)  # (batch, channels)

        return {name: head(pooled) for name, head in self.heads.items()}
