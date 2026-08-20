import torch
import torch.nn as nn
import torch.nn.functional as F

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig, ConvBottleneckEncoder

ATTRS = ("shape", "location", "extent", "intensity")


def build_position_channel(x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
    """Per-sample normalized temporal coordinate: position[t] = t / (L-1),
    where L is THIS sample's own valid length (pad_mask.sum()), not the
    batch's max_len -- so the first valid timestep is always 0.0 and the
    last valid timestep is always 1.0 regardless of how much trailing
    padding a given sample has. Padded positions are exactly zero (masked
    out), not left as an out-of-[0,1] ramp value.

    x: (batch, 1, T). pad_mask: (batch, 1, T) with 1=real, 0=padding, or
    None (every sample assumed fully valid, length T)."""
    batch, _, time = x.shape
    if pad_mask is None:
        pad_mask = torch.ones(batch, 1, time, device=x.device, dtype=x.dtype)

    lengths = pad_mask[:, 0, :].sum(dim=1)  # (batch,)
    denom = (lengths - 1).clamp_min(1)  # avoid /0 when L==1 (single valid step -> position 0)
    idx = torch.arange(time, device=x.device, dtype=x.dtype).view(1, time)
    pos = idx / denom.view(batch, 1)
    pos = pos * pad_mask[:, 0, :]  # zero out padding (also handles idx>=L overshoot before clamping)
    return pos.unsqueeze(1)


class AttributeHead(nn.Module):
    """Generic, task-agnostic head: every attribute (shape/location/extent/
    intensity/...) uses the SAME architecture with independent parameters --
    no hand-crafted per-task pooling or positional logic. What temporal
    region(s) and features each head's queries end up specializing in is
    left entirely to its own loss to discover during training.

    1x1 conv projection -> K learned-query attention pool -> flatten ->
    small MLP -> embedding_dim. Intentionally shallow (see MTL_V2_REPORT.md
    Section 9): the shared trunk stays the real feature extractor, this head
    only selects/aggregates from what the trunk already produced."""

    def __init__(self, in_channels: int = 128, proj_channels: int = 32, num_queries: int = 4,
                 mlp_hidden: int = 64, embedding_dim: int = 32, num_heads: int = 1, dropout: float = 0.0,
                 normalize_embedding: bool = False, normalize_eps: float = 1e-8):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, proj_channels, kernel_size=1)
        self.queries = nn.Parameter(torch.randn(1, num_queries, proj_channels) * 0.02)
        self.pool_attn = nn.MultiheadAttention(proj_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(num_queries * proj_channels, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embedding_dim),
        )
        # V2.1 (see MTL_V21_REPORT.md): the ONLY architecture change from V2
        # -- constrains every attribute's final embedding to unit L2 norm, to
        # test whether V2's unconstrained embedding scale was contributing to
        # intensity's observed gradient-norm runaway. Default False = exact
        # V2 behavior.
        self.normalize_embedding = normalize_embedding
        self.normalize_eps = normalize_eps
        self.last_raw_embedding = None

    def forward(self, feat: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # feat: (batch, in_channels, T'). pad_mask: (batch, 1, T') or None.
        h = self.proj(feat)
        if pad_mask is not None:
            h = h * pad_mask
        h_t = h.transpose(1, 2)  # (batch, T', proj_channels)

        batch = h.shape[0]
        query = self.queries.expand(batch, -1, -1)
        key_padding_mask = (pad_mask[:, 0, :] < 0.5) if pad_mask is not None else None
        pooled, _ = self.pool_attn(query, h_t, h_t, key_padding_mask=key_padding_mask, need_weights=False)
        # pooled: (batch, num_queries, proj_channels)
        flat = pooled.reshape(batch, -1)
        raw_embedding = self.mlp(flat)
        # detached copy for diagnostic introspection only (Section 5/9 of
        # MTL_V21_REPORT.md) -- does not affect the backward graph below.
        self.last_raw_embedding = raw_embedding.detach()
        if self.normalize_embedding:
            return F.normalize(raw_embedding, p=2, dim=-1, eps=self.normalize_eps)
        return raw_embedding


class ContrastiveEncoderV2(nn.Module):
    """V2 architecture (see MTL_V2_REPORT.md): removes the shared Conv1d
    squeeze + single-query attention pool + shared z=4 bottleneck that Phase
    2 diagnostics traced location/extent's information loss to. The shared
    trunk (unchanged Stem/Stage0-3 conv+self-attention) now feeds its raw
    (batch, 128, T') feature map directly to FOUR independent, identically-
    architected AttributeHead instances -- no shared pooling bottleneck sits
    between the trunk and any attribute's own temporal aggregation.

    Input gets a second channel: a per-sample normalized temporal position
    (see build_position_channel), so absolute position is available to the
    trunk from the very first layer rather than only being injected at a
    shared pooling step. All attributes see the same trunk output and the
    same architecture -- only their own loss decides what to use."""

    def __init__(self, config: ConvBottleneckConfig, attributes=ATTRS, embedding_dim: int = 32,
                 head_proj_channels: int = 32, head_num_queries: int = 4, head_mlp_hidden: int = 64,
                 normalize_embedding: bool = False):
        super().__init__()
        if config.n_features != 2:
            raise ValueError(
                f"ContrastiveEncoderV2 requires config.n_features=2 (signal + position channel), "
                f"got {config.n_features}"
            )
        self.attributes = tuple(attributes)
        self.encoder = ConvBottleneckEncoder(
            config.n_features, config.num_filters, config.bottleneck_channels,
            kernel_size=config.kernel_size, stride=config.stride, padding=config.padding,
            padding_mode=config.padding_mode, num_stem_layers=config.num_stem_layers,
            n_time_max=config.n_time_max, attention_max_resolution=config.attention_max_resolution,
            attention_heads=config.attention_heads,
            dropout=config.dropout, normalization=config.normalization, num_groups=config.num_groups,
            include_squeeze=False,
        )
        trunk_channels = config.num_filters[-1]
        self.attribute_heads = nn.ModuleDict({
            name: AttributeHead(
                in_channels=trunk_channels, proj_channels=head_proj_channels,
                num_queries=head_num_queries, mlp_hidden=head_mlp_hidden, embedding_dim=embedding_dim,
                normalize_embedding=normalize_embedding,
            )
            for name in self.attributes
        })

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> dict:
        pos = build_position_channel(x, pad_mask)
        x2 = torch.cat([x, pos], dim=1)  # (batch, 2, T)
        feat, lengths, masks = self.encoder(x2, pad_mask=pad_mask)
        trunk_mask = masks[-1]
        return {name: head(feat, pad_mask=trunk_mask) for name, head in self.attribute_heads.items()}


def count_parameters(model: ContrastiveEncoderV2) -> dict:
    """Shared-vs-task-specific parameter breakdown for MTL_V2_REPORT.md
    Section 3/9."""
    shared_trunk = sum(p.numel() for p in model.encoder.parameters())
    per_head = {name: sum(p.numel() for p in head.parameters()) for name, head in model.attribute_heads.items()}
    single_attribute_head = next(iter(per_head.values()))
    all_attribute_heads = sum(per_head.values())
    total = shared_trunk + all_attribute_heads
    return {
        "shared_trunk": shared_trunk,
        "single_attribute_head": single_attribute_head,
        "per_head": per_head,
        "all_attribute_heads": all_attribute_heads,
        "total": total,
        "shared_ratio": shared_trunk / total,
        "task_specific_ratio": all_attribute_heads / total,
    }
