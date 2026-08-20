import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_num_stages(n_time_max: int, target_bottleneck_len: int = 32, stride: int = 2) -> int:
    """How many stride>1 downsampling stages are needed so that an input of
    length n_time_max compresses to roughly target_bottleneck_len. Rounds to
    the CLOSEST stage count (not ceil) so it doesn't systematically
    over-compress relative to the target. Always at least 3, so there are
    enough distinct resolutions for self-attention placement later."""
    ratio = n_time_max / target_bottleneck_len
    if ratio <= 1:
        return 3
    return max(3, round(math.log(ratio, stride)))


def compute_num_filters(
    n_time_max: int, target_bottleneck_len: int = 32, stride: int = 2,
    channel_base: int = 16, channel_max: int = 128,
) -> List[int]:
    """Channel width per downsampling stage: doubles from channel_base each
    stage, capped at channel_max, one entry per stage computed by
    compute_num_stages."""
    n_stages = compute_num_stages(n_time_max, target_bottleneck_len, stride)
    return [min(channel_base * (2 ** i), channel_max) for i in range(n_stages)]


@dataclass
class ConvBottleneckConfig:
    """n_time_max is used for (a) padding-width validation/collate and (b)
    -- when num_filters is not given explicitly -- computing how many
    downsampling stages/channel widths are needed via compute_num_filters.
    It is NOT baked into any layer's weight shape the way RedLamp's original
    ConvDecoder (nn.Linear(1, compressed_len) tied to a training-time
    seq_len) was; variable length is recovered dynamically every forward
    pass via ConvTranspose1d's output_size argument."""
    n_time_max: int
    n_features: int = 1
    name: str = "ConvBottleneckAEC"
    num_filters: Optional[List[int]] = None
    bottleneck_channels: int = 4
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    padding_mode: str = "reflect"
    num_stem_layers: int = 1
    target_bottleneck_len: int = 32
    channel_base: int = 16
    channel_max: int = 128
    attention_max_resolution: int = 256
    attention_heads: int = 4
    dropout: float = 0.2
    normalization: str = "group"
    num_groups: int = 8
    bce_loss_ratio: float = 0.1

    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = compute_num_filters(
                self.n_time_max, self.target_bottleneck_len, self.stride,
                self.channel_base, self.channel_max,
            )


def _make_norm(normalization, channels, num_groups):
    if normalization == 'group':
        g = min(num_groups, channels)
        while channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, channels)
    if normalization == 'layer':
        return nn.GroupNorm(1, channels)
    if normalization == 'batch':
        return nn.BatchNorm1d(channels)
    return None


class ConvEncoderBlock(nn.Module):
    """Plain (non-dilated) stride-2 Conv1d block -- genuinely shrinks the
    temporal axis each block, unlike TCNResidualBlock's 'same' padding.
    This IS the information bottleneck: a downsampled representation
    cannot losslessly carry every input timestep's exact value through,
    unlike a same-resolution residual path."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 padding_mode='reflect', dropout=0.2, normalization='group', num_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, padding_mode=padding_mode)
        self.norm = _make_norm(normalization, out_channels, num_groups)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        return self.drop(self.act(out))


class ConvDecoderBlock(nn.Module):
    """Mirrors ConvEncoderBlock with ConvTranspose1d. output_size is passed
    at every forward call (not fixed at construction time), so the exact
    reconstructed length always matches the corresponding encoder stage's
    input length for THIS call, regardless of the series' actual length."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dropout=0.2, normalization='group', num_groups=8):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = _make_norm(normalization, out_channels, num_groups)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, output_size=None):
        out = self.conv_t(x, output_size=[output_size] if output_size is not None else None)
        if self.norm is not None:
            out = self.norm(out)
        return self.drop(self.act(out))


class SelfAttentionBlock(nn.Module):
    """Standard pre-norm Transformer encoder block (MultiheadAttention +
    feedforward, each with a residual). Placed only at stages whose nominal
    length is <= attention_max_resolution -- self-attention is O(T^2), so
    it's only affordable once the sequence has already been compressed by
    the conv stages (see ConvBottleneckEncoder's attn_by_stage)."""

    def __init__(self, channels, num_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ff_mult, channels),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, channels, time) -> attention operates on (batch, time, channels)
        x_t = x.transpose(1, 2)
        h = self.norm1(x_t)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x_t = x_t + self.drop(attn_out)
        h2 = self.norm2(x_t)
        x_t = x_t + self.drop(self.ff(h2))
        return x_t.transpose(1, 2)


class ConvBottleneckEncoder(nn.Module):
    """Stack of ConvEncoderBlock (num_filters growing while stride-2 shrinks
    the temporal axis each stage, for feature-extraction depth), followed
    by a final 1x1 "waist" conv squeezing down to bottleneck_channels.

    The waist squeeze is what actually enforces a narrow bottleneck: growing
    channel width while only shrinking time can otherwise increase total
    capacity (channels*time) rather than reduce it, since a scalar channel
    can in principle carry a value losslessly with no quantization -- the
    squeeze forces genuinely few numbers per (already-shrunk) timestep."""

    def __init__(self, num_inputs, num_filters, bottleneck_channels, kernel_size=3, stride=2, padding=1,
                 padding_mode='reflect', num_stem_layers=1,
                 n_time_max: Optional[int] = None, attention_max_resolution: int = 0,
                 attention_heads: int = 4,
                 dropout=0.2, normalization='group', num_groups=8, include_squeeze: bool = True):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

        stem_channels = num_filters[0]
        stem = []
        for i in range(num_stem_layers):
            in_channels = num_inputs if i == 0 else stem_channels
            # stride=1 (no downsampling): a short period (e.g. 3-5 timesteps)
            # would otherwise risk aliasing at the very first strided hop --
            # this stem lets the network see it at full resolution first.
            stem.append(ConvEncoderBlock(
                in_channels, stem_channels, kernel_size, stride=1, padding=padding,
                padding_mode=padding_mode, dropout=dropout,
                normalization=normalization, num_groups=num_groups))
        self.stem = nn.ModuleList(stem)

        blocks = []
        for i, out_channels in enumerate(num_filters):
            in_channels = stem_channels if i == 0 else num_filters[i - 1]
            blocks.append(ConvEncoderBlock(
                in_channels, out_channels, kernel_size, stride, padding,
                padding_mode=padding_mode, dropout=dropout,
                normalization=normalization, num_groups=num_groups))
        self.blocks = nn.ModuleList(blocks)
        # include_squeeze=False: used by V2 (models_contrastive_v2.py), whose
        # shared representation is the raw last-stage feature map itself, not
        # a channel-squeezed bottleneck -- skipping construction entirely (not
        # just skipping the call) avoids dead, never-trained parameters.
        self.squeeze = nn.Conv1d(num_filters[-1], bottleneck_channels, 1) if include_squeeze else None

        # Attention placement is decided ONCE at construction from the
        # NOMINAL per-stage length (derived from n_time_max), not from
        # whatever actual input length a given forward() call happens to
        # see -- attention modules are learned parameters and can't be
        # created dynamically per call. Self-attention is O(T^2), so it's
        # only attached where the nominal length is already cheap.
        self.attn_by_stage = nn.ModuleDict()
        if n_time_max is not None and attention_max_resolution > 0:
            nominal_len = n_time_max
            for i, out_channels in enumerate(num_filters):
                nominal_len = (nominal_len + 2 * padding - kernel_size) // stride + 1
                if nominal_len <= attention_max_resolution:
                    self.attn_by_stage[str(i)] = SelfAttentionBlock(
                        out_channels, num_heads=attention_heads, dropout=dropout)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None):
        lengths = [x.shape[2]]
        masks = [pad_mask]
        h, m = x, pad_mask

        for block in self.stem:
            h = block(h)
            if m is not None:
                h = h * m  # stride=1: length unchanged, mask needs no re-pooling
            lengths.append(h.shape[2])
            masks.append(m)

        for i, block in enumerate(self.blocks):
            h = block(h)
            if m is not None:
                m = F.max_pool1d(m, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
                h = h * m
            if str(i) in self.attn_by_stage:
                key_padding_mask = (m[:, 0, :] < 0.5) if m is not None else None
                h = self.attn_by_stage[str(i)](h, key_padding_mask=key_padding_mask)
                if m is not None:
                    h = h * m  # re-zero: attention's linear layers have bias too
            lengths.append(h.shape[2])
            masks.append(m)

        if self.squeeze is not None:
            h = self.squeeze(h)
            if m is not None:
                h = h * m  # re-zero: a 1x1 conv's bias would otherwise un-zero padded positions
        return h, lengths, masks


class ConvBottleneckDecoder(nn.Module):
    """Mirrors ConvBottleneckEncoder in reverse. Recovers each stage's
    exact original length via output_size (from the encoder's own
    recorded `lengths`), so the final output always matches the true
    input length -- computed at runtime, never baked into a weight."""

    def __init__(self, num_filters, bottleneck_channels, kernel_size=3, stride=2, padding=1,
                 dropout=0.2, normalization='group', num_groups=8):
        super().__init__()
        reversed_filters = list(reversed(num_filters))
        self.unsqueeze = nn.Conv1d(bottleneck_channels, reversed_filters[0], 1)
        blocks = []
        for i in range(len(reversed_filters)):
            in_channels = reversed_filters[i]
            out_channels = reversed_filters[i + 1] if i + 1 < len(reversed_filters) else reversed_filters[i]
            blocks.append(ConvDecoderBlock(
                in_channels, out_channels, kernel_size, stride, padding,
                dropout=dropout, normalization=normalization, num_groups=num_groups))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, feat, lengths, masks):
        h = self.unsqueeze(feat)
        if masks[-1] is not None:
            h = h * masks[-1]
        for i, block in enumerate(self.blocks):
            target_len = lengths[-(i + 2)]
            h = block(h, output_size=target_len)
            mask = masks[-(i + 2)]
            if mask is not None:
                h = h * mask
        return h


class ConvBottleneckAEC(nn.Module):
    """Dual-head whole-series model with a genuine temporal-downsample
    bottleneck (replaces the earlier dilated, no-downsample TCN backbone).
    Same interface as the TCN version: forward(x, pad_mask) -> (recon,
    anomaly_logits, feat); calculate_loss(...) unchanged."""

    def __init__(self, config: ConvBottleneckConfig):
        super().__init__()
        self.name = config.name
        self.bce_loss_ratio = config.bce_loss_ratio

        self.encoder = ConvBottleneckEncoder(
            config.n_features, config.num_filters, config.bottleneck_channels,
            kernel_size=config.kernel_size, stride=config.stride, padding=config.padding,
            padding_mode=config.padding_mode, num_stem_layers=config.num_stem_layers,
            n_time_max=config.n_time_max, attention_max_resolution=config.attention_max_resolution,
            attention_heads=config.attention_heads,
            dropout=config.dropout, normalization=config.normalization, num_groups=config.num_groups,
        )
        self.decoder = ConvBottleneckDecoder(
            config.num_filters, config.bottleneck_channels,
            kernel_size=config.kernel_size, stride=config.stride, padding=config.padding,
            dropout=config.dropout, normalization=config.normalization, num_groups=config.num_groups,
        )
        feat_channels = config.num_filters[0]
        self.recon_head = nn.Conv1d(feat_channels, config.n_features, 1)
        self.anomaly_head = nn.Conv1d(feat_channels, config.n_features, 1)

    def forward(self, x, pad_mask=None):
        bottleneck, lengths, masks = self.encoder(x, pad_mask=pad_mask)
        feat = self.decoder(bottleneck, lengths, masks)
        recon = self.recon_head(feat)
        anomaly_logits = self.anomaly_head(feat)
        return recon, anomaly_logits, feat

    def calculate_loss(self, Y, recon, anomaly_logits, is_anomaly, anomaly_mask, pad_mask):
        """
        All tensors: (batch, n_features, time).
        anomaly_mask: 1=normal, 0=anomalous(injected)   -- MSE gate
        is_anomaly:   0=normal, 1=anomalous             -- BCE target (OPPOSITE convention)
        pad_mask:     1=real timestep, 0=right-padding  -- excluded from BOTH losses
        """
        mse_gate = anomaly_mask * pad_mask
        sq_err = (Y - recon) ** 2
        denom_ae = mse_gate.sum().clamp_min(1.0)
        loss_ae = (sq_err * mse_gate).sum() / denom_ae

        bce_raw = F.binary_cross_entropy_with_logits(anomaly_logits, is_anomaly, reduction='none')
        denom_c = pad_mask.sum().clamp_min(1.0)
        loss_c = (bce_raw * pad_mask).sum() / denom_c

        loss = (1 - self.bce_loss_ratio) * loss_ae + self.bce_loss_ratio * loss_c
        return loss, loss_ae, loss_c
