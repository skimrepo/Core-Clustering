import torch
import torch.nn as nn

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig, ConvBottleneckEncoder
from core_clustering.models_contrastive_v2 import ATTRS, AttributeHead, build_position_channel
from core_clustering.prob_heads import ScalarPredictionAdapter, ShapeUncertaintyAdapter
from core_clustering.reference_context import ContextFusion, ReferenceContextEncoder

SCALAR_ATTRS = ("location", "extent", "intensity")
_LINK_BY_ATTR = {"location": "sigmoid", "extent": "sigmoid", "intensity": "softplus"}


class ContrastiveEncoderV3(nn.Module):
    """V3 (see MTL_V3_REPORT.md): adds OPTIONAL local reference-set
    conditioning and probabilistic (mean + scale) scalar outputs on top of
    V2.1's architecture. The shared trunk, the four Generic AttributeHeads
    (imported unchanged from models_contrastive_v2), and final L2
    normalization are all reused exactly as-is -- the SAME trunk encodes
    reference samples, normal queries, and anomalous queries; there is no
    separate normal/anomaly trunk.

    Reference conditioning is OPTIONAL evidence, not a requirement: with
    ref_x=None (K=0), ContextFusion's gate is hard-forced to zero (see
    reference_context.py), so the model falls back exactly to its
    globally-learned, reference-free behavior -- global knowledge is
    learned through the shared weights across the whole training set, not
    defined by whatever local reference subset happens to be available at
    inference time.

    Each scalar attribute (location/extent/intensity) gets an independent
    ScalarPredictionAdapter reading its own head's normalized embedding --
    the embedding itself stays unit-norm (keeps the V2.1 gradient-stability
    benefit), but the adapter's OUTPUT is allowed to be unbounded
    (intensity) or merely [0,1]-bounded (location/extent) by its link
    function, never by embedding-distance geometry. Shape keeps its
    existing contrastive embedding and gains only a lightweight per-sample
    uncertainty scale from the same embedding."""

    def __init__(self, config: ConvBottleneckConfig, attributes=ATTRS, embedding_dim: int = 32,
                 head_proj_channels: int = 32, head_num_queries: int = 4, head_mlp_hidden: int = 64):
        super().__init__()
        if config.n_features != 2:
            raise ValueError(
                f"ContrastiveEncoderV3 requires config.n_features=2 (signal + position channel), "
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
                normalize_embedding=True,
            )
            for name in self.attributes
        })

        # Shared (not per-task) reference-context machinery -- applied
        # BEFORE branching into the four heads, per the spec.
        self.reference_encoder = ReferenceContextEncoder(channels=trunk_channels)
        self.context_fusion = ContextFusion(channels=trunk_channels)

        self.scalar_adapters = nn.ModuleDict({
            name: ScalarPredictionAdapter(embedding_dim, link=_LINK_BY_ATTR[name])
            for name in SCALAR_ATTRS if name in self.attributes
        })
        self.shape_uncertainty = ShapeUncertaintyAdapter(embedding_dim) if "shape" in self.attributes else None

    def _trunk_forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        pos = build_position_channel(x, pad_mask)
        x2 = torch.cat([x, pos], dim=1)
        feat, lengths, masks = self.encoder(x2, pad_mask=pad_mask)
        return feat, masks[-1]

    def forward(self, query_x: torch.Tensor, query_pad_mask: torch.Tensor = None,
                ref_x: torch.Tensor = None, ref_pad_mask: torch.Tensor = None,
                ref_k_valid_mask: torch.Tensor = None, has_reference: torch.Tensor = None) -> dict:
        Hq, Hq_mask = self._trunk_forward(query_x, query_pad_mask)
        B = Hq.shape[0]
        device = Hq.device

        if ref_x is not None and ref_x.shape[1] > 0:
            K = ref_x.shape[1]
            T = ref_x.shape[-1]
            ref_x_flat = ref_x.reshape(B * K, 1, T)
            ref_mask_flat = ref_pad_mask.reshape(B * K, 1, T) if ref_pad_mask is not None else None

            # A batch mixing different per-item K pads reference slots up to
            # the batch's own max K -- those padding slots are sometimes
            # (K=0 items, or any item with K < max K) 100% padding, every
            # timestep masked. Running a fully-masked sequence through the
            # trunk's self-attention triggers a real PyTorch issue: under
            # torch.no_grad() specifically, the dispatched fused-attention
            # kernel can return NaN for an all-masked row (a grad-tracked
            # forward of the SAME input does not hit this) -- and since
            # ReferenceContextEncoder's weighting later multiplies by a
            # (correctly near-zero, but not exactly zero until after
            # multiplication) weight, 0 * NaN is still NaN, silently
            # corrupting the whole batch through downstream cross-sample
            # losses. Fix: never feed a fully-padding slot through the
            # trunk at all -- only the genuinely valid slots are computed;
            # invalid slots are zero-filled directly, bypassing the trunk.
            if ref_k_valid_mask is not None:
                valid_flat = ref_k_valid_mask.reshape(B * K) > 0.5
            else:
                valid_flat = torch.ones(B * K, dtype=torch.bool, device=device)

            if valid_flat.any():
                valid_feat, valid_feat_mask = self._trunk_forward(
                    ref_x_flat[valid_flat], ref_mask_flat[valid_flat] if ref_mask_flat is not None else None
                )
                Tp = valid_feat.shape[-1]
                Cc = valid_feat.shape[1]
                ref_feat_flat = torch.zeros(B * K, Cc, Tp, device=device, dtype=valid_feat.dtype)
                ref_feat_flat[valid_flat] = valid_feat
                if valid_feat_mask is not None:
                    ref_feat_mask_flat = torch.zeros(B * K, 1, Tp, device=device, dtype=valid_feat_mask.dtype)
                    ref_feat_mask_flat[valid_flat] = valid_feat_mask
                else:
                    ref_feat_mask_flat = None
            else:
                # Degenerate: ref_x.shape[1] > 0 but every slot in the whole
                # batch is padding (shouldn't happen via episodic_pad_collate,
                # which only pads up to a real max K>=1, but guarded anyway).
                Tp = Hq.shape[-1]
                ref_feat_flat = torch.zeros(B * K, Hq.shape[1], Tp, device=device, dtype=Hq.dtype)
                ref_feat_mask_flat = torch.zeros(B * K, 1, Tp, device=device, dtype=Hq.dtype)

            ref_feat = ref_feat_flat.reshape(B, K, -1, Tp)
            ref_feat_mask = ref_feat_mask_flat.reshape(B, K, 1, Tp) if ref_feat_mask_flat is not None else None
            ref_ctx = self.reference_encoder(ref_feat, ref_mask=ref_feat_mask, k_valid_mask=ref_k_valid_mask)
            if has_reference is None:
                has_reference = (
                    (ref_k_valid_mask.sum(dim=1) > 0).float()
                    if ref_k_valid_mask is not None else torch.ones(B, device=device)
                )
            H_fused, gate = self.context_fusion(
                Hq, ref_ctx["mean_ref"], ref_ctx["log_var_ref"], ref_ctx["count_feature"],
                has_reference, query_mask=Hq_mask,
            )
            reference_weights = ref_ctx["weights"]
        else:
            zeros = torch.zeros_like(Hq)
            has_reference = torch.zeros(B, device=device)
            H_fused, gate = self.context_fusion(Hq, zeros, zeros, 0.0, has_reference, query_mask=Hq_mask)
            reference_weights = None

        embeddings = {name: head(H_fused, pad_mask=Hq_mask) for name, head in self.attribute_heads.items()}

        outputs = {"embeddings": embeddings, "gate": gate, "reference_weights": reference_weights}
        for name in SCALAR_ATTRS:
            if name not in self.scalar_adapters:
                continue
            mu, scale = self.scalar_adapters[name](embeddings[name])
            outputs[f"{name}_mu"] = mu
            outputs[f"{name}_scale"] = scale
        if self.shape_uncertainty is not None:
            outputs["shape_scale"] = self.shape_uncertainty(embeddings["shape"])
        return outputs
