"""MTL_V2_REPORT.md Section 9-10: V2 trunk-vs-head gradient norms and
trunk gradient-conflict (cosine similarity) re-measurement -- adapted from
diagnostics/phase2_gradient_analysis.py for V2's architecture (trunk =
model.encoder only; no pool_attn/pool_query since V2 has no shared pooling
step). Also records each task's OWN AttributeHead gradient norm alongside
its shared-trunk gradient norm (Section 18's "is the trunk still receiving
a real update signal, not just the head" question).

Not a full Phase 2-scale gradient study -- same early/middle/late x ~15
batch sampling as Phase 2, just re-run once against V2 to see whether
extent-vs-intensity's conflict (the most consistent negative-cosine pair
in V1) changed.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v2_gradient_analysis.py \\
        --n_instances 1000 --epochs 20 --seed 0 --device cpu \\
        --output_dir diagnostics/outputs/v2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.losses_contrastive import DEFAULT_WEIGHTS, MultiHeadContrastiveLoss
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import ATTRS, ContrastiveEncoderV2

from diagnostics.phase1_baselines import build_loaders

PAIRS = [(a, b) for i, a in enumerate(ATTRS) for b in ATTRS[i + 1:]]


def flatten_grads(grads, params):
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def measure_batch_gradients(model, loss_fn, batch, device, trunk_params, head_params_by_attr,
                             optimizer, max_grad_norm=1.0):
    Y = batch["Y"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    shape = batch["shape_label"].to(device)
    loc = batch["location_value"].to(device)
    ext = batch["extent_value"].to(device)
    inten = batch["intensity_value"].to(device)

    emb = model(Y, pad_mask=pad_mask)
    comp = loss_fn.compute_components(emb, shape, loc, ext, inten)

    # Embedding-norm snapshot (MTL_V21_REPORT.md Section 5/9): normalized
    # (public) embedding norm from emb[attr] itself; raw pre-normalization
    # norm from each head's stashed last_raw_embedding (identical to
    # emb[attr] when normalize_embedding=False). Read-only, no effect on
    # the gradient measurement or training step below.
    embedding_norms = {}
    for attr in ATTRS:
        raw = model.attribute_heads[attr].last_raw_embedding
        embedding_norms[attr] = {
            "raw_norm": raw.norm(dim=-1).detach().cpu().numpy(),
            "normalized_norm": emb[attr].norm(dim=-1).detach().cpu().numpy(),
        }

    trunk_grad_flat = {}
    head_grad_norm = {}
    for i, attr in enumerate(ATTRS):
        weighted = loss_fn.weights[i] * comp[attr]
        g_trunk = torch.autograd.grad(weighted, trunk_params, retain_graph=True, allow_unused=True)
        trunk_grad_flat[attr] = flatten_grads(g_trunk, trunk_params).detach().cpu().numpy()

        own_head_params = head_params_by_attr[attr]
        g_head = torch.autograd.grad(weighted, own_head_params, retain_graph=True, allow_unused=True)
        head_grad_norm[attr] = float(flatten_grads(g_head, own_head_params).norm().item())

    total = sum(loss_fn.weights[i] * comp[attr] for i, attr in enumerate(ATTRS))
    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), max_grad_norm)
    optimizer.step()

    return trunk_grad_flat, head_grad_norm, embedding_norms


def summarize_embedding_norms(embedding_norm_samples):
    raw = {a: [] for a in ATTRS}
    normalized = {a: [] for a in ATTRS}
    for sample in embedding_norm_samples:
        for a in ATTRS:
            raw[a].extend(sample[a]["raw_norm"].tolist())
            normalized[a].extend(sample[a]["normalized_norm"].tolist())
    return {
        a: {
            "raw_norm": {"mean": float(np.mean(raw[a])), "std": float(np.std(raw[a]))},
            "normalized_norm": {"mean": float(np.mean(normalized[a])), "std": float(np.std(normalized[a]))},
        }
        for a in ATTRS
    }


def summarize_segment(trunk_samples, head_norm_samples):
    trunk_norms = {a: [] for a in ATTRS}
    cos_by_pair = {p: [] for p in PAIRS}
    for sample in trunk_samples:
        for a in ATTRS:
            trunk_norms[a].append(float(np.linalg.norm(sample[a])))
        for a, b in PAIRS:
            ga, gb = sample[a], sample[b]
            na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
            cos = float(np.dot(ga, gb) / (na * nb)) if na > 0 and nb > 0 else float("nan")
            cos_by_pair[(a, b)].append(cos)

    trunk_norm_summary = {a: {"mean": float(np.mean(v)), "std": float(np.std(v))} for a, v in trunk_norms.items()}
    head_norms = {a: [] for a in ATTRS}
    for sample in head_norm_samples:
        for a in ATTRS:
            head_norms[a].append(sample[a])
    head_norm_summary = {a: {"mean": float(np.mean(v)), "std": float(np.std(v))} for a, v in head_norms.items()}

    ratio_summary = {
        a: (trunk_norm_summary[a]["mean"] / head_norm_summary[a]["mean"]
            if head_norm_summary[a]["mean"] > 0 else float("nan"))
        for a in ATTRS
    }

    cos_summary = {}
    for (a, b), vals in cos_by_pair.items():
        vals = np.array(vals)
        cos_summary[f"{a}_vs_{b}"] = {
            "mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals)),
            "frac_negative": float((vals < 0).mean()), "n": len(vals),
        }
    return {
        "trunk_grad_norms": trunk_norm_summary,
        "head_grad_norms": head_norm_summary,
        "trunk_to_head_ratio": ratio_summary,
        "cosine_similarity": cos_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v2")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--normalize_embedding", action="store_true",
                         help="V2.1: L2-normalize every AttributeHead's final embedding. Default off (V2).")
    parser.add_argument("--attention_max_resolution", type=int, default=256)
    parser.add_argument("--intensity_mode", default="legacy_native_intensity",
                         choices=["legacy_native_intensity", "universal_deviation_intensity"])
    parser.add_argument("--intensity_min", type=float, default=0.05)
    parser.add_argument("--intensity_max", type=float, default=8.0)
    parser.add_argument("--intensity_sampling", default="log_uniform", choices=["log_uniform"])
    parser.add_argument("--experiment_id_prefix", default=None,
                         help="Override the auto-derived output-filename prefix (e.g. 'v22a').")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches_per_segment", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, _, train_dl, _ = build_loaders(args, args.seed)
    n_batches_per_epoch = len(train_dl)
    total_batches = n_batches_per_epoch * args.epochs
    segment_starts = {
        "early": int(0.10 * total_batches),
        "middle": int(0.50 * total_batches),
        "late": int(0.90 * total_batches),
    }
    print(f"n_batches_per_epoch={n_batches_per_epoch}  total_batches={total_batches}")
    print(f"segment_starts={segment_starts}  batches_per_segment={args.batches_per_segment}")

    config = ConvBottleneckConfig(n_time_max=args.max_len, n_features=2,
                                   attention_max_resolution=args.attention_max_resolution)
    model = ContrastiveEncoderV2(config, embedding_dim=args.embedding_dim,
                                  normalize_embedding=args.normalize_embedding).to(args.device)
    loss_fn = MultiHeadContrastiveLoss(weights=DEFAULT_WEIGHTS).to(args.device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=args.lr)
    trunk_params = list(model.encoder.parameters())
    head_params_by_attr = {attr: list(model.attribute_heads[attr].parameters()) for attr in ATTRS}

    segment_trunk_samples = {name: [] for name in segment_starts}
    segment_head_norm_samples = {name: [] for name in segment_starts}
    segment_embedding_norm_samples = {name: [] for name in segment_starts}
    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            active_segment = None
            for name, start in segment_starts.items():
                if start <= global_step < start + args.batches_per_segment:
                    active_segment = name
            if active_segment is not None:
                trunk_grads, head_norms, embedding_norms = measure_batch_gradients(
                    model, loss_fn, batch, args.device, trunk_params, head_params_by_attr, optimizer
                )
                segment_trunk_samples[active_segment].append(trunk_grads)
                segment_head_norm_samples[active_segment].append(head_norms)
                segment_embedding_norm_samples[active_segment].append(embedding_norms)
            else:
                Y = batch["Y"].to(args.device)
                pad_mask = batch["pad_mask"].to(args.device)
                shape = batch["shape_label"].to(args.device)
                loc = batch["location_value"].to(args.device)
                ext = batch["extent_value"].to(args.device)
                inten = batch["intensity_value"].to(args.device)
                optimizer.zero_grad()
                emb = model(Y, pad_mask=pad_mask)
                total, _ = loss_fn(emb, shape, loc, ext, inten)
                total.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), 1.0)
                optimizer.step()
            global_step += 1
        print(f"epoch {epoch}: global_step={global_step}  "
              + "  ".join(f"{name}={len(s)}" for name, s in segment_trunk_samples.items()))

    result = {
        name: {
            **summarize_segment(segment_trunk_samples[name], segment_head_norm_samples[name]),
            "embedding_norms": summarize_embedding_norms(segment_embedding_norm_samples[name]),
        }
        for name in segment_starts if segment_trunk_samples[name]
    }
    if args.experiment_id_prefix is not None:
        out_name = f"{args.experiment_id_prefix}_gradient_analysis.json"
    elif args.intensity_mode == "universal_deviation_intensity":
        out_name = "v22_gradient_analysis.json"
    elif args.normalize_embedding:
        out_name = "v21_gradient_analysis.json"
    else:
        out_name = "v2_gradient_analysis.json"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
