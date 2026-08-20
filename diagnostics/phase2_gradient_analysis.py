"""Phase 2, Problem B.4: extent-centric gradient norm/cosine analysis.
Trains a multi-task model normally (same config as Phase 1's multitask
baseline), and at early/mid/late points in training, samples ~15 batches
and decomposes each task's gradient w.r.t. the SHARED TRUNK (encoder +
pool_attn + pool_query) via torch.autograd.grad -- without disturbing the
actual training step (the real combined-loss backward+step still runs
normally on every batch; the decomposition is an extra measurement using
the same retained graph, not a replacement for training).

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/phase2_gradient_analysis.py \\
        --n_instances 1000 --epochs 20 --seed 0 --device cpu \\
        --output_dir diagnostics/outputs/phase2
"""
import argparse
import functools
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.dataset_contrastive import BalancedBatchSampler, contrastive_pad_collate
from core_clustering.losses_contrastive import DEFAULT_WEIGHTS, MultiHeadContrastiveLoss
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive import ContrastiveEncoder

from diagnostics.phase1_baselines import build_loaders

ATTRS = ("shape", "location", "extent", "intensity")
PAIRS = [(a, b) for i, a in enumerate(ATTRS) for b in ATTRS[i + 1:]]


def flatten_grads(grads, params):
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def measure_batch_gradients(model, loss_fn, batch, device, trunk_params, optimizer, max_grad_norm=1.0):
    """Decomposes this batch's per-task gradient on trunk_params, THEN
    still performs the normal combined training step (so training
    proceeds exactly as it would without this measurement)."""
    Y = batch["Y"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    shape = batch["shape_label"].to(device)
    loc = batch["location_value"].to(device)
    ext = batch["extent_value"].to(device)
    inten = batch["intensity_value"].to(device)

    emb = model(Y, pad_mask=pad_mask)
    comp = loss_fn.compute_components(emb, shape, loc, ext, inten)

    task_grad_flat = {}
    for i, attr in enumerate(ATTRS):
        weighted = loss_fn.weights[i] * comp[attr]
        g = torch.autograd.grad(weighted, trunk_params, retain_graph=True, allow_unused=True)
        task_grad_flat[attr] = flatten_grads(g, trunk_params).detach().numpy()

    # normal training step, using the same retained graph
    total = sum(loss_fn.weights[i] * comp[attr] for i, attr in enumerate(ATTRS))
    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), max_grad_norm)
    optimizer.step()

    return task_grad_flat, float(total.item())


def summarize_segment(grad_samples):
    """grad_samples: list of {attr: flat_grad_np} dicts (one per sampled batch)."""
    norms = {a: [] for a in ATTRS}
    cos_by_pair = {p: [] for p in PAIRS}
    for sample in grad_samples:
        for a in ATTRS:
            norms[a].append(float(np.linalg.norm(sample[a])))
        for a, b in PAIRS:
            ga, gb = sample[a], sample[b]
            na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
            cos = float(np.dot(ga, gb) / (na * nb)) if na > 0 and nb > 0 else float("nan")
            cos_by_pair[(a, b)].append(cos)

    norm_summary = {a: {"mean": float(np.mean(v)), "std": float(np.std(v))} for a, v in norms.items()}
    cos_summary = {}
    for (a, b), vals in cos_by_pair.items():
        vals = np.array(vals)
        cos_summary[f"{a}_vs_{b}"] = {
            "mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals)),
            "frac_negative": float((vals < 0).mean()), "n": len(vals),
        }
    return {"grad_norms": norm_summary, "cosine_similarity": cos_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/phase2")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=4)
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

    config = ConvBottleneckConfig(n_time_max=args.max_len, bottleneck_channels=args.z_dim)
    model = ContrastiveEncoder(config, embedding_dim=args.embedding_dim).to(args.device)
    loss_fn = MultiHeadContrastiveLoss(weights=DEFAULT_WEIGHTS).to(args.device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=args.lr)
    trunk_params = list(model.encoder.parameters()) + list(model.pool_attn.parameters()) + [model.pool_query]

    segment_samples = {name: [] for name in segment_starts}
    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            active_segment = None
            for name, start in segment_starts.items():
                if start <= global_step < start + args.batches_per_segment:
                    active_segment = name
            if active_segment is not None:
                grads, total_loss = measure_batch_gradients(
                    model, loss_fn, batch, args.device, trunk_params, optimizer
                )
                segment_samples[active_segment].append(grads)
            else:
                # normal step, no measurement overhead
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
              + "  ".join(f"{name}={len(s)}" for name, s in segment_samples.items()))

    result = {name: summarize_segment(samples) for name, samples in segment_samples.items() if samples}
    out_path = os.path.join(args.output_dir, "gradient_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
