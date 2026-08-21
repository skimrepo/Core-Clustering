"""MTL_V3_REPORT.md Section J: per-task shared-trunk / head gradient norms
across early/mid/late training, mirroring v2_gradient_analysis.py's
established sampling convention, adapted for V3's four probabilistic
losses + reference-context/fusion modules. Verifies that Intensity's new
UNBOUNDED D target does not recreate the old raw-distance gradient
explosion.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v3_gradient_analysis.py \\
        --n_instances 1000 --epochs 20 --seed 0 --device cpu \\
        --output_dir diagnostics/outputs/v3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.losses_contrastive import NormalRelativeRegressionLoss, PairwiseGapRegressionLoss, ShapeContrastiveLoss
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ATTRS, ContrastiveEncoderV3
from core_clustering.prob_heads import heteroscedastic_weight, laplace_nll

from diagnostics.v3_baseline import build_v3_loaders

ATTR_ORDER = ("shape", "location", "extent", "intensity")


def flatten_grads(grads, params):
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def compute_task_losses(model, shape_loss, loc_geom, ext_geom, batch, device, lambda_geom):
    Y = batch["Y"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    shape = batch["shape_label"].to(device)
    loc = batch["location_value"].to(device)
    ext = batch["extent_value"].to(device)
    D = batch["D"].to(device)
    ref_x = batch["ref_x"].to(device)
    ref_pad_mask = batch["ref_pad_mask"].to(device)
    ref_k_valid_mask = batch["ref_k_valid_mask"].to(device)

    out = model(Y, query_pad_mask=pad_mask, ref_x=ref_x, ref_pad_mask=ref_pad_mask, ref_k_valid_mask=ref_k_valid_mask)
    is_anom = shape == 1

    mean_shape, per_anchor, valid_anchor = shape_loss(out["embeddings"]["shape"], shape, return_per_sample=True)
    l_shape = heteroscedastic_weight(per_anchor[valid_anchor], out["shape_scale"][valid_anchor]) \
        if valid_anchor.any() else mean_shape

    if is_anom.any():
        l_loc = laplace_nll(loc[is_anom], out["location_mu"][is_anom], out["location_scale"][is_anom])
        l_ext = laplace_nll(ext[is_anom], out["extent_mu"][is_anom], out["extent_scale"][is_anom])
    else:
        l_loc = Y.new_tensor(0.0)
        l_ext = Y.new_tensor(0.0)

    anomaly_pair_mask = is_anom.unsqueeze(0) & is_anom.unsqueeze(1)
    geom_loc = loc_geom(out["embeddings"]["location"], loc, anomaly_pair_mask)
    geom_ext = ext_geom(out["embeddings"]["extent"], is_anom, ext)

    l_loc_total = l_loc + lambda_geom * geom_loc
    l_ext_total = l_ext + lambda_geom * geom_ext
    l_int = laplace_nll(D, out["intensity_mu"], out["intensity_scale"])

    return {"shape": l_shape, "location": l_loc_total, "extent": l_ext_total, "intensity": l_int}, out


def measure_batch(model, shape_loss, loc_geom, ext_geom, batch, device, trunk_params, head_params_by_attr,
                   optimizer, lambda_geom, max_grad_norm=1.0):
    losses, out = compute_task_losses(model, shape_loss, loc_geom, ext_geom, batch, device, lambda_geom)

    trunk_grad_flat, head_grad_norm = {}, {}
    for attr in ATTR_ORDER:
        g_trunk = torch.autograd.grad(losses[attr], trunk_params, retain_graph=True, allow_unused=True)
        trunk_grad_flat[attr] = flatten_grads(g_trunk, trunk_params).detach().cpu().numpy()
        own_head_params = head_params_by_attr[attr]
        g_head = torch.autograd.grad(losses[attr], own_head_params, retain_graph=True, allow_unused=True)
        head_grad_norm[attr] = float(flatten_grads(g_head, own_head_params).norm().item())

    total = sum(losses[a] for a in ATTR_ORDER)
    optimizer.zero_grad()
    total.backward()
    all_params = list(model.parameters()) + list(shape_loss.parameters()) + list(loc_geom.parameters()) \
        + list(ext_geom.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
    optimizer.step()

    return trunk_grad_flat, head_grad_norm


def summarize_segment(trunk_samples, head_norm_samples):
    norms = {a: [] for a in ATTR_ORDER}
    head_norms = {a: [] for a in ATTR_ORDER}
    for sample in trunk_samples:
        for a in ATTR_ORDER:
            norms[a].append(float(np.linalg.norm(sample[a])))
    for sample in head_norm_samples:
        for a in ATTR_ORDER:
            head_norms[a].append(sample[a])
    return {
        "trunk_grad_norms": {a: {"mean": float(np.mean(v)), "std": float(np.std(v)), "max": float(np.max(v))}
                              for a, v in norms.items()},
        "head_grad_norms": {a: {"mean": float(np.mean(v)), "std": float(np.std(v)), "max": float(np.max(v))}
                             for a, v in head_norms.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v3")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--attention_max_resolution", type=int, default=256)
    parser.add_argument("--intensity_min", type=float, default=0.2)
    parser.add_argument("--intensity_max", type=float, default=4.0)
    parser.add_argument("--k_regimes", type=int, nargs="+", default=[0, 3, 10, 30, 100])
    parser.add_argument("--contamination_prob", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda_geom", type=float, default=0.1)
    parser.add_argument("--batches_per_segment", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, _, train_dl, _ = build_v3_loaders(args, args.seed)
    n_batches_per_epoch = len(train_dl)
    total_batches = n_batches_per_epoch * args.epochs
    segment_starts = {"early": int(0.10 * total_batches), "middle": int(0.50 * total_batches),
                       "late": int(0.90 * total_batches)}
    print(f"n_batches_per_epoch={n_batches_per_epoch}  total_batches={total_batches}")

    config = ConvBottleneckConfig(n_time_max=args.max_len, n_features=2,
                                   attention_max_resolution=args.attention_max_resolution)
    model = ContrastiveEncoderV3(config, embedding_dim=args.embedding_dim).to(args.device)
    shape_loss = ShapeContrastiveLoss().to(args.device)
    loc_geom = PairwiseGapRegressionLoss().to(args.device)
    ext_geom = NormalRelativeRegressionLoss().to(args.device)
    all_params = list(model.parameters()) + list(shape_loss.parameters()) + list(loc_geom.parameters()) \
        + list(ext_geom.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr)
    trunk_params = list(model.encoder.parameters())
    head_params_by_attr = {attr: list(model.attribute_heads[attr].parameters()) for attr in ATTR_ORDER}

    segment_trunk, segment_head = {n: [] for n in segment_starts}, {n: [] for n in segment_starts}
    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            active = next((n for n, s in segment_starts.items() if s <= global_step < s + args.batches_per_segment),
                           None)
            if active is not None:
                trunk_g, head_g = measure_batch(model, shape_loss, loc_geom, ext_geom, batch, args.device,
                                                  trunk_params, head_params_by_attr, optimizer, args.lambda_geom)
                segment_trunk[active].append(trunk_g)
                segment_head[active].append(head_g)
            else:
                losses, _ = compute_task_losses(model, shape_loss, loc_geom, ext_geom, batch, args.device,
                                                  args.lambda_geom)
                total = sum(losses[a] for a in ATTR_ORDER)
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
            global_step += 1
        print(f"epoch {epoch}: global_step={global_step}  "
              + "  ".join(f"{n}={len(s)}" for n, s in segment_trunk.items()))

    result = {n: summarize_segment(segment_trunk[n], segment_head[n]) for n in segment_starts if segment_trunk[n]}
    out_path = os.path.join(args.output_dir, "v3_gradient_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
