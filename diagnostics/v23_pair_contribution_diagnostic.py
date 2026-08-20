"""MTL_V23_PAIR_CONTRIBUTION_REPORT.md: diagnostic-only decomposition of
RadialOrdinalLoss's valid pairs into Normal-Anomaly (NA) and Anomaly-
Anomaly (AA) groups, to test why V2.3's intensity severity collapsed
almost all anomalies into a narrow band despite perfect normal-vs-anomaly
separation (MTL_V23_ORDINAL_INTENSITY_REPORT.md).

Does NOT modify RadialOrdinalLoss, the model, the optimizer, or the
training procedure in any way -- reproduces V2.3's exact seed=0 training
(same dataset config, same architecture, same combined 4-task loss actually
used for every optimizer.step()) and, at sampled batches only, ADDITIONALLY
computes NA/AA sub-losses and their gradients purely for measurement
(mirroring phase2_gradient_analysis.py / v2_gradient_analysis.py's
established "measure via a separate autograd.grad call, then still take
the normal step" pattern). The hypothetical "balanced" NA/AA aggregate is
computed as a NUMBER ONLY, never backpropagated.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v23_pair_contribution_diagnostic.py \\
        --n_instances 1000 --epochs 20 --seed 0 --device cpu \\
        --output_dir diagnostics/outputs/v23
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.losses_contrastive import DEFAULT_WEIGHTS, MultiHeadContrastiveLoss
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import ATTRS, ContrastiveEncoderV2

from diagnostics.phase1_baselines import build_loaders


def flatten_grads(grads, params):
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def decompose_intensity_pairs(embeddings, is_anomalous, value, eps=1e-9):
    """Replicates RadialOrdinalLoss.forward's exact math (same centroid,
    same stop-gradient convention, same pair validity/direction/softplus
    formula) but additionally exposes per-pair NA/AA masks, margins, and
    the two loss terms separately -- for measurement only. Verified against
    the real loss module's output (see verify_equivalence)."""
    normal_emb = embeddings[~is_anomalous]
    centroid = normal_emb.mean(dim=0)
    normal_pull = ((normal_emb - centroid) ** 2).sum(dim=-1).mean()

    y = torch.zeros(embeddings.shape[0], dtype=value.dtype, device=embeddings.device)
    y[is_anomalous] = value[is_anomalous]
    s = (embeddings - centroid.detach()).norm(dim=-1)

    n = s.shape[0]
    y_diff = y.unsqueeze(1) - y.unsqueeze(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    valid = (y_diff.abs() > eps) & ~eye

    direction = torch.sign(y_diff)
    pair_losses = F.softplus(-direction * s_diff)
    margins = direction * s_diff  # >0 = correct ordering, per the spec's own definition

    is_anom_i = is_anomalous.unsqueeze(1).expand(n, n)
    is_anom_j = is_anomalous.unsqueeze(0).expand(n, n)
    na_mask = valid & (is_anom_i != is_anom_j)
    aa_mask = valid & is_anom_i & is_anom_j

    return {
        "s": s, "y": y, "valid": valid, "na_mask": na_mask, "aa_mask": aa_mask,
        "pair_losses": pair_losses, "margins": margins, "normal_pull": normal_pull,
    }


def verify_equivalence(diag, real_intensity_loss, atol=1e-5):
    valid = diag["valid"]
    rank_loss = diag["pair_losses"][valid].mean() if valid.any() else diag["pair_losses"].new_tensor(0.0)
    reconstructed = rank_loss + diag["normal_pull"]
    diff = float((reconstructed - real_intensity_loss).abs().item())
    return diff, diff < atol


def summarize_group(losses, margins):
    if len(losses) == 0:
        nan = float("nan")
        return {"count": 0, "mean_loss": nan, "median_loss": nan, "std_loss": nan, "total_loss": nan,
                "ordering_accuracy": nan,
                "margin": {"mean": nan, "median": nan, "p5": nan, "p25": nan, "p75": nan, "p95": nan}}
    return {
        "count": len(losses),
        "mean_loss": float(np.mean(losses)), "median_loss": float(np.median(losses)),
        "std_loss": float(np.std(losses)), "total_loss": float(np.sum(losses)),
        "ordering_accuracy": float(np.mean(margins > 0)),
        "margin": {
            "mean": float(np.mean(margins)), "median": float(np.median(margins)),
            "p5": float(np.percentile(margins, 5)), "p25": float(np.percentile(margins, 25)),
            "p75": float(np.percentile(margins, 75)), "p95": float(np.percentile(margins, 95)),
        },
    }


def measure_batch(model, loss_fn, batch, device, trunk_params, head_params, optimizer, max_grad_norm=1.0):
    Y = batch["Y"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    shape = batch["shape_label"].to(device)
    loc = batch["location_value"].to(device)
    ext = batch["extent_value"].to(device)
    inten = batch["intensity_value"].to(device)

    emb = model(Y, pad_mask=pad_mask)
    comp = loss_fn.compute_components(emb, shape, loc, ext, inten)

    is_anomalous = shape == 1
    diag = decompose_intensity_pairs(emb["intensity"], is_anomalous, inten)
    equiv_diff, equiv_ok = verify_equivalence(diag, comp["intensity"])

    na_mask, aa_mask, valid = diag["na_mask"], diag["aa_mask"], diag["valid"]
    pair_losses, margins = diag["pair_losses"], diag["margins"]

    n_na, n_aa = int(na_mask.sum().item()), int(aa_mask.sum().item())
    na_losses_np = pair_losses[na_mask].detach().cpu().numpy() if n_na > 0 else np.array([])
    aa_losses_np = pair_losses[aa_mask].detach().cpu().numpy() if n_aa > 0 else np.array([])
    na_margins_np = margins[na_mask].detach().cpu().numpy() if n_na > 0 else np.array([])
    aa_margins_np = margins[aa_mask].detach().cpu().numpy() if n_aa > 0 else np.array([])

    na_summary = summarize_group(na_losses_np, na_margins_np)
    aa_summary = summarize_group(aa_losses_np, aa_margins_np)

    n_valid = n_na + n_aa
    l_current = float(pair_losses[valid].mean().item()) if n_valid > 0 else float("nan")
    l_balanced = (
        0.5 * na_summary["mean_loss"] + 0.5 * aa_summary["mean_loss"]
        if n_na > 0 and n_aa > 0 else float("nan")
    )

    zero = emb["intensity"].new_tensor(0.0)
    L_NA = pair_losses[na_mask].mean() if n_na > 0 else zero
    L_AA = pair_losses[aa_mask].mean() if n_aa > 0 else zero

    g_trunk_na = torch.autograd.grad(L_NA, trunk_params, retain_graph=True, allow_unused=True)
    g_trunk_aa = torch.autograd.grad(L_AA, trunk_params, retain_graph=True, allow_unused=True)
    g_head_na = torch.autograd.grad(L_NA, head_params, retain_graph=True, allow_unused=True)
    g_head_aa = torch.autograd.grad(L_AA, head_params, retain_graph=True, allow_unused=True)

    trunk_na_flat = flatten_grads(g_trunk_na, trunk_params)
    trunk_aa_flat = flatten_grads(g_trunk_aa, trunk_params)
    head_na_flat = flatten_grads(g_head_na, head_params)
    head_aa_flat = flatten_grads(g_head_aa, head_params)

    na_norm, aa_norm = trunk_na_flat.norm(), trunk_aa_flat.norm()
    cos_trunk = (
        float((trunk_na_flat @ trunk_aa_flat / (na_norm * aa_norm)).item())
        if na_norm > 0 and aa_norm > 0 else float("nan")
    )

    s_np = diag["s"].detach().cpu().numpy()
    is_anom_np = is_anomalous.cpu().numpy()
    s_normal, s_anomaly = s_np[~is_anom_np], s_np[is_anom_np]

    batch_result = {
        "equivalence_diff": equiv_diff, "equivalence_ok": equiv_ok,
        "n_na": n_na, "n_aa": n_aa,
        "na": na_summary, "aa": aa_summary,
        "l_current": l_current, "l_balanced": l_balanced,
        "trunk_grad_norm_na": float(trunk_na_flat.norm().item()),
        "trunk_grad_norm_aa": float(trunk_aa_flat.norm().item()),
        "head_grad_norm_na": float(head_na_flat.norm().item()),
        "head_grad_norm_aa": float(head_aa_flat.norm().item()),
        "cos_trunk_na_aa": cos_trunk,
        "normal_severity": {"mean": float(s_normal.mean()), "std": float(s_normal.std())} if len(s_normal) else None,
        "anomaly_severity": {
            "mean": float(s_anomaly.mean()), "std": float(s_anomaly.std()),
            "min": float(s_anomaly.min()), "max": float(s_anomaly.max()),
            "p5": float(np.percentile(s_anomaly, 5)), "p25": float(np.percentile(s_anomaly, 25)),
            "p75": float(np.percentile(s_anomaly, 75)), "p95": float(np.percentile(s_anomaly, 95)),
        } if len(s_anomaly) else None,
        "fraction_anomaly_gt_normal": (
            float(np.mean([a > b for a in s_anomaly for b in s_normal]))
            if len(s_anomaly) and len(s_normal) else float("nan")
        ),
    }

    # REAL training step -- unmodified original combined loss, same as V2.3's actual run.
    total = sum(loss_fn.weights[i] * comp[attr] for i, attr in enumerate(ATTRS))
    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), max_grad_norm)
    optimizer.step()

    return batch_result


def aggregate_segment(batch_results):
    def mean_of(key_path):
        vals = []
        for r in batch_results:
            v = r
            for k in key_path:
                v = v[k] if v is not None else None
                if v is None:
                    break
            if v is not None and v == v:  # not NaN
                vals.append(v)
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)

    def pack(path):
        m, s, n = mean_of(path)
        return {"mean": m, "std": s, "n_batches": n}

    return {
        "n_na": pack(["n_na"]), "n_aa": pack(["n_aa"]),
        "na_mean_loss": pack(["na", "mean_loss"]), "aa_mean_loss": pack(["aa", "mean_loss"]),
        "na_total_loss": pack(["na", "total_loss"]), "aa_total_loss": pack(["aa", "total_loss"]),
        "na_ordering_accuracy": pack(["na", "ordering_accuracy"]),
        "aa_ordering_accuracy": pack(["aa", "ordering_accuracy"]),
        "na_margin_mean": pack(["na", "margin", "mean"]), "aa_margin_mean": pack(["aa", "margin", "mean"]),
        "l_current": pack(["l_current"]), "l_balanced": pack(["l_balanced"]),
        "trunk_grad_norm_na": pack(["trunk_grad_norm_na"]), "trunk_grad_norm_aa": pack(["trunk_grad_norm_aa"]),
        "head_grad_norm_na": pack(["head_grad_norm_na"]), "head_grad_norm_aa": pack(["head_grad_norm_aa"]),
        "cos_trunk_na_aa": pack(["cos_trunk_na_aa"]),
        "normal_severity_mean": pack(["normal_severity", "mean"]),
        "anomaly_severity_mean": pack(["anomaly_severity", "mean"]),
        "anomaly_severity_std": pack(["anomaly_severity", "std"]),
        "fraction_anomaly_gt_normal": pack(["fraction_anomaly_gt_normal"]),
        "equivalence_max_diff": max((r["equivalence_diff"] for r in batch_results), default=float("nan")),
        "n_batches_sampled": len(batch_results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v23")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--attention_max_resolution", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches_per_segment", type=int, default=15)
    # V2.3's exact dataset config -- not overridable, this diagnostic is specifically about V2.3
    parser.add_argument("--intensity_min", type=float, default=0.2)
    parser.add_argument("--intensity_max", type=float, default=4.0)
    args = parser.parse_args()
    args.intensity_mode = "universal_deviation_intensity"
    args.intensity_sampling = "log_uniform"
    args.intensity_metric_transform = "identity"

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
    model = ContrastiveEncoderV2(config, embedding_dim=args.embedding_dim, normalize_embedding=True).to(args.device)
    loss_fn = MultiHeadContrastiveLoss(weights=DEFAULT_WEIGHTS, intensity_objective="radial_ordinal").to(args.device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=args.lr)
    trunk_params = list(model.encoder.parameters())
    head_params = list(model.attribute_heads["intensity"].parameters())

    segment_batches = {name: [] for name in segment_starts}
    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_dl:
            active_segment = None
            for name, start in segment_starts.items():
                if start <= global_step < start + args.batches_per_segment:
                    active_segment = name
            if active_segment is not None:
                result = measure_batch(model, loss_fn, batch, args.device, trunk_params, head_params, optimizer)
                segment_batches[active_segment].append(result)
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
              + "  ".join(f"{name}={len(s)}" for name, s in segment_batches.items()))

    result = {name: aggregate_segment(batches) for name, batches in segment_batches.items() if batches}
    max_equiv_diff = max((r["equivalence_max_diff"] for r in result.values()), default=float("nan"))
    print(f"\nMax |reconstructed - real intensity loss| across all sampled batches: {max_equiv_diff:.3e}")

    out_path = os.path.join(args.output_dir, "v23_pair_contribution.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
