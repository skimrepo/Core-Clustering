"""MTL_V23_ORDINAL_INTENSITY_REPORT.md Sections 4-9: loads the V2.3
checkpoint (already trained by diagnostics/v2_baseline.py) and measures the
learned severity geometry -- ordering-based metrics only (V2.3 explicitly
does not try to make embedding distance equal I_raw, so absolute-distance
calibration is not the right lens here; see Section 20 of the spec).

Quantile-based (not fixed-threshold) grouping throughout, so this doesn't
silently assume the current 0.2-4.0 training range.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v23_ordinal_intensity_diagnostic.py \\
        --output_dir diagnostics/outputs/v23
"""
import argparse
import csv
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.dataset_dynamic_contrastive import DynamicContrastiveDataset, generate_entity_manifest
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import ContrastiveEncoderV2


def load_model(checkpoint_path, device="cpu"):
    config = ConvBottleneckConfig(n_time_max=550, n_features=2, attention_max_resolution=256)
    model = ContrastiveEncoderV2(config, embedding_dim=32, head_proj_channels=32,
                                  head_num_queries=4, head_mlp_hidden=64, normalize_embedding=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_val_dataset(seed=0, n_instances=1000):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)
    return DynamicContrastiveDataset(
        entities, split="val", train=False, base_seed=seed, length_range=(500, 550),
        intensity_mode="universal_deviation_intensity", intensity_min=0.2, intensity_max=4.0,
        intensity_metric_transform="identity",
    )


@torch.no_grad()
def extract(model, dataset, device="cpu"):
    embs, is_anom, i_raw = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        e = model(item["Y"].unsqueeze(0).to(device))["intensity"][0].cpu().numpy()
        embs.append(e)
        is_anom.append(item["shape_label"] == 1)
        i_raw.append(item["intensity_value_raw"])
    return np.array(embs), np.array(is_anom), np.array(i_raw)


def ordering_accuracy(i_vals, s_vals, eps=1e-9):
    n = len(i_vals)
    if n < 2:
        return float("nan"), 0
    correct, total = 0, 0
    for a, b in itertools.combinations(range(n), 2):
        di = i_vals[a] - i_vals[b]
        if abs(di) <= eps:
            continue
        total += 1
        if np.sign(di) == np.sign(s_vals[a] - s_vals[b]):
            correct += 1
    return (correct / total if total > 0 else float("nan")), total


def cross_group_ordering(i_vals, s_vals, mask_a, mask_b, same_group, eps=1e-9):
    idx_a, idx_b = np.where(mask_a)[0], np.where(mask_b)[0]
    correct, total = 0, 0
    seen = set()
    for a in idx_a:
        for b in idx_b:
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if same_group and key in seen:
                continue
            seen.add(key)
            di = i_vals[a] - i_vals[b]
            if abs(di) <= eps:
                continue
            total += 1
            if np.sign(di) == np.sign(s_vals[a] - s_vals[b]):
                correct += 1
    return (correct / total if total > 0 else float("nan")), total


def dist_summary(d):
    return {
        "min": float(d.min()), "p5": float(np.percentile(d, 5)), "p25": float(np.percentile(d, 25)),
        "median": float(np.median(d)), "p75": float(np.percentile(d, 75)), "p95": float(np.percentile(d, 95)),
        "max": float(d.max()), "mean": float(d.mean()), "std": float(d.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v23")
    parser.add_argument("--checkpoint", default="diagnostics/outputs/v2/v23_multitask_seed0/bestmodel.pkl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    dataset = build_val_dataset()
    embs, is_anom, i_raw_all = extract(model, dataset, device=args.device)

    normal_emb, anomaly_emb = embs[~is_anom], embs[is_anom]
    centroid = normal_emb.mean(axis=0)
    s_all = np.linalg.norm(embs - centroid, axis=1)
    s_anom = s_all[is_anom]
    s_normal = s_all[~is_anom]
    i_raw = i_raw_all[is_anom]

    # Section 5: global ordering metrics
    n = len(i_raw)
    pearson = float(pearsonr(i_raw, s_anom)[0])
    spearman = float(spearmanr(i_raw, s_anom)[0])
    kendall = float(kendalltau(i_raw, s_anom)[0])
    overall_acc, overall_n = ordering_accuracy(i_raw, s_anom)
    global_result = {
        "pearson": pearson, "spearman": spearman, "kendall_tau": kendall,
        "overall_ordering_accuracy": overall_acc, "overall_n_pairs": overall_n, "n": n,
    }
    print("Global:", json.dumps(global_result, indent=2))

    # Section 6: quantile (tertile) ordering
    t1, t2 = np.percentile(i_raw, [33.33, 66.67])
    low_mask, mid_mask, high_mask = i_raw <= t1, (i_raw > t1) & (i_raw <= t2), i_raw > t2
    group_masks = {"low": low_mask, "mid": mid_mask, "high": high_mask}
    quantile_result = {"tertile_boundaries": [float(t1), float(t2)]}
    for name_a, name_b in [("low", "low"), ("mid", "mid"), ("high", "high"),
                            ("low", "mid"), ("mid", "high"), ("low", "high")]:
        acc, npair = cross_group_ordering(i_raw, s_anom, group_masks[name_a], group_masks[name_b],
                                           same_group=(name_a == name_b))
        quantile_result[f"{name_a}_{name_b}"] = {"accuracy": acc, "n_pairs": npair}
    print("Quantile ordering:", json.dumps(quantile_result, indent=2))

    # Section 7: gap resolution by percentile bucket of |delta intensity|
    pairs = list(itertools.combinations(range(n), 2))
    gaps = np.array([abs(i_raw[a] - i_raw[b]) for a, b in pairs])
    g1, g2 = np.percentile(gaps, [33.33, 66.67])
    gap_result = {"gap_tertile_boundaries": [float(g1), float(g2)]}
    for label, lo, hi in [("small_gap", -np.inf, g1), ("medium_gap", g1, g2), ("large_gap", g2, np.inf)]:
        correct, total = 0, 0
        for (a, b), gap in zip(pairs, gaps):
            if not (lo < gap <= hi):
                continue
            di = i_raw[a] - i_raw[b]
            if abs(di) <= 1e-9:
                continue
            total += 1
            if np.sign(di) == np.sign(s_anom[a] - s_anom[b]):
                correct += 1
        gap_result[label] = {"accuracy": (correct / total if total > 0 else float("nan")), "n_pairs": total}
    print("Gap resolution:", json.dumps(gap_result, indent=2))

    # Section 8: normal vs anomaly geometry
    frac_anom_gt_normal = float(np.mean([a > b for a in s_anom for b in s_normal]))
    normal_vs_anomaly = {
        "normal_distance_distribution": dist_summary(s_normal),
        "anomaly_distance_distribution": dist_summary(s_anom),
        "fraction_anomaly_severity_gt_normal_severity": frac_anom_gt_normal,
    }
    print("Normal vs anomaly:", json.dumps(normal_vs_anomaly, indent=2))

    # Section 9: geometry utilization
    embedding_norms = np.linalg.norm(embs, axis=1)
    geometry = {
        "embedding_norm": dist_summary(embedding_norms),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "severity_distance_distribution": dist_summary(s_anom),
    }
    print("Geometry utilization:", json.dumps(geometry, indent=2))

    # sample-level export
    csv_path = os.path.join(args.output_dir, "v23_ordinal_intensity_samples.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "I_raw", "severity"])
        for idx in range(n):
            writer.writerow([idx, i_raw[idx], s_anom[idx]])

    # Section: learned severity curve (scatter + binned mean, NO reference curve)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(i_raw, s_anom, s=10, alpha=0.4, label="severity (predicted)")
    order = np.argsort(i_raw)
    n_bins = 8
    bin_edges = np.percentile(i_raw, np.linspace(0, 100, n_bins + 1))
    bin_centers, bin_means = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (i_raw >= lo) & (i_raw <= hi)
        if mask.sum() > 0:
            bin_centers.append(i_raw[mask].mean())
            bin_means.append(s_anom[mask].mean())
    ax.plot(bin_centers, bin_means, "o-", color="black", label="binned mean severity")
    ax.axhline(float(s_normal.mean()), color="green", linestyle=":", label="mean normal severity")
    ax.set_xlabel("I_raw (universal realized deviation)")
    ax.set_ylabel("learned severity ||e - c||")
    ax.set_title("V2.3: learned monotonic severity mapping (no reference curve imposed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "v23_severity_curve.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    result = {
        "global": global_result, "quantile_ordering": quantile_result, "gap_resolution": gap_result,
        "normal_vs_anomaly": normal_vs_anomaly, "geometry_utilization": geometry,
    }
    out_path = os.path.join(args.output_dir, "v23_ordinal_intensity_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}, {csv_path}, {plot_path}")


if __name__ == "__main__":
    main()
