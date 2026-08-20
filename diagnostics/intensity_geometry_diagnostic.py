"""MTL_INTENSITY_GEOMETRY_REPORT.md: observation-only diagnostic comparing
how V2.1 (raw intensity regression target) and V2.2a (I_raw/(1+I_raw)
metric-space target) actually placed intensity embeddings in distance-from-
normal-centroid space. Loads EXISTING checkpoints -- no retraining, no
architecture/loss/transform change. Centroid is computed the same way
diagnostics/metrics.py's normal_relative_metrics already does (mean of the
eval set's normal-instance embeddings) -- the established convention this
whole project's "intensity Pearson" numbers have always used, not a new
definition.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/intensity_geometry_diagnostic.py \\
        --output_dir diagnostics/outputs/intensity_geometry
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
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.dataset_dynamic_contrastive import DynamicContrastiveDataset, generate_entity_manifest
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import ContrastiveEncoderV2

BINS = [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
DELTA_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, float("inf"))]


def load_model(checkpoint_path, device="cpu"):
    config = ConvBottleneckConfig(n_time_max=550, n_features=2, attention_max_resolution=256)
    model = ContrastiveEncoderV2(config, embedding_dim=32, head_proj_channels=32,
                                  head_num_queries=4, head_mlp_hidden=64, normalize_embedding=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_val_dataset(intensity_mode, intensity_min=0.2, intensity_max=4.0, seed=0, n_instances=1000):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)
    kwargs = dict(base_seed=seed, length_range=(500, 550))
    if intensity_mode == "universal_deviation_intensity":
        kwargs.update(intensity_mode=intensity_mode, intensity_min=intensity_min, intensity_max=intensity_max)
    else:
        kwargs.update(min_magnitude_std_multiplier=intensity_min, max_magnitude_std_multiplier=intensity_max)
    return DynamicContrastiveDataset(entities, split="val", train=False, **kwargs)


@torch.no_grad()
def extract(model, dataset, device="cpu"):
    embs, is_anom, i_raw, targets, loc, ext = [], [], [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        e = model(item["Y"].unsqueeze(0).to(device))["intensity"][0].cpu().numpy()
        embs.append(e)
        is_anom.append(item["shape_label"] == 1)
        i_raw.append(item["intensity_value_raw"])
        targets.append(item["intensity_value"])
        loc.append(item["location_value"])
        ext.append(item["extent_value"])
    return {
        "embs": np.array(embs), "is_anom": np.array(is_anom), "i_raw": np.array(i_raw),
        "target": np.array(targets), "location": np.array(loc), "extent": np.array(ext),
    }


def regression_stats(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"),
                "mae": float("nan"), "rmse": float("nan"), "n": len(x)}
    return {
        "pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]),
        "mae": float(np.mean(np.abs(x - y))), "rmse": float(np.sqrt(np.mean((x - y) ** 2))), "n": len(x),
    }


def dist_summary(d):
    return {
        "min": float(d.min()), "p5": float(np.percentile(d, 5)), "p25": float(np.percentile(d, 25)),
        "median": float(np.median(d)), "p75": float(np.percentile(d, 75)), "p95": float(np.percentile(d, 95)),
        "max": float(d.max()), "mean": float(d.mean()), "std": float(d.std()),
    }


def ordering_accuracy(i_vals, d_vals):
    n = len(i_vals)
    if n < 2:
        return float("nan"), 0
    correct, total = 0, 0
    for a, b in itertools.combinations(range(n), 2):
        di = i_vals[a] - i_vals[b]
        dd = d_vals[a] - d_vals[b]
        if di == 0:
            continue
        total += 1
        if np.sign(di) == np.sign(dd):
            correct += 1
    return (correct / total if total > 0 else float("nan")), total


def analyze_one(name, model, dataset, device="cpu"):
    data = extract(model, dataset, device=device)
    embs, is_anom = data["embs"], data["is_anom"]
    normal_emb = embs[~is_anom]
    anomaly_emb = embs[is_anom]
    centroid = normal_emb.mean(axis=0)  # matches diagnostics/metrics.py's normal_relative_metrics convention
    d = np.linalg.norm(anomaly_emb - centroid, axis=1)

    i_raw = data["i_raw"][is_anom]
    target = data["target"][is_anom]
    loc = data["location"][is_anom]
    ext = data["extent"][is_anom]
    pred_error = d - target

    # unit-sphere utilization
    embedding_norms = np.linalg.norm(embs, axis=1)
    centroid_norm = float(np.linalg.norm(centroid))

    # Section 5: global relationship (I_raw vs predicted distance)
    global_stats = regression_stats(i_raw, d)
    global_stats["distance_distribution"] = dist_summary(d)

    # Section 6: bin analysis
    bin_rows = []
    for lo, hi in BINS:
        mask = (i_raw >= lo) & (i_raw <= hi if hi == BINS[-1][1] else i_raw < hi)
        n = int(mask.sum())
        row = {"bin": f"[{lo},{hi}{']' if hi == BINS[-1][1] else ')'}", "count": n}
        if n > 0:
            row["mean_i_raw"] = float(i_raw[mask].mean())
            row["mean_target"] = float(target[mask].mean())
            row["mean_distance"] = float(d[mask].mean())
            row["std_distance"] = float(d[mask].std())
            row["mae"] = float(np.mean(np.abs(d[mask] - target[mask])))
            row["rmse"] = float(np.sqrt(np.mean((d[mask] - target[mask]) ** 2)))
            if n >= 3 and np.std(i_raw[mask]) > 0:
                row["pearson"] = float(pearsonr(i_raw[mask], d[mask])[0])
                row["spearman"] = float(spearmanr(i_raw[mask], d[mask])[0])
            else:
                row["pearson"] = row["spearman"] = float("nan")
        bin_rows.append(row)

    # Section 9: high-intensity (I_raw>=2.0) vs low-intensity (I_raw<1.0)
    high_mask = i_raw >= 2.0
    low_mask = i_raw < 1.0

    def subset_stats(mask):
        n = int(mask.sum())
        if n < 2:
            return {"n": n}
        sub_d = d[mask]
        pairs = list(itertools.combinations(range(n), 2))
        pairwise_abs_diff = np.mean([abs(sub_d[a] - sub_d[b]) for a, b in pairs]) if pairs else float("nan")
        stats = regression_stats(i_raw[mask], sub_d)
        stats["std_distance"] = float(sub_d.std())
        stats["mean_pairwise_abs_distance_diff"] = float(pairwise_abs_diff)
        return stats

    high_intensity_stats = subset_stats(high_mask)
    low_intensity_stats = subset_stats(low_mask)

    # Section 10: pairwise ordering accuracy (overall + low-low/low-high/high-high)
    overall_ordering, overall_n = ordering_accuracy(i_raw, d)

    def paired_ordering(mask_a, mask_b, same_group):
        idx_a = np.where(mask_a)[0]
        idx_b = np.where(mask_b)[0]
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
                di, dd = i_raw[a] - i_raw[b], d[a] - d[b]
                if di == 0:
                    continue
                total += 1
                if np.sign(di) == np.sign(dd):
                    correct += 1
        return (correct / total if total > 0 else float("nan")), total

    low_low_acc, low_low_n = paired_ordering(low_mask, low_mask, same_group=True)
    high_high_acc, high_high_n = paired_ordering(high_mask, high_mask, same_group=True)
    low_high_acc, low_high_n = paired_ordering(low_mask, high_mask, same_group=False)

    # Section 11: local resolution by |delta intensity|
    delta_bins = []
    n = len(i_raw)
    for lo, hi in DELTA_BINS:
        correct, total = 0, 0
        for a, b in itertools.combinations(range(n), 2):
            delta = abs(i_raw[a] - i_raw[b])
            if not (lo <= delta < hi):
                continue
            di, dd = i_raw[a] - i_raw[b], d[a] - d[b]
            if di == 0:
                continue
            total += 1
            if np.sign(di) == np.sign(dd):
                correct += 1
        delta_bins.append({
            "range": f"[{lo},{hi if hi != float('inf') else 'inf'})",
            "n_pairs": total, "ordering_accuracy": (correct / total if total > 0 else float("nan")),
        })

    # Section 13: V2.1 "impossible target" (I_raw > 2) subset
    impossible_mask = i_raw > 2.0
    impossible_stats = subset_stats(impossible_mask)
    impossible_ordering, impossible_ordering_n = paired_ordering(impossible_mask, impossible_mask, same_group=True)
    impossible_stats["ordering_accuracy_within_subset"] = impossible_ordering
    impossible_stats["mean_target"] = float(target[impossible_mask].mean()) if impossible_mask.sum() else float("nan")
    impossible_stats["mean_distance"] = float(d[impossible_mask].mean()) if impossible_mask.sum() else float("nan")
    impossible_stats["distance_range"] = (
        [float(d[impossible_mask].min()), float(d[impossible_mask].max())] if impossible_mask.sum() else None
    )

    # Section 14: location/extent leakage (quick correlation only)
    leakage = {
        "corr_distance_location": float(pearsonr(d, loc)[0]) if np.std(loc) > 0 else float("nan"),
        "corr_distance_extent": float(pearsonr(d, ext)[0]) if np.std(ext) > 0 else float("nan"),
    }

    result = {
        "name": name,
        "global": global_stats,
        "bins": bin_rows,
        "high_intensity_ge2": high_intensity_stats,
        "low_intensity_lt1": low_intensity_stats,
        "ordering": {
            "overall": {"accuracy": overall_ordering, "n_pairs": overall_n},
            "low_low": {"accuracy": low_low_acc, "n_pairs": low_low_n},
            "high_high": {"accuracy": high_high_acc, "n_pairs": high_high_n},
            "low_high": {"accuracy": low_high_acc, "n_pairs": low_high_n},
        },
        "local_resolution_by_delta": delta_bins,
        "unit_sphere": {
            "embedding_norm": dist_summary(embedding_norms),
            "centroid_norm": centroid_norm,
        },
        "impossible_target_gt2": impossible_stats,
        "leakage": leakage,
    }
    samples = {
        "i_raw": i_raw, "target": target, "distance": d, "error": pred_error, "location": loc, "extent": ext,
    }
    return result, samples


def write_samples_csv(path, samples):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "I_raw", "training_target", "predicted_distance",
                          "prediction_error", "location", "extent"])
        for i in range(len(samples["i_raw"])):
            writer.writerow([
                i, samples["i_raw"][i], samples["target"][i], samples["distance"][i],
                samples["error"][i], samples["location"][i], samples["extent"][i],
            ])


def make_plot(out_path, results_v21, samples_v21, results_v22a, samples_v22a):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    x_ref = np.linspace(0.2, 4.0, 200)

    for ax, name, samples, ref_fn in (
        (axes[0], "V2.1 (raw target)", samples_v21, lambda x: x),
        (axes[1], "V2.2a (I/(1+I) target)", samples_v22a, lambda x: x / (1 + x)),
    ):
        ax.scatter(samples["i_raw"], samples["distance"], s=10, alpha=0.4, label="predicted distance")
        bin_centers, bin_means = [], []
        for lo, hi in BINS:
            mask = (samples["i_raw"] >= lo) & (samples["i_raw"] < hi + 1e-9)
            if mask.sum() > 0:
                bin_centers.append((lo + hi) / 2)
                bin_means.append(samples["distance"][mask].mean())
        ax.plot(bin_centers, bin_means, "o-", color="black", label="binned mean distance")
        ax.plot(x_ref, ref_fn(x_ref), "--", color="red", label="training target reference")
        ax.axhline(2.0, color="gray", linestyle=":", label="theoretical max distance (unit sphere)")
        ax.set_xlabel("I_raw")
        ax.set_ylabel("embedding distance from normal centroid")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/intensity_geometry")
    parser.add_argument("--v21_checkpoint", default="diagnostics/outputs/v2/v21_multitask_seed0/bestmodel.pkl")
    parser.add_argument("--v22a_checkpoint", default="diagnostics/outputs/v2/v22a_multitask_seed0/bestmodel.pkl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading V2.1...")
    model_v21 = load_model(args.v21_checkpoint, device=args.device)
    ds_v21 = build_val_dataset("legacy_native_intensity", intensity_min=0.2, intensity_max=4.0)
    results_v21, samples_v21 = analyze_one("v21", model_v21, ds_v21, device=args.device)

    print("Loading V2.2a...")
    model_v22a = load_model(args.v22a_checkpoint, device=args.device)
    ds_v22a = build_val_dataset("universal_deviation_intensity", intensity_min=0.2, intensity_max=4.0)
    results_v22a, samples_v22a = analyze_one("v22a", model_v22a, ds_v22a, device=args.device)

    print("V2.1 global:", json.dumps(results_v21["global"], indent=2))
    print("V2.2a global:", json.dumps(results_v22a["global"], indent=2))
    print("V2.1 ordering:", json.dumps(results_v21["ordering"], indent=2))
    print("V2.2a ordering:", json.dumps(results_v22a["ordering"], indent=2))

    write_samples_csv(os.path.join(args.output_dir, "intensity_geometry_samples_v21.csv"), samples_v21)
    write_samples_csv(os.path.join(args.output_dir, "intensity_geometry_samples_v22a.csv"), samples_v22a)

    with open(os.path.join(args.output_dir, "intensity_geometry_results.json"), "w") as f:
        json.dump({"v21": results_v21, "v22a": results_v22a}, f, indent=2)

    plot_path = os.path.join(args.output_dir, "intensity_geometry_v21_vs_v22a.png")
    make_plot(plot_path, results_v21, samples_v21, results_v22a, samples_v22a)
    print(f"\nWrote {args.output_dir}/intensity_geometry_results.json, sample CSVs, and {plot_path}")


if __name__ == "__main__":
    main()
