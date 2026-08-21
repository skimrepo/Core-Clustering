"""MTL_V3_REPORT.md Sections E/F/G/H (reference-subset sensitivity,
contamination robustness, uncertainty calibration) plus the optional
Section 21 clustering sanity check. All evaluation-only against an
EXISTING checkpoint -- no retraining, no new checkpoint.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v3_eval_diagnostics.py \\
        --checkpoint diagnostics/outputs/v3/v3_multitask_seed0/bestmodel.pkl \\
        --output_dir diagnostics/outputs/v3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.dataset_dynamic_contrastive import generate_entity_manifest
from core_clustering.dataset_episodic import EpisodicContrastiveDataset
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ContrastiveEncoderV3
from core_clustering.prob_heads import laplace_nll

from diagnostics.metrics import shape_metrics

K_SWEEP = (0, 3, 10, 30, 100)

# Below this, per-dimension embedding std / mu std is considered collapsed.
# V3's actual collapse measured ~1e-7-1e-8 (see MTL_V3_REPORT.md); this
# threshold sits comfortably above that floor and well below the O(0.1-1)
# variation a healthy 32-dim unit-sphere embedding or an unconstrained
# scalar head would be expected to show.
COLLAPSE_STD_THRESHOLD = 1e-3


def load_model(checkpoint_path, device="cpu", embedding_dim=32, max_len=550, attention_max_resolution=256,
                num_filters=None, head_proj_channels=32, head_num_queries=4, head_mlp_hidden=64):
    config = ConvBottleneckConfig(n_time_max=max_len, n_features=2, num_filters=num_filters,
                                   attention_max_resolution=attention_max_resolution)
    model = ContrastiveEncoderV3(config, embedding_dim=embedding_dim, head_proj_channels=head_proj_channels,
                                  head_num_queries=head_num_queries, head_mlp_hidden=head_mlp_hidden)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_val_dataset(seed=0, n_instances=1000, length_range=(500, 550), intensity_min=0.2, intensity_max=4.0):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)
    return EpisodicContrastiveDataset(
        entities, split="val", train=False, base_seed=seed, length_range=length_range,
        intensity_mode="universal_deviation_intensity", intensity_min=intensity_min, intensity_max=intensity_max,
        intensity_metric_transform="identity",
    )


def _pack_reference_set(refs, T, device):
    K = len(refs)
    if K == 0:
        return None, None, None
    ref_x = torch.zeros(1, K, 1, T)
    ref_pad_mask = torch.zeros(1, K, 1, T)
    ref_k_valid_mask = torch.ones(1, K)
    for k, (Y_ref, n_ref) in enumerate(refs):
        ref_x[0, k, 0, :n_ref] = torch.from_numpy(Y_ref[0]).float()
        ref_pad_mask[0, k, 0, :n_ref] = 1.0
    return ref_x.to(device), ref_pad_mask.to(device), ref_k_valid_mask.to(device)


@torch.no_grad()
def _forward_with_k(model, dataset, idx, K, device, refs=None, max_len=550):
    # Query and references must share the SAME padded length T -- the model
    # concatenates their post-trunk feature maps along the channel dim,
    # which requires matching time dimensions (guaranteed during training
    # by collate padding everything to max_len; this eval script must do
    # the same instead of using each sequence's own raw, differing length).
    item = dataset[idx]
    n_q = item["Y"].shape[-1]
    Y = torch.zeros(1, 1, max_len)
    Y[0, :, :n_q] = item["Y"]
    query_pad_mask = torch.zeros(1, 1, max_len)
    query_pad_mask[:, :, :n_q] = 1.0
    Y, query_pad_mask = Y.to(device), query_pad_mask.to(device)

    if refs is None:
        refs, _ = ([], []) if K == 0 else dataset.sample_alternate_references(idx, K=K)
    ref_x, ref_pad_mask, ref_k_valid_mask = _pack_reference_set(refs, max_len, device)
    out = model(Y, query_pad_mask=query_pad_mask, ref_x=ref_x, ref_pad_mask=ref_pad_mask,
                ref_k_valid_mask=ref_k_valid_mask)
    gate = float(out["gate"][0].item())
    return {
        "location_mu": float(out["location_mu"][0]), "extent_mu": float(out["extent_mu"][0]),
        "intensity_mu": float(out["intensity_mu"][0]), "intensity_scale": float(out["intensity_scale"][0]),
        "location_scale": float(out["location_scale"][0]), "extent_scale": float(out["extent_scale"][0]),
        "gate": gate, "item": item,
    }


# --- COLLAPSE CHECK: must be run BEFORE interpreting any downstream metric --

def collapse_check(model, dataset, device="cpu", n_queries=150, k_for_gate=10, n_draws_gate=3, max_len=550):
    """Quantitative, output-level check for the exact failure mode V3
    exhibited (near-zero variance in every task output regardless of
    input). Deliberately independent of task-metric quality: a model can
    look "collapsed" by this check even if some downstream metric happens
    to look reasonable, and vice versa."""
    shape_embs, loc_mu, loc_scale, ext_mu, ext_scale = [], [], [], [], []
    int_mu, int_scale, d_vals, shape_labels = [], [], [], []
    for idx in range(min(n_queries, len(dataset))):
        item = dataset[idx]
        with torch.no_grad():
            out = model(item["Y"].unsqueeze(0).to(device))
        shape_embs.append(out["embeddings"]["shape"][0].cpu().numpy())
        loc_mu.append(float(out["location_mu"][0]))
        loc_scale.append(float(out["location_scale"][0]))
        ext_mu.append(float(out["extent_mu"][0]))
        ext_scale.append(float(out["extent_scale"][0]))
        int_mu.append(float(out["intensity_mu"][0]))
        int_scale.append(float(out["intensity_scale"][0]))
        d_vals.append(item["D"])
        shape_labels.append(item["shape_label"])

    shape_embs = np.array(shape_embs)
    shape_labels = np.array(shape_labels)
    d_vals = np.array(d_vals)
    loc_mu, loc_scale, ext_mu, ext_scale, int_mu, int_scale = map(
        np.array, (loc_mu, loc_scale, ext_mu, ext_scale, int_mu, int_scale)
    )

    norm = shape_embs / (np.linalg.norm(shape_embs, axis=1, keepdims=True) + 1e-12)
    cos_sim = norm @ norm.T
    n = cos_sim.shape[0]
    off_diag = cos_sim[~np.eye(n, dtype=bool)]
    per_dim_std = shape_embs.std(axis=0)
    shape_block = {
        "mean_pairwise_cosine_similarity": float(off_diag.mean()),
        "std_pairwise_cosine_similarity": float(off_diag.std()),
        "mean_per_dim_std": float(per_dim_std.mean()),
        "min_per_dim_std": float(per_dim_std.min()),
        "separation_metrics": shape_metrics(shape_embs, shape_labels),
    }

    location_block = {"std_mu": float(loc_mu.std()), "std_scale": float(loc_scale.std()),
                       "mean_scale": float(loc_scale.mean())}
    extent_block = {"std_mu": float(ext_mu.std()), "std_scale": float(ext_scale.std()),
                     "mean_scale": float(ext_scale.mean())}

    if len(int_mu) >= 3 and int_mu.std() > 0 and d_vals.std() > 0:
        mu_vs_d_corr = float(pearsonr(int_mu, d_vals)[0])
    else:
        mu_vs_d_corr = float("nan")
    intensity_block = {"std_mu": float(int_mu.std()), "std_scale": float(int_scale.std()),
                        "mean_scale": float(int_scale.mean()), "mu_vs_D_pearson_corr": mu_vs_d_corr}

    gates = []
    for idx in range(min(n_queries, len(dataset))):
        for _ in range(max(n_draws_gate, 1)):
            r = _forward_with_k(model, dataset, idx, k_for_gate, device, max_len=max_len)
            gates.append(r["gate"])
    reference_block = {"K": k_for_gate, "mean_gate": float(np.mean(gates)), "std_gate": float(np.std(gates))}

    collapsed = bool(
        shape_block["mean_per_dim_std"] < COLLAPSE_STD_THRESHOLD
        or (location_block["std_mu"] < COLLAPSE_STD_THRESHOLD
            and extent_block["std_mu"] < COLLAPSE_STD_THRESHOLD
            and intensity_block["std_mu"] < COLLAPSE_STD_THRESHOLD)
    )

    result = {
        "collapsed": collapsed, "collapse_std_threshold": COLLAPSE_STD_THRESHOLD,
        "shape": shape_block, "location": location_block, "extent": extent_block,
        "intensity": intensity_block, "reference": reference_block,
    }
    print(json.dumps(result, indent=2))
    return result


# --- Section 18: reference-subset sensitivity -------------------------------

def reference_sensitivity_sweep(model, dataset, device="cpu", n_queries=40, n_draws=5, k_values=K_SWEEP,
                                 max_len=550):
    query_indices = list(range(min(n_queries, len(dataset))))
    results = {}
    for K in k_values:
        per_query_std = {"location_mu": [], "extent_mu": [], "intensity_mu": []}
        gates = []
        pred_change = []
        for idx in query_indices:
            draws = [_forward_with_k(model, dataset, idx, K, device, max_len=max_len)
                     for _ in range(max(n_draws, 1))]
            for key in per_query_std:
                vals = np.array([d[key] for d in draws])
                per_query_std[key].append(float(vals.std()))
            gates.append(np.mean([d["gate"] for d in draws]))
            first = draws[0]
            pred_change.append(np.mean([
                abs(d["location_mu"] - first["location_mu"]) + abs(d["extent_mu"] - first["extent_mu"])
                + abs(d["intensity_mu"] - first["intensity_mu"])
                for d in draws[1:]
            ]) if len(draws) > 1 else 0.0)

        results[f"K={K}"] = {
            "mean_std_location_mu": float(np.mean(per_query_std["location_mu"])),
            "mean_std_extent_mu": float(np.mean(per_query_std["extent_mu"])),
            "mean_std_intensity_mu": float(np.mean(per_query_std["intensity_mu"])),
            "mean_gate": float(np.mean(gates)),
            "mean_prediction_change_from_resampling": float(np.mean(pred_change)),
        }
        print(f"K={K}: {json.dumps(results[f'K={K}'], indent=2)}")
    return results


# --- Section 19: reference contamination ------------------------------------

def contamination_test(model, dataset, device="cpu", n_queries=40, K=10, contamination_prob=0.3, max_len=550):
    clean_preds, contam_preds = [], []
    old_prob = dataset.contamination_prob
    for idx in range(min(n_queries, len(dataset))):
        dataset.contamination_prob = 0.0
        clean_refs, _ = dataset.sample_alternate_references(idx, K=K)
        dataset.contamination_prob = contamination_prob
        contam_refs, contam_flags = dataset.sample_alternate_references(idx, K=K)

        item = dataset[idx]
        n_q = item["Y"].shape[-1]
        Y = torch.zeros(1, 1, max_len)
        Y[0, :, :n_q] = item["Y"]
        query_pad_mask = torch.zeros(1, 1, max_len)
        query_pad_mask[:, :, :n_q] = 1.0
        Y, query_pad_mask = Y.to(device), query_pad_mask.to(device)
        with torch.no_grad():
            rx, rm, rk = _pack_reference_set(clean_refs, max_len, device)
            out_clean = model(Y, query_pad_mask=query_pad_mask, ref_x=rx, ref_pad_mask=rm, ref_k_valid_mask=rk)
            rx, rm, rk = _pack_reference_set(contam_refs, max_len, device)
            out_contam = model(Y, query_pad_mask=query_pad_mask, ref_x=rx, ref_pad_mask=rm, ref_k_valid_mask=rk)
        clean_preds.append({
            "location_mu": float(out_clean["location_mu"][0]), "extent_mu": float(out_clean["extent_mu"][0]),
            "intensity_mu": float(out_clean["intensity_mu"][0]), "intensity_scale": float(out_clean["intensity_scale"][0]),
        })
        contam_preds.append({
            "location_mu": float(out_contam["location_mu"][0]), "extent_mu": float(out_contam["extent_mu"][0]),
            "intensity_mu": float(out_contam["intensity_mu"][0]), "intensity_scale": float(out_contam["intensity_scale"][0]),
            "n_contaminated": int(sum(contam_flags)),
        })
    dataset.contamination_prob = old_prob

    mean_abs_change = {
        key: float(np.mean([abs(c[key] - cl[key]) for c, cl in zip(contam_preds, clean_preds)]))
        for key in ("location_mu", "extent_mu", "intensity_mu")
    }
    mean_scale_change = float(np.mean([c["intensity_scale"] - cl["intensity_scale"]
                                        for c, cl in zip(contam_preds, clean_preds)]))
    mean_n_contaminated = float(np.mean([c["n_contaminated"] for c in contam_preds]))
    result = {
        "K": K, "contamination_prob": contamination_prob, "mean_n_contaminated_refs": mean_n_contaminated,
        "mean_abs_prediction_change": mean_abs_change, "mean_intensity_scale_change": mean_scale_change,
    }
    print(json.dumps(result, indent=2))
    return result


# --- Section 17: uncertainty diagnostics ------------------------------------

def uncertainty_diagnostics(model, dataset, device="cpu", n_queries=150):
    loc_err, loc_scale, ext_err, ext_scale, int_err, int_scale = [], [], [], [], [], []
    loc_nll, ext_nll, int_nll = [], [], []
    loc_covered = {0.5: [], 0.8: [], 0.95: []}
    int_covered = {0.5: [], 0.8: [], 0.95: []}

    for idx in range(min(n_queries, len(dataset))):
        item = dataset[idx]
        Y = item["Y"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(Y)
        is_anom = item["shape_label"] == 1
        D = item["D"]

        mu_i, s_i = float(out["intensity_mu"][0]), float(out["intensity_scale"][0])
        int_err.append(abs(D - mu_i))
        int_scale.append(s_i)
        int_nll.append(float(laplace_nll(torch.tensor([D]), torch.tensor([mu_i]), torch.tensor([s_i])).item()))
        for p in int_covered:
            half_width = -s_i * np.log(1 - p)  # Laplace CDF inverse for a two-sided interval
            int_covered[p].append(abs(D - mu_i) <= half_width)

        if is_anom:
            mu_l, s_l = float(out["location_mu"][0]), float(out["location_scale"][0])
            loc_err.append(abs(item["location_value"] - mu_l))
            loc_scale.append(s_l)
            loc_nll.append(float(laplace_nll(torch.tensor([item["location_value"]]), torch.tensor([mu_l]),
                                              torch.tensor([s_l])).item()))
            for p in loc_covered:
                half_width = -s_l * np.log(1 - p)
                loc_covered[p].append(abs(item["location_value"] - mu_l) <= half_width)

            mu_e, s_e = float(out["extent_mu"][0]), float(out["extent_scale"][0])
            ext_err.append(abs(item["extent_value"] - mu_e))
            ext_scale.append(s_e)
            ext_nll.append(float(laplace_nll(torch.tensor([item["extent_value"]]), torch.tensor([mu_e]),
                                              torch.tensor([s_e])).item()))

    def corr_block(err, scale):
        err, scale = np.array(err), np.array(scale)
        if len(err) < 3 or np.std(err) == 0 or np.std(scale) == 0:
            return {"pearson": float("nan"), "spearman": float("nan")}
        return {"pearson": float(pearsonr(err, scale)[0]), "spearman": float(spearmanr(err, scale)[0])}

    result = {
        "location": {
            "error_vs_uncertainty_corr": corr_block(loc_err, loc_scale),
            "mean_uncertainty": float(np.mean(loc_scale)) if loc_scale else float("nan"),
            "mean_nll": float(np.mean(loc_nll)) if loc_nll else float("nan"),
            "coverage": {str(p): float(np.mean(v)) for p, v in loc_covered.items() if v},
        },
        "extent": {
            "error_vs_uncertainty_corr": corr_block(ext_err, ext_scale),
            "mean_uncertainty": float(np.mean(ext_scale)) if ext_scale else float("nan"),
            "mean_nll": float(np.mean(ext_nll)) if ext_nll else float("nan"),
        },
        "intensity": {
            "error_vs_uncertainty_corr": corr_block(int_err, int_scale),
            "mean_uncertainty": float(np.mean(int_scale)),
            "mean_nll": float(np.mean(int_nll)),
            "coverage": {str(p): float(np.mean(v)) for p, v in int_covered.items()},
        },
    }
    print(json.dumps(result, indent=2))
    return result


# --- Section 21: light clustering sanity check ------------------------------

def clustering_sanity_check(model, dataset, device="cpu", n_queries=150):
    from sklearn.cluster import KMeans

    embs, labels = [], []
    for idx in range(min(n_queries, len(dataset))):
        item = dataset[idx]
        with torch.no_grad():
            out = model(item["Y"].unsqueeze(0).to(device))
        embs.append(out["embeddings"]["shape"][0].cpu().numpy())
        labels.append(item["shape_label"])
    embs, labels = np.array(embs), np.array(labels)

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(embs)
    assigned = km.labels_
    # label-agnostic agreement: try both cluster<->label pairings, take the better one
    agree_a = float(np.mean(assigned == labels))
    agree_b = float(np.mean(assigned == (1 - labels)))
    agreement = max(agree_a, agree_b)

    normal_mask = labels == 0
    cluster_purity_normal = float(np.mean(assigned[normal_mask] == np.bincount(assigned[normal_mask]).argmax())) \
        if normal_mask.any() else float("nan")

    result = {
        "n_samples": len(labels), "kmeans_label_agreement": agreement,
        "normal_majority_cluster_purity": cluster_purity_normal,
        "note": "Diagnostic only -- NOT used inside training. Does NOT assume largest cluster == normal; "
                "agreement is computed label-aware here purely because ground truth is available for THIS "
                "evaluation, not because the model or a future deployment would have it.",
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="diagnostics/outputs/v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_filters", default=None)
    parser.add_argument("--head_proj_channels", type=int, default=32)
    parser.add_argument("--head_num_queries", type=int, default=4)
    parser.add_argument("--head_mlp_hidden", type=int, default=64)
    parser.add_argument("--attention_max_resolution", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_if_collapsed", action="store_true",
                         help="Stop after the collapse check if it reports collapsed=true "
                              "(downstream diagnostics are not meaningfully interpretable then).")
    parser.add_argument("--output_name", default="v3_eval_diagnostics.json")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_filters = [int(c) for c in args.num_filters.split(",")] if args.num_filters else None
    model = load_model(args.checkpoint, device=args.device, embedding_dim=args.embedding_dim,
                        max_len=args.max_len, attention_max_resolution=args.attention_max_resolution,
                        num_filters=num_filters, head_proj_channels=args.head_proj_channels,
                        head_num_queries=args.head_num_queries, head_mlp_hidden=args.head_mlp_hidden)
    dataset = build_val_dataset(seed=args.seed, n_instances=args.n_instances,
                                 length_range=(args.length_min, args.length_max))

    print("=== COLLAPSE CHECK (must precede any downstream interpretation) ===")
    collapse = collapse_check(model, dataset, device=args.device, max_len=args.max_len)
    out_path = os.path.join(args.output_dir, args.output_name)
    if collapse["collapsed"] and args.skip_if_collapsed:
        result = {"collapse_check": collapse,
                   "note": "Downstream diagnostics skipped: collapse_check reported collapsed=true."}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\ncollapsed=true and --skip_if_collapsed set -- stopping. Wrote {out_path}")
        return

    print("\n=== Section 18: reference-subset sensitivity ===")
    sensitivity = reference_sensitivity_sweep(model, dataset, device=args.device, max_len=args.max_len)
    print("\n=== Section 19: reference contamination ===")
    contamination = contamination_test(model, dataset, device=args.device, max_len=args.max_len)
    print("\n=== Section 17: uncertainty diagnostics ===")
    uncertainty = uncertainty_diagnostics(model, dataset, device=args.device)
    print("\n=== Section 21: clustering sanity check ===")
    clustering = clustering_sanity_check(model, dataset, device=args.device)

    result = {
        "collapse_check": collapse,
        "reference_sensitivity": sensitivity, "contamination": contamination,
        "uncertainty": uncertainty, "clustering": clustering,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
