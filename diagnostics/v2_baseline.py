"""MTL_V2_REPORT.md Section 5-8: V2 seed0 multitask baseline, directly
comparable to Phase 1's multitask baseline (phase1_baselines.py) -- same
build_loaders/evaluate_all_metrics, same n_instances/epochs/patience/seed
defaults, only the model+trainer classes differ (V2's ContrastiveEncoderV2/
ContrastiveTrainerV2 instead of V1's ContrastiveEncoder/SimpleTrainer).

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v2_baseline.py \\
        --n_instances 1000 --epochs 20 --patience 5 --seed 0 \\
        --output_dir diagnostics/outputs/v2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import ATTRS, ContrastiveEncoderV2, count_parameters
from core_clustering.target_transforms import ScalarMetricTargetTransform
from core_clustering.trainer_contrastive_v2 import ContrastiveTrainerV2

from diagnostics.metrics import regression_metrics
from diagnostics.phase1_baselines import WEIGHTS_BY_MODE, build_loaders, evaluate_all_metrics

assert ATTRS == ("shape", "location", "extent", "intensity")


def make_experiment_id(mode, seed, normalize_embedding, intensity_mode="legacy_native_intensity",
                        experiment_id_prefix=None):
    # V2.2 (universal_deviation_intensity) gets its own id prefix, layered on
    # top of V2.1's normalize_embedding prefix, so all variants' results can
    # coexist in the same manifest/output_dir without collision.
    # experiment_id_prefix overrides the auto-derived prefix entirely -- used
    # by V2.2a (same intensity_mode as V2.2, only the sampling range differs,
    # so the auto-derived prefix alone can't distinguish them).
    if experiment_id_prefix is not None:
        prefix = experiment_id_prefix
    elif intensity_mode == "universal_deviation_intensity":
        prefix = "v22"
    elif normalize_embedding:
        prefix = "v21"
    else:
        prefix = "v2"
    return f"{prefix}_{mode}_seed{seed}"


@torch.no_grad()
def evaluate_intensity_dual(model, dataset, device="cpu", metric_transform_mode="positive_unbounded_to_unit"):
    """MTL_V22_REPORT.md Section 11: evaluates the intensity head against
    BOTH the bounded metric-space target it was actually trained on
    (I_metric) and, via ScalarMetricTargetTransform's inverse, the semantic
    raw universal intensity (I_raw) -- so a change in label semantics alone
    (V2 -> V2.2) can be judged on the scale a human actually cares about,
    not just the bounded training target. No-op-safe for V2/V2.1 datasets
    (whose intensity_value_raw always equals intensity_value, i.e. the
    identity transform), which just makes the two evaluations identical.

    metric_transform_mode MUST match whatever transform the dataset actually
    applied (see intensity_metric_transform) -- V2.3's radial_ordinal
    objective never applies positive_unbounded_to_unit at all, so passing
    "identity" here makes the inverse a no-op and both evaluations agree
    (there IS no separate metric space to invert out of)."""
    model.eval()
    embs, is_anom, i_metric_true, i_raw_true = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        e = model(item["Y"].unsqueeze(0).to(device))["intensity"][0].cpu().numpy()
        embs.append(e)
        is_anom.append(item["shape_label"] == 1)
        i_metric_true.append(item["intensity_value"])
        i_raw_true.append(item.get("intensity_value_raw", item["intensity_value"]))

    embs = np.array(embs)
    is_anom = np.array(is_anom)
    i_metric_true = np.array(i_metric_true)
    i_raw_true = np.array(i_raw_true)
    normal, anomaly = embs[~is_anom], embs[is_anom]
    if len(normal) == 0 or len(anomaly) == 0:
        nan_metrics = {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"),
                       "spearman": float("nan"), "n": 0}
        return nan_metrics, nan_metrics

    centroid = normal.mean(axis=0)
    d = np.linalg.norm(anomaly - centroid, axis=1)
    metric_space = regression_metrics(d, i_metric_true[is_anom])

    transform = ScalarMetricTargetTransform(mode=metric_transform_mode)
    # The [0,1) clip is only meaningful for positive_unbounded_to_unit's
    # inverse (d/(1-d) blows up as d->1) -- identity's "inverse" is a
    # pure passthrough, so clipping it here would silently flatten every
    # d>=1 sample to the same value (a real bug this fixes: it previously
    # made V2.3's raw-space Pearson/Spearman come out NaN, since MOST
    # anomalous distances exceed 1 under the ordinal objective).
    if metric_transform_mode == "identity":
        pred_raw = d
    else:
        d_safe = np.clip(d, 0.0, 1.0 - 1e-6)  # d can exceed 1 (unit-sphere embeddings allow distance up to 2)
        pred_raw = np.array([transform.inverse(float(x)) for x in d_safe])
    raw_space = regression_metrics(pred_raw, i_raw_true[is_anom])
    return metric_space, raw_space


def run_experiment(experiment_id, mode, args, seed):
    out_dir = os.path.join(args.output_dir, experiment_id)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path) and not args.force:
        print(f"skip {experiment_id} (metrics.json exists, pass --force to rerun)")
        with open(metrics_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds, val_ds, train_dl, val_dl = build_loaders(args, seed)

    config = ConvBottleneckConfig(
        n_time_max=args.max_len, n_features=2,
        attention_max_resolution=args.attention_max_resolution,
    )
    model = ContrastiveEncoderV2(
        config, embedding_dim=args.embedding_dim, head_proj_channels=args.head_proj_channels,
        head_num_queries=args.head_num_queries, head_mlp_hidden=args.head_mlp_hidden,
        normalize_embedding=args.normalize_embedding,
    )
    param_counts = count_parameters(model)
    trainer = ContrastiveTrainerV2(model, device=args.device, lr=args.lr, patience=args.patience,
                                    weights=WEIGHTS_BY_MODE[mode], output_dir=out_dir,
                                    intensity_objective=args.intensity_objective)

    t0 = time.time()
    history = trainer.train(train_dl, val_dl, epochs=args.epochs)
    runtime = time.time() - t0

    model.load_state_dict(torch.load(os.path.join(out_dir, "bestmodel.pkl"), map_location=args.device))
    task_metrics = evaluate_all_metrics(model, val_ds, device=args.device)

    metric_transform_mode = "identity" if args.intensity_metric_transform == "identity" else (
        "positive_unbounded_to_unit")
    intensity_metric_space, intensity_raw_space = evaluate_intensity_dual(
        model, val_ds, device=args.device, metric_transform_mode=metric_transform_mode
    )
    task_metrics["intensity_metric_space"] = intensity_metric_space
    task_metrics["intensity_raw_space"] = intensity_raw_space

    config_dict = {
        "mode": mode, "seed": seed, "embedding_dim": args.embedding_dim,
        "head_proj_channels": args.head_proj_channels, "head_num_queries": args.head_num_queries,
        "head_mlp_hidden": args.head_mlp_hidden,
        "n_instances": args.n_instances, "batch_size": args.batch_size, "lr": args.lr,
        "epochs_requested": args.epochs, "patience": args.patience,
        "intensity_mode": args.intensity_mode, "intensity_min": args.intensity_min,
        "intensity_max": args.intensity_max, "intensity_objective": args.intensity_objective,
        "intensity_metric_transform": args.intensity_metric_transform,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    if args.experiment_id_prefix is not None:
        architecture = args.experiment_id_prefix
    elif args.intensity_mode == "universal_deviation_intensity":
        architecture = "v2.2"
    else:
        architecture = "v2.1" if args.normalize_embedding else "v2"
    result = {
        "experiment_id": experiment_id,
        "architecture": architecture,
        "task": mode,
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "runtime_seconds": runtime,
        "task_metrics": task_metrics,
        "param_counts": param_counts,
        "status": "completed",
    }
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v2")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--head_proj_channels", type=int, default=32)
    parser.add_argument("--head_num_queries", type=int, default=4)
    parser.add_argument("--head_mlp_hidden", type=int, default=64)
    parser.add_argument("--normalize_embedding", action="store_true",
                         help="V2.1: L2-normalize every AttributeHead's final embedding. Default off (V2).")
    parser.add_argument("--attention_max_resolution", type=int, default=256)
    parser.add_argument("--intensity_mode", default="legacy_native_intensity",
                         choices=["legacy_native_intensity", "universal_deviation_intensity"],
                         help="V2.2: 'universal_deviation_intensity' replaces the native generator "
                              "parameter with a type-agnostic realized-deviation intensity target.")
    parser.add_argument("--intensity_min", type=float, default=0.05)
    parser.add_argument("--intensity_max", type=float, default=8.0)
    parser.add_argument("--intensity_sampling", default="log_uniform", choices=["log_uniform"])
    parser.add_argument("--intensity_metric_transform", default=None,
                         choices=[None, "identity", "positive_unbounded_to_unit"],
                         help="Override the dataset's auto-derived intensity target transform. "
                              "V2.3 (--intensity_objective radial_ordinal) should pass 'identity' "
                              "so the raw, unbounded I_raw flows through untransformed.")
    parser.add_argument("--intensity_objective", default="radial_regression",
                         choices=["radial_regression", "radial_ordinal"],
                         help="V2.3: 'radial_ordinal' uses RadialOrdinalLoss (ordering-only "
                              "supervision). Default 'radial_regression' is unchanged V1-V2.2a behavior.")
    parser.add_argument("--experiment_id_prefix", default=None,
                         help="Override the auto-derived v2_/v21_/v22_ experiment_id prefix "
                              "(e.g. 'v22a' for a variant sharing intensity_mode with v22).")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["multitask"], choices=list(WEIGHTS_BY_MODE.keys()))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "v2_experiment_results.json")

    all_results = []
    for mode in args.modes:
        experiment_id = make_experiment_id(mode, args.seed, args.normalize_embedding, args.intensity_mode,
                                            experiment_id_prefix=args.experiment_id_prefix)
        print(f"=== {experiment_id} ===")
        t0 = time.time()
        result = run_experiment(experiment_id, mode, args, args.seed)
        print(f"  runtime={time.time()-t0:.1f}s  best_val_loss={result['best_val_loss']:.4f}")
        print(f"  task_metrics: {json.dumps(result['task_metrics'], indent=2)}")
        all_results.append(result)

    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
    by_id = {r["experiment_id"]: r for r in existing}
    for r in all_results:
        by_id[r["experiment_id"]] = r
    with open(manifest_path, "w") as f:
        json.dump(list(by_id.values()), f, indent=2)
    print(f"\nWrote manifest to {manifest_path} ({len(by_id)} experiments total)")


if __name__ == "__main__":
    main()
