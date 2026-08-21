"""MTL_V3_REPORT.md Sections C/D/I: trains + evaluates V3 (reference-set
conditioning + probabilistic outputs) at the SAME scale/split/simulator
config as the V2.1 baseline, across 3 seeds. Also backfills V2.1's own
seed1/seed2 (only seed0 existed from earlier reports) by reusing
diagnostics/v2_baseline.py's run_experiment directly -- same code path that
already produced every prior "V2.1" number in this report series.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v3_baseline.py \\
        --n_instances 1000 --epochs 20 --patience 5 --seeds 0 1 2 \\
        --output_dir diagnostics/outputs/v3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.dataset_contrastive import BalancedBatchSampler
from core_clustering.dataset_dynamic_contrastive import generate_entity_manifest
from core_clustering.dataset_episodic import DEFAULT_K_REGIMES, EpisodicContrastiveDataset, episodic_pad_collate
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ATTRS, ContrastiveEncoderV3
from core_clustering.trainer_contrastive_v3 import ContrastiveTrainerV3

from diagnostics.metrics import regression_metrics, shape_metrics
from diagnostics.v2_baseline import run_experiment as run_v21_experiment

def build_v3_loaders(args, seed):
    entities = generate_entity_manifest(n_instances=args.n_instances, anomaly_ratio=0.5, base_seed=seed)
    kwargs = dict(
        base_seed=seed, length_range=(args.length_min, args.length_max),
        intensity_mode="universal_deviation_intensity", intensity_min=args.intensity_min,
        intensity_max=args.intensity_max, intensity_metric_transform="identity",
        k_regimes=tuple(args.k_regimes), contamination_prob=args.contamination_prob,
        include_alternate_references=True,
    )
    train_ds = EpisodicContrastiveDataset(entities, split="train", train=True, **kwargs)
    val_ds = EpisodicContrastiveDataset(entities, split="val", train=False, **kwargs)

    collate = lambda batch: episodic_pad_collate(batch, max_len=args.max_len)  # noqa: E731
    train_labels = [1 if e.is_anomalous else 0 for e in train_ds.entities]
    train_sampler = BalancedBatchSampler(train_labels, args.batch_size, seed=seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate)

    val_dl = None
    val_labels = [1 if e.is_anomalous else 0 for e in val_ds.entities]
    if len(set(val_labels)) > 1:
        val_sampler = BalancedBatchSampler(val_labels, args.batch_size, seed=seed)
        if len(val_sampler) > 0:
            val_dl = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate)
    return train_ds, val_ds, train_dl, val_dl


@torch.no_grad()
def evaluate_v3(model, dataset, device="cpu"):
    model.eval()
    embs = {a: [] for a in ATTRS}
    loc_mu, ext_mu, int_mu, int_scale = [], [], [], []
    shape_labels, loc_vals, ext_vals, d_vals = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        out = model(item["Y"].unsqueeze(0).to(device))  # K=0 eval: global-only behavior
        for a in ATTRS:
            embs[a].append(out["embeddings"][a][0].cpu().numpy())
        loc_mu.append(float(out["location_mu"][0]))
        ext_mu.append(float(out["extent_mu"][0]))
        int_mu.append(float(out["intensity_mu"][0]))
        int_scale.append(float(out["intensity_scale"][0]))
        shape_labels.append(item["shape_label"])
        loc_vals.append(item["location_value"])
        ext_vals.append(item["extent_value"])
        d_vals.append(item["D"])

    shape_labels = np.array(shape_labels)
    is_anom = shape_labels == 1
    loc_vals, ext_vals, d_vals = np.array(loc_vals), np.array(ext_vals), np.array(d_vals)
    loc_mu, ext_mu, int_mu, int_scale = map(np.array, (loc_mu, ext_mu, int_mu, int_scale))
    for a in ATTRS:
        embs[a] = np.array(embs[a])

    result = {
        "shape": shape_metrics(embs["shape"], shape_labels),
        "location": regression_metrics(loc_mu[is_anom], loc_vals[is_anom]),
        "extent": regression_metrics(ext_mu[is_anom], ext_vals[is_anom]),
        "intensity": regression_metrics(int_mu, d_vals),
    }

    # Quantile-based bins over the anomalous (D>0) subset -- D's range
    # scales with each instance's own physical signal amplitude (sigma_ref
    # is log-uniform up to ~31.6), so a FIXED absolute range like the old
    # sigma-normalized I_raw's [0.2,4.0] would silently drop most of the
    # distribution. Normal (D=0) instances get their own explicit bin.
    bins = [{"bin": "normal (D=0)", "count": int((~is_anom).sum())}]
    if (~is_anom).any():
        bins[0]["mean_target_D"] = float(d_vals[~is_anom].mean())
        bins[0]["mean_predicted_mu"] = float(int_mu[~is_anom].mean())
        bins[0]["mean_predicted_scale"] = float(int_scale[~is_anom].mean())

    anom_d = d_vals[is_anom]
    if len(anom_d) >= 5:
        edges = np.percentile(anom_d, [0, 20, 40, 60, 80, 100])
        edges[0] -= 1e-9  # include the minimum in the first bin
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = is_anom & (d_vals > lo) & (d_vals <= hi)
            n = int(mask.sum())
            row = {"bin": f"({lo:.4g},{hi:.4g}]", "count": n}
            if n > 0:
                row["mean_target_D"] = float(d_vals[mask].mean())
                row["mean_predicted_mu"] = float(int_mu[mask].mean())
                row["mean_predicted_scale"] = float(int_scale[mask].mean())
            bins.append(row)
    result["intensity_binned"] = bins
    return result


def run_v3_experiment(experiment_id, args, seed):
    out_dir = os.path.join(args.output_dir, experiment_id)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path) and not args.force:
        print(f"skip {experiment_id} (metrics.json exists, pass --force to rerun)")
        with open(metrics_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds, val_ds, train_dl, val_dl = build_v3_loaders(args, seed)

    config = ConvBottleneckConfig(n_time_max=args.max_len, n_features=2,
                                   attention_max_resolution=args.attention_max_resolution)
    model = ContrastiveEncoderV3(config, embedding_dim=args.embedding_dim)
    n_params_total = sum(p.numel() for p in model.parameters())
    trainer = ContrastiveTrainerV3(model, device=args.device, lr=args.lr, patience=args.patience,
                                    output_dir=out_dir, seed=seed)

    t0 = time.time()
    history = trainer.train(train_dl, val_dl, epochs=args.epochs)
    runtime = time.time() - t0

    model.load_state_dict(torch.load(os.path.join(out_dir, "bestmodel.pkl"), map_location=args.device))
    task_metrics = evaluate_v3(model, val_ds, device=args.device)

    result = {
        "experiment_id": experiment_id, "architecture": "v3", "seed": seed,
        "epochs_run": len(history), "best_epoch": trainer.best_epoch, "best_val_loss": trainer.best_val_loss,
        "runtime_seconds": runtime, "task_metrics": task_metrics, "n_params_total": n_params_total,
        "status": "completed",
    }
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


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
    parser.add_argument("--k_regimes", type=int, nargs="+", default=list(DEFAULT_K_REGIMES))
    parser.add_argument("--contamination_prob", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_v21_backfill", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_v3_results = []
    for seed in args.seeds:
        experiment_id = f"v3_multitask_seed{seed}"
        print(f"=== {experiment_id} ===")
        t0 = time.time()
        result = run_v3_experiment(experiment_id, args, seed)
        print(f"  runtime={time.time()-t0:.1f}s  best_val_loss={result['best_val_loss']}")
        print(f"  task_metrics: {json.dumps(result['task_metrics'], indent=2)}")
        all_v3_results.append(result)

    manifest_path = os.path.join(args.output_dir, "v3_experiment_results.json")
    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
    by_id = {r["experiment_id"]: r for r in existing}
    for r in all_v3_results:
        by_id[r["experiment_id"]] = r
    with open(manifest_path, "w") as f:
        json.dump(list(by_id.values()), f, indent=2)
    print(f"\nWrote {manifest_path}")

    if not args.skip_v21_backfill:
        print("\n=== backfilling V2.1 seeds (reusing diagnostics/v2_baseline.py) ===")

        class _V21Args:
            pass

        v21_args = _V21Args()
        v21_args.output_dir = "diagnostics/outputs/v2"
        v21_args.n_instances = args.n_instances
        v21_args.length_min = args.length_min
        v21_args.length_max = args.length_max
        v21_args.max_len = args.max_len
        v21_args.batch_size = args.batch_size
        v21_args.embedding_dim = 32
        v21_args.head_proj_channels = 32
        v21_args.head_num_queries = 4
        v21_args.head_mlp_hidden = 64
        v21_args.normalize_embedding = True
        v21_args.attention_max_resolution = args.attention_max_resolution
        v21_args.intensity_mode = "legacy_native_intensity"
        v21_args.intensity_min = args.intensity_min
        v21_args.intensity_max = args.intensity_max
        v21_args.intensity_sampling = "log_uniform"
        v21_args.intensity_metric_transform = None
        v21_args.intensity_objective = "radial_regression"
        v21_args.experiment_id_prefix = None
        v21_args.lr = args.lr
        v21_args.epochs = args.epochs
        v21_args.patience = args.patience
        v21_args.device = args.device
        v21_args.force = args.force

        for seed in args.seeds:
            experiment_id = f"v21_multitask_seed{seed}"
            print(f"--- backfilling {experiment_id} ---")
            run_v21_experiment(experiment_id, "multitask", v21_args, seed)


if __name__ == "__main__":
    main()
