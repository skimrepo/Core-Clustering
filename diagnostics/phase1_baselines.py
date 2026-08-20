"""Phase 1 screening (MTL_DIAGNOSTIC_REPORT.md plan): 4 single-task
baselines + 1 multi-task baseline, 1 seed, a short screening budget.
Writes one checkpoint/epoch_history/metrics.json per experiment under
--output_dir, plus a combined experiment_results.json manifest.

Independent experiments (the 5 modes) can run in parallel -- see the
reproduction commands in MTL_DIAGNOSTIC_REPORT.md for how to launch them
as separate background processes.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/phase1_baselines.py \\
        --n_instances 1000 --epochs 20 --patience 5 --seed 0 \\
        --output_dir diagnostics/outputs/phase1

    # single mode only (for parallel launch):
    PYTHONPATH=.:../AnomSim python3 diagnostics/phase1_baselines.py \\
        --modes location_only --seed 0
"""
import argparse
import functools
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_clustering.dataset_contrastive import BalancedBatchSampler, contrastive_pad_collate
from core_clustering.dataset_dynamic_contrastive import DynamicContrastiveDataset, generate_entity_manifest
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive import ATTRS, ContrastiveEncoder

from diagnostics.metrics import location_metrics, normal_relative_metrics, shape_metrics
from diagnostics.simple_trainer import SimpleTrainer

WEIGHTS_BY_MODE = {
    "shape_only": (1.0, 0.0, 0.0, 0.0),
    "location_only": (0.0, 1.0, 0.0, 0.0),
    "extent_only": (0.0, 0.0, 1.0, 0.0),
    "intensity_only": (0.0, 0.0, 0.0, 1.0),
    "multitask": (1.0, 1.0, 1.0, 1.0),
}


def build_loaders(args, seed):
    entities = generate_entity_manifest(n_instances=args.n_instances, anomaly_ratio=0.5, base_seed=seed)
    kwargs = dict(base_seed=seed, length_range=(args.length_min, args.length_max))
    train_ds = DynamicContrastiveDataset(entities, split="train", train=True, **kwargs)
    val_ds = DynamicContrastiveDataset(entities, split="val", train=False, **kwargs)

    collate = functools.partial(contrastive_pad_collate, max_len=args.max_len)
    train_labels = [1 if e.is_anomalous else 0 for e in train_ds.entities]
    train_sampler = BalancedBatchSampler(train_labels, args.batch_size, seed=seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate)

    val_labels = [1 if e.is_anomalous else 0 for e in val_ds.entities]
    val_dl = None
    if len(set(val_labels)) > 1:
        val_sampler = BalancedBatchSampler(val_labels, args.batch_size, seed=seed)
        if len(val_sampler) > 0:
            val_dl = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate)
    return train_ds, val_ds, train_dl, val_dl


@torch.no_grad()
def evaluate_all_metrics(model, dataset, device="cpu"):
    model.eval()
    embs = {a: [] for a in ATTRS}
    shape_labels, loc_vals, ext_vals, int_vals = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        e = model(item["Y"].unsqueeze(0).to(device))
        for a in ATTRS:
            embs[a].append(e[a][0].cpu().numpy())
        shape_labels.append(item["shape_label"])
        loc_vals.append(item["location_value"])
        ext_vals.append(item["extent_value"])
        int_vals.append(item["intensity_value"])

    shape_labels = np.array(shape_labels)
    is_anom = shape_labels == 1
    loc_vals, ext_vals, int_vals = np.array(loc_vals), np.array(ext_vals), np.array(int_vals)
    for a in ATTRS:
        embs[a] = np.array(embs[a])

    return {
        "shape": shape_metrics(embs["shape"], shape_labels),
        "location": location_metrics(embs["location"][is_anom], loc_vals[is_anom]),
        "extent": normal_relative_metrics(embs["extent"], is_anom, ext_vals),
        "intensity": normal_relative_metrics(embs["intensity"], is_anom, int_vals),
    }


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

    config = ConvBottleneckConfig(n_time_max=args.max_len, bottleneck_channels=args.z_dim)
    model = ContrastiveEncoder(config, embedding_dim=args.embedding_dim)
    trainer = SimpleTrainer(model, device=args.device, lr=args.lr, patience=args.patience,
                             weights=WEIGHTS_BY_MODE[mode], output_dir=out_dir)

    t0 = time.time()
    history, stop_reason = trainer.train(train_dl, val_dl, epochs=args.epochs)
    runtime = time.time() - t0

    model.load_state_dict(torch.load(os.path.join(out_dir, "bestmodel.pkl"), map_location=args.device))
    task_metrics = evaluate_all_metrics(model, val_ds, device=args.device)

    config_dict = {
        "mode": mode, "seed": seed, "z_dim": args.z_dim, "embedding_dim": args.embedding_dim,
        "n_instances": args.n_instances, "batch_size": args.batch_size, "lr": args.lr,
        "epochs_requested": args.epochs, "patience": args.patience,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    result = {
        "experiment_id": experiment_id,
        "task": mode,
        "seed": seed,
        "z_dim": args.z_dim,
        "epochs_run": len(history),
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "early_stop_reason": stop_reason,
        "runtime_seconds": runtime,
        "task_metrics": task_metrics,
        "status": "completed",
    }
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/phase1")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=4, help="bottleneck_channels")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--modes", nargs="+", default=list(WEIGHTS_BY_MODE.keys()), choices=list(WEIGHTS_BY_MODE.keys()))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "experiment_results.json")

    all_results = []
    for mode in args.modes:
        experiment_id = f"phase1_{mode}_seed{args.seed}"
        print(f"=== {experiment_id} ===")
        t0 = time.time()
        result = run_experiment(experiment_id, mode, args, args.seed)
        print(f"  runtime={time.time()-t0:.1f}s  best_val_loss={result['best_val_loss']:.4f}  "
              f"stop_reason={result['early_stop_reason']}")
        print(f"  task_metrics: {json.dumps(result['task_metrics'], indent=2)}")
        all_results.append(result)

    # merge with any existing manifest entries from prior partial runs
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
