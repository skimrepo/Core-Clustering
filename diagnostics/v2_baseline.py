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
from core_clustering.trainer_contrastive_v2 import ContrastiveTrainerV2

from diagnostics.phase1_baselines import WEIGHTS_BY_MODE, build_loaders, evaluate_all_metrics

assert ATTRS == ("shape", "location", "extent", "intensity")


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
    )
    param_counts = count_parameters(model)
    trainer = ContrastiveTrainerV2(model, device=args.device, lr=args.lr, patience=args.patience,
                                    weights=WEIGHTS_BY_MODE[mode], output_dir=out_dir)

    t0 = time.time()
    history = trainer.train(train_dl, val_dl, epochs=args.epochs)
    runtime = time.time() - t0

    model.load_state_dict(torch.load(os.path.join(out_dir, "bestmodel.pkl"), map_location=args.device))
    task_metrics = evaluate_all_metrics(model, val_ds, device=args.device)

    config_dict = {
        "mode": mode, "seed": seed, "embedding_dim": args.embedding_dim,
        "head_proj_channels": args.head_proj_channels, "head_num_queries": args.head_num_queries,
        "head_mlp_hidden": args.head_mlp_hidden,
        "n_instances": args.n_instances, "batch_size": args.batch_size, "lr": args.lr,
        "epochs_requested": args.epochs, "patience": args.patience,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    result = {
        "experiment_id": experiment_id,
        "architecture": "v2",
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
    parser.add_argument("--attention_max_resolution", type=int, default=256)
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
        experiment_id = f"v2_{mode}_seed{args.seed}"
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
