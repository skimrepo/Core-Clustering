"""Re-runs evaluate_v3 (fixed quantile-based intensity binning) against
ALREADY-TRAINED V3 checkpoints without retraining, updating each
experiment's metrics.json and the manifest in place.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v3_reeval.py \\
        --seeds 0 1 2 --output_dir diagnostics/outputs/v3
"""
import argparse
import json
import os
import sys

import torch

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ContrastiveEncoderV3

from diagnostics.v3_baseline import build_v3_loaders, evaluate_v3


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
    parser.add_argument("--k_regimes", type=int, nargs="+", default=[0, 3, 10, 30])
    parser.add_argument("--contamination_prob", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = os.path.join(args.output_dir, "v3_experiment_results.json")
    with open(manifest_path) as f:
        existing = json.load(f)
    by_id = {r["experiment_id"]: r for r in existing}

    for seed in args.seeds:
        experiment_id = f"v3_multitask_seed{seed}"
        out_dir = os.path.join(args.output_dir, experiment_id)
        ckpt_path = os.path.join(out_dir, "bestmodel.pkl")
        if not os.path.exists(ckpt_path):
            print(f"skip {experiment_id}: no checkpoint found at {ckpt_path}")
            continue

        _, val_ds, _, _ = build_v3_loaders(args, seed)
        config = ConvBottleneckConfig(n_time_max=args.max_len, n_features=2,
                                       attention_max_resolution=args.attention_max_resolution)
        model = ContrastiveEncoderV3(config, embedding_dim=args.embedding_dim)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        task_metrics = evaluate_v3(model, val_ds, device=args.device)
        print(f"=== {experiment_id} ===")
        print(json.dumps(task_metrics, indent=2))

        if experiment_id in by_id:
            by_id[experiment_id]["task_metrics"] = task_metrics
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(by_id[experiment_id], f, indent=2)

    with open(manifest_path, "w") as f:
        json.dump(list(by_id.values()), f, indent=2)
    print(f"\nUpdated {manifest_path}")


if __name__ == "__main__":
    main()
