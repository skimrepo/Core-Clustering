"""Phase 2, Problem B.3: cheap pairwise task ablation for extent. Reuses
phase1_baselines.py's run_experiment/build_loaders/evaluate_all_metrics
directly (same SimpleTrainer, same config) -- only the weight combination
changes, so results are directly comparable to Phase 1's extent_only and
multitask rows (which should be reused, not rerun).

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/phase2_pairs.py \\
        --pairs extent_shape --seed 0 \\
        --output_dir diagnostics/outputs/phase2/pairs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from diagnostics.phase1_baselines import run_experiment

PAIR_WEIGHTS = {
    # (shape, location, extent, intensity)
    "extent_shape": (1.0, 0.0, 1.0, 0.0),
    "extent_location": (0.0, 1.0, 1.0, 0.0),
    "extent_intensity": (0.0, 0.0, 1.0, 1.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/phase2/pairs")
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pairs", nargs="+", default=list(PAIR_WEIGHTS.keys()), choices=list(PAIR_WEIGHTS.keys()))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "experiment_results.json")

    # run_experiment reads WEIGHTS_BY_MODE by mode name from phase1_baselines;
    # monkeypatch-extend it here rather than editing that file.
    import diagnostics.phase1_baselines as p1
    p1.WEIGHTS_BY_MODE.update(PAIR_WEIGHTS)

    all_results = []
    for pair in args.pairs:
        experiment_id = f"phase2_{pair}_seed{args.seed}"
        print(f"=== {experiment_id} ===")
        t0 = time.time()
        result = run_experiment(experiment_id, pair, args, args.seed)
        print(f"  runtime={time.time()-t0:.1f}s  best_val_loss={result['best_val_loss']:.4f}  "
              f"stop_reason={result['early_stop_reason']}")
        print(f"  extent_metric: {json.dumps(result['task_metrics']['extent'], indent=2)}")
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
