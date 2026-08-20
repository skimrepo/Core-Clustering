"""Phase 2, Problem A.2: frozen-representation location probe across
Stage2/Stage3/Squeeze/Pool z, Linear vs small MLP. Uses the location_only
checkpoint by default (the model that actually tried to learn location,
giving the best chance of finding location info if it exists anywhere in
the architecture's representational capacity).

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/phase2_location_probe.py \\
        --checkpoint diagnostics/outputs/phase1/phase1_location_only_seed0/bestmodel.pkl \\
        --n_instances 1000 --seed 0 \\
        --output_dir diagnostics/outputs/phase2
"""
import argparse
import json
import os

import numpy as np
import torch

from core_clustering.dataset_dynamic_contrastive import DynamicContrastiveDataset, generate_entity_manifest
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive import ContrastiveEncoder

from diagnostics.metrics import regression_metrics
from diagnostics.representation_probe import cache_all_representations, train_probe

REPRESENTATIONS = ("stage2", "stage3", "squeeze", "pool_z")
PROBE_TYPES = ("linear", "mlp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--z_dim", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--output_dir", default="diagnostics/outputs/phase2")
    parser.add_argument("--target", default="location", choices=["location", "extent", "intensity"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entities = generate_entity_manifest(n_instances=args.n_instances, anomaly_ratio=0.5, base_seed=args.seed)
    train_ds = DynamicContrastiveDataset(entities, split="train", train=False, base_seed=args.seed,
                                          length_range=(args.length_min, args.length_max))
    val_ds = DynamicContrastiveDataset(entities, split="val", train=False, base_seed=args.seed,
                                        length_range=(args.length_min, args.length_max))

    config = ConvBottleneckConfig(n_time_max=args.max_len, bottleneck_channels=args.z_dim)
    model = ContrastiveEncoder(config, embedding_dim=args.embedding_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    print("Caching representations (train)...")
    train_cache = cache_all_representations(model, train_ds, device="cpu", max_len=args.max_len)
    print("Caching representations (val)...")
    val_cache = cache_all_representations(model, val_ds, device="cpu", max_len=args.max_len)

    is_anom_train = train_cache["shape_label"] == 1
    is_anom_val = val_cache["shape_label"] == 1
    y_train = train_cache[f"{args.target}_value"][is_anom_train].astype(np.float32)
    y_val = val_cache[f"{args.target}_value"][is_anom_val].astype(np.float32)

    results = {}
    for rep_name in REPRESENTATIONS:
        X_train = train_cache[rep_name][is_anom_train].astype(np.float32)
        X_val = val_cache[rep_name][is_anom_val].astype(np.float32)
        # standardize using train stats (helps both probe types converge)
        mu, sigma = X_train.mean(axis=0, keepdims=True), X_train.std(axis=0, keepdims=True).clip(min=1e-6)
        X_train_n = (X_train - mu) / sigma
        X_val_n = (X_val - mu) / sigma

        for probe_type in PROBE_TYPES:
            key = f"{rep_name}_{probe_type}"
            print(f"Training probe: {key}  (input_dim={X_train.shape[1]})")
            pred_val = train_probe(X_train_n, y_train, X_val_n, y_val, probe_type=probe_type,
                                    epochs=args.probe_epochs)
            metrics = regression_metrics(pred_val, y_val)
            results[key] = metrics
            print(f"  {key}: pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f} "
                  f"mae={metrics['mae']:.4f}")

    out = {
        "checkpoint": args.checkpoint, "target": args.target, "seed": args.seed,
        "n_train": int(is_anom_train.sum()), "n_val": int(is_anom_val.sum()),
        "results": results,
    }
    out_path = os.path.join(args.output_dir, f"{args.target}_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'Representation':<12} {'Probe':<8} {'Pearson':>9} {'Spearman':>9} {'MAE':>9}")
    for rep_name in REPRESENTATIONS:
        for probe_type in PROBE_TYPES:
            m = results[f"{rep_name}_{probe_type}"]
            print(f"{rep_name:<12} {probe_type:<8} {m['pearson']:>9.4f} {m['spearman']:>9.4f} {m['mae']:>9.4f}")


if __name__ == "__main__":
    main()
