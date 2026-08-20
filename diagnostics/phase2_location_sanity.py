"""Phase 2, Problem A.1: location loss sanity checks. Answers: is the LOSS
itself (and its optimization via SGD) sane, independent of the encoder?

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/phase2_location_sanity.py \\
        --checkpoint diagnostics/outputs/phase1/phase1_location_only_seed0/bestmodel.pkl \\
        --n_instances 1000 --seed 0 \\
        --output_dir diagnostics/outputs/phase2
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from core_clustering.dataset_dynamic_contrastive import DynamicContrastiveDataset, generate_entity_manifest
from core_clustering.losses_contrastive import PairwiseGapRegressionLoss
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive import ContrastiveEncoder

from diagnostics.metrics import embedding_collapse_stats, regression_metrics


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
        "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
        "n": len(x),
    }


def oracle_embedding_check(location_values, embedding_dim):
    """e_i = [location_i, 0, ..., 0] -- a hand-built 'perfect' embedding.
    If PairwiseGapRegressionLoss is sane, this should give ~0 loss (the
    embedding distance IS exactly the location gap along dim 0)."""
    n = len(location_values)
    emb = torch.zeros(n, embedding_dim)
    emb[:, 0] = torch.from_numpy(location_values).float()
    valid_mask = torch.ones(n, n, dtype=torch.bool)
    loss_fn = PairwiseGapRegressionLoss()
    loss = loss_fn(emb, torch.from_numpy(location_values).float(), valid_mask)
    return float(loss.item())


def tiny_network_optimizability_check(location_values, embedding_dim, epochs=300, lr=0.01):
    """Bypasses the encoder entirely: a tiny MLP takes the RAW location
    value as its only input and must learn (via the SAME loss, same
    optimizer family) to produce embeddings satisfying
    PairwiseGapRegressionLoss. Tests whether the loss is optimizable via
    gradient descent at all, given a maximally-informative input."""
    n = len(location_values)
    x = torch.from_numpy(location_values).float().unsqueeze(-1)  # (n, 1)
    net = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, embedding_dim))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    valid_mask = torch.ones(n, n, dtype=torch.bool)
    loss_fn = PairwiseGapRegressionLoss()
    target = torch.from_numpy(location_values).float()

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        emb = net(x)
        loss = loss_fn(emb, target, valid_mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


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
    parser.add_argument("--output_dir", default="diagnostics/outputs/phase2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entities = generate_entity_manifest(n_instances=args.n_instances, anomaly_ratio=0.5, base_seed=args.seed)
    val_ds = DynamicContrastiveDataset(entities, split="val", train=False, base_seed=args.seed,
                                        length_range=(args.length_min, args.length_max))

    config = ConvBottleneckConfig(n_time_max=args.max_len, bottleneck_channels=args.z_dim)
    model = ContrastiveEncoder(config, embedding_dim=args.embedding_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    embs, loc_vals = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            if item["shape_label"] != 1:
                continue
            e = model(item["Y"].unsqueeze(0))
            embs.append(e["location"][0].numpy())
            loc_vals.append(item["location_value"])
    embs = np.array(embs)
    loc_vals = np.array(loc_vals)
    n = len(loc_vals)

    D = np.linalg.norm(embs[:, None, :] - embs[None, :, :], axis=-1)
    gap = np.abs(loc_vals[:, None] - loc_vals[None, :])
    iu = np.triu_indices(n, k=1)

    result = {
        "checkpoint": args.checkpoint,
        "n_anomalous_val_instances": n,
        "target_gap_distribution": describe(gap[iu]),
        "predicted_distance_distribution": describe(D[iu]),
        "regression_metrics_dist_vs_gap": regression_metrics(D[iu], gap[iu]),
        "embedding_collapse_stats": embedding_collapse_stats(embs),
        "oracle_embedding_loss": oracle_embedding_check(loc_vals, args.embedding_dim),
    }

    tiny_losses = tiny_network_optimizability_check(loc_vals, args.embedding_dim)
    result["tiny_network_optimizability"] = {
        "initial_loss": tiny_losses[0],
        "final_loss": tiny_losses[-1],
        "min_loss": min(tiny_losses),
        "loss_every_50_epochs": tiny_losses[::50],
    }

    print(json.dumps(result, indent=2))
    with open(os.path.join(args.output_dir, "location_sanity.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {os.path.join(args.output_dir, 'location_sanity.json')}")


if __name__ == "__main__":
    main()
