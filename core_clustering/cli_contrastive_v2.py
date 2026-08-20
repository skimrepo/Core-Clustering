import argparse
import functools
import os
import random
import time
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_clustering.dataset_contrastive import BalancedBatchSampler, contrastive_pad_collate
from core_clustering.dataset_dynamic_contrastive import (
    DynamicContrastiveDataset,
    dynamic_worker_init_fn,
    generate_entity_manifest,
)
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v2 import ContrastiveEncoderV2
from core_clustering.trainer_contrastive_v2 import ContrastiveTrainerV2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-clustering-contrastive-pretrain-v2",
        description=(
            "V2 multi-head contrastive pretraining (see MTL_V2_REPORT.md): removes the "
            "shared Conv1d(128->4) squeeze + single-query attention pool + shared z=4 "
            "bottleneck that Phase 2 diagnostics traced location/extent's information "
            "loss to. The shared trunk (unchanged Stem/Stage0-3) now feeds its raw "
            "(B,128,T') feature map directly to four independent, identically-"
            "architected AttributeHead instances (1x1 proj -> multi-query attention "
            "pool -> small MLP). Input gets a second channel: a per-sample normalized "
            "temporal position. Single AdamW over model+loss params (no per-attribute "
            "optimizer split). Loss functions themselves are unchanged from V1."
        ),
    )
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="also the base_seed for the entity manifest/injection")
    parser.add_argument("--batch_size", type=int, default=32, help="must be even (split evenly normal/anomalous)")
    parser.add_argument("--n_instances", type=int, default=1000,
                         help="background entities (fixed split/role); shift instances are injected on the "
                              "fly -- train resamples location/extent/intensity every call, val/test cache "
                              "them once so early-stopping/val_loss stay comparable across epochs.")
    parser.add_argument("--anomaly_ratio", type=float, default=0.5)
    parser.add_argument("--length_min", type=int, default=500)
    parser.add_argument("--length_max", type=int, default=550)
    parser.add_argument("--shift_min_range_ratio", type=float, default=0.05)
    parser.add_argument("--shift_max_range_ratio", type=float, default=0.5)
    parser.add_argument("--shift_min_magnitude_std_multiplier", type=float, default=0.2)
    parser.add_argument("--shift_max_magnitude_std_multiplier", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0, help="-1 = cpu")
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--embedding_dim", type=int, default=32, help="output dim of every AttributeHead")
    parser.add_argument("--head_proj_channels", type=int, default=32)
    parser.add_argument("--head_num_queries", type=int, default=4)
    parser.add_argument("--head_mlp_hidden", type=int, default=64)
    parser.add_argument("--num_filters", default=None,
                         help="Comma-separated channel widths. Default: auto-computed from --max_len.")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--padding_mode", default="reflect", choices=["reflect", "replicate", "circular", "zeros"])
    parser.add_argument("--num_stem_layers", type=int, default=1)
    parser.add_argument("--target_bottleneck_len", type=int, default=32)
    parser.add_argument("--channel_base", type=int, default=16)
    parser.add_argument("--channel_max", type=int, default=128)
    parser.add_argument("--attention_max_resolution", type=int, default=256)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--normalization", default="group", choices=["group", "layer", "batch"])
    parser.add_argument("--num_groups", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_shape", type=float, default=1.0)
    parser.add_argument("--weight_location", type=float, default=1.0)
    parser.add_argument("--weight_extent", type=float, default=1.0)
    parser.add_argument("--weight_intensity", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Retrain even if bestmodel.pkl already exists")
    return parser


def _resolve_device(gpu: int) -> str:
    if gpu >= 0 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S", time.gmtime())
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    device = _resolve_device(args.gpu)

    bestmodel_path = os.path.join(output_dir, "bestmodel.pkl")
    if os.path.exists(bestmodel_path) and not args.force:
        print(f"skip: {bestmodel_path} already exists (pass --force to retrain)")
        return

    num_filters = [int(c) for c in args.num_filters.split(",")] if args.num_filters else None

    entities = generate_entity_manifest(
        n_instances=args.n_instances, anomaly_ratio=args.anomaly_ratio, base_seed=args.seed,
    )
    dataset_kwargs = dict(
        base_seed=args.seed, length_range=(args.length_min, args.length_max),
        min_range_ratio=args.shift_min_range_ratio, max_range_ratio=args.shift_max_range_ratio,
        min_magnitude_std_multiplier=args.shift_min_magnitude_std_multiplier,
        max_magnitude_std_multiplier=args.shift_max_magnitude_std_multiplier,
    )
    train_ds = DynamicContrastiveDataset(entities, split="train", train=True, **dataset_kwargs)
    val_ds = DynamicContrastiveDataset(entities, split="val", train=False, **dataset_kwargs)
    print(f"train: {len(train_ds)} entities  |  val: {len(val_ds)} entities")

    worker_init_fn = dynamic_worker_init_fn if args.num_workers > 0 else None
    collate = functools.partial(contrastive_pad_collate, max_len=args.max_len)

    train_labels = [1 if e.is_anomalous else 0 for e in train_ds.entities]
    train_sampler = BalancedBatchSampler(train_labels, args.batch_size, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                           num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    val_dl = None
    val_labels = [1 if e.is_anomalous else 0 for e in val_ds.entities]
    if len(set(val_labels)) > 1:
        val_sampler = BalancedBatchSampler(val_labels, args.batch_size, seed=args.seed)
        if len(val_sampler) > 0:
            val_dl = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate,
                                 num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    config = ConvBottleneckConfig(
        n_time_max=args.max_len, n_features=2, num_filters=num_filters,
        kernel_size=args.kernel_size, stride=args.stride, padding=args.padding,
        padding_mode=args.padding_mode, num_stem_layers=args.num_stem_layers,
        target_bottleneck_len=args.target_bottleneck_len,
        channel_base=args.channel_base, channel_max=args.channel_max,
        attention_max_resolution=args.attention_max_resolution, attention_heads=args.attention_heads,
        dropout=args.dropout, normalization=args.normalization, num_groups=args.num_groups,
    )
    model = ContrastiveEncoderV2(
        config, embedding_dim=args.embedding_dim, head_proj_channels=args.head_proj_channels,
        head_num_queries=args.head_num_queries, head_mlp_hidden=args.head_mlp_hidden,
    )
    weights = (args.weight_shape, args.weight_location, args.weight_extent, args.weight_intensity)
    trainer = ContrastiveTrainerV2(model, device=device, lr=args.lr, patience=args.patience,
                                    weights=weights, output_dir=output_dir)
    trainer.train(train_dl, val_dl, epochs=args.epochs)

    print(f"Wrote run '{run_id}' to {output_dir}")


if __name__ == "__main__":
    main()
