import argparse
import functools
import os
import random
import time
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_clustering.dataset_tcn import WholeSeriesDataset, load_whole_series_pool, pad_collate
from core_clustering.models_conv_bottleneck import ConvBottleneckAEC
from core_clustering.trainer_tcn import TCNTrainer, default_tcn_hyperparameters
from core_clustering.trainer import write_run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-clustering-train-tcn",
        description=(
            "Train the whole-series conv-bottleneck dual-head model (masked "
            "reconstruction + per-timepoint anomaly BCE) on an AnomSim_v3-style "
            "dataset (_manifest.jsonl + per-instance Y.npy/Z.npy/label.npy, split "
            "baked into the manifest -- no windowing, no domain holdout). Uses a "
            "plain (non-dilated) stride-2 conv encoder/decoder with a genuine "
            "temporal-downsample bottleneck (channel-squeezed waist layer), not "
            "the earlier dilated 'same'-padding TCN backbone -- that backbone had "
            "no capacity constraint, so masked-MSE reconstruction could trivially "
            "learn to copy the input through residual connections."
        ),
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0, help="-1 = cpu")
    parser.add_argument("--max_len", type=int, default=550)
    parser.add_argument("--num_filters", default=None,
                         help="Comma-separated channel widths, one per stride-2 encoder stage. "
                              "Default: auto-computed from --max_len/--target_bottleneck_len/--stride/"
                              "--channel_base/--channel_max.")
    parser.add_argument("--bottleneck_channels", type=int, default=4,
                         help="Channel width at the compressed waist -- the actual enforced capacity limit")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--padding_mode", default="reflect", choices=["reflect", "replicate", "circular", "zeros"])
    parser.add_argument("--num_stem_layers", type=int, default=1,
                         help="stride=1 layers before any downsampling starts, to avoid aliasing short periods")
    parser.add_argument("--target_bottleneck_len", type=int, default=32,
                         help="used to auto-compute stage count/channels from --max_len when --num_filters is not given")
    parser.add_argument("--channel_base", type=int, default=16)
    parser.add_argument("--channel_max", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--normalization", default="group", choices=["group", "layer", "batch"])
    parser.add_argument("--num_groups", type=int, default=8)
    parser.add_argument("--bce_loss_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Retrain even if bestmodel.pkl already exists")
    return parser


def _resolve_device(gpu: int) -> str:
    if gpu >= 0 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Seeding order matters: must happen before model construction and
    # DataLoader(shuffle=True), matching cli.py/online_cli.py exactly.
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

    train_records, train_stats = load_whole_series_pool(args.dataset_dir, split="train")
    val_records, val_stats = load_whole_series_pool(args.dataset_dir, split="val")
    print(train_stats.summary())
    print(val_stats.summary())

    collate = functools.partial(pad_collate, max_len=args.max_len)
    train_dl = DataLoader(WholeSeriesDataset(train_records), batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_dl = None
    if len(val_records) > 0:
        val_dl = DataLoader(WholeSeriesDataset(val_records), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    config = default_tcn_hyperparameters(
        n_features=1, n_time_max=args.max_len, num_filters=num_filters,
        bottleneck_channels=args.bottleneck_channels,
        kernel_size=args.kernel_size, stride=args.stride, padding=args.padding,
        padding_mode=args.padding_mode, num_stem_layers=args.num_stem_layers,
        target_bottleneck_len=args.target_bottleneck_len,
        channel_base=args.channel_base, channel_max=args.channel_max,
        dropout=args.dropout, normalization=args.normalization, num_groups=args.num_groups,
        bce_loss_ratio=args.bce_loss_ratio,
    )
    model = ConvBottleneckAEC(config)
    trainer = TCNTrainer(model, device=device, lr=args.lr, patience=args.patience, output_dir=output_dir)
    history = trainer.train(train_dl, val_dl, epochs=args.epochs)

    n_train_total = train_stats.n_manifest_lines if train_stats.n_manifest_lines else train_stats.n_loaded
    val_fraction_actual = (
        len(val_records) / (len(train_records) + len(val_records))
        if (len(train_records) + len(val_records)) > 0 else 0.0
    )

    write_run_summary(
        os.path.join(output_dir, "run_summary.json"),
        run_id=run_id,
        dataset_dir=args.dataset_dir,
        seed=args.seed,
        device=device,
        included_domains=["white_noise"],
        held_out_domains=[],
        val_fraction_requested=val_fraction_actual,
        val_fraction_actual=val_fraction_actual,
        n_entities_attempted=train_stats.n_attempted + val_stats.n_attempted,
        n_entities_loaded=train_stats.n_loaded + val_stats.n_loaded,
        n_entities_failed=train_stats.n_failed + val_stats.n_failed,
        domain_window_counts=[{
            "domain": "white_noise", "role": "included",
            "n_windows_train": len(train_records), "n_windows_val": len(val_records),
            "n_windows_eval": None, "n_entities_loaded": len(train_records) + len(val_records),
        }],
        epochs=history,
        epochs_requested=args.epochs,
        early_stop_patience=args.patience,
        model_hyperparameters={"model": "ConvBottleneckAEC", **asdict(config)},
        held_out_accuracy=[],
    )

    print(f"Wrote run '{run_id}' to {output_dir}")


if __name__ == "__main__":
    main()
