import os

# Must happen before numpy/torch/sklearn are imported anywhere in the process
# (including transitively, via online_dataset/plots below) -- OpenBLAS reads
# these once, at first use, and picks nproc threads by default. On a
# many-core server that alone can exceed OpenBLAS's compiled-in thread
# ceiling (observed: "maximum of 128 threads" -> segfault), especially with
# --num_workers > 0 spawning additional worker processes that each try the
# same default. setdefault() so an operator's own env var still wins.
for _env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_var, "4")

import argparse
import csv
import time
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from core_clustering.colors import SURFACE
from core_clustering.metrics import evaluate_classification
from core_clustering.models import ConvAEC
from core_clustering.online_dataset import BasePool, OnlineWindowedDataset, get_anomaly, load_base_pool, materialize_windows
from core_clustering.single_entity import load_single_entity_split
from core_clustering.plots import plot_example_window, plot_tsne_by_class, plot_tsne_by_domain
from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
from core_clustering.splits import make_cross_domain_split
from core_clustering.trainer import Trainer, default_model_hyperparameters, write_run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-clustering-train-online",
        description=(
            "Train a cross-domain anomaly-type classifier with windowing + anomaly "
            "injection computed on the fly (not precomputed to disk) from an AnomSim "
            "base-pool dataset (anomsim-base-pool-dataset output). The injection itself "
            "is fixed once per (row, window, type) -- matching RedLamp's own Loader_aug, "
            "which injects once at construction and never re-injects -- only the actual "
            "array materialization is deferred/lazy, to keep memory bounded for large pools."
        ),
    )
    parser.add_argument("--dataset_dir", required=True, help="Output of anomsim-base-pool-dataset")
    parser.add_argument("--held_out_domains", nargs="*", default=[])
    parser.add_argument(
        "--single_entity", default=None,
        help="Train on exactly ONE entity_dir (e.g. 'sine_b0') instead of the whole pool -- "
             "RedLamp's own per-entity 'Self' model convention (temporal 90/10 split of that "
             "entity's own timeline, not a cross-entity group split). Mutually exclusive with "
             "--held_out_domains.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--window_step", type=int, default=10)
    parser.add_argument(
        "--class_list", default="redlamp",
        help="'redlamp' (default) pins RedLamp's own 12-class order -- required for the "
             "resulting checkpoint to be cross-loadable into RedLamp's ConvAEC. Or pass a "
             "comma-separated list for a custom order.",
    )
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0, help="-1 = cpu")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--c_loss_ratio", type=float, default=0.1)
    parser.add_argument("--apply_anomaly_mask", action="store_false", default=True)
    parser.add_argument("--label_smoothing", action="store_false", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--eval_max_samples", type=int, default=5000,
        help="Cap on how many (window, type) samples to materialize per domain for "
             "classification accuracy / t-SNE / sample plots -- these are one-off reporting "
             "steps, not training, so they don't need to touch every window.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30)
    parser.add_argument("--n_sample_plots", type=int, default=3)
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if output_dir/bestmodel.pkl already exists. Default: skip "
             "training and go straight to evaluation/plotting if a checkpoint from a "
             "prior run is already there (e.g. training finished but a later reporting "
             "step crashed) -- avoids silently redoing a long training run.",
    )
    return parser


def _load_epoch_history(output_dir: str):
    from core_clustering.trainer import EpochRecord

    path = os.path.join(output_dir, "epoch_history.json")
    if not os.path.exists(path):
        return []
    import json as _json

    with open(path) as f:
        raw = _json.load(f)
    return [EpochRecord(**r) for r in raw]


def _resolve_class_list(raw) -> list:
    if raw == "redlamp":
        return list(REDLAMP_ANOMALY_TYPES)
    return raw.split(",")


def _resolve_device(gpu: int) -> str:
    if gpu >= 0 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def _domain_window_counts(pool: BasePool, split, held_out_domains, window_size, window_step, class_list):
    rows = []
    for domain in pool.load_stats.domains:
        is_held_out = domain in held_out_domains

        def count(idx_array):
            domain_idx = idx_array[pool.domain[idx_array] == domain]
            if len(domain_idx) == 0:
                return 0
            return len(OnlineWindowedDataset(pool, domain_idx, window_size, window_step, class_list))

        rows.append({
            "domain": domain,
            "role": "held_out" if is_held_out else "included",
            "n_windows_train": count(split.train_idx) if not is_held_out else None,
            "n_windows_val": count(split.val_idx) if not is_held_out else None,
            "n_windows_eval": count(split.holdout_idx) if is_held_out else None,
            "n_base_instances_loaded": int(np.sum(pool.domain == domain)),
        })
    return rows


def _materialize_domain_batch(pool, idx_array, domain, window_size, window_step, class_list, max_samples, seed):
    """Filters idx_array down to one domain, then delegates to the
    domain-agnostic materialize_windows() (also reused directly by
    scripts that score an external model against a single AnomSim entity)."""
    domain_row_idx = idx_array[pool.domain[idx_array] == domain]
    return materialize_windows(pool, domain_row_idx, window_size, window_step, class_list, max_samples, seed)


def _extract_embeddings(model, Y: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    if len(Y) == 0:
        return np.zeros((0, 1))
    Y_t = torch.from_numpy(Y).float().transpose(2, 1)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(Y_t), batch_size):
            batch = Y_t[start : start + batch_size].to(device)
            _, _, x_enc = model(batch)
            embeddings.append(x_enc.reshape(x_enc.size(0), -1).cpu().numpy())
    return np.concatenate(embeddings)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.single_entity and args.held_out_domains:
        build_parser().error("--single_entity and --held_out_domains are mutually exclusive")
    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S", time.gmtime())
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    device = _resolve_device(args.gpu)
    class_list = _resolve_class_list(args.class_list)

    if args.single_entity:
        pool, split = load_single_entity_split(args.dataset_dir, args.single_entity, val_fraction=args.val_fraction)
        print(f"Loaded single entity {args.single_entity!r} from {args.dataset_dir} "
              f"(domain={split.included_domains[0]}, train/val timeline split)")
    else:
        pool = load_base_pool(args.dataset_dir)
        print(
            f"Loaded {pool.load_stats.n_loaded}/{pool.load_stats.n_attempted} base instances "
            f"({pool.load_stats.n_failed} failed) from {args.dataset_dir} -- domains={pool.load_stats.domains}"
        )
        split = make_cross_domain_split(pool, args.held_out_domains, val_fraction=args.val_fraction, seed=args.seed)
    for w in split.warnings:
        print(f"Warning: {w}")

    model_config = default_model_hyperparameters(
        n_features=1, n_time=args.window_size, classes=len(class_list),
        embedding_dim=args.embedding_dim, c_loss_ratio=args.c_loss_ratio,
        apply_anomaly_mask=args.apply_anomaly_mask, label_smoothing=args.label_smoothing,
    )
    model = ConvAEC(model_config)

    bestmodel_path = os.path.join(output_dir, "bestmodel.pkl")
    if os.path.isfile(bestmodel_path) and not args.force:
        print(f"[resume] {bestmodel_path} already exists -- skipping training, loading existing "
              f"checkpoint straight into evaluation/plotting. Pass --force to retrain from scratch.")
        model.load_state_dict(torch.load(bestmodel_path, map_location=device))
        model.to(device)
        history = _load_epoch_history(output_dir)
        if not history:
            print(f"[resume] no epoch_history.json found next to {bestmodel_path} (checkpoint predates "
                  f"this feature) -- run_summary.json's per-epoch fields will be empty for this run.")
    else:
        train_ds = OnlineWindowedDataset(pool, split.train_idx, args.window_size, args.window_step, class_list, base_seed=args.seed)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dl = None
        if len(split.val_idx) > 0:
            val_ds = OnlineWindowedDataset(pool, split.val_idx, args.window_size, args.window_step, class_list, base_seed=args.seed)
            val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"train windows/epoch={len(train_ds)}"
              + (f", val windows/epoch={len(val_ds)}" if val_dl is not None else ""))

        trainer = Trainer(model, device=device, patience=args.patience, output_dir=output_dir)
        history = trainer.train(train_dl, val_dl, epochs=args.epochs)

    classification_rows = []
    held_out_accuracy = []
    pooled_embeddings, pooled_class_idx, pooled_domain_idx = [], [], []
    all_domains = pool.load_stats.domains
    domain_to_idx = {d: i for i, d in enumerate(all_domains)}

    for domain in split.included_domains:
        data = _materialize_domain_batch(pool, split.val_idx, domain, args.window_size, args.window_step,
                                          class_list, args.eval_max_samples, args.seed)
        if data is None:
            continue
        Y, labels, _ds, _idx = data
        result = evaluate_classification(model, Y, labels, domain, device=device)
        classification_rows.append({"role": "included", **result.compact_dict()})
        emb = _extract_embeddings(model, Y, device)
        pooled_embeddings.append(emb)
        pooled_class_idx.append(labels.argmax(axis=1))
        pooled_domain_idx.append(np.full(len(emb), domain_to_idx[domain]))

    for domain in args.held_out_domains:
        data = _materialize_domain_batch(pool, split.holdout_idx, domain, args.window_size, args.window_step,
                                          class_list, args.eval_max_samples, args.seed)
        if data is None:
            continue
        Y, labels, _ds, _idx = data
        result = evaluate_classification(model, Y, labels, domain, device=device)
        classification_rows.append({"role": "held_out", **result.compact_dict()})
        held_out_accuracy.append(result.compact_dict())
        emb = _extract_embeddings(model, Y, device)
        pooled_embeddings.append(emb)
        pooled_class_idx.append(labels.argmax(axis=1))
        pooled_domain_idx.append(np.full(len(emb), domain_to_idx[domain]))

    csv_path = os.path.join(output_dir, "classification_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "role", "n_total", "n_correct", "n_incorrect", "accuracy"])
        writer.writeheader()
        for row in classification_rows:
            writer.writerow(row)

    domain_window_counts = _domain_window_counts(pool, split, args.held_out_domains, args.window_size, args.window_step, class_list)

    write_run_summary(
        os.path.join(output_dir, "run_summary.json"),
        run_id=run_id,
        dataset_dir=args.dataset_dir,
        seed=args.seed,
        device=device,
        included_domains=split.included_domains,
        held_out_domains=split.holdout_domains,
        val_fraction_requested=split.val_fraction_requested,
        val_fraction_actual=split.val_fraction_actual,
        n_entities_attempted=pool.load_stats.n_attempted,
        n_entities_loaded=pool.load_stats.n_loaded,
        n_entities_failed=pool.load_stats.n_failed,
        domain_window_counts=domain_window_counts,
        epochs=history,
        epochs_requested=args.epochs,
        early_stop_patience=args.patience,
        model_hyperparameters={"model": "ConvAEC", **asdict(model_config)},
        held_out_accuracy=held_out_accuracy,
    )

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if pooled_embeddings:
        embeddings = np.concatenate(pooled_embeddings)
        class_idx = np.concatenate(pooled_class_idx)
        domain_idx = np.concatenate(pooled_domain_idx)
        plot_tsne_by_class(
            embeddings, class_idx, class_list, os.path.join(plots_dir, "tsne_by_class.png"),
            title=f"{run_id}: embeddings by anomaly type", perplexity=args.tsne_perplexity, seed=args.seed,
        )
        plot_tsne_by_domain(
            embeddings, class_idx, domain_idx, class_list, all_domains,
            os.path.join(plots_dir, "tsne_by_domain.png"),
            title=f"{run_id}: embeddings by domain", perplexity=args.tsne_perplexity, seed=args.seed,
        )

    samples_dir = os.path.join(plots_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for domain in all_domains:
        is_held_out = domain in args.held_out_domains
        idx_array = split.holdout_idx if is_held_out else split.val_idx
        domain_row_idx = idx_array[pool.domain[idx_array] == domain]
        if len(domain_row_idx) == 0:
            continue
        sample_ds = OnlineWindowedDataset(pool, domain_row_idx, args.window_size, args.window_step, class_list)
        n = len(sample_ds)
        if n == 0:
            continue
        k = min(args.n_sample_plots, n)
        chosen = rng.choice(n, size=k, replace=False)
        for i, item_idx in enumerate(chosen):
            row_idx, window_idx, start, end, type_idx = sample_ds.index[int(item_idx)]
            anomaly_type = class_list[type_idx]
            window = pool.Y[row_idx][:, start:end]
            # Must match OnlineWindowedDataset.__getitem__'s own seed formula
            # exactly (base_seed, row, window, type -- no epoch, since
            # injection is now fixed once) or this plot would show a
            # different anomaly draw than what training/eval actually used.
            item_rng = np.random.default_rng([sample_ds.base_seed, row_idx, window_idx, type_idx])
            y, z, mask = get_anomaly(anomaly_type)().apply(window, item_rng)

            fig, ax = plt.subplots(figsize=(9, 2.3))
            fig.patch.set_facecolor(SURFACE)
            plot_example_window(ax, y, z, mask, anomaly_type, waveform_type=domain)
            fig.tight_layout()
            fig.savefig(os.path.join(samples_dir, f"{domain}_{i + 1}.png"), dpi=150, facecolor=SURFACE)
            plt.close(fig)

    print(f"Wrote run '{run_id}' to {output_dir}")


if __name__ == "__main__":
    main()
