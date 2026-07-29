import argparse
import csv
import os
import time
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_clustering.dataset import load_windowed_dataset
from core_clustering.metrics import evaluate_classification
from core_clustering.models import ConvAEC
from core_clustering.plots import plot_representative_samples, plot_tsne_by_class, plot_tsne_by_domain
from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
from core_clustering.splits import make_cross_domain_split
from core_clustering.trainer import Trainer, default_model_hyperparameters, make_torch_dataset, write_run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-clustering-train",
        description="Train a cross-domain anomaly-type classifier on an AnomSim windowed dataset.",
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--held_out_domains", nargs="*", default=[])
    parser.add_argument("--val_fraction", type=float, default=0.2)
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
    parser.add_argument("--tsne_perplexity", type=float, default=30)
    parser.add_argument("--tsne_max_samples", type=int, default=2000)
    parser.add_argument("--n_sample_plots", type=int, default=3)
    parser.add_argument(
        "--class_list", default=None,
        help="Pin the classifier's class order instead of deriving it alphabetically from "
             "the dataset. Pass the literal 'redlamp' to use RedLamp's own 12-class order "
             "(required for a checkpoint trained here to be cross-loadable into RedLamp's "
             "ConvAEC with matching classifier semantics — see core_clustering/redlamp_compat.py), "
             "or a comma-separated list for a custom order. Default: alphabetical, derived from "
             "whatever anomaly types are present in the dataset.",
    )
    return parser


def _resolve_class_list(raw):
    if raw is None:
        return None
    if raw == "redlamp":
        return REDLAMP_ANOMALY_TYPES
    return raw.split(",")


def _resolve_device(gpu: int) -> str:
    if gpu >= 0 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def _domain_window_counts(dataset, split, held_out_domains):
    rows = []
    for domain in dataset.load_stats.domains:
        is_held_out = domain in held_out_domains
        n_train = int(np.sum(dataset.domain[split.train_idx] == domain)) if not is_held_out else None
        n_val = int(np.sum(dataset.domain[split.val_idx] == domain)) if not is_held_out else None
        n_eval = int(np.sum(dataset.domain[split.holdout_idx] == domain)) if is_held_out else None
        n_loaded = int(np.sum(dataset.domain == domain))
        rows.append({
            "domain": domain,
            "role": "held_out" if is_held_out else "included",
            "n_windows_train": n_train,
            "n_windows_val": n_val,
            "n_windows_eval": n_eval,
            "n_entities_loaded": n_loaded,
        })
    return rows


def _extract_embeddings(model, dataset, indices, device, batch_size=256):
    if len(indices) == 0:
        return np.zeros((0, 1)), np.zeros((0,), dtype=np.int64)
    Y_t = torch.from_numpy(dataset.Y[indices]).float().transpose(2, 1)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(Y_t), batch_size):
            batch = Y_t[start : start + batch_size].to(device)
            _, _, x_enc = model(batch)
            embeddings.append(x_enc.reshape(x_enc.size(0), -1).cpu().numpy())
    return np.concatenate(embeddings), indices


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S", time.gmtime())
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    device = _resolve_device(args.gpu)

    dataset = load_windowed_dataset(args.dataset_dir, class_list=_resolve_class_list(args.class_list))
    print(dataset.load_stats.summary())

    split = make_cross_domain_split(dataset, args.held_out_domains, val_fraction=args.val_fraction, seed=args.seed)
    for w in split.warnings:
        print(f"Warning: {w}")

    train_ds = make_torch_dataset(dataset, split.train_idx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = None
    if len(split.val_idx) > 0:
        val_ds = make_torch_dataset(dataset, split.val_idx)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model_config = default_model_hyperparameters(
        n_features=1,
        n_time=dataset.window_size,
        classes=len(dataset.class_list),
        embedding_dim=args.embedding_dim,
        c_loss_ratio=args.c_loss_ratio,
        apply_anomaly_mask=args.apply_anomaly_mask,
        label_smoothing=args.label_smoothing,
    )
    model = ConvAEC(model_config)
    trainer = Trainer(model, device=device, patience=args.patience, output_dir=output_dir)
    history = trainer.train(train_dl, val_dl, epochs=args.epochs)

    classification_rows = []
    held_out_accuracy = []
    one_hot = dataset.one_hot_labels()

    for domain in split.included_domains:
        idx = split.val_idx[dataset.domain[split.val_idx] == domain]
        if len(idx) == 0:
            continue
        result = evaluate_classification(model, dataset.Y[idx], one_hot[idx], domain, device=device)
        classification_rows.append({"role": "included", **result.compact_dict()})

    for domain in args.held_out_domains:
        idx = split.holdout_idx[dataset.domain[split.holdout_idx] == domain]
        if len(idx) == 0:
            continue
        result = evaluate_classification(model, dataset.Y[idx], one_hot[idx], domain, device=device)
        classification_rows.append({"role": "held_out", **result.compact_dict()})
        held_out_accuracy.append(result.compact_dict())

    csv_path = os.path.join(output_dir, "classification_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "role", "n_total", "n_correct", "n_incorrect", "accuracy"])
        writer.writeheader()
        for row in classification_rows:
            writer.writerow(row)

    domain_window_counts = _domain_window_counts(dataset, split, args.held_out_domains)

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
        n_entities_attempted=dataset.load_stats.n_attempted,
        n_entities_loaded=dataset.load_stats.n_loaded,
        n_entities_failed=dataset.load_stats.n_failed,
        domain_window_counts=domain_window_counts,
        epochs=history,
        epochs_requested=args.epochs,
        early_stop_patience=args.patience,
        model_hyperparameters={"model": "ConvAEC", **asdict(model_config)},
        held_out_accuracy=held_out_accuracy,
    )

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    eval_idx = np.concatenate([split.val_idx, split.holdout_idx]) if len(split.holdout_idx) else split.val_idx
    if len(eval_idx) > args.tsne_max_samples:
        rng = np.random.default_rng(args.seed)
        eval_idx = rng.choice(eval_idx, size=args.tsne_max_samples, replace=False)

    if len(eval_idx) > 0:
        embeddings, used_idx = _extract_embeddings(model, dataset, eval_idx, device)
        class_to_idx = {c: i for i, c in enumerate(dataset.class_list)}
        class_idx = np.array([class_to_idx[a] for a in dataset.anomaly_type[used_idx]])
        all_domains = sorted(set(dataset.domain.tolist()))
        domain_to_idx = {d: i for i, d in enumerate(all_domains)}
        domain_idx = np.array([domain_to_idx[d] for d in dataset.domain[used_idx]])

        plot_tsne_by_class(
            embeddings, class_idx, dataset.class_list, os.path.join(plots_dir, "tsne_by_class.png"),
            title=f"{run_id}: embeddings by anomaly type", perplexity=args.tsne_perplexity, seed=args.seed,
        )
        plot_tsne_by_domain(
            embeddings, class_idx, domain_idx, dataset.class_list, all_domains,
            os.path.join(plots_dir, "tsne_by_domain.png"),
            title=f"{run_id}: embeddings by domain", perplexity=args.tsne_perplexity, seed=args.seed,
        )

    samples_dir = os.path.join(plots_dir, "samples")
    plot_representative_samples(dataset, split.val_idx, samples_dir, n_per_domain=args.n_sample_plots, seed=args.seed)

    print(f"Wrote run '{run_id}' to {output_dir}")


if __name__ == "__main__":
    main()
