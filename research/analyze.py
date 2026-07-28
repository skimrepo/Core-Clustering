import argparse
import json
import os
import time
from dataclasses import fields
from typing import List, Optional, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from core_clustering.colors import SURFACE
from core_clustering.dataset import load_windowed_dataset
from core_clustering.metrics import evaluate_classification
from core_clustering.models import ConvAEC, ModelConfig
from core_clustering.plots import plot_example_window

_MODEL_CONFIG_FIELDS = {f.name for f in fields(ModelConfig)}


def write_examples_pdf(dataset, global_indices: np.ndarray, out_path: str, n_examples: int, seed: int, domain: str, category: str) -> None:
    if len(global_indices) == 0:
        print(f"Note: domain '{domain}' has 0 {category} examples; skipping {os.path.basename(out_path)}.")
        return

    rng = np.random.default_rng(seed)
    k = min(n_examples, len(global_indices))
    if k < n_examples:
        print(f"Note: domain '{domain}' has only {k} {category} example(s) (< requested {n_examples}); using all available.")
    chosen = rng.choice(global_indices, size=k, replace=False)

    with PdfPages(out_path) as pdf:
        for idx in chosen:
            fig, ax = plt.subplots(figsize=(9, 2.3))
            fig.patch.set_facecolor(SURFACE)
            plot_example_window(
                ax, dataset.Y[idx], dataset.Z[idx], dataset.labels[idx], dataset.anomaly_type[idx],
                waveform_type=dataset.domain[idx],
            )
            fig.tight_layout()
            pdf.savefig(fig, facecolor=SURFACE)
            plt.close(fig)


def run_research_analysis(
    run_dir: str,
    dataset_dir: Optional[str] = None,
    domains: Optional[List[str]] = None,
    research_root: str = "./research",
    n_examples: int = 10,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> None:
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        run_summary = json.load(f)

    run_id = run_summary["run_id"]
    dataset_dir = dataset_dir or run_summary["dataset_dir"]
    domains = domains or run_summary["held_out_domains"]
    seed = seed if seed is not None else run_summary["seed"]

    dataset = load_windowed_dataset(dataset_dir)

    model_kwargs = {k: v for k, v in run_summary["model_hyperparameters"].items() if k in _MODEL_CONFIG_FIELDS}
    model = ConvAEC(ModelConfig(**model_kwargs))
    state_dict = torch.load(os.path.join(run_dir, "bestmodel.pkl"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    one_hot = dataset.one_hot_labels()

    for domain in domains:
        idx = np.where(dataset.domain == domain)[0]
        if len(idx) == 0:
            print(f"Note: domain '{domain}' not present in {dataset_dir}; skipping.")
            continue

        result = evaluate_classification(model, dataset.Y[idx], one_hot[idx], domain, device=device)

        domain_out_dir = os.path.join(research_root, run_id, domain)
        os.makedirs(domain_out_dir, exist_ok=True)

        accuracy_record = {
            "run_id": run_id,
            "source_run_dir": run_dir,
            **result.compact_dict(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(domain_out_dir, "accuracy.json"), "w") as f:
            json.dump(accuracy_record, f, indent=2)

        write_examples_pdf(
            dataset, idx[result.correct_indices], os.path.join(domain_out_dir, "correct_examples.pdf"),
            n_examples, seed, domain, "correct",
        )
        write_examples_pdf(
            dataset, idx[result.incorrect_indices], os.path.join(domain_out_dir, "incorrect_examples.pdf"),
            n_examples, seed, domain, "incorrect",
        )
        print(f"Wrote research analysis for domain '{domain}' to {domain_out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-analyze",
        description="Per-domain classification-accuracy breakdown + correct/incorrect example plots for a trained run.",
    )
    parser.add_argument("--run_dir", required=True, help="e.g. ./outputs/run-20260728-142310")
    parser.add_argument("--dataset_dir", default=None, help="default: read from run_dir/run_summary.json")
    parser.add_argument("--domains", nargs="*", default=None, help="default: run's held_out_domains")
    parser.add_argument("--research_root", default="./research")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="default: reuse the run's own seed")
    parser.add_argument("--gpu", type=int, default=0, help="-1 = cpu")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    run_research_analysis(
        args.run_dir,
        dataset_dir=args.dataset_dir,
        domains=args.domains,
        research_root=args.research_root,
        n_examples=args.n_examples,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()
