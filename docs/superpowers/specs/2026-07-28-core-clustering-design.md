# Core-Clustering Design Spec

Date: 2026-07-28

## Purpose

RedLamp (time-series anomaly detection) is being split into two independent
projects:

- **AnomSim** (already built, sibling repo) — synthetic waveform generation +
  RedLamp-style windowed multi-type anomaly injection. Produces a directory
  of labeled window entities plus a `_manifest.jsonl` index.
- **Core-Clustering** (this project) — ONLY the model training/inference
  part of RedLamp. No data generation or anomaly injection lives here; it
  reads AnomSim's output directly from disk (plain `.npy`/JSON files, no
  runtime dependency on the `anomsim` Python package — a deliberate
  decoupling between the two projects).

Primary goal: evaluate **cross-domain generalization** — train a model while
excluding one or more domains entirely (domain = AnomSim waveform type, e.g.
`sine`, `binary_state`, `quantized_sensor`), then evaluate the trained model
against those held-out domains to measure how well it generalizes to
waveform types it never saw during training.

## Scope

- Univariate only (`n_features=1`), matching AnomSim's current output.
- Reuses RedLamp's model architecture (`ConvEncoder`/`ConvDecoder`/
  `NonLinClassifier`/`ConvAEC`) near-verbatim, with one fix: `NonLinClassifier`
  no longer applies `Softmax` before its output — `CrossEntropyLoss` expects
  raw logits and internally does numerically-stable `log_softmax`, so
  RedLamp's original double-softmax was a strict regression in gradient
  quality, not a deliberate choice. A `predict_proba()` helper is added for
  the few places actual probabilities are needed for display/thresholding.
- Classification target is the anomaly type AnomSim already injected
  (`meta["anomaly"]["type"]`, real ground truth) — no need to re-derive
  labels the way RedLamp's `Loader_aug` did from its own injection
  bookkeeping.
- Train/val split for early stopping is grouped by the compound key
  `(waveform.type, base_instance_id)` — every window from the same base
  series goes entirely to one of {train, val}, never split across, to avoid
  leakage between highly-correlated overlapping windows (confirmed with the
  user as the intentional, methodologically-correct choice over a faster but
  leakier pure-random split).
- Logging/summaries must be **compact and structured**, not verbose prose —
  the user's explicit, repeated requirement. RedLamp's own
  `stage_summary.json` (5 flat scalar fields) is the right *shape* to aim
  for, just extended with more fields relevant here.
- Per-domain classification accuracy breakdown (correct/incorrect window
  counts) plus correct-example and incorrect-example sample plots are
  **research-only** output, kept in a directory structurally separate from
  the core training/inference code and its own results — the user's explicit
  requirement that the codebase read cleanly as "just training/inference."

## Architecture

Flat package layout, mirroring AnomSim's own minimal style (no
`pyproject.toml`/packaging metadata, just `conftest.py` + `requirements.txt`):

```
Core-Clustering/
  core_clustering/
    dataset.py     # LoadStats, LoadedDataset, load_windowed_dataset()
    splits.py      # SplitResult, make_cross_domain_split()
    models.py      # ConvEncoder, ConvDecoder, NonLinClassifier, MetaAEC,
                    # ConvAEC, ModelConfig, predict_proba()
    colors.py       # shared 7 hex color constants (RedLamp/AnomSim convention)
    trainer.py       # EpochRecord, Trainer, write_run_summary(),
                      # default_model_hyperparameters()
    metrics.py        # ClassificationResult, evaluate_classification()
    plots.py           # plot_example_window() (shared single-window renderer,
                        # reused by research/analyze.py), plot_tsne_by_class(),
                        # plot_tsne_by_domain(), plot_representative_samples()
    cli.py               # `train` command: load -> split -> train -> eval -> save
  research/
    analyze.py            # run_research_analysis() + its own CLI entrypoint
    <run_id>/<domain>/     # generated output (gitignored)
  outputs/                 # generated core run output (gitignored)
    <run_id>/
      bestmodel.pkl
      run_summary.json
      classification_accuracy.csv
      plots/{tsne_by_class.png, tsne_by_domain.png, samples/*.png}
  tests/
```

## Data flow

1. `dataset.load_windowed_dataset(out_dir)` reads `out_dir/_manifest.jsonl`
   line by line, loading each entity's `Y.npy`/`labels.npy`/`Z.npy` directly
   (plain numpy/json, no `anomsim` import). A bad line (missing/corrupt file,
   unparsable JSON, missing required manifest field, or a `Y.npy` shape that
   doesn't match the dataset's modal window size) is recorded as a load
   failure with a reason and skipped — never crashes the whole load. Returns
   a `LoadedDataset` (stacked `Y`/`labels`/`Z`/`domain`/`anomaly_type`/
   `base_instance_id`/`window_index`/`entity_dir` arrays, plus `class_list`
   derived from whatever anomaly types are actually present) and a
   `LoadStats` (attempted/loaded/failed counts, failure reasons, domains
   present) — this is the source of the "몇 개의 entity를 로드성공/실패했는지"
   requirement.
2. `splits.make_cross_domain_split(dataset, held_out_domains, val_fraction,
   seed)` excludes `held_out_domains` entirely, then splits the remaining
   domains' `(domain, base_instance_id)` groups into train/val (proportional
   per domain, degenerate single-group domains go entirely to train with a
   logged warning, never silently mishandled).
3. `trainer.Trainer.train(train_dl, val_dl)` runs the REDLAMP-equivalent
   epoch loop (early stopping patience, `bestmodel.pkl` on improvement),
   accumulating one `EpochRecord` per epoch (10 fields: epoch, three train
   losses, three val losses, epoch_seconds, is_best, early_stop_counter) —
   this list becomes the `"epochs"` array in the final summary, replacing
   RedLamp's seven separate flat `.txt` arrays with one structured artifact.
   A one-line, numbers-only console print per epoch is kept for live
   feedback (not "verbose prose" — it's the *saved* artifact that changes).
4. `metrics.evaluate_classification(model, Y, labels, domain)` runs
   eval-mode inference and returns a `ClassificationResult` (counts,
   accuracy, and `correct_indices`/`incorrect_indices` into the caller's own
   arrays — cheap to recompute later without retraining).
5. `cli.py`'s `train` command writes, per run, into `outputs/<run_id>/`:
   `bestmodel.pkl`, `run_summary.json` (full schema below),
   `classification_accuracy.csv` (one row per domain, both included and
   held-out, columns `domain, role, n_total, n_correct, n_incorrect,
   accuracy`), and three plots (`tsne_by_class.png`, `tsne_by_domain.png`,
   `samples/{domain}_{1..3}.png`).
6. `research/analyze.py`'s `run_research_analysis(run_dir, dataset_dir,
   domains, ...)` re-loads a domain's windows, re-runs
   `evaluate_classification` against the already-trained `bestmodel.pkl`
   (cheap inference, no retraining), and writes
   `research/<run_id>/<domain>/{accuracy.json, correct_examples.pdf,
   incorrect_examples.pdf}` — `run_id` read from `run_summary.json` so
   multiple analyses never collide.

## `run_summary.json` schema (compact, flat where possible)

```jsonc
{
  "schema_version": 1,
  "run_id": "run-20260728-142310",
  "created_at": "2026-07-28T14:23:10Z",
  "dataset_dir": "...", "seed": 0, "device": "cpu",
  "included_domains": ["quantized_sensor", "binary_state"],
  "held_out_domains": ["sine"],
  "val_fraction_requested": 0.2, "val_fraction_actual": 0.187,
  "n_entities_attempted": 240, "n_entities_loaded": 236, "n_entities_failed": 4,
  "n_windows_train": 18422, "n_windows_val": 4230, "n_windows_total": 22652,
  "domain_window_counts": [
    {"domain": "binary_state", "role": "included", "n_windows_train": 9100,
     "n_windows_val": 2050, "n_windows_eval": null,
     "n_entities_loaded": 118, "n_entities_failed": 2},
    {"domain": "sine", "role": "held_out", "n_windows_train": null,
     "n_windows_val": null, "n_windows_eval": 3100,
     "n_entities_loaded": 40, "n_entities_failed": 1}
  ],
  "epochs_requested": 100, "epochs_ran": 34, "early_stopped": true,
  "early_stop_patience": 10, "early_stop_epoch": 33,
  "best_epoch": 23, "best_val_loss": 0.4021,
  "best_val_loss_ae": 0.3814, "best_val_loss_c": 0.0207,
  "total_train_seconds": 512.4, "mean_epoch_seconds": 15.07,
  "model_hyperparameters": {"model": "ConvAEC", "n_features": 1, "n_time": 100,
    "num_filters": [128,128,256,256], "embedding_dim": 128, "kernel_size": 4,
    "dropout": 0.2, "normalization": "batch", "stride": 2, "padding": 2,
    "classes": 12, "classifier_dim": 32, "c_loss_ratio": 0.1,
    "apply_anomaly_mask": true, "label_smoothing": true, "alpha": 0.1, "beta": 0.01},
  "held_out_accuracy": [
    {"domain": "sine", "n_total": 3100, "n_correct": 2108, "n_incorrect": 992, "accuracy": 0.68}
  ],
  "epochs": [
    {"epoch": 0, "train_loss": 0.91, "train_loss_ae": 0.85, "train_loss_c": 0.06,
     "val_loss": 0.88, "val_loss_ae": 0.83, "val_loss_c": 0.05,
     "epoch_seconds": 14.9, "is_best": true, "early_stop_counter": 0}
  ]
}
```

Every one-per-run field is a bare scalar. `domain_window_counts`,
`held_out_accuracy`, and `epochs` are the only arrays — each is genuinely
one-row-per-{domain,epoch} data, which is exactly as compact as a CSV row,
not prose. `model_hyperparameters` is the only nested dict, to avoid ~16
flat hyperparameter keys polluting the top-level namespace.

Held-out-domain accuracy numbers live in *both* `classification_accuracy.csv`
(full detail, all domains) and `run_summary.json`'s `held_out_accuracy` (just
the held-out subset) — intentional small duplication, since held-out
accuracy is the headline number this whole project exists to produce and
belongs in the one-stop-shop summary. Only `correct_indices`/
`incorrect_indices` and the actual example plots are reserved for
`research/`.

## Explicit Non-Goals (v1)

- Multivariate support (`n_features > 1`) — the model keeps `n_features` as
  a general constructor parameter (zero extra cost), but nothing in this
  project generates or tests multivariate input yet.
- A config/DSL layer — CLI flags only, matching AnomSim's own YAGNI stance.
- Re-deriving anomaly labels from scratch — AnomSim's ground truth is used
  directly.
