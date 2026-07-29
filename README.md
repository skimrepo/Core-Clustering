# Core-Clustering

Trains RedLamp's `ConvAEC` model (CNN autoencoder + 12-class pseudo-anomaly
classifier) on a windowed dataset produced by
[AnomSim](https://github.com/skimrepo/AnomSim), with cross-domain
generalization evaluation (train on some waveform domains, test on
held-out ones) and RedLamp-compatible checkpoints.

## Install

```bash
pip install numpy matplotlib scikit-learn pytest
```

`torch` is intentionally left out of this list — install/verify it separately,
matching whatever CUDA version is already set up on your machine. Don't blindly
`pip install -r requirements.txt` on a server that already has a working
CUDA-matched torch; it may get silently replaced.

Run everything from the repo root (`python -m ...`, no install step needed).

## Quick start: train

```bash
python -m core_clustering.cli \
  --dataset_dir /path/to/AnomSim/data/windowed_v1 \
  --val_fraction 0.1 \
  --output_dir ./outputs \
  --run_id sim_v1 \
  --epochs 100 \
  --batch_size 128 \
  --gpu 0 \
  --embedding_dim 128 \
  --class_list redlamp
```

**`--class_list redlamp` is required if you plan to ever load this checkpoint
into RedLamp itself** (e.g. for cross-domain testing against real datasets via
RedLamp_Check's `simulation_cross_domain_metrics.py`). It pins the 12-class
order to match RedLamp's own (`normal` at index 0) — RedLamp's scoring code
hardcodes that assumption, and without this flag the class order is instead
derived alphabetically from whatever's in the dataset, which silently breaks
scoring semantics (no error, just meaningless results). Omit it only for a
pure internal Core-Clustering experiment that never touches RedLamp.

Other useful flags:
- `--held_out_domains sine trend` — exclude these AnomSim waveform domains from training
  entirely, to measure generalization to unseen domains.
- `--gpu -1` — force CPU.
- `--force` isn't needed here — a fresh `--run_id` just makes a new output folder.

### Output layout, `outputs/<run_id>/`

- `bestmodel.pkl` — the trained model's `state_dict` (loadable into RedLamp's own
  `ConvAEC` directly, if trained with `--class_list redlamp` and default architecture).
- `run_summary.json` — one structured file with everything: epoch history, split info,
  per-domain window counts, held-out classification accuracy, full model hyperparameters.
- `classification_accuracy.csv` — per-domain accuracy (included vs. held-out).
- `plots/tsne_by_class.png`, `plots/tsne_by_domain.png`, `plots/samples/` — embedding
  visualizations + representative example windows.

## Per-domain research analysis (deeper accuracy breakdown)

Kept separate from the core training code/output on purpose:

```bash
python -m research.analyze \
  --run_dir ./outputs/sim_v1 \
  --domains sine trend \
  --research_root ./research \
  --n_examples 10
```

Writes `research/sim_v1/<domain>/accuracy.json` +
`correct_examples.pdf`/`incorrect_examples.pdf` per domain — exact
correct/incorrect window counts and example plots, for whichever domains you
ask about (default: the run's own held-out domains).

## Tests

```bash
pytest
```
