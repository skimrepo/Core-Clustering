"""
For each AnomSim_v1 waveform type (9 registered in the sibling AnomSim
repo's anomsim/waveforms/basic.py: sine, random_walk, white_noise, arma,
trend, square, sawtooth, binary_state, quantized_sensor), picks one
representative entity and plots exactly ONE "normal" (unmodified) window and
ONE anomaly-injected window (default: spike) -- the Cross-AnomSim analogue
of RedLamp_Check's main.save_anomaly_type_examples, used to visually check
whether Cross-AnomSim's actual training data resembles what it's evaluated
against (see RedLamp_Check's DS_1/DS_2 gap analysis).

Randomized per entity (like online_cli.py's own --n_sample_plots sampling,
not the "always take the very first match" bug an earlier version of this
script had -- that made every waveform type's anomaly-example land at the
exact same relative position, since row_idx=window_idx=0 and a shared
base_seed=0 made the injection RNG identical across all 9 types): for the
chosen entity, enumerates every (row, window, type) combo via
OnlineWindowedDataset's own indexing, then --rng_seed-seeded rng.choice's
ONE candidate matching type_idx=0 (normal) and one matching --anomaly_type's
index. Each waveform type also gets its own OnlineWindowedDataset base_seed
(derived from its position in the sorted type list) so the same type_idx at
the same (row, window) would still diverge across entities.

Runs entirely from local data files -- no GPU, no trained model needed,
pure data + injection + plotting. Does not modify online_cli.py, online_dataset.py,
single_entity.py, plots.py, or redlamp_compat.py -- only imports from them.
"""
import argparse
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, '..', 'AnomSim'))  # sibling repo, for `anomsim.*`

from core_clustering.colors import SURFACE
from core_clustering.online_dataset import OnlineWindowedDataset, get_anomaly
from core_clustering.single_entity import load_single_entity_split
from core_clustering.plots import plot_example_window
from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES

WINDOW_SIZE = 100
WINDOW_STEP = 10


def entities_by_type(dataset_dir, manifest_name='_manifest.jsonl'):
    by_type = {}
    with open(os.path.join(dataset_dir, manifest_name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            by_type.setdefault(meta['type'], []).append(meta['entity_dir'])
    return by_type


def random_window_of_type(sample_ds, type_idx, rng):
    candidates = [(row_idx, window_idx, start, end)
                  for row_idx, window_idx, start, end, t_idx in sample_ds.index
                  if t_idx == type_idx]
    if not candidates:
        return None
    return candidates[rng.integers(len(candidates))]


def plot_one(pool, sample_ds, row_idx, window_idx, start, end, type_idx, domain, save_path):
    anomaly_type = REDLAMP_ANOMALY_TYPES[type_idx]
    window = pool.Y[row_idx][:, start:end]
    # Must match OnlineWindowedDataset.__getitem__'s own seed formula exactly
    # (base_seed, row, window, type -- no epoch) or this would show a
    # different anomaly draw than training/eval actually used.
    item_rng = np.random.default_rng([sample_ds.base_seed, row_idx, window_idx, type_idx])
    y, z, mask = get_anomaly(anomaly_type)().apply(window, item_rng)

    fig, ax = plt.subplots(figsize=(9, 2.3))
    fig.patch.set_facecolor(SURFACE)
    plot_example_window(ax, y, z, mask, anomaly_type, waveform_type=domain)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--anomaly_type', default='spike', choices=REDLAMP_ANOMALY_TYPES[1:])
    parser.add_argument('--out_dir', default='./outputs/anomsim_domain_samples')
    parser.add_argument('--rng_seed', type=int, default=0,
                         help='Seeds both which (row, window) candidate is picked per type and each '
                              "type's OnlineWindowedDataset base_seed -- change to get a different "
                              'set of representative samples.')
    args = parser.parse_args()

    anomaly_type_idx = REDLAMP_ANOMALY_TYPES.index(args.anomaly_type)
    by_type = entities_by_type(args.dataset_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    for i, wf_type in enumerate(sorted(by_type)):
        entity_dir = sorted(by_type[wf_type])[0]
        pool, split = load_single_entity_split(args.dataset_dir, entity_dir)
        all_idx = np.concatenate([split.train_idx, split.val_idx])
        # Distinct base_seed per waveform type -- otherwise every type's
        # OnlineWindowedDataset shares the same default (0), and picking the
        # same (row_idx=0, window_idx=0, type_idx) candidate across all of
        # them would still produce an identical injection draw.
        sample_ds = OnlineWindowedDataset(pool, all_idx, WINDOW_SIZE, WINDOW_STEP, REDLAMP_ANOMALY_TYPES,
                                           base_seed=args.rng_seed * 1000 + i)

        type_dir = os.path.join(args.out_dir, wf_type)
        os.makedirs(type_dir, exist_ok=True)

        normal_hit = random_window_of_type(sample_ds, 0, rng)
        if normal_hit:
            plot_one(pool, sample_ds, *normal_hit, 0, wf_type, os.path.join(type_dir, 'normal.png'))
        anomaly_hit = random_window_of_type(sample_ds, anomaly_type_idx, rng)
        if anomaly_hit:
            plot_one(pool, sample_ds, *anomaly_hit, anomaly_type_idx, wf_type,
                     os.path.join(type_dir, f'{args.anomaly_type}.png'))

        print(f'{wf_type} (entity={entity_dir}): normal={bool(normal_hit)}, {args.anomaly_type}={bool(anomaly_hit)}')

    print(f'Done. Wrote samples for {len(by_type)} waveform types to {args.out_dir}')


if __name__ == '__main__':
    run()
