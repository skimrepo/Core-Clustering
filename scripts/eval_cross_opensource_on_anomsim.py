"""
Score the Cross-OpenSource model (RedLamp-trained, real open-source data
only -- continuous_n697_excl_ucr, i.e. SMD+SMAP+MSL, never saw UCR or KPI)
against every AnomSim_v1 entity's own validation split. Uses the exact same
temporal 90/10 split every entity's own Self model was evaluated against
(core_clustering.single_entity.load_single_entity_split) so the comparison
against train_self_all.py's per-entity Self accuracy is apples-to-apples.

Pure inference -- no training. The model is RedLamp's own ConvAEC checkpoint
(layer-identical port, loads directly via load_state_dict); this script
constructs the same architecture and loads external weights into it,
mirroring RedLamp_Check/scripts/simulation_cross_domain_metrics.py's pattern
in the opposite direction (that script loads a Core-Clustering checkpoint
into RedLamp's ConvAEC; this one loads a RedLamp checkpoint into
Core-Clustering's ConvAEC).

Resumable: reruns skip entities already present in --out_csv unless --force.
Incremental save after every entity.
"""
import argparse
import os
import sys

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from core_clustering.models import ConvAEC
from core_clustering.trainer import default_model_hyperparameters
from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
from core_clustering.online_dataset import materialize_windows
from core_clustering.metrics import evaluate_classification
from core_clustering.single_entity import list_entities, load_single_entity_split


def build_model(bestmodel_path, window_size, embedding_dim, device):
    class_list = list(REDLAMP_ANOMALY_TYPES)
    model_config = default_model_hyperparameters(
        n_features=1, n_time=window_size, classes=len(class_list), embedding_dim=embedding_dim,
    )
    model = ConvAEC(model_config)
    model.load_state_dict(torch.load(bestmodel_path, map_location=device))
    model.to(device)
    return model, class_list


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, help='AnomSim_v1 root')
    parser.add_argument('--bestmodel_path', required=True,
                         help='Path to the Cross-OpenSource model, e.g. '
                              '.../continuous_n697_excl_ucr/{seed}/bestmodel.pkl')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                         help='Must match what train_self_all.py used, for a fair per-entity comparison.')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--window_step', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--eval_max_samples', type=int, default=5000)
    parser.add_argument('--gpu', type=int, default=0, help='-1 = cpu')
    parser.add_argument('--out_csv', default='outputs/cross_opensource_accuracy.csv')
    parser.add_argument('--force', action='store_true',
                         help='Recompute every entity even if already present in out_csv.')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    model, class_list = build_model(args.bestmodel_path, args.window_size, args.embedding_dim, device)
    print(f'Loaded Cross-OpenSource model from {args.bestmodel_path} onto {device}')

    entities = list_entities(args.dataset_dir)
    print(f'{len(entities)} entities found in {args.dataset_dir}')

    rows = []
    already_done = set()
    if not args.force and os.path.isfile(args.out_csv):
        prior = pd.read_csv(args.out_csv)
        rows = prior.to_dict('records')
        already_done = set(prior['entity'])
        print(f'Resuming from {args.out_csv}: {len(already_done)} entities already scored, skipping those.')

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for entity in entities:
        if entity in already_done:
            continue
        pool, split = load_single_entity_split(args.dataset_dir, entity, val_fraction=args.val_fraction)
        data = materialize_windows(pool, split.val_idx, args.window_size, args.window_step, class_list,
                                    max_samples=args.eval_max_samples)
        if data is None:
            print(f'[skip] {entity}: no val windows')
            continue
        Y, labels, _ds, _idx = data
        result = evaluate_classification(model, Y, labels, split.included_domains[0], device=device)
        row = dict(entity=entity, **result.compact_dict())
        rows.append(row)
        print(f'  {entity}: accuracy={result.accuracy:.4f} (n={result.n_total})')
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    if rows:
        avg = pd.DataFrame(rows)['accuracy'].mean()
        print(f'Done. {len(rows)} entities scored. Average accuracy={avg:.4f}. Wrote {args.out_csv}')
    else:
        print('Done. No entities scored.')


if __name__ == '__main__':
    run()
