"""
Train one "Self" model per AnomSim entity: 144 separate
`core_clustering.online_cli --single_entity` runs (RedLamp's own per-entity
Self convention -- one model trained+tested only on that one entity's own
timeline), launched as concurrent subprocesses to actually use GPU headroom
instead of one at a time (mirrors RedLamp_Check's own
scripts/run_multiseed_training.py, which observed ~30% GPU utilization for a
single training job on its own).

Every job is naturally idempotent (online_cli.py skips training and reuses
bestmodel.pkl if it already exists), so this launcher keeps no separate
progress state -- if interrupted, just rerun the same command.

After each entity finishes, its classification_accuracy.csv (exactly one row
in --single_entity mode) is appended to a running
outputs/self_accuracy_all_entities.csv, rewritten after every completion so
a crash partway through never loses already-finished entities.

Usage:
    python scripts/train_self_all.py \
        --dataset_dir /path/to/AnomSim/data/AnomSim_v1 \
        --output_root outputs/self --seed 0 --val_fraction 0.1 \
        --max_parallel 3
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from core_clustering.single_entity import list_entities


def launch(entity, args):
    output_dir = os.path.join(args.output_root, entity)
    log_path = os.path.join(args.log_dir, f'{entity}.log')
    log_file = open(log_path, 'w')
    cmd = [sys.executable, '-m', 'core_clustering.online_cli',
           '--dataset_dir', args.dataset_dir,
           '--single_entity', entity,
           '--val_fraction', str(args.val_fraction),
           '--output_dir', output_dir,
           '--run_id', str(args.seed),
           '--seed', str(args.seed),
           '--epochs', str(args.epochs),
           '--patience', str(args.patience),
           '--batch_size', str(args.batch_size),
           '--gpu', str(args.gpu),
           '--class_list', 'redlamp']
    print(f'[launch] {entity} -> {log_path}', flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def append_accuracy(entity, args, rows):
    csv_path = os.path.join(args.output_root, entity, str(args.seed), 'classification_accuracy.csv')
    if not os.path.isfile(csv_path):
        print(f'[warn] {entity}: no classification_accuracy.csv found at {csv_path} — not appended')
        return
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        rows.append(dict(domain=row['domain'], entity=entity, seed=args.seed,
                          n_total=row['n_total'], n_correct=row['n_correct'],
                          n_incorrect=row['n_incorrect'], accuracy=row['accuracy']))
    pd.DataFrame(rows).to_csv(args.accuracy_csv, index=False)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_root', default='outputs/self')
    parser.add_argument('--accuracy_csv', default=None)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_fraction', type=float, default=0.1,
                         help='0.1 matches RedLamp\'s own per-entity Self convention '
                              '(int(Y.shape[1]*0.9) train / remainder val).')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_parallel', type=int, default=3,
                         help='How many online_cli.py jobs to run at once on the same GPU. A single '
                              'job runs at ~30%% GPU utilization on its own (observed on RedLamp\'s '
                              'analogous per-entity training) -- raise this if nvidia-smi still shows '
                              'headroom, lower it if jobs start visibly slowing each other down.')
    parser.add_argument('--poll_seconds', type=int, default=10)
    parser.add_argument('--force', action='store_true',
                         help='Recompute every entity even if already in accuracy_csv (default: '
                              'resume, skipping entities already scored in a prior run).')
    args = parser.parse_args()

    args.accuracy_csv = args.accuracy_csv or os.path.join(args.output_root, '..', 'self_accuracy_all_entities.csv')
    args.accuracy_csv = os.path.normpath(args.accuracy_csv)
    args.log_dir = args.log_dir or os.path.join(args.output_root, '_logs')
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    entities = list_entities(args.dataset_dir)
    print(f'{len(entities)} entities found in {args.dataset_dir}', flush=True)

    rows = []
    already_done = set()
    if not args.force and os.path.isfile(args.accuracy_csv):
        prior = pd.read_csv(args.accuracy_csv)
        rows = prior.to_dict('records')
        already_done = set(zip(prior['entity'], prior['seed']))
        print(f'Resuming from {args.accuracy_csv}: {len(already_done)} entities already scored, skipping those.',
              flush=True)

    pending = [e for e in entities if (e, args.seed) not in already_done]
    running = {}  # entity -> (proc, log_file, start_time)
    finished = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            entity = pending.pop(0)
            proc, log_file = launch(entity, args)
            running[entity] = (proc, log_file, time.time())

        time.sleep(args.poll_seconds)

        for entity in list(running.keys()):
            proc, log_file, start_time = running[entity]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                elapsed_min = (time.time() - start_time) / 60
                status = 'OK' if ret == 0 else f'FAILED (exit code {ret})'
                print(f'[done] {entity}: {status} after {elapsed_min:.1f} min '
                      f'({len(pending) + len(running) - 1} left)', flush=True)
                if ret == 0:
                    append_accuracy(entity, args, rows)
                finished.append((entity, ret))
                del running[entity]

    n_failed = sum(1 for _, ret in finished if ret != 0)
    print(f'All {len(entities)} entities processed this run: {len(finished)} attempted, {n_failed} failed.',
          flush=True)
    for entity, ret in finished:
        if ret != 0:
            print(f'  FAILED: {entity} — see {args.log_dir}/{entity}.log', flush=True)
    if rows:
        avg = pd.DataFrame(rows)['accuracy'].mean()
        print(f'Self accuracy so far: {len(rows)} entities, average={avg:.4f}. Wrote {args.accuracy_csv}')


if __name__ == '__main__':
    run()
