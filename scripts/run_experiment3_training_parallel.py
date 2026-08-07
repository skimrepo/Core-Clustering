"""
Launch all Experiment 3 training jobs (3 models x N seeds) as concurrent
subprocesses against a single GPU, mirroring this repo's own
scripts/train_self_all.py and RedLamp_Check's scripts/run_multiseed_training.py
-- both observed a single online_cli.py job sitting at ~30% GPU utilization
on its own, so running several concurrently meaningfully cuts wall-clock
time. The ConvAEC used here is tiny (1D conv, window_size=100, n_features=1),
so GPU memory is not a real constraint at this job count on an otherwise-idle
GPU -- unlike an earlier OOM this project hit, which was many more
concurrent processes contending for a GPU that OTHER jobs had already
mostly filled.

Every job is naturally idempotent (online_cli.py skips training and reuses
bestmodel.pkl if --output_dir/bestmodel.pkl already exists, unless --force),
so this launcher keeps no separate progress state of its own -- if
interrupted, just rerun the same command.

The three models' --held_out_domains follow the Experiment 3 design (see
RedLamp_Check's docs/DS3_handoff_context.md and result/Experiment_3/):
  Self_A           -- trained ONLY on domain A (square); every other domain held out.
  Cross_without_E  -- trained on every domain except A (square) and E (smoothed_pulse).
  Cross_without_D  -- trained on every domain except A (square) and D (white_noise).
--exclude_entity_dirs_file (from AnomSim's scripts/carve_experiment3_fixed_test_ids.py)
is passed to all three identically, so A's fixed test instances stay invisible
to every one of them.

Usage:
    python scripts/run_experiment3_training_parallel.py \
        --dataset_dir /path/to/AnomSim/data/AnomSim_v2 \
        --exclude_entity_dirs_file /path/to/AnomSim/data/AnomSim_v2/_experiment3_exclude_entity_dirs.json \
        --seeds 0 1 2 --batch_size 256 --max_parallel 9 --gpu 0
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_DOMAINS = ['sine', 'random_walk', 'white_noise', 'arma', 'trend', 'square',
               'sawtooth', 'binary_state', 'quantized_sensor', 'smoothed_pulse']
TARGET_DOMAIN = 'square'            # A
SIMILAR_DOMAIN = 'smoothed_pulse'   # E
DISSIMILAR_DOMAIN = 'white_noise'   # D

MODEL_HELD_OUT_DOMAINS = {
    'Self_A': [d for d in ALL_DOMAINS if d != TARGET_DOMAIN],
    'Cross_without_E': [TARGET_DOMAIN, SIMILAR_DOMAIN],
    'Cross_without_D': [TARGET_DOMAIN, DISSIMILAR_DOMAIN],
}

MODEL_RUN_NAMES = {
    'Self_A': 'experiment3_self_A',
    'Cross_without_E': 'experiment3_without_E',
    'Cross_without_D': 'experiment3_without_D',
}


def build_jobs(seeds):
    jobs = []
    for model_name, held_out in MODEL_HELD_OUT_DOMAINS.items():
        for seed in seeds:
            jobs.append(dict(model_name=model_name, held_out_domains=held_out, seed=seed))
    return jobs


def launch(job, args, log_dir):
    tag = f"{job['model_name']}_seed{job['seed']}"
    log_path = os.path.join(log_dir, f'{tag}.log')
    log_file = open(log_path, 'w')
    output_dir = os.path.join(args.output_root, MODEL_RUN_NAMES[job['model_name']])

    cmd = [sys.executable, '-m', 'core_clustering.online_cli',
           '--dataset_dir', args.dataset_dir,
           '--held_out_domains', *job['held_out_domains'],
           '--exclude_entity_dirs_file', args.exclude_entity_dirs_file,
           '--output_dir', output_dir,
           '--run_id', str(job['seed']),
           '--seed', str(job['seed']),
           '--batch_size', str(args.batch_size),
           '--epochs', str(args.epochs),
           '--patience', str(args.patience),
           '--val_fraction', str(args.val_fraction),
           '--gpu', str(args.gpu),
           '--class_list', 'redlamp']
    if args.force:
        cmd.append('--force')

    # Cap CPU-side thread usage per subprocess -- data loading/anomaly
    # injection still runs on CPU even though the model itself trains on
    # GPU. Overrides online_cli.py's own os.environ.setdefault(..., "4") so
    # this launcher controls it explicitly, sized to how many jobs run at
    # once on a shared box.
    env = os.environ.copy()
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[var] = str(args.cpu_threads_per_job)

    print(f'[launch] {tag} -> {log_path}  (held_out={job["held_out_domains"]})', flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return tag, proc, log_file


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_dir', required=True, help='AnomSim_v2 base-pool directory')
    parser.add_argument('--exclude_entity_dirs_file', required=True,
                         help="From AnomSim's scripts/carve_experiment3_fixed_test_ids.py")
    parser.add_argument('--output_root', default='./outputs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch_size', type=int, default=256,
                         help="Doubled from online_cli.py's own default of 128.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--max_parallel', type=int, default=9,
        help='How many jobs to run at once on the same GPU. A single online_cli.py job was observed at '
             '~30%% GPU utilization on its own (see train_self_all.py) -- lower this if nvidia-smi shows '
             'jobs visibly queueing/slowing each other down once several are running.',
    )
    parser.add_argument(
        '--cpu_threads_per_job', type=int, default=16,
        help='OMP/MKL/OPENBLAS thread cap per subprocess -- keeps N concurrent jobs from oversubscribing '
             "a shared box's CPU (this box already runs other tenants' work).",
    )
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--poll_seconds', type=int, default=15)
    parser.add_argument('--log_dir', default=None)
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(REPO_ROOT, 'logs', 'experiment3_training')
    os.makedirs(log_dir, exist_ok=True)

    jobs = build_jobs(args.seeds)
    print(f'{len(jobs)} jobs queued (models={list(MODEL_HELD_OUT_DOMAINS)}, seeds={args.seeds})', flush=True)
    print(f'Running up to {args.max_parallel} at a time on GPU {args.gpu}, '
          f'{args.cpu_threads_per_job} CPU threads/job, batch_size={args.batch_size}. '
          f'Per-job logs in {log_dir}/', flush=True)

    pending = list(jobs)
    running = {}  # tag -> (proc, log_file, start_time)
    finished = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            tag, proc, log_file = launch(job, args, log_dir)
            running[tag] = (proc, log_file, time.time())

        time.sleep(args.poll_seconds)

        for tag in list(running.keys()):
            proc, log_file, start_time = running[tag]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                elapsed_min = (time.time() - start_time) / 60
                status = 'OK' if ret == 0 else f'FAILED (exit code {ret})'
                print(f'[done] {tag}: {status} after {elapsed_min:.1f} min '
                      f'({len(running) - 1 + len(pending)} left)', flush=True)
                finished.append((tag, ret))
                del running[tag]

    n_failed = sum(1 for _, ret in finished if ret != 0)
    print(f'All {len(jobs)} jobs finished. {n_failed} failed.', flush=True)
    for tag, ret in finished:
        if ret != 0:
            print(f'  FAILED: {tag} -- see {log_dir}/{tag}.log', flush=True)


if __name__ == '__main__':
    run()
