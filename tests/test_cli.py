import json
import os

import numpy as np

from core_clustering.cli import main

WINDOW_SIZE = 20


def _write_entity(base_dir, name, meta, window_size=WINDOW_SIZE, seed=0):
    entity_dir = os.path.join(base_dir, name)
    os.makedirs(entity_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(1, window_size)).astype(np.float32)
    labels = np.ones((1, window_size), dtype=np.float32)
    labels[0, window_size // 2] = 0
    Z = Y.copy()
    np.save(os.path.join(entity_dir, "Y.npy"), Y)
    np.save(os.path.join(entity_dir, "labels.npy"), labels)
    np.save(os.path.join(entity_dir, "Z.npy"), Z)
    np.save(os.path.join(entity_dir, "mask.npy"), np.ones_like(Y))
    with open(os.path.join(entity_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(entity_dir, "_name.txt"), "w") as f:
        f.write(name)


def _build_dataset(tmp_path, domains=("a", "b"), n_base=3, atypes=("normal", "spike", "noise"), n_windows=3):
    out_dir = str(tmp_path / "dataset")
    os.makedirs(out_dir, exist_ok=True)
    manifest_lines = []
    seed = 0
    for domain in domains:
        for base_id in range(n_base):
            for atype in atypes:
                for w in range(n_windows):
                    name = f"{domain}_b{base_id}_w{w}_{atype}"
                    meta = {
                        "waveform": {"type": domain, "params": {}},
                        "anomaly": {"type": atype, "params": {}},
                        "window": {"index": w, "start": w * 10, "end": w * 10 + WINDOW_SIZE, "size": WINDOW_SIZE, "step": 10},
                        "base_instance_id": base_id,
                        "base_seed": base_id,
                        "base_n_time": 100,
                        "seed": seed,
                    }
                    _write_entity(out_dir, name, meta, seed=seed)
                    manifest_lines.append({**meta, "entity_dir": name})
                    seed += 1
    with open(os.path.join(out_dir, "_manifest.jsonl"), "w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")
    return out_dir


def test_train_cli_end_to_end(tmp_path):
    dataset_dir = _build_dataset(tmp_path)
    output_root = str(tmp_path / "outputs")

    main(
        [
            "--dataset_dir", dataset_dir,
            "--held_out_domains", "b",
            "--val_fraction", "0.3",
            "--output_dir", output_root,
            "--run_id", "test-run",
            "--epochs", "2",
            "--patience", "10",
            "--seed", "0",
            "--batch_size", "8",
            "--gpu", "-1",
            "--embedding_dim", "4",
            "--tsne_perplexity", "2",
            "--n_sample_plots", "2",
        ]
    )

    run_dir = os.path.join(output_root, "test-run")
    assert os.path.exists(os.path.join(run_dir, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(run_dir, "run_summary.json"))
    assert os.path.exists(os.path.join(run_dir, "classification_accuracy.csv"))
    assert os.path.exists(os.path.join(run_dir, "plots", "tsne_by_class.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "tsne_by_domain.png"))

    with open(os.path.join(run_dir, "run_summary.json")) as f:
        summary = json.load(f)

    assert summary["included_domains"] == ["a"]
    assert summary["held_out_domains"] == ["b"]
    assert summary["n_windows_total"] > 0
    assert len(summary["held_out_accuracy"]) == 1
    assert summary["held_out_accuracy"][0]["domain"] == "b"
    assert summary["epochs_ran"] == 2

    with open(os.path.join(run_dir, "classification_accuracy.csv")) as f:
        header = f.readline().strip()
    assert header == "domain,role,n_total,n_correct,n_incorrect,accuracy"
