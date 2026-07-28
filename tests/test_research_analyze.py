import json
import os

import numpy as np

from core_clustering.cli import main as train_main
from core_clustering.dataset import LoadedDataset, LoadStats
from research.analyze import run_research_analysis, write_examples_pdf

WINDOW_SIZE = 20


def _write_entity(base_dir, name, meta, seed=0):
    entity_dir = os.path.join(base_dir, name)
    os.makedirs(entity_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(1, WINDOW_SIZE)).astype(np.float32)
    labels = np.ones((1, WINDOW_SIZE), dtype=np.float32)
    labels[0, WINDOW_SIZE // 2] = 0
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


def test_write_examples_pdf_skips_and_notes_when_zero_examples(tmp_path, capsys):
    load_stats = LoadStats(
        manifest_path="fake", n_manifest_lines=1, n_attempted=1, n_loaded=1, n_failed=0,
        failures=[], failures_by_reason={}, domains=["a"], anomaly_types=["normal"],
    )
    dataset = LoadedDataset(
        Y=np.zeros((1, 1, 10), dtype=np.float32), labels=np.ones((1, 1, 10), dtype=np.float32),
        Z=np.zeros((1, 1, 10), dtype=np.float32), domain=np.array(["a"]), anomaly_type=np.array(["normal"]),
        base_instance_id=np.array([0]), window_index=np.array([0]), entity_dir=np.array(["e0"]),
        window_size=10, class_list=["normal"], load_stats=load_stats,
    )
    out_path = str(tmp_path / "incorrect_examples.pdf")
    write_examples_pdf(dataset, np.array([], dtype=np.int64), out_path, n_examples=10, seed=0, domain="a", category="incorrect")

    assert not os.path.exists(out_path)
    captured = capsys.readouterr()
    assert "0 incorrect examples" in captured.out


def test_run_research_analysis_creates_expected_output(tmp_path):
    dataset_dir = _build_dataset(tmp_path)
    output_root = str(tmp_path / "outputs")

    train_main(
        [
            "--dataset_dir", dataset_dir,
            "--held_out_domains", "b",
            "--val_fraction", "0.3",
            "--output_dir", output_root,
            "--run_id", "test-run",
            "--epochs", "2",
            "--seed", "0",
            "--batch_size", "8",
            "--gpu", "-1",
            "--embedding_dim", "4",
            "--tsne_perplexity", "2",
            "--n_sample_plots", "1",
        ]
    )

    run_dir = os.path.join(output_root, "test-run")
    research_root = str(tmp_path / "research")

    run_research_analysis(run_dir, dataset_dir=dataset_dir, domains=["b"], research_root=research_root, n_examples=5, seed=0, device="cpu")

    domain_dir = os.path.join(research_root, "test-run", "b")
    accuracy_path = os.path.join(domain_dir, "accuracy.json")
    assert os.path.exists(accuracy_path)

    with open(accuracy_path) as f:
        accuracy = json.load(f)
    for key in ["run_id", "source_run_dir", "domain", "n_total", "n_correct", "n_incorrect", "accuracy", "generated_at"]:
        assert key in accuracy
    assert accuracy["domain"] == "b"
    assert accuracy["n_total"] == accuracy["n_correct"] + accuracy["n_incorrect"]

    if accuracy["n_correct"] > 0:
        correct_path = os.path.join(domain_dir, "correct_examples.pdf")
        assert os.path.exists(correct_path)
        assert os.path.getsize(correct_path) > 500
    if accuracy["n_incorrect"] > 0:
        incorrect_path = os.path.join(domain_dir, "incorrect_examples.pdf")
        assert os.path.exists(incorrect_path)
        assert os.path.getsize(incorrect_path) > 500
