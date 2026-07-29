import json
import os

import numpy as np

from core_clustering.online_cli import main


def _write_base_instance(base_dir, name, meta, n_time, seed):
    entity_dir = os.path.join(base_dir, name)
    os.makedirs(entity_dir, exist_ok=True)
    Y = np.random.default_rng(seed).normal(size=(1, n_time)).astype(np.float64)
    np.save(os.path.join(entity_dir, "Y.npy"), Y)
    with open(os.path.join(entity_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def _build_base_pool(tmp_path, domains=("a", "b"), n_base=6, n_time=300):
    out_dir = str(tmp_path / "base_pool")
    os.makedirs(out_dir, exist_ok=True)
    manifest_lines = []
    for domain in domains:
        for base_id in range(n_base):
            name = f"{domain}_b{base_id}"
            meta = {"type": domain, "params": {}, "n_time": n_time,
                     "base_instance_id": base_id, "base_seed": base_id}
            _write_base_instance(out_dir, name, meta, n_time=n_time, seed=base_id)
            manifest_lines.append({**meta, "entity_dir": name})
    with open(os.path.join(out_dir, "_manifest.jsonl"), "w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")
    return out_dir


def test_online_train_cli_end_to_end(tmp_path):
    dataset_dir = _build_base_pool(tmp_path)
    output_root = str(tmp_path / "outputs")

    main(
        [
            "--dataset_dir", dataset_dir,
            "--held_out_domains", "b",
            "--val_fraction", "0.3",
            "--window_size", "50",
            "--window_step", "20",
            "--class_list", "normal,spike,noise",
            "--output_dir", output_root,
            "--run_id", "test-run",
            "--epochs", "2",
            "--patience", "10",
            "--seed", "0",
            "--batch_size", "16",
            "--gpu", "-1",
            "--embedding_dim", "4",
            "--tsne_perplexity", "2",
            "--n_sample_plots", "2",
            "--eval_max_samples", "200",
        ]
    )

    run_dir = os.path.join(output_root, "test-run")
    assert os.path.exists(os.path.join(run_dir, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(run_dir, "run_summary.json"))
    assert os.path.exists(os.path.join(run_dir, "classification_accuracy.csv"))
    assert os.path.exists(os.path.join(run_dir, "plots", "tsne_by_class.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "tsne_by_domain.png"))
    assert len(os.listdir(os.path.join(run_dir, "plots", "samples"))) > 0

    with open(os.path.join(run_dir, "run_summary.json")) as f:
        summary = json.load(f)

    assert summary["included_domains"] == ["a"]
    assert summary["held_out_domains"] == ["b"]
    assert summary["n_windows_total"] > 0
    assert len(summary["held_out_accuracy"]) == 1
    assert summary["held_out_accuracy"][0]["domain"] == "b"
    assert summary["epochs_ran"] == 2
    assert summary["model_hyperparameters"]["classes"] == 3
    assert summary["model_hyperparameters"]["n_features"] == 1

    domain_rows = {row["domain"]: row for row in summary["domain_window_counts"]}
    assert domain_rows["a"]["role"] == "included"
    assert domain_rows["a"]["n_windows_train"] is not None
    assert domain_rows["b"]["role"] == "held_out"
    assert domain_rows["b"]["n_windows_eval"] is not None

    with open(os.path.join(run_dir, "classification_accuracy.csv")) as f:
        header = f.readline().strip()
    assert header == "domain,role,n_total,n_correct,n_incorrect,accuracy"


def test_online_train_cli_skips_retraining_if_bestmodel_already_exists(tmp_path, monkeypatch):
    dataset_dir = _build_base_pool(tmp_path)
    output_root = str(tmp_path / "outputs")
    args = [
        "--dataset_dir", dataset_dir,
        "--held_out_domains", "b",
        "--val_fraction", "0.3",
        "--window_size", "50",
        "--window_step", "20",
        "--class_list", "normal,spike,noise",
        "--output_dir", output_root,
        "--run_id", "test-run",
        "--epochs", "2",
        "--seed", "0",
        "--batch_size", "16",
        "--gpu", "-1",
        "--embedding_dim", "4",
        "--tsne_perplexity", "2",
        "--n_sample_plots", "1",
        "--eval_max_samples", "200",
    ]
    main(args)  # first run: trains for real
    run_dir = os.path.join(output_root, "test-run")
    bestmodel_mtime_1 = os.path.getmtime(os.path.join(run_dir, "bestmodel.pkl"))

    import core_clustering.trainer as trainer_module

    def _fail_if_called(*a, **kw):
        raise AssertionError("Trainer.train() should not be called on a resumed run")

    monkeypatch.setattr(trainer_module.Trainer, "train", _fail_if_called)

    main(args)  # second run: same output dir, should skip training entirely
    bestmodel_mtime_2 = os.path.getmtime(os.path.join(run_dir, "bestmodel.pkl"))
    assert bestmodel_mtime_1 == bestmodel_mtime_2  # untouched -- not retrained
    assert os.path.exists(os.path.join(run_dir, "classification_accuracy.csv"))


def test_online_train_cli_force_retrains_even_if_bestmodel_exists(tmp_path):
    dataset_dir = _build_base_pool(tmp_path)
    output_root = str(tmp_path / "outputs")
    args = [
        "--dataset_dir", dataset_dir,
        "--val_fraction", "0.3",
        "--window_size", "50",
        "--window_step", "20",
        "--class_list", "normal,spike,noise",
        "--output_dir", output_root,
        "--run_id", "test-run",
        "--epochs", "1",
        "--seed", "0",
        "--batch_size", "16",
        "--gpu", "-1",
        "--embedding_dim", "4",
        "--n_sample_plots", "1",
        "--eval_max_samples", "200",
    ]
    main(args)
    run_dir = os.path.join(output_root, "test-run")
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        assert json.load(f)["epochs_ran"] == 1

    main(args + ["--force"])
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        assert json.load(f)["epochs_ran"] == 1  # retrained, not resumed (still ran epochs, didn't crash)


def test_online_train_cli_class_list_redlamp_pins_twelve_classes(tmp_path):
    dataset_dir = _build_base_pool(tmp_path, domains=("a",), n_base=6, n_time=300)
    output_root = str(tmp_path / "outputs")

    main(
        [
            "--dataset_dir", dataset_dir,
            "--val_fraction", "0.3",
            "--window_size", "50",
            "--window_step", "20",
            "--output_dir", output_root,
            "--run_id", "test-run",
            "--epochs", "1",
            "--seed", "0",
            "--batch_size", "16",
            "--gpu", "-1",
            "--embedding_dim", "4",
            "--n_sample_plots", "1",
            "--eval_max_samples", "100",
        ]
    )
    run_dir = os.path.join(output_root, "test-run")
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        summary = json.load(f)
    assert summary["model_hyperparameters"]["classes"] == 12
