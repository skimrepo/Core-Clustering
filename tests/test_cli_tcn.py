import json
import os

import numpy as np

from core_clustering.cli_tcn import main


def _write_instance(out_dir, name, n_time, split, is_anomalous, Y, Z, label):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "Y.npy"), Y)
    np.save(os.path.join(d, "Z.npy"), Z)
    np.save(os.path.join(d, "label.npy"), label)
    meta = {"n_time": n_time, "split": split, "is_anomalous": is_anomalous, "entity_dir": name}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _build_fixture_dataset(out_dir, n_time=20):
    rng = np.random.default_rng(0)
    manifest_rows = []
    splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
    for i, split in enumerate(splits):
        Z = rng.normal(size=(1, n_time))
        Y = Z.copy()
        label = np.zeros((1, n_time))
        is_anomalous = i % 2 == 0
        if is_anomalous:
            Y[0, 5:10] += 4.0
            label[0, 5:10] = 1.0
        manifest_rows.append(
            _write_instance(out_dir, f"inst_{i}", n_time, split, is_anomalous, Y, Z, label)
        )
    with open(os.path.join(out_dir, "_manifest.jsonl"), "w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")


def test_cli_tcn_trains_and_writes_run_summary(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    output_dir = str(tmp_path / "outputs")
    _build_fixture_dataset(dataset_dir)

    main([
        "--dataset_dir", dataset_dir,
        "--output_dir", output_dir,
        "--run_id", "test_run",
        "--epochs", "2",
        "--batch_size", "2",
        "--max_len", "20",
        "--num_filters", "4,4",
        "--bottleneck_channels", "2",
        "--num_groups", "2",
        "--gpu", "-1",
        "--seed", "0",
    ])

    run_dir = os.path.join(output_dir, "test_run")
    assert os.path.exists(os.path.join(run_dir, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(run_dir, "run_summary.json"))
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        summary = json.load(f)
    assert summary["n_entities_loaded"] == 6  # train(4) + val(2); test(2) intentionally excluded


def test_cli_tcn_force_skips_existing_checkpoint(tmp_path, capsys):
    dataset_dir = str(tmp_path / "dataset")
    output_dir = str(tmp_path / "outputs")
    _build_fixture_dataset(dataset_dir)

    argv = [
        "--dataset_dir", dataset_dir,
        "--output_dir", output_dir,
        "--run_id", "test_run",
        "--epochs", "1",
        "--batch_size", "2",
        "--max_len", "20",
        "--num_filters", "4,4",
        "--bottleneck_channels", "2",
        "--num_groups", "2",
        "--gpu", "-1",
    ]
    main(argv)
    main(argv)  # second call: should skip (bestmodel.pkl already exists), not error
    out = capsys.readouterr().out
    assert "skip" in out.lower()
