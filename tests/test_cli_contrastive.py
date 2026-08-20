import os

from core_clustering.cli_contrastive import main


def _base_argv(output_dir, run_id):
    return [
        "--output_dir", output_dir,
        "--run_id", run_id,
        "--n_instances", "40",
        "--length_min", "20",
        "--length_max", "20",
        "--epochs", "2",
        "--batch_size", "4",
        "--max_len", "20",
        "--num_filters", "4,4",
        "--bottleneck_channels", "2",
        "--num_groups", "2",
        "--embedding_dim", "8",
        "--attention_max_resolution", "0",
        "--gpu", "-1",
        "--seed", "0",
    ]


def test_cli_contrastive_trains_and_saves_checkpoint(tmp_path):
    output_dir = str(tmp_path / "outputs")

    main(_base_argv(output_dir, "test_run"))

    run_dir = os.path.join(output_dir, "test_run")
    assert os.path.exists(os.path.join(run_dir, "bestmodel.pkl"))
    assert os.path.exists(os.path.join(run_dir, "epoch_history.json"))


def test_cli_contrastive_force_skips_existing_checkpoint(tmp_path, capsys):
    output_dir = str(tmp_path / "outputs")
    argv = _base_argv(output_dir, "test_run")

    main(argv)
    main(argv)
    out = capsys.readouterr().out
    assert "skip" in out.lower()
