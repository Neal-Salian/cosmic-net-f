import json

import pytest
import torch
import yaml

from training.full_graph_baseline import run_baseline


def smoke_config():
    with open("config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config["model"].update(hidden_dim=8, output_dim=8, num_layers=1,
                           dropout=0.0, mc_dropout=False)
    config["training"].update(epochs=7, early_stopping_patience=1,
                              save_every=1, save_best=True)
    config["data"].update(batch_size=4, num_workers=0)
    config["graph"].update(method="knn", k_neighbors=2,
                           edge_features=["distance", "delta_v"])
    config["physics"]["use_virial_loss"] = True
    config["wandb"]["enabled"] = False
    return config


def test_synthetic_full_graph_smoke_emits_actual_metrics_and_hashed_artifacts(tmp_path):
    output = tmp_path / "baseline"
    result = run_baseline(smoke_config(), output_dir=output, smoke=True)

    assert result["research_label"] == "synthetic"
    assert result["physics_use_virial_loss"] is False
    assert set(result["metrics"]) >= {"val/rmse", "val/r2", "test/rmse", "test/r2"}
    assert all(isinstance(value, float) for value in result["metrics"].values())
    for name in ("results.json", "provenance.json", "split_manifest.json",
                 "checkpoints/best_model.pt"):
        assert (output / name).is_file()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["checkpoint_sha256"]
    assert provenance["model_state_sha256"]
    assert provenance["source_code_files"]
    assert provenance["train_catalog"]["research_label"] == "synthetic"
    assert provenance["evaluation_catalog"]["research_label"] == "synthetic"
    assert provenance["resolved_config"]["training"]["epochs"] == 1
    assert provenance["resolved_config"]["data"]["source"] == "synthetic_smoke"
    assert provenance["resolved_config"]["training"]["train_source"] == "synthetic_smoke"
    assert provenance["resolved_config"]["training"]["test_source"] == "synthetic_smoke"
    assert provenance["data_generator"] == {
        "name": "training.full_graph_baseline._smoke_catalog",
        "seed": provenance["resolved_config"]["seed"],
        "halo_count": 12,
        "subhalos_per_halo": 5,
        "generator": "numpy.random.Generator(PCG64)",
    }


def test_baseline_requires_new_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="new output directory"):
        run_baseline(smoke_config(), output_dir=output, smoke=True)


def test_non_smoke_corrected_run_requires_positive_catalog_contract(tmp_path):
    config = smoke_config()
    config["data"]["strict_research"] = True
    with pytest.raises(ValueError, match="catalog_contract"):
        run_baseline(config, output_dir=tmp_path / "strict", smoke=False)


def test_seeded_smoke_baseline_is_reproducible(tmp_path):
    """Two smoke baselines with identical seed/config must produce identical
    model-state hashes and held-out predictions."""
    from data.provenance import model_state_hash

    def _run(label):
        cfg = smoke_config()
        cfg["model"]["dropout"] = 0.3
        out = tmp_path / label
        result = run_baseline(cfg, output_dir=out, smoke=True)
        ckpt_path = out / "checkpoints" / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_hash = model_state_hash(ckpt["model_state_dict"])
        return state_hash, result["metrics"]

    hash1, metrics1 = _run("run1")
    hash2, metrics2 = _run("run2")

    assert hash1 == hash2, (
        f"model-state hashes differ across identical seeded runs: "
        f"{hash1} != {hash2}"
    )
    for key in ("test/rmse", "test/r2", "val/rmse", "val/r2"):
        assert metrics1[key] == pytest.approx(metrics2[key], rel=1e-6), (
            f"metric {key} differs: {metrics1[key]} != {metrics2[key]}"
        )


def test_baseline_source_code_files_include_data_loaders(tmp_path):
    """Provenance source_code_files must include the data loader modules
    that parse scientific fields."""
    from training.full_graph_baseline import SOURCE_FILES
    assert "data/loaders/base_loader.py" in SOURCE_FILES
    assert "data/loaders/tng_loader.py" in SOURCE_FILES
    assert "data/loaders/camels_loader.py" in SOURCE_FILES
    assert "data/loaders/synthetic_loader.py" in SOURCE_FILES
