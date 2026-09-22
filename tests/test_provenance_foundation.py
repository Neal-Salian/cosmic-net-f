import json

import pytest
import torch
import yaml

from data.provenance import (
    build_run_provenance,
    canonical_json,
    hash_payload,
    model_state_hash,
    validate_checkpoint_file,
    validate_inference_compatibility,
    validate_run_provenance,
)
from model.model import build_model, load_model
from training.train import Trainer


def tiny_config(tmp_path):
    with open("config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config["model"].update(hidden_dim=8, output_dim=8, num_layers=1, dropout=0.0)
    config["training"].update(checkpoint_dir=str(tmp_path), epochs=1,
                              save_every=1, save_best=False)
    config["wandb"]["enabled"] = False
    config["physics"]["use_virial_loss"] = False
    return config


def test_canonical_hash_is_order_independent_and_rejects_nonfinite_values():
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_checkpoint_validation_rejects_lfs_pointer_before_torch_load(tmp_path):
    pointer = tmp_path / "missing.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\nsize 1000\n"
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        validate_checkpoint_file(pointer)
    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_model(pointer, tiny_config(tmp_path), torch.device("cpu"),
                   research_label="legacy_total_radius")


def test_tiny_valid_checkpoint_is_allowed_but_legacy_label_is_explicit(tmp_path):
    config = tiny_config(tmp_path)
    model = build_model(config)
    path = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), path)
    validate_checkpoint_file(path)
    with pytest.raises(ValueError, match="legacy"):
        load_model(path, config, torch.device("cpu"))
    loaded = load_model(path, config, torch.device("cpu"),
                        research_label="legacy_total_radius")
    assert model_state_hash(loaded.state_dict()) == model_state_hash(model.state_dict())


def test_run_provenance_detects_split_and_config_reuse():
    split = {"manifest_sha256": "s" * 64}
    config = {"seed": 4, "graph": {"method": "knn"}}
    provenance = build_run_provenance(
        source_files=[{"path": "input", "size_bytes": 1, "sha256": "a" * 64}],
        catalog_contract={"research_label": "synthetic"},
        split_manifest=split,
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    validate_run_provenance(provenance, split_manifest=split,
                            resolved_config=config)
    with pytest.raises(ValueError, match="split"):
        validate_run_provenance(
            provenance, split_manifest={"manifest_sha256": "x" * 64})
    with pytest.raises(ValueError, match="config"):
        validate_run_provenance(provenance, resolved_config={"seed": 9})


def test_external_inference_compatibility_uses_semantics_not_catalog_identity():
    training = {
        "source": "illustris_tng", "target_definition": "log10(M200c/M_sun)",
        "mass_units": "M_sun", "position_units": "Mpc", "radius_units": "Mpc",
        "velocity_units": "km/s", "coordinate_frame": "physical",
        "velocity_convention": "peculiar_physical",
        "radius_semantic": "stellar_half_mass_radius_type4",
    }
    external = {**training, "source": "camels", "simulation": "LH_0"}
    validate_inference_compatibility(training, external)
    external["radius_units"] = "ckpc/h"
    with pytest.raises(ValueError, match="radius_units"):
        validate_inference_compatibility(training, external)


def test_trainer_checkpoint_writes_safe_provenance_manifest(tmp_path):
    config = tiny_config(tmp_path)
    config["provenance"] = build_run_provenance(
        source_files=[], catalog_contract={"research_label": "synthetic"},
        split_manifest={"manifest_sha256": "s" * 64},
        resolved_config=config, source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None, research_label="synthetic",
    )
    model = build_model(config)
    trainer = Trainer(config, model, train_loader=[], val_loader=[])
    trainer._save_checkpoint("tiny.pt", 0, {"val/mse": 1.0})

    path = tmp_path / "tiny.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    manifest = json.loads((tmp_path / "tiny.pt.manifest.json").read_text())
    assert checkpoint["provenance"]["model_state_sha256"] == model_state_hash(
        model.state_dict())
    assert manifest["checkpoint_sha256"] == validate_checkpoint_file(path)["sha256"]
    assert manifest["model_state_sha256"] == checkpoint["provenance"]["model_state_sha256"]


def test_validate_run_provenance_rejects_mutated_embedded_catalog():
    """Mutating embedded catalog_contract without updating its hash must fail."""
    split = {"manifest_sha256": "s" * 64}
    config = {"seed": 4, "graph": {"method": "knn"}}
    catalog = {"research_label": "synthetic", "target_definition": "log10(M)"}
    provenance = build_run_provenance(
        source_files=[], catalog_contract=catalog,
        split_manifest=split, resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None, research_label="synthetic",
    )
    # Unmodified record must pass
    validate_run_provenance(provenance)
    # Mutate embedded catalog without updating hash
    provenance["catalog_contract"]["target_definition"] = "log10(M_vir)"
    with pytest.raises(ValueError, match="catalog contract embedded"):
        validate_run_provenance(provenance)


def test_validate_run_provenance_rejects_mutated_embedded_config():
    """Mutating embedded resolved_config without updating its hash must fail."""
    split = {"manifest_sha256": "s" * 64}
    config = {"seed": 4, "graph": {"method": "knn"}}
    catalog = {"research_label": "synthetic"}
    provenance = build_run_provenance(
        source_files=[], catalog_contract=catalog,
        split_manifest=split, resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None, research_label="synthetic",
    )
    provenance["resolved_config"]["seed"] = 999
    with pytest.raises(ValueError, match="resolved config embedded"):
        validate_run_provenance(provenance)


def test_validate_run_provenance_rejects_mutated_embedded_augmentation():
    """Mutating embedded augmentation_manifest without updating its hash must fail."""
    split = {"manifest_sha256": "s" * 64}
    config = {"seed": 4}
    catalog = {"research_label": "synthetic"}
    aug = {"copies": 1, "jitter_std": 0.0}
    provenance = build_run_provenance(
        source_files=[], catalog_contract=catalog,
        split_manifest=split, resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=aug, research_label="synthetic",
    )
    provenance["augmentation_manifest"]["copies"] = 5
    with pytest.raises(ValueError, match="augmentation manifest embedded"):
        validate_run_provenance(provenance)


def test_trainer_load_checkpoint_rejects_provenance_mismatch(tmp_path):
    """Trainer._load_checkpoint must reject checkpoint whose provenance
    disagrees with the active run's provenance (catalog/split/config/label)."""
    config = tiny_config(tmp_path)
    model_ckpt = build_model(config)
    # Build checkpoint provenance with split hash "original-split"
    ckpt_provenance = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "original-split"},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    ckpt_provenance["model_state_sha256"] = model_state_hash(
        model_ckpt.state_dict())
    config_ckpt = {**config, "provenance": ckpt_provenance}
    path = tmp_path / "ckpt.pt"
    torch.save({
        "model_state_dict": model_ckpt.state_dict(),
        "provenance": ckpt_provenance,
        "config": config_ckpt,
        "epoch": 0,
    }, path)

    # Active run has a different split hash
    active_provenance = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "different-split"},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    active_provenance["model_state_sha256"] = model_state_hash(
        model_ckpt.state_dict())
    config_active = {**config, "provenance": active_provenance}
    model_active = build_model(config_active)
    from torch_geometric.loader import DataLoader
    trainer = Trainer(config_active, model_active,
                      train_loader=DataLoader([], batch_size=1),
                      val_loader=DataLoader([], batch_size=1))

    with pytest.raises(ValueError, match="provenance mismatch"):
        trainer._load_checkpoint(path)


def test_trainer_load_checkpoint_accepts_matching_provenance(tmp_path):
    """Trainer._load_checkpoint must accept checkpoint whose provenance
    matches the active run's provenance."""
    config = tiny_config(tmp_path)
    model_ckpt = build_model(config)
    shared_provenance = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "same-split"},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    shared_provenance["model_state_sha256"] = model_state_hash(
        model_ckpt.state_dict())
    config_both = {**config, "provenance": shared_provenance}
    path = tmp_path / "ckpt.pt"
    torch.save({
        "model_state_dict": model_ckpt.state_dict(),
        "provenance": shared_provenance,
        "config": config_both,
        "epoch": 0,
    }, path)

    model_active = build_model(config)
    from torch_geometric.loader import DataLoader
    trainer = Trainer(config_both, model_active,
                      train_loader=DataLoader([], batch_size=1),
                      val_loader=DataLoader([], batch_size=1))
    trainer._load_checkpoint(path)


def test_trainer_load_checkpoint_rejects_missing_provenance_when_required(tmp_path):
    """When active run declares provenance, a checkpoint without provenance
    must be rejected before loading weights."""
    config = tiny_config(tmp_path)
    model_ckpt = build_model(config)
    active_provenance = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "s" * 64},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    config_active = {**config, "provenance": active_provenance}
    # Save checkpoint WITHOUT provenance
    path = tmp_path / "no_prov.pt"
    torch.save({
        "model_state_dict": model_ckpt.state_dict(),
        "config": config,
        "epoch": 0,
    }, path)

    model_active = build_model(config_active)
    from torch_geometric.loader import DataLoader
    trainer = Trainer(config_active, model_active,
                      train_loader=DataLoader([], batch_size=1),
                      val_loader=DataLoader([], batch_size=1))

    weights_before = {
        k: v.clone() for k, v in model_active.state_dict().items()
    }
    with pytest.raises(ValueError, match="checkpoint has no provenance"):
        trainer._load_checkpoint(path)
    for k, v in model_active.state_dict().items():
        assert torch.equal(v, weights_before[k]), (
            f"Weights changed after rejected load: {k}"
        )


def test_trainer_load_checkpoint_rejects_empty_provenance_when_required(tmp_path):
    """A checkpoint with empty provenance {} must be rejected when active
    run declares provenance."""
    config = tiny_config(tmp_path)
    model_ckpt = build_model(config)
    active_provenance = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "s" * 64},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    config_active = {**config, "provenance": active_provenance}
    path = tmp_path / "empty_prov.pt"
    torch.save({
        "model_state_dict": model_ckpt.state_dict(),
        "provenance": {},
        "config": config,
        "epoch": 0,
    }, path)

    model_active = build_model(config_active)
    from torch_geometric.loader import DataLoader
    trainer = Trainer(config_active, model_active,
                      train_loader=DataLoader([], batch_size=1),
                      val_loader=DataLoader([], batch_size=1))

    weights_before = {
        k: v.clone() for k, v in model_active.state_dict().items()
    }
    with pytest.raises(ValueError, match="checkpoint has no provenance"):
        trainer._load_checkpoint(path)
    for k, v in model_active.state_dict().items():
        assert torch.equal(v, weights_before[k]), (
            f"Weights changed after rejected load: {k}"
        )


def test_trainer_load_checkpoint_no_active_provenance_allows_no_prov_ckpt(tmp_path):
    """When no active provenance is declared, a checkpoint without provenance
    must be accepted (legacy workflow)."""
    config = tiny_config(tmp_path)
    model_ckpt = build_model(config)
    path = tmp_path / "legacy.pt"
    torch.save({
        "model_state_dict": model_ckpt.state_dict(),
        "epoch": 0,
    }, path)

    model_active = build_model(config)
    from torch_geometric.loader import DataLoader
    trainer = Trainer(config, model_active,
                      train_loader=DataLoader([], batch_size=1),
                      val_loader=DataLoader([], batch_size=1))
    trainer._load_checkpoint(path)
    assert torch.equal(
        model_active.state_dict()["input_proj.0.weight"],
        model_ckpt.state_dict()["input_proj.0.weight"],
    )


def test_load_model_config_optin_legacy_label_loads_weights(tmp_path):
    """load_model with no keyword but data.checkpoint_research_label in config
    must load historical weights successfully."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    path = tmp_path / "historical.pt"
    torch.save(model.state_dict(), path)

    cfg_with_label = {**config, "data": {
        **config.get("data", {}),
        "checkpoint_research_label": "legacy_total_radius",
    }}
    loaded = load_model(str(path), cfg_with_label, torch.device("cpu"))
    assert model_state_hash(loaded.state_dict()) == model_state_hash(
        model.state_dict())


def test_load_model_config_optin_wrong_label_still_rejected(tmp_path):
    """Config option with a non-legacy label must still be rejected for
    unprovenanced checkpoints."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    path = tmp_path / "historical.pt"
    torch.save(model.state_dict(), path)

    cfg_with_label = {**config, "data": {
        **config.get("data", {}),
        "checkpoint_research_label": "corrected_v2",
    }}
    with pytest.raises(ValueError, match="legacy"):
        load_model(str(path), cfg_with_label, torch.device("cpu"))


def test_load_model_explicit_keyword_overrides_config(tmp_path):
    """Explicit research_label keyword must take precedence over config."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    path = tmp_path / "historical.pt"
    torch.save(model.state_dict(), path)

    cfg_with_label = {**config, "data": {
        **config.get("data", {}),
        "checkpoint_research_label": "legacy_uncertified",
    }}
    loaded = load_model(str(path), cfg_with_label, torch.device("cpu"),
                        research_label="legacy_total_radius")
    assert model_state_hash(loaded.state_dict()) == model_state_hash(
        model.state_dict())


def test_load_model_no_config_no_keyword_rejects(tmp_path):
    """Without explicit keyword or config option, unprovenanced checkpoint
    must still be rejected."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    path = tmp_path / "historical.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="legacy"):
        load_model(str(path), config, torch.device("cpu"))


def test_load_model_expected_provenance_enforced_with_legacy_optin(tmp_path):
    """When expected_provenance is explicitly supplied, checkpoint must have
    provenance even if legacy research_label is also supplied."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), path)  # no provenance

    with pytest.raises(ValueError, match="expected_provenance was supplied"):
        load_model(str(path), config, torch.device("cpu"),
                   research_label="legacy_total_radius",
                   expected_provenance={"research_label": "corrected_research",
                                        "split_manifest_sha256": "x" * 64})


def test_load_model_expected_provenance_validates_matching_checkpoint(tmp_path):
    """When expected_provenance is supplied and checkpoint has provenance,
    mismatch must be rejected."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    prov = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "actual-split"},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    prov["model_state_sha256"] = model_state_hash(model.state_dict())
    path = tmp_path / "provenanced.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "provenance": prov, "config": config, "epoch": 0}, path)

    # expected_provenance with correct research_label but wrong split
    with pytest.raises(ValueError, match="split_manifest_sha256"):
        load_model(str(path), config, torch.device("cpu"),
                   expected_provenance={
                       "research_label": "synthetic",
                       "catalog_contract_sha256": prov["catalog_contract_sha256"],
                       "config_sha256": prov["config_sha256"],
                       "split_manifest_sha256": "wrong-split",
                   })


def test_load_model_expected_provenance_accepts_matching(tmp_path):
    """When expected_provenance matches checkpoint provenance, load succeeds."""
    config = tiny_config(tmp_path)
    model = build_model(config)
    prov = build_run_provenance(
        source_files=[],
        catalog_contract={"research_label": "synthetic",
                           "target_definition": "log10(M)"},
        split_manifest={"manifest_sha256": "the-split"},
        resolved_config=config,
        source_revision="dc045d4",
        graph={"backend": "torch", "tie_policy": "inclusive"},
        augmentation_manifest=None,
        research_label="synthetic",
    )
    prov["model_state_sha256"] = model_state_hash(model.state_dict())
    path = tmp_path / "provenanced.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "provenance": prov, "config": config, "epoch": 0}, path)

    loaded = load_model(str(path), config, torch.device("cpu"),
                        expected_provenance={
                            "research_label": "synthetic",
                            "catalog_contract_sha256": prov["catalog_contract_sha256"],
                            "config_sha256": prov["config_sha256"],
                            "split_manifest_sha256": "the-split",
                        })
    assert model_state_hash(loaded.state_dict()) == model_state_hash(
        model.state_dict())


def test_trainer_saves_final_model_before_best_weight_reload(tmp_path):
    """final_model.pt must contain last-epoch weights and metrics, saved
    BEFORE best weights are reloaded for test evaluation.  Deterministic
    fixture: no real GNN forward pass, no random data, no scheduler."""
    config = tiny_config(tmp_path)
    config["training"].update(epochs=2, save_every=1, save_best=True,
                              early_stopping_patience=10)
    model = build_model(config)

    trainer = Trainer(config, model,
                      train_loader=[], val_loader=[], test_loader=[])
    trainer.lr_scheduler = None

    w_param = next(model.parameters())

    def controlled_train_epoch(epoch):
        with torch.no_grad():
            w_param.fill_(float(epoch + 1))
        return {"train/mse": (epoch + 1) ** 2,
                "train/loss": (epoch + 1) ** 2,
                "train/virial_loss": 0.0,
                "train/lambda": 0.0,
                "train/lr": 0.001}

    def controlled_validate(loader, prefix="val"):
        w = w_param[0, 0].item()
        return {f"{prefix}/mse": w * w,
                f"{prefix}/r2": -w * w}

    trainer.train_epoch = controlled_train_epoch
    trainer.validate = controlled_validate
    result = trainer.train()

    ckpt_dir = tmp_path
    final_path = ckpt_dir / "final_model.pt"
    best_path = ckpt_dir / "best_model.pt"
    assert final_path.exists(), "final_model.pt must exist"
    assert best_path.exists(), "best_model.pt must exist"

    final_ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)

    # final_model.pt: epoch=1, weight=2, val/mse=4, no test metrics
    assert final_ckpt["epoch"] == 1
    final_w = final_ckpt["model_state_dict"]["input_proj.0.weight"][0, 0].item()
    assert final_w == pytest.approx(2.0, abs=1e-6), (
        f"final_model weight should be 2.0 (last epoch), got {final_w}"
    )
    assert final_ckpt["metrics"]["val/mse"] == pytest.approx(4.0, abs=1e-6)
    assert "test/mse" not in final_ckpt["metrics"]

    # best_model.pt: epoch=0, weight=1, val/mse=1
    assert best_ckpt["epoch"] == 0
    best_w = best_ckpt["model_state_dict"]["input_proj.0.weight"][0, 0].item()
    assert best_w == pytest.approx(1.0, abs=1e-6), (
        f"best_model weight should be 1.0 (best epoch), got {best_w}"
    )
    assert best_ckpt["metrics"]["val/mse"] == pytest.approx(1.0, abs=1e-6)

    # returned metrics
    assert result["final_test_metrics"]["test/mse"] == pytest.approx(1.0, abs=1e-6)
    assert result["best_epoch"] == 0
    assert result["best_val_loss"] == pytest.approx(1.0, abs=1e-6)

    # current model weight after best-weight reload = 1
    assert w_param[0, 0].item() == pytest.approx(1.0, abs=1e-6)

    # model-state hashes match actual weights
    final_hash = model_state_hash(final_ckpt["model_state_dict"])
    best_hash = model_state_hash(best_ckpt["model_state_dict"])
    assert final_hash != best_hash, "final and best should have different state hashes"
