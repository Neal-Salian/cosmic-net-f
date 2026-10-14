"""Audited baseline tests use tiny explicitly synthetic HDF5 inputs only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from data.astronomy_audit import audit_astronomy_manifest
from data.astronomy_loader import load_audited_catalog
from data.astronomy_manifest import load_astronomy_manifest
from data.observations import ObservationConfig
from data.projected_graph import ProjectedGraphConfig, checkpoint_compatibility_record
from data.provenance import hash_payload, model_state_hash, validate_run_provenance
from model.model import CosmicNetGNN
from rls.astronomy_preflight import _load_artifact, _read_split_manifest
from training.full_graph_baseline import run_baseline


def _catalog(path, index, *, kind="synthetic_fixture", role="development", ic=None):
    with h5py.File(path, "w") as h5:
        h5.attrs["h"], h5.attrs["Time"] = .5, 1.
        h5.create_dataset("Group/Group_M_Crit200", data=[5. + index, 10. + index])
        h5.create_dataset("Subhalo/SubhaloMass", data=np.ones(6))
        h5.create_dataset("Subhalo/SubhaloPos", data=np.arange(18).reshape(6, 3) * 10.)
        h5.create_dataset("Subhalo/SubhaloVel", data=np.arange(18).reshape(6, 3) * 2.)
        masses = np.zeros((6, 6)); masses[:, 4] = .1 + np.arange(6) / 10
        radii = np.zeros((6, 6)); radii[:, 4] = 20.
        h5.create_dataset("Subhalo/SubhaloMassType", data=masses)
        h5.create_dataset("Subhalo/SubhaloHalfmassRadType", data=radii)
        h5.create_dataset("Subhalo/SubhaloGrNr", data=np.repeat([0, 1], 3))
    return {
        "id": f"fixture-{index}", "role": role, "suite": "IllustrisTNG", "simulation": f"fixture-{index}",
        "simulation_set": "LH", "initial_condition_id": ic or f"fixture-ic-{index}", "snapshot": "099",
        "source": {"kind": kind, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "epoch": {"h": .5, "scale_factor": 1., "redshift": 0.},
        "conventions": {"coordinates": "comoving", "velocities": "peculiar_km_per_s", "periodic_box": False},
        "fields": {
            "group_mass": {"path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "shape": [2]},
            "subhalo_mass_anchor": {"path": "Subhalo/SubhaloMass", "unit": "1e10_Msun_per_h", "shape": [6]},
            "position": {"path": "Subhalo/SubhaloPos", "unit": "ckpc_per_h", "shape": [6, 3]},
            "velocity": {"path": "Subhalo/SubhaloVel", "unit": "km_per_s", "shape": [6, 3]},
            "stellar_mass": {"path": "Subhalo/SubhaloMassType", "unit": "1e10_Msun_per_h", "shape": [6, 6], "component": 4},
            "half_mass_radius": {"path": "Subhalo/SubhaloHalfmassRadType", "unit": "ckpc_per_h", "shape": [6, 6], "component": 4},
            "group_index": {"path": "Subhalo/SubhaloGrNr", "unit": "index", "shape": [6]},
        },
        "target": {"name": "halo_mass", "path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h",
                   "definition": "log10(M200c/M_sun)", "shape": [2]},
        "selection": {"predicate": "stellar_mass_type4_gt_zero"}, "release": "synthetic baseline test fixture",
    }


def _config(tmp_path, count=4, *, kind="synthetic_fixture"):
    entries = [_catalog(tmp_path / f"fixture-{i}.h5", i, kind=kind) for i in range(count)]
    manifest_path = tmp_path / "catalogs.yaml"
    manifest_path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": entries}))
    return {
        "seed": 7,
        "data": {"source": "audited_astronomy", "strict_research": True,
                 "manifest_path": str(manifest_path), "catalog_ids": [e["id"] for e in entries],
                 "split_group_key": "initial_condition_id", "train_ratio": .5, "val_ratio": .25,
                 "test_ratio": .25, "batch_size": 2, "num_workers": 0},
        "observation": {"line_of_sight": "z", "seed": 7, "realization_id": "baseline-fixture", "min_richness": 2},
        "graph": {"method": "knn", "k_neighbors": 2, "self_loops": True,
                  "edge_features": ["projected_distance", "los_delta_v", "mass_ratio"]},
        "model": {"node_features": 4, "hidden_dim": 8, "output_dim": 8, "num_layers": 1,
                  "dropout": 0., "mc_dropout": False, "pooling": "mean"},
        "training": {"epochs": 1, "evidence_mode": "fixture_smoke", "device": "cpu",
                     "learning_rate": .001, "optimizer": "adam", "lr_scheduler": "none"},
        "physics": {"use_virial_loss": False}, "wandb": {"enabled": False},
    }


def _rewrite_manifest(config, mutator):
    path = Path(config["data"]["manifest_path"])
    payload = yaml.safe_load(path.read_text())
    mutator(payload)
    path.write_text(yaml.safe_dump(payload))


def test_explicit_audited_fixture_trains_projected_schema_and_exports_bound_backbone(tmp_path):
    config = _config(tmp_path)
    output = tmp_path / "run"
    result = run_baseline(config, output_dir=output)
    assert result["research_label"] == "fixture_smoke"
    assert result["scientific_result"] is False
    assert result["physics_use_virial_loss"] is False
    assert {"val/rmse", "test/rmse", "test/r2"} <= set(result["metrics"])
    assert all(np.isfinite(v) for v in result["metrics"].values() if v is not None)
    split, _ = _read_split_manifest(output / "split_manifest.json")
    sections = split["splits"]
    assert [sections[k]["group_count"] for k in ("train", "val", "test")] == [2, 1, 1]
    assert len(set.union(*(set(sections[k]["cluster_ids"]) for k in sections))) == 8
    artifact_path = output / "projected_backbone.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact["artifact_type"] == "projected_backbone"
    assert artifact["provenance"]["research_label"] == "fixture_smoke"
    assert artifact["provenance"]["catalog_contract"]["synthetic"] is True
    assert artifact["projected_compatibility"] == checkpoint_compatibility_record(
        ProjectedGraphConfig(method="knn", k_neighbors=2), "z")
    observation = asdict(ObservationConfig(**config["observation"]))
    assert artifact["training_observation_config"] == observation
    assert artifact["training_observation_config_sha256"] == hash_payload(observation)
    provenance = artifact["provenance"]
    validate_run_provenance(provenance, model_state=artifact["model_state_dict"], split_manifest=split)
    assert len(provenance["source_files"]) == 4
    assert set(provenance["training_catalogs"]) == set(config["data"]["catalog_ids"])
    assert len(provenance["source_code_files"]) >= 10
    resolved = provenance["resolved_config"]
    assert resolved["model"]["edge_features"] == 3
    assert resolved["projected_compatibility"] == artifact["projected_compatibility"]
    assert resolved["training_observation_config"] == observation
    model = CosmicNetGNN(resolved)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    predictions = json.loads((output / "heldout_predictions.json").read_text())
    assert {row["cluster_id"] for row in predictions["test"]} == set(sections["test"]["cluster_ids"])
    assert all(row["split"] == "test" for row in predictions["test"])
    reports = json.loads((output / "observation_reports.json").read_text())
    assert len(reports) == 8
    assert not any(output.glob("*policy*.pt"))
    _load_artifact(artifact_path, "projected_backbone", "model_state_dict", evaluation_mode="fixture_smoke")
    with pytest.raises(ValueError, match="fixture|corrected_research"):
        _load_artifact(artifact_path, "projected_backbone", "model_state_dict")


@pytest.mark.parametrize("case,match", [
    ("one_ic", "independent|nonempty"),
    ("wrong_role", "development"),
    ("schema", "projected|unknown"),
    ("target", "target"),
    ("bad_data", "finite"),
    ("missing_id", "catalog"),
    ("duplicate_id", "unique"),
    ("physics", "physics|virial"),
    ("bad_epoch", "epochs"),
    ("fixture_science", "synthetic_fixture|fixture"),
])
def test_invalid_audited_requests_fail_before_output_creation(tmp_path, case, match):
    config = _config(tmp_path)
    if case == "one_ic":
        _rewrite_manifest(config, lambda p: [e.update(initial_condition_id="same-ic") for e in p["catalogs"]])
    elif case == "wrong_role":
        _rewrite_manifest(config, lambda p: p["catalogs"][0].update(role="external_development"))
    elif case == "schema":
        config["graph"]["edge_features"] = ["metallicity"]
    elif case == "target":
        _rewrite_manifest(config, lambda p: p["catalogs"][0]["target"].update(definition="wrong"))
    elif case == "bad_data":
        path = tmp_path / "fixture-0.h5"
        with h5py.File(path, "a") as h5:
            h5["Subhalo/SubhaloVel"][0, 0] = np.nan
        _rewrite_manifest(config, lambda p: p["catalogs"][0]["source"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    elif case == "missing_id":
        config["data"]["catalog_ids"][0] = "absent"
    elif case == "duplicate_id":
        config["data"]["catalog_ids"][0] = config["data"]["catalog_ids"][1]
    elif case == "physics":
        config["physics"]["use_virial_loss"] = True
    elif case == "bad_epoch":
        config["training"]["epochs"] = 0
    elif case == "fixture_science":
        config["training"]["evidence_mode"] = "scientific"
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        run_baseline(config, output_dir=output)
    assert not output.exists()


def test_synthetic_fixture_loading_requires_explicit_optin_and_keeps_partial_schema(tmp_path):
    config = _config(tmp_path)
    manifest = load_astronomy_manifest(config["data"]["manifest_path"])
    audit = audit_astronomy_manifest(manifest)
    with pytest.raises(ValueError, match="synthetic_fixture"):
        load_audited_catalog(manifest, audit, "fixture-0")
    halo = load_audited_catalog(manifest, audit, "fixture-0", allow_synthetic_fixture=True)[0]
    assert halo.metadata["catalog_contract"]["research_label"] == "synthetic"
    assert halo.metadata["catalog_contract"]["synthetic"] is True
    assert not hasattr(halo.members[0], "velocity_dispersion")
    assert not hasattr(halo.members[0], "metallicity")


def test_fixture_mode_never_upgrades_local_hdf5_evidence(tmp_path):
    config = _config(tmp_path, kind="local_hdf5")
    result = run_baseline(config, output_dir=tmp_path / "local-fixture")
    assert result["research_label"] == "fixture_smoke"
    assert not result["scientific_result"]


def test_legacy_loader_contract_cannot_bypass_audited_source_or_create_output(tmp_path):
    config = _config(tmp_path, kind="local_hdf5")
    manifest = load_astronomy_manifest(config["data"]["manifest_path"])
    audit = audit_astronomy_manifest(manifest)
    contract = load_audited_catalog(manifest, audit, "fixture-0")[0].metadata["catalog_contract"]
    config["data"].update(source="tng", catalog_contract=contract)
    output = tmp_path / "legacy"
    with pytest.raises(ValueError, match="audited_astronomy"):
        run_baseline(config, output_dir=output)
    assert not output.exists()


def test_richness_rejection_requires_nonempty_ic_splits_before_output(tmp_path):
    config = _config(tmp_path)
    config["observation"]["min_richness"] = 4
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="independent|nonempty"):
        run_baseline(config, output_dir=output)
    assert not output.exists()
