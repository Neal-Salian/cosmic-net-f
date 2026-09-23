"""Offline scientific-readiness checks; catalog/model files are labeled fixtures."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from data.astronomy_manifest import load_astronomy_manifest
from data.astronomy_audit import audit_astronomy_manifest
from data.projected_graph import ProjectedGraphConfig, checkpoint_compatibility_record
from data.observations import ObservationConfig
from data.provenance import hash_payload, model_state_hash
from rls.astronomy_preflight import preflight_astronomy
from rls.matched_benchmark import BudgetSpec


def _source(path):
    with h5py.File(path, "w") as f:
        f.attrs["h"], f.attrs["Time"] = 0.6774, 1.0
        f.create_dataset("Group/Group_M_Crit200", data=[10.])
        f.create_dataset("Subhalo/SubhaloMass", data=[1., 2.])
        f.create_dataset("Subhalo/SubhaloPos", data=np.zeros((2, 3)))
        f.create_dataset("Subhalo/SubhaloVel", data=np.zeros((2, 3)))
        sm = np.ones((2, 6)); sm[:, :4] = 0
        f.create_dataset("Subhalo/SubhaloMassType", data=sm)
        f.create_dataset("Subhalo/SubhaloHalfmassRadType", data=np.ones((2, 6)))
        f.create_dataset("Subhalo/SubhaloGrNr", data=[0, 0])


def _entry(source, kind="local_hdf5", role="development"):
    return {
        "id": "fixture", "role": role, "suite": "IllustrisTNG", "simulation": "L25n256",
        "simulation_set": "LH", "initial_condition_id": "fixture-ic", "snapshot": "099",
        "source": {"kind": kind, "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "epoch": {"redshift": 0., "scale_factor": 1., "h": .6774},
        "conventions": {"coordinates": "comoving", "velocities": "peculiar_km_per_s", "periodic_box": False},
        "fields": {
            "group_mass": {"path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "shape": [1]},
            "subhalo_mass_anchor": {"path": "Subhalo/SubhaloMass", "unit": "1e10_Msun_per_h", "shape": [2]},
            "position": {"path": "Subhalo/SubhaloPos", "unit": "ckpc_per_h", "shape": [2, 3]},
            "velocity": {"path": "Subhalo/SubhaloVel", "unit": "km_per_s", "shape": [2, 3]},
            "stellar_mass": {"path": "Subhalo/SubhaloMassType", "unit": "1e10_Msun_per_h", "shape": [2, 6], "component": 4},
            "half_mass_radius": {"path": "Subhalo/SubhaloHalfmassRadType", "unit": "ckpc_per_h", "shape": [2, 6], "component": 4},
            "group_index": {"path": "Subhalo/SubhaloGrNr", "unit": "index", "shape": [2]},
        },
        "target": {"name": "halo_mass", "path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "definition": "log10(M200c/M_sun)", "shape": [1]},
        "selection": {"predicate": "stellar_mass_type4_gt_zero"}, "release": "synthetic test fixture",
    }


def _setup(tmp_path, *, kind="local_hdf5", role="development"):
    source = tmp_path / "fixture.h5"; _source(source)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": [_entry(source, kind, role)]}))
    split = {"schema_version": 1, "group_key": "initial_condition_id", "seed": 4,
             "ratios": {"train": .7, "val": .15, "test": .15},
             "splits": {"train": {"group_ids": ["train"], "cluster_ids": ["a"], "parent_ids": ["a"], "group_count": 1, "halo_count": 1},
                        "val": {"group_ids": ["val"], "cluster_ids": ["b"], "parent_ids": ["b"], "group_count": 1, "halo_count": 1},
                        "test": {"group_ids": ["test"], "cluster_ids": ["c"], "parent_ids": ["c"], "group_count": 1, "halo_count": 1}},
             "generator": "fixture"}
    split["manifest_sha256"] = hash_payload({k: v for k, v in split.items() if k != "manifest_sha256"})
    split_path = tmp_path / "split.json"; split_path.write_text(json.dumps(split))
    graph_config, observation = ProjectedGraphConfig(), ObservationConfig(line_of_sight="z")
    compat = checkpoint_compatibility_record(graph_config, "z")
    state = {"weight": torch.tensor([1.])}
    catalog_contract = {
        "schema_version": 1, "research_label": "corrected_research", "source": "IllustrisTNG/L25n256/099",
        "suite": "IllustrisTNG", "simulation": "L25n256", "snapshot": "099", "volume_or_ic_group": "train-ic",
        "target_field": "Group_M_Crit200", "target_definition": "log10(M200c/M_sun)", "mass_units": "M_sun",
        "position_units": "physical_Mpc", "radius_units": "physical_Mpc", "velocity_units": "peculiar_km_per_s",
        "coordinate_frame": "physical_proper", "velocity_convention": "peculiar_km_per_s", "hubble_param": .6774,
        "scale_factor": 1., "radius_source_field": "SubhaloHalfmassRadType[4]", "radius_semantic": "stellar_half_mass_radius_type4",
        "synthetic": False, "fallback": False, "source_files": [{"path": "train.h5", "size_bytes": 1, "sha256": "a"*64}],
        "field_mapping": {"group_mass": "Group/Group_M_Crit200"}, "conversion_record": {"mass": "raw*1e10/h"}}
    training_observation = {"line_of_sight": "z", "missing_fraction": 0., "noise": {}}
    config = {"projected_compatibility": compat, "training_observation_config": training_observation}
    prov = {"schema_version": 1, "catalog_contract": catalog_contract,
            "catalog_contract_sha256": hash_payload(catalog_contract), "split_manifest_sha256": split["manifest_sha256"],
            "resolved_config": config, "config_sha256": hash_payload(config), "source_revision": "test-only",
            "research_label": "corrected_research",
            "source_code_files": [{"path": "synthetic-test.py", "size_bytes": 1, "sha256": "c"*64}]}
    prov["model_state_sha256"] = model_state_hash(state)
    policy_state = {"weight": torch.tensor([2.])}
    training_observation_sha = hash_payload(training_observation)
    torch.save({"artifact_type": "projected_backbone", "model_state_dict": state, "provenance": prov,
                "projected_compatibility": compat, "training_observation_config": training_observation,
                "training_observation_config_sha256": training_observation_sha}, tmp_path / "backbone.pt")
    pprov = dict(prov)
    pprov["model_state_sha256"] = model_state_hash(policy_state)
    torch.save({"artifact_type": "projected_policy", "policy_state_dict": policy_state, "provenance": pprov,
                "projected_compatibility": compat, "backbone_state_sha256": model_state_hash(state),
                "training_observation_config": training_observation,
                "training_observation_config_sha256": training_observation_sha}, tmp_path / "policy.pt")
    return manifest_path, split_path, graph_config, observation, tmp_path / "backbone.pt", tmp_path / "policy.pt"


def _call(paths, **kw):
    manifest, split, graph, observation, backbone, policy = paths
    args = dict(manifest_path=manifest, catalog_id="fixture", expected_role="development",
                split_manifest_path=split, checkpoint_path=backbone, policy_path=policy,
                graph_config=graph, observation_config=observation,
                transform_config={"scenario": "baseline"},
                budget=BudgetSpec(pair_count=1, constraint="no_isolates"),
                evaluation_mode="fixture_smoke")
    args.update(kw)
    return preflight_astronomy(**args)


def test_preflight_returns_labeled_fixture_smoke_record_without_scientific_readiness(tmp_path):
    result = _call(_setup(tmp_path))
    assert result["fixture_smoke_ready"] and result["metadata_preflight_passed"]
    assert not result["scientific_evaluation_allowed"] and result["evaluation_execution_ready"] is False
    assert result["evaluation_executed"] is False and result["model_load_verified"] is False
    assert result["catalog"]["role"] == "development"
    assert result["catalog"]["source_file"]["sha256"] == result["catalog"]["source_sha256"]
    assert result["artifacts"]["training_source_revision"] == "test-only"
    assert result["artifacts"]["training_source_code_files"]
    assert not result["training_source_files_rehashed"]
    assert result["budget"] == {"pair_count": 1, "keep_fraction": None, "constraint": "no_isolates", "decoder": "direct"}
    assert result["artifacts"]["backbone_sha256"] and result["artifacts"]["policy_sha256"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["synthetic_fixture"])
def test_preflight_rejects_synthetic_catalog_before_loading_artifacts(tmp_path, kind):
    paths = _setup(tmp_path, kind=kind)
    paths[4].unlink(); paths[5].unlink()
    with pytest.raises(ValueError, match="synthetic"):
        _call(paths)


def test_preflight_rejects_role_mismatch_before_artifacts(tmp_path):
    paths = _setup(tmp_path, role="external_development")
    paths[4].unlink(); paths[5].unlink()
    with pytest.raises(ValueError, match="role"):
        _call(paths)


def test_preflight_requires_content_verified_catalog_and_split_hash(tmp_path):
    paths = list(_setup(tmp_path)); paths[1].write_text("{}")
    with pytest.raises(ValueError, match="split"):
        _call(tuple(paths))


def test_preflight_rejects_incompatible_projected_semantics_and_legacy_state_dict(tmp_path):
    paths = list(_setup(tmp_path))
    torch.save({"weight": torch.tensor([1.])}, paths[4])
    with pytest.raises(ValueError, match="projected|artifact"):
        _call(tuple(paths))


def test_preflight_accepts_fraction_and_records_unresolved_per_graph_count(tmp_path):
    paths = _setup(tmp_path)
    result = _call(paths, budget=BudgetSpec(keep_fraction=.5))
    assert result["budget"]["keep_fraction"] == .5
    assert result["achieved_pair_counts_available"] is False


def test_preflight_fixture_exercises_simba_confirmation_role_without_claiming_data(tmp_path):
    paths = list(_setup(tmp_path, role="confirmation"))
    manifest_path = paths[0]
    payload = yaml.safe_load(manifest_path.read_text())
    payload["catalogs"][0]["suite"] = "SIMBA"
    payload["catalogs"][0]["simulation"] = "SIMBA fixture simulation"
    payload["catalogs"][0]["initial_condition_id"] = "fixture-confirmation-ic"
    manifest_path.write_text(yaml.safe_dump(payload))
    result = _call(tuple(paths), expected_role="confirmation")
    assert result["catalog"]["suite"] == "SIMBA"
    assert result["catalog"]["initial_condition_id"] == "fixture-confirmation-ic"
    assert result["fixture_smoke_ready"] and not result["scientific_evaluation_allowed"]


def test_release_text_does_not_control_fixture_science_status(tmp_path):
    paths = list(_setup(tmp_path))
    manifest_path = paths[0]
    payload = yaml.safe_load(manifest_path.read_text())
    payload["catalogs"][0]["release"] = "v1 test release"
    manifest_path.write_text(yaml.safe_dump(payload))
    result = _call(tuple(paths), evaluation_mode="scientific")
    assert result["metadata_preflight_passed"]
    assert not result["scientific_evaluation_allowed"]
    assert not result["fixture_smoke_ready"]


def test_preflight_rejects_tampered_training_observation_identity(tmp_path):
    paths = list(_setup(tmp_path))
    artifact = torch.load(paths[4], map_location="cpu", weights_only=True)
    artifact["training_observation_config_sha256"] = "d"*64
    torch.save(artifact, paths[4])
    with pytest.raises(ValueError, match="observation"):
        _call(tuple(paths))


def test_preflight_rejects_rehashed_but_overlapping_split_membership(tmp_path):
    paths = list(_setup(tmp_path))
    split = json.loads(paths[1].read_text())
    split["splits"]["val"]["cluster_ids"] = ["a"]
    split["manifest_sha256"] = hash_payload({k: v for k, v in split.items() if k != "manifest_sha256"})
    paths[1].write_text(json.dumps(split))
    with pytest.raises(ValueError, match="overlap|disjoint"):
        _call(tuple(paths))


def test_preflight_rejects_snapshot_grouped_split_even_with_valid_hash(tmp_path):
    paths = list(_setup(tmp_path))
    split = json.loads(paths[1].read_text())
    split["group_key"] = "snapshot"
    split["manifest_sha256"] = hash_payload({k: v for k, v in split.items() if k != "manifest_sha256"})
    paths[1].write_text(json.dumps(split))
    with pytest.raises(ValueError, match="initial-condition/lineage"):
        _call(tuple(paths))


def test_preflight_rejects_feature_record_that_differs_from_resolved_config(tmp_path):
    paths = list(_setup(tmp_path))
    artifact = torch.load(paths[4], map_location="cpu", weights_only=True)
    artifact["provenance"]["resolved_config"]["projected_compatibility"] = checkpoint_compatibility_record(
        ProjectedGraphConfig(), "x")
    artifact["provenance"]["config_sha256"] = hash_payload(artifact["provenance"]["resolved_config"])
    torch.save(artifact, paths[4])
    with pytest.raises(ValueError, match="resolved training config"):
        _call(tuple(paths))
