"""Offline, fail-closed readiness checks for projected astronomy evaluation.

This module validates inputs and artifact identity only. It does not build a
graph, load a model into an architecture, or run inference/evaluation.
"""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re

import torch

from data.astronomy_audit import audit_astronomy_manifest
from data.astronomy_loader import load_audited_catalog
from data.astronomy_manifest import load_astronomy_manifest
from data.catalog import CatalogContract
from data.observations import ObservationConfig
from data.projected_graph import (ProjectedGraphConfig,
                                  checkpoint_compatibility_record,
                                  validate_checkpoint_compatibility)
from data.provenance import (hash_payload, model_state_hash, sha256_file,
                             validate_checkpoint_file, validate_inference_compatibility,
                             validate_run_provenance)
from rls.matched_benchmark import BudgetSpec


def _load_artifact(path, artifact_type, state_key):
    file_record = validate_checkpoint_file(path)
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{artifact_type} artifact is not a safely readable structured checkpoint") from exc
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != artifact_type:
        raise ValueError(f"{artifact_type} requires an explicit structured projected artifact")
    state = artifact.get(state_key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{artifact_type} artifact is missing {state_key}")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{artifact_type} artifact is missing positive provenance")
    if provenance.get("research_label") != "corrected_research":
        raise ValueError(f"{artifact_type} provenance must identify corrected_research data")
    code_files = provenance.get("source_code_files")
    if not isinstance(code_files, list) or not code_files:
        raise ValueError(f"{artifact_type} provenance requires source_code_files identities")
    for record in code_files:
        if (not isinstance(record, dict) or not record.get("path")
                or not isinstance(record.get("size_bytes"), int)
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
            raise ValueError(f"{artifact_type} source_code_files entries need path, size and SHA-256")
    for name in ("source_revision", "catalog_contract", "catalog_contract_sha256",
                 "split_manifest_sha256", "resolved_config", "config_sha256"):
        if not provenance.get(name):
            raise ValueError(f"{artifact_type} provenance is missing {name}")
    validate_run_provenance(provenance, model_state=state)
    contract = CatalogContract.from_mapping(provenance["catalog_contract"]).validate(strict_research=True)
    return artifact, state, provenance, contract, file_record


def _read_split_manifest(path):
    if path is None:
        raise ValueError("an explicit training split manifest file is required")
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:
        raise ValueError("training split manifest must be readable JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("training split manifest has unsupported or missing schema_version")
    digest = data.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("training split manifest requires a SHA-256 identity")
    payload = {key: value for key, value in data.items() if key != "manifest_sha256"}
    if hash_payload(payload) != digest:
        raise ValueError("training split manifest content hash mismatch")
    sections = data.get("splits")
    names = ("train", "val", "test")
    if not isinstance(sections, dict) or set(sections) != set(names):
        raise ValueError("training split manifest must explicitly define train, val and test")
    if data.get("group_key") not in {"initial_condition_id", "lineage_group", "volume_or_ic_group"}:
        raise ValueError("training split must group by initial-condition/lineage identity")
    if not sections["train"].get("cluster_ids"):
        raise ValueError("training split manifest has no explicit training membership")
    for field in ("group_ids", "cluster_ids", "parent_ids"):
        seen = set()
        for name in names:
            if not isinstance(sections[name], dict):
                raise ValueError(f"split {name} must be a mapping")
            values = sections[name].get(field)
            if not isinstance(values, list) or len(values) != len(set(values)):
                raise ValueError(f"split {name} requires unique {field}")
            overlap = seen.intersection(values)
            if overlap:
                raise ValueError(f"split {field} must be disjoint; overlap: {sorted(overlap)}")
            seen.update(values)
    for name in names:
        section = sections[name]
        for count_key, field in (("group_count", "group_ids"), ("halo_count", "cluster_ids")):
            if section.get(count_key) != len(section[field]):
                raise ValueError(f"split {name} {count_key} disagrees with {field}")
    return data, sha256_file(path)


def _config_payload(value, label):
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be an explicit nonempty mapping/configuration")
    # Round-trip through canonical JSON so output and identity are JSON-safe.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def preflight_astronomy(*, manifest_path, catalog_id, expected_role,
                        split_manifest_path, checkpoint_path, policy_path,
                        graph_config, observation_config, transform_config,
                        budget, evaluation_mode="scientific"):
    """Verify an explicitly selected real catalog and projected artifacts.

    Returns a JSON-safe readiness record. Readiness means inputs are internally
    identified and semantically compatible; it is not evidence that evaluation
    has run or that the requested budget is feasible for every projected graph.
    """
    if expected_role not in {"development", "external_development", "confirmation"}:
        raise ValueError("expected_role must be an explicit supported scientific role")
    if evaluation_mode not in {"scientific", "fixture_smoke"}:
        raise ValueError("evaluation_mode must be scientific or fixture_smoke")
    if not isinstance(budget, BudgetSpec):
        raise ValueError("budget must be an explicit matched-benchmark BudgetSpec")

    # Audit and role/source checks deliberately precede any checkpoint file access.
    manifest = load_astronomy_manifest(manifest_path)
    matches = [item for item in manifest.catalogs if item["id"] == catalog_id]
    if len(matches) != 1:
        raise ValueError(f"catalog_id {catalog_id!r} must select exactly one manifest entry")
    catalog = matches[0]
    if catalog["role"] != expected_role:
        raise ValueError(f"catalog role {catalog['role']!r} does not match requested role {expected_role!r}")
    if catalog["source"]["kind"] == "synthetic_fixture":
        raise ValueError("scientific preflight rejects synthetic_fixture catalogs")
    if catalog["source"]["kind"] != "local_hdf5":
        raise ValueError("scientific preflight requires an explicit local_hdf5 source")
    audit = audit_astronomy_manifest(manifest, roles=[expected_role], inspect_hdf5=True)
    audit.require_content_verified()
    evidence = [row for row in audit.catalogs if row.get("catalog_id") == catalog_id]
    if len(evidence) != 1 or evidence[0].get("source_kind") != "local_hdf5":
        raise ValueError("selected catalog has no matching content-verified local source evidence")
    # Strict loader rechecks the manifest and catalog hash at its boundary.
    halos = load_audited_catalog(manifest, audit, catalog_id)
    if not halos:
        raise ValueError("selected catalog produced no halos")
    evaluation_contract = CatalogContract.from_mapping(halos[0].metadata["catalog_contract"]).validate(strict_research=True)
    evaluation_source_files = evaluation_contract.source_files
    if (len(evaluation_source_files) != 1
            or evaluation_source_files[0].get("sha256") != evidence[0]["source"].get("sha256")):
        raise ValueError("strict loader source-file identity disagrees with successful catalog content audit")

    split, split_file = _read_split_manifest(split_manifest_path)
    backbone, backbone_state, backbone_prov, training_contract, backbone_file = _load_artifact(
        checkpoint_path, "projected_backbone", "model_state_dict")
    policy, policy_state, policy_prov, policy_contract, policy_file = _load_artifact(
        policy_path, "projected_policy", "policy_state_dict")
    validate_inference_compatibility(training_contract.to_dict(), evaluation_contract.to_dict())
    validate_inference_compatibility(training_contract.to_dict(), policy_contract.to_dict())

    split_hash = split["manifest_sha256"]
    if backbone_prov["split_manifest_sha256"] != split_hash or policy_prov["split_manifest_sha256"] != split_hash:
        raise ValueError("artifact training split identity does not match the validated split manifest")
    if (backbone_prov["split_manifest_sha256"] != policy_prov["split_manifest_sha256"]
            or backbone_prov["catalog_contract_sha256"] != policy_prov["catalog_contract_sha256"]):
        raise ValueError("policy and backbone provenance disagree on training catalog or split")
    backbone_state_hash = model_state_hash(backbone_state)
    policy_state_hash = model_state_hash(policy_state)
    if backbone_prov.get("model_state_sha256") != backbone_state_hash:
        raise ValueError("backbone model-state provenance hash mismatch")
    if policy_prov.get("model_state_sha256") != policy_state_hash:
        raise ValueError("policy model-state provenance hash mismatch")
    if policy.get("backbone_state_sha256") != backbone_state_hash:
        raise ValueError("policy was not trained against the supplied backbone state")

    if not isinstance(graph_config, ProjectedGraphConfig):
        raise ValueError("graph_config must be an explicit ProjectedGraphConfig")
    if not isinstance(observation_config, ObservationConfig):
        raise ValueError("observation_config must be an explicit ObservationConfig")
    scenario = _config_payload(transform_config, "transform_config")
    observation_payload = asdict(observation_config)
    conflicting = set(scenario).intersection(observation_payload)
    if conflicting:
        raise ValueError("transform_config is scenario metadata only; actual transform fields belong in observation_config: "
                         f"{sorted(conflicting)}")
    transform_sha = hash_payload(observation_payload)
    expected_schema = checkpoint_compatibility_record(graph_config, observation_config.line_of_sight)
    for artifact in (backbone, policy):
        validate_checkpoint_compatibility(expected_schema, artifact.get("projected_compatibility"))
        artifact_config = artifact.get("provenance", {}).get("resolved_config", {})
        if artifact_config.get("projected_compatibility") != artifact.get("projected_compatibility"):
            raise ValueError("projected feature compatibility differs from embedded resolved training config")
        training_observation = artifact.get("training_observation_config")
        training_observation_hash = artifact.get("training_observation_config_sha256")
        if (not isinstance(training_observation, dict) or not training_observation
                or hash_payload(training_observation) != training_observation_hash):
            raise ValueError("training observation configuration payload/hash mismatch")
        if artifact_config.get("training_observation_config") != training_observation:
            raise ValueError("training observation config differs from embedded resolved training config")
        if training_observation_hash != backbone.get("training_observation_config_sha256"):
            raise ValueError("policy and backbone observation-transform training identities disagree")
    training_transform_sha = backbone.get("training_observation_config_sha256")
    if not isinstance(training_transform_sha, str) or len(training_transform_sha) != 64:
        raise ValueError("projected artifacts require a positive training observation-config identity")

    fixture_labeled = evaluation_mode == "fixture_smoke"
    return {
        "metadata_preflight_passed": True,
        "fixture_smoke_ready": fixture_labeled,
        "evaluation_execution_ready": False,
        "scientific_evaluation_allowed": False,
        "evaluation_executed": False,
        "model_load_verified": False,
        "readiness_scope": ("fixture_contract_smoke_only" if fixture_labeled
                            else "offline_input_and_artifact_compatibility_only"),
        "evaluation_mode": evaluation_mode,
        "catalog": {"catalog_id": catalog_id, "role": catalog["role"], "suite": catalog["suite"],
                    "simulation": catalog["simulation"], "initial_condition_id": catalog["initial_condition_id"],
                    "snapshot": catalog["snapshot"], "release": catalog["release"],
                    "source_sha256": evidence[0]["source"]["sha256"],
                    "source_file": evaluation_source_files[0],
                    "manifest_sha256": manifest.sha256, "target_definition": evaluation_contract.target_definition,
                    "target_field": evaluation_contract.target_field, "target_units": evaluation_contract.mass_units},
        "observation": {"feature_schema": expected_schema,
                        "training_transform_config_sha256": training_transform_sha,
                        "evaluation_transform_config": observation_payload,
                        "evaluation_transform_config_sha256": transform_sha,
                        "scenario_metadata": scenario,
                        "scenario_metadata_sha256": hash_payload(scenario)},
        "artifacts": {"backbone_sha256": backbone_file["sha256"],
                      "backbone_model_state_sha256": backbone_state_hash,
                      "policy_sha256": policy_file["sha256"], "policy_model_state_sha256": policy_state_hash,
                      "training_catalog_contract_sha256": backbone_prov["catalog_contract_sha256"],
                      "training_source_revision": backbone_prov["source_revision"],
                      "training_source_code_files": backbone_prov["source_code_files"],
                      "training_catalog_source_files": training_contract.source_files,
                      "training_split_manifest_sha256": split_hash,
                      "split_manifest_file_sha256": split_file["sha256"]},
        "training_split_membership_verified": False,
        "artifact_tensor_loadability_verified": False,
        "budget": budget.config_dict(),
        "achieved_pair_counts_available": False,
        "catalog_halo_count": len(halos),
        "training_source_files_rehashed": False,
        "limitations": ["model architecture construction and state-dict compatibility are not checked; the downstream loader must load both artifacts before execution",
                        "budget feasibility is graph-specific and not evaluated by preflight",
                        "training membership is identified by a content-checked split manifest but not revalidated against training catalog rows",
                        "training source-file hashes are recorded in the artifact catalog contract but are not rehashed by this evaluation preflight"],
    }
