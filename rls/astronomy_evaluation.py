"""Fail-closed runner for matched-budget projected astronomy evaluation.

Metadata readiness is delegated to ``preflight_astronomy``. This module then
loads both real architectures, checks their complete state dicts, rebuilds
projected graphs from freshly rehashed source rows, and delegates every mask
and prediction comparison to the shared matched-budget evaluator.
"""
from copy import deepcopy
import json
import math

import torch

from data.astronomy_audit import audit_astronomy_manifest
from data.astronomy_loader import load_audited_catalog
from data.astronomy_manifest import load_astronomy_manifest
from data.astronomy_observations import observe_astronomy_halo
from data.projected_graph import build_projected_graph, checkpoint_compatibility_record
from data.provenance import hash_payload, model_state_hash, sha256_file
from data.observations import ObservationConfig
from model.model import CosmicNetGNN
from rls.astronomy_preflight import preflight_astronomy, _read_split_manifest
from rls.benchmark_scorers import PairScorer
from rls.matched_benchmark import BudgetSpec, evaluate_matched_budget
from rls.policy import EdgePolicyNet


def _safe_artifact(path, expected_hash, artifact_type, state_key):
    record = sha256_file(path)
    if record["sha256"] != expected_hash:
        raise ValueError(f"{artifact_type} artifact changed after preflight")
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{artifact_type} artifact could not be safely loaded") from exc
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != artifact_type:
        raise ValueError(f"{artifact_type} artifact type changed after preflight")
    state = artifact.get(state_key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{artifact_type} artifact has no state dict")
    if model_state_hash(state) != artifact["provenance"].get("model_state_sha256"):
        raise ValueError(f"{artifact_type} state hash does not match its provenance")
    return artifact, state


def _load_module(module, state, name):
    try:
        incompatible = module.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"{name} state is incompatible with its declared architecture") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"{name} state is incomplete for its declared architecture")
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _membership_check(halos, split, role, training_contract):
    sections = split["splits"]
    if role == "development":
        allowed = set(sections["val"]["cluster_ids"]) | set(sections["test"]["cluster_ids"])
        allowed_groups = set(sections["val"]["group_ids"]) | set(sections["test"]["group_ids"])
        training = set(sections["train"]["cluster_ids"])
        training_groups = set(sections["train"]["group_ids"])
        ids = {f"{halo.catalog_id}:group:{halo.group_index}" for halo in halos}
        groups = {str(halo.metadata.get("initial_condition_id", "")) for halo in halos}
        if not groups or "" in groups or len(groups) != 1:
            raise ValueError("development evaluation requires one explicit catalog IC group")
        if groups & training_groups or not groups.issubset(allowed_groups):
            raise ValueError("development catalog IC group is not exclusively in validation/test split")
        if ids & training:
            raise ValueError(f"development evaluation overlaps training halo IDs: {sorted(ids & training)}")
        if ids - allowed:
            raise ValueError("development evaluation halo IDs cannot be reconciled to explicit validation/test split membership")
        return {"membership_basis": "explicit_validation_or_test_cluster_ids",
                "evaluation_halo_ids": sorted(ids), "evaluation_initial_condition_ids": sorted(groups),
                "split_role": "development"}
    training_ics = {str(value) for value in sections["train"]["group_ids"]}
    training_ic = training_contract.get("volume_or_ic_group")
    if training_ic:
        training_ics.add(str(training_ic))
    evaluation_ics = {str(halo.metadata.get("initial_condition_id", "")) for halo in halos}
    if not training_ics or not evaluation_ics or "" in evaluation_ics:
        raise ValueError("external evaluation requires identifiable training and evaluation IC groups")
    overlap = training_ics & evaluation_ics
    if overlap:
        raise ValueError(f"external evaluation initial-condition groups overlap training: {sorted(overlap)}")
    return {"membership_basis": "initial_condition_group_disjointness",
            "training_initial_condition_ids": sorted(training_ics),
            "evaluation_initial_condition_ids": sorted(evaluation_ics), "split_role": role}


def _matched_success_summary(rows, methods, budget):
    successes = {method: {row["graph_id"]: row for row in rows
                          if row.get("method") == method and row.get("status") == "success"}
                 for method in methods}
    counts = {method: len(successes[method]) for method in methods}
    common = set.intersection(*(set(successes[method]) for method in methods))
    for graph_id in common:
        rows_for_graph = [successes[method][graph_id] for method in methods]
        final_counts = {row.get("final_pair_count") for row in rows_for_graph}
        if len(final_counts) != 1:
            raise ValueError(f"matched methods do not share one final pair count for {graph_id}")
        available = rows_for_graph[0].get("available_pair_count")
        requested = (budget.pair_count if budget.pair_count is not None
                     else math.ceil(budget.keep_fraction * available))
        final_count = next(iter(final_counts))
        if final_count != requested:
            raise ValueError(f"matched final physical-pair count {final_count} != requested {requested} for {graph_id}")
    return counts, sorted(common)


def run_astronomy_evaluation(*, manifest_path, catalog_id, expected_role,
                             split_manifest_path, checkpoint_path, policy_path,
                             graph_config, observation_config, transform_config,
                             budget, evaluation_mode="scientific", seed=0):
    """Run matched random/distance/degree/provided-policy comparisons.

    ``fixture_smoke`` outputs are permanently labeled smoke and never expose a
    scientific-result flag. Every preflight failure occurs before graph/model
    construction or state loading.
    """
    if not isinstance(budget, BudgetSpec):
        raise ValueError("budget must be a BudgetSpec")
    readiness = preflight_astronomy(
        manifest_path=manifest_path, catalog_id=catalog_id, expected_role=expected_role,
        split_manifest_path=split_manifest_path, checkpoint_path=checkpoint_path,
        policy_path=policy_path, graph_config=graph_config,
        observation_config=observation_config, transform_config=transform_config,
        budget=budget, evaluation_mode=evaluation_mode)

    # Re-read only the hash-bound structured artifacts; architecture construction
    # remains after catalog and split leakage checks below.
    backbone_artifact, backbone_state = _safe_artifact(
        checkpoint_path, readiness["artifacts"]["backbone_sha256"],
        "projected_backbone", "model_state_dict")
    policy_artifact, policy_state = _safe_artifact(
        policy_path, readiness["artifacts"]["policy_sha256"],
        "projected_policy", "policy_state_dict")

    # Re-audit content and split membership before constructing any model or graph.
    manifest = load_astronomy_manifest(manifest_path)
    if manifest.sha256 != readiness["catalog"]["manifest_sha256"]:
        raise ValueError("astronomy manifest changed after preflight")
    audit = audit_astronomy_manifest(manifest, roles=[expected_role], inspect_hdf5=True)
    audit.require_content_verified()
    halos = load_audited_catalog(manifest, audit, catalog_id)
    selected = [row for row in audit.catalogs if row.get("catalog_id") == catalog_id]
    if len(selected) != 1 or selected[0]["source"].get("sha256") != readiness["catalog"]["source_sha256"]:
        raise ValueError("evaluation catalog source changed after preflight")
    split, split_file_record = _read_split_manifest(split_manifest_path)
    if split_file_record["sha256"] != readiness["artifacts"]["split_manifest_file_sha256"]:
        raise ValueError("training split file changed after preflight")
    membership = _membership_check(halos, split, expected_role,
                                   backbone_artifact["provenance"]["catalog_contract"])

    # Only after the second content audit and leakage check do we load/construct models.
    resolved = backbone_artifact["provenance"]["resolved_config"]
    model_cfg = resolved.get("model")
    if not isinstance(model_cfg, dict) or not model_cfg:
        raise ValueError("backbone resolved_config must declare its model architecture")
    policy_cfg = policy_artifact.get("policy_config")
    if not isinstance(policy_cfg, dict):
        raise ValueError("projected policy artifact must declare policy_config architecture")
    feature_names = list(graph_config.edge_features)
    resolved_policy_cfg = policy_artifact["provenance"].get("resolved_config", {}).get("policy_config")
    resolved_compatibility = policy_artifact["provenance"].get("resolved_config", {}).get("projected_compatibility")
    if policy_cfg != resolved_policy_cfg:
        raise ValueError("policy_config differs from its hashed resolved training configuration")
    if (policy_artifact.get("policy_edge_feature_names") != feature_names
            or not isinstance(resolved_compatibility, dict)
            or resolved_compatibility.get("edge_features") != feature_names):
        raise ValueError("policy edge schema differs from hashed projected training configuration")
    if (policy_cfg.get("edge_dim") != len(feature_names)
            or policy_cfg.get("node_emb_dim") != model_cfg.get("output_dim", 64)):
        raise ValueError("policy architecture dimensions disagree with projected graph/backbone schema")
    model = _load_module(CosmicNetGNN({"model": model_cfg,
                                       "graph": {"edge_features": list(graph_config.edge_features)}}),
                         backbone_state, "backbone")
    policy = _load_module(EdgePolicyNet(**policy_cfg), policy_state, "policy")

    graphs, reports, rejected = [], [], []
    for halo in halos:
        observed, report = observe_astronomy_halo(halo, observation_config)
        reports.append(report)
        if observed is None:
            rejected.append({"halo_id": f"{halo.catalog_id}:group:{halo.group_index}",
                             "reason": report.get("rejection_reason", "richness_selection")})
            continue
        graph = build_projected_graph(observed, graph_config, report)
        if graph.compatibility_record != checkpoint_compatibility_record(
                graph_config, observation_config.line_of_sight):
            raise ValueError("rebuilt graph schema differs from preflighted projected compatibility")
        graphs.append(graph)
    if not graphs:
        raise ValueError("observation selection produced no graph eligible for evaluation")

    training_prov = backbone_artifact["provenance"]
    evaluation_contract = deepcopy(halos[0].metadata["catalog_contract"])
    label = "fixture_smoke" if evaluation_mode == "fixture_smoke" else "corrected_research"
    config_record = {"evaluation_mode": evaluation_mode, "observation_config": readiness["observation"]["evaluation_transform_config"],
                     "scenario_metadata": readiness["observation"]["scenario_metadata"],
                     "graph_config": graph_config.to_dict(), "budget": budget.config_dict(),
                     "training_split_sha256": split["manifest_sha256"]}
    runner_identity = sha256_file(__file__)
    provenance = {
        "schema_version": 1, "research_label": label,
        "source_files": [deepcopy(readiness["catalog"]["source_file"])],
        "catalog_contract": evaluation_contract,
        "catalog_contract_sha256": hash_payload(evaluation_contract),
        "split_manifest_sha256": split["manifest_sha256"],
        "resolved_config": config_record, "config_sha256": hash_payload(config_record),
        "source_revision": f"astronomy_evaluation_sha256:{runner_identity['sha256']}",
        "source_code_files": [runner_identity],
        "training_source_revision": training_prov.get("source_revision"),
        "training_source_code_files": deepcopy(training_prov.get("source_code_files", [])),
        "augmentation_manifest": None, "augmentation_manifest_sha256": None,
        "training_artifact_hashes": {"backbone_file_sha256": readiness["artifacts"]["backbone_sha256"],
                                      "backbone_state_sha256": readiness["artifacts"]["backbone_model_state_sha256"],
                                      "policy_file_sha256": readiness["artifacts"]["policy_sha256"],
                                      "policy_state_sha256": readiness["artifacts"]["policy_model_state_sha256"]},
        "training_split_manifest_sha256": split["manifest_sha256"],
        "training_split_file_sha256": split_file_record["sha256"],
        "evaluation_source_sha256": readiness["catalog"]["source_sha256"],
        "membership_audit": membership,
        "observation_reports": reports,
        "rejected_halos": rejected,
        "feature_schema": readiness["observation"]["feature_schema"],
        "exact_budget": budget.config_dict(),
    }
    scorers = [PairScorer("random"), PairScorer("distance"), PairScorer("degree"),
               PairScorer("provided_rl", policy=policy, edge_feature_names=feature_names)]
    summaries, rows = evaluate_matched_budget(
        model, graphs, scorers, budget, seed=seed, provenance=provenance,
        backbone="projected_frozen")
    expected_budget_hash = hash_payload(budget.config_dict())
    if any(row.get("budget_config_hash") != expected_budget_hash for row in rows):
        raise ValueError("matched evaluator returned rows not bound to the declared exact budget")
    methods = ("random", "distance", "degree", "provided_rl")
    success_counts, common_graph_ids = _matched_success_summary(rows, methods, budget)
    scientific_result = evaluation_mode == "scientific" and bool(common_graph_ids)
    result = {
        "evaluation_mode": evaluation_mode,
        "research_label": label,
        "scientific_result": scientific_result,
        "method_success_counts": success_counts,
        "common_success_graph_ids": common_graph_ids,
        "model_load_verified": True,
        "evaluation_executed": True,
        "preflight": readiness,
        "provenance": provenance,
        "budget_sha256": expected_budget_hash,
        "summary_rows": summaries,
        "per_example_rows": rows,
    }
    json.dumps(result, allow_nan=False)
    return result
