"""Frozen local evaluation. Usage: python -m experiments.paper_generalization.run --help"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path

from .contracts import checked_file, file_record, read_json, validate_pair

COLUMNS = ["graph_id", "method", "status", "reason", "target", "full_prediction", "pruned_prediction",
           "absolute_error", "squared_error", "available_pair_count", "requested_pair_count", "final_pair_count",
           "physical_retention", "constraint_satisfied", "budget_satisfied", "model_state_hash", "research_label", "seed"]


def execute(manifest_path, output, *, fixture_smoke=False):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    expected_status = "fixture_smoke" if fixture_smoke else "ready"
    if manifest.get("schema_version") != 1 or manifest.get("status") != expected_status:
        raise ValueError("manifest is pending_inputs; supply audited real inputs and mark ready")
    if Path(output).exists():
        raise ValueError("output directory must be new")
    base = path.parent
    dataset = manifest["dataset"]
    source = checked_file(dataset["source"], base)
    if source["sha256"] != dataset["source_sha256"]:
        raise ValueError("dataset source identity mismatch")
    records = {"catalog": source, "manifest": file_record(path)}
    for name in ("parameters", "schema_review"):
        records[name] = checked_file(dataset[name], base)
    for name in ("training_receipt", "backbone", "policy"):
        records[name] = checked_file(manifest[name], base)
    training = read_json(records["training_receipt"]["path"])
    claim = validate_pair(training, dataset, fixture_smoke=fixture_smoke)
    # An attestation without the actual hashed training/split/transform evidence is not provenance.
    evidence = training.get("evidence_files", {})
    if set(evidence) != {"training_catalog", "split", "semantic_audit", "training_code"}:
        raise ValueError("training receipt requires catalog, split, semantic_audit and training_code evidence files")
    records["training_evidence"] = {name: checked_file(spec, Path(records["training_receipt"]["path"]).parent)
                                    for name, spec in evidence.items()}
    if records["training_evidence"]["training_catalog"]["sha256"] not in {
            c["source_sha256"] for c in training["training_catalogs"]}:
        raise ValueError("training evidence catalog hash is not bound to the identified training population")
    for name in ("backbone", "policy"):
        if training.get(name + "_sha256") != records[name]["sha256"]:
            raise ValueError(f"{name} receipt does not bind this frozen artifact")
    if training.get("policy_backbone_sha256") != records["backbone"]["sha256"]:
        raise ValueError("policy was not audited against this backbone")

    code_root = Path(__file__).resolve().parents[2]
    def source_records():
        return {str(p.relative_to(code_root)): file_record(p)
                for directory in ("experiments/paper_generalization", "data", "graph", "model", "rls")
                for p in sorted((code_root / directory).rglob("*.py"))}
    code_before = source_records()
    import torch
    from .catalog import load_catalog
    from graph.graph_builder import GraphBuilder
    from model.model import CosmicNetGNN
    from rls.policy import build_policy
    from rls.benchmark_scorers import PairScorer
    from rls.matched_benchmark import BudgetSpec, evaluate_matched_budget
    from data.provenance import model_state_hash, hash_payload

    cfg = training["config"]
    if (cfg["model"]["node_features"] != 4 or cfg["model"]["edge_features"] != 5
            or cfg["graph"]["edge_features"] != training["feature_contract"]["edge_features"]):
        raise ValueError("architecture dimensions/order disagree with audited intrinsic schema")
    backbone_blob = torch.load(records["backbone"]["path"], map_location="cpu", weights_only=True)
    if not isinstance(backbone_blob, dict) or "model_state_dict" not in backbone_blob:
        raise ValueError("backbone must carry model_state_dict and original config")
    saved = backbone_blob.get("config", {})
    if any(saved.get(key) != cfg.get(key) for key in ("model", "graph")):
        raise ValueError("receipt architecture/graph differs from original backbone config")
    model = CosmicNetGNN(cfg)
    model.load_state_dict(backbone_blob["model_state_dict"], strict=True)
    policy = build_policy(cfg, node_emb_dim=model.output_dim)
    policy_state = torch.load(records["policy"]["path"], map_location="cpu", weights_only=True)
    policy.load_state_dict(policy_state, strict=True)
    for module in (model, policy):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    before = [model_state_hash(m.state_dict()) for m in (model, policy)]
    halos, rejected = load_catalog(source["path"], catalog_id=dataset["id"],
                                   expected_h=dataset["epoch"]["h"], expected_a=dataset["epoch"]["scale_factor"],
                                   **training["population_contract"])
    graphs = GraphBuilder(cfg).build_graphs(halos)
    for g in graphs:
        g.metadata["initial_condition_id"] = dataset["initial_condition_id"]
        g.lineage_group = dataset["initial_condition_id"]
        if g.x.shape[1] != 4 or g.edge_attr.shape[1] != 5 or not torch.isfinite(g.x).all() or not torch.isfinite(g.edge_attr).all():
            raise ValueError("rebuilt graph feature schema or finiteness mismatch")
    budget = BudgetSpec(**manifest["budget"])
    scorers = [PairScorer("random"), PairScorer("distance"), PairScorer("degree"),
               PairScorer("provided_rl", policy=policy, edge_feature_names=cfg["graph"]["edge_features"])]
    # Bind the shared evaluator's row provenance to the same checked files.
    resolved = {"config": cfg, "budget": budget.config_dict(), "dataset": dataset}
    contract = training["feature_contract"]
    provenance = {"schema_version": 1, "research_label": "fixture_smoke" if fixture_smoke else "audited_intrinsic_total_radius",
                  "source_files": [source], "catalog_contract": contract, "catalog_contract_sha256": hash_payload(contract),
                  "split_manifest_sha256": records["training_evidence"]["split"]["sha256"],
                  "resolved_config": resolved, "config_sha256": hash_payload(resolved),
                  "source_revision": "runner_sha256:" + file_record(__file__)["sha256"],
                  "augmentation_manifest": None, "augmentation_manifest_sha256": None}
    summaries, rows = evaluate_matched_budget(model, graphs, scorers, budget, seed=manifest["seed"],
                                               backbone="frozen_intrinsic_audited", provenance=provenance)
    if before != [model_state_hash(m.state_dict()) for m in (model, policy)]:
        raise ValueError("model or policy state changed during frozen evaluation")
    for name in ("catalog", "manifest", "backbone", "policy", "training_receipt", "parameters", "schema_review"):
        if file_record(records[name]["path"])["sha256"] != records[name]["sha256"]:
            raise ValueError(f"{name} content changed during evaluation")
    for record in records["training_evidence"].values():
        if file_record(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("training evidence content changed during evaluation")
    if code_before != source_records():
        raise ValueError("source code changed during evaluation")
    methods = [s.name for s in scorers]
    successes = {m: {r["graph_id"] for r in rows if r["method"] == m and r["status"] == "success"} for m in methods}
    common = sorted(set.intersection(*successes.values()))
    payload = {"evaluation_executed": True, "scientific_result": bool(common) and not fixture_smoke,
               "claim": claim["claim"] if common else "none_no_common_successes", "claim_scope": claim,
               "domain_generalization_established": False, "eligible_halos": len(graphs),
               "common_success_halos": common, "rejected_halos": rejected, "records": records,
               "feature_contract": training["feature_contract"], "state_hashes": before,
               "budget": budget.config_dict(), "seed": manifest["seed"],
               "runtime": {"python": platform.python_version(), "torch": torch.__version__},
               "source_code": code_before,
               "summary_rows": summaries, "per_halo_rows": rows}
    low, high = training["target_range_log10_msun"]
    payload["mass_support"] = {"training_target_log10_range": [low, high],
                               "evaluation_target_log10_range": [min(h.log_halo_mass for h in halos), max(h.log_halo_mass for h in halos)],
                               "outside_training_mass_range": [h.cluster_id for h in halos if not low <= h.log_halo_mass <= high],
                               "note": "all eligible halos retained; mass-range overlap does not establish feature-distribution overlap"}
    payload["population_contract"] = training["population_contract"]
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(serialized + "\n")
    with (out / "per_halo_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-smoke", action="store_true", help="synthetic software tests only; scientific_result is always false")
    args = parser.parse_args()
    try:
        result = execute(args.manifest, args.output, fixture_smoke=args.fixture_smoke)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ["evaluation_executed", "scientific_result", "eligible_halos", "claim"]}))
    return 0 if result["scientific_result"] or (args.fixture_smoke and result["common_success_halos"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
