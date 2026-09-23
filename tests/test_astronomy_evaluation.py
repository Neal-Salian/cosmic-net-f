"""Tiny fixture tests for the strict astronomy evaluation runner."""
import json
import importlib.util
from pathlib import Path

import pytest
import torch

from model.model import CosmicNetGNN
from rls.policy import EdgePolicyNet
from data.provenance import hash_payload, model_state_hash
from rls.astronomy_evaluation import run_astronomy_evaluation, _matched_success_summary
_PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "astronomy_preflight_test_helpers", Path(__file__).with_name("test_astronomy_preflight.py"))
_PREFLIGHT_HELPERS = importlib.util.module_from_spec(_PREFLIGHT_SPEC)
_PREFLIGHT_SPEC.loader.exec_module(_PREFLIGHT_HELPERS)
_setup = _PREFLIGHT_HELPERS._setup


def _runnable_fixture(tmp_path):
    paths = _setup(tmp_path)
    manifest, split_path, graph, observation, backbone_path, policy_path = paths
    split = json.loads(split_path.read_text())
    split["splits"]["val"]["group_ids"] = ["fixture-ic"]
    split["splits"]["val"]["cluster_ids"] = ["fixture:group:0"]
    split["splits"]["val"]["halo_count"] = 1
    split["manifest_sha256"] = hash_payload({k: v for k, v in split.items() if k != "manifest_sha256"})
    split_path.write_text(json.dumps(split))

    model_config = {"node_features": 4, "hidden_dim": 8, "output_dim": 8,
                    "num_layers": 1, "dropout": 0., "residual": True,
                    "activation": "leaky_relu", "pooling": "mean"}
    model = CosmicNetGNN({"model": model_config,
                          "graph": {"edge_features": list(graph.edge_features)}})
    policy_config = {"edge_dim": 3, "node_emb_dim": 8, "hidden_dim": 4, "normalize": False}
    policy = EdgePolicyNet(**policy_config)
    artifacts = [torch.load(backbone_path, map_location="cpu", weights_only=True),
                 torch.load(policy_path, map_location="cpu", weights_only=True)]
    artifacts[0]["model_state_dict"] = model.state_dict()
    artifacts[1]["policy_state_dict"] = policy.state_dict()
    artifacts[1]["policy_config"] = policy_config
    artifacts[1]["policy_edge_feature_names"] = list(graph.edge_features)
    for artifact, state_key in zip(artifacts, ("model_state_dict", "policy_state_dict")):
        prov = artifact["provenance"]
        prov["model_state_sha256"] = model_state_hash(artifact[state_key])
        prov["split_manifest_sha256"] = split["manifest_sha256"]
        prov["resolved_config"]["model"] = model_config
        if state_key == "policy_state_dict":
            prov["resolved_config"]["policy_config"] = policy_config
        prov["config_sha256"] = hash_payload(prov["resolved_config"])
    artifacts[1]["backbone_state_sha256"] = model_state_hash(artifacts[0]["model_state_dict"])
    torch.save(artifacts[0], backbone_path)
    torch.save(artifacts[1], policy_path)
    return {"manifest_path": manifest, "catalog_id": "fixture", "expected_role": "development",
            "split_manifest_path": split_path, "checkpoint_path": backbone_path,
            "policy_path": policy_path, "graph_config": graph,
            "observation_config": observation, "transform_config": {"scenario": "test"},
            "budget": __import__("rls.matched_benchmark", fromlist=["BudgetSpec"]).BudgetSpec(
                pair_count=1, constraint="no_isolates"), "evaluation_mode": "fixture_smoke"}


def test_fixture_runner_loads_real_weights_and_returns_matched_labeled_rows(tmp_path):
    result = run_astronomy_evaluation(**_runnable_fixture(tmp_path))
    assert result["evaluation_mode"] == "fixture_smoke"
    assert result["scientific_result"] is False
    assert result["model_load_verified"] is True
    assert {row["method"] for row in result["per_example_rows"]} >= {
        "random", "distance", "degree", "provided_rl"}
    assert all(row["budget_config_hash"] == result["budget_sha256"]
               for row in result["per_example_rows"] if row["comparison_axis"] == "scorer")
    assert result["provenance"]["evaluation_source_sha256"]
    assert result["common_success_graph_ids"] == ["fixture:group:0"]
    assert all(row["final_pair_count"] == 1 for row in result["per_example_rows"]
               if row["status"] == "success" and row["comparison_axis"] == "scorer")


def test_preflight_failure_happens_before_model_or_graph_construction(monkeypatch, tmp_path):
    import rls.astronomy_evaluation as runner
    events = []
    monkeypatch.setattr(runner, "preflight_astronomy", lambda **kw: (_ for _ in ()).throw(ValueError("preflight stop")))
    monkeypatch.setattr(runner, "CosmicNetGNN", lambda *a, **k: events.append("model"))
    monkeypatch.setattr(runner, "build_projected_graph", lambda *a, **k: events.append("graph"))
    with pytest.raises(ValueError, match="preflight stop"):
        runner.run_astronomy_evaluation(**_runnable_fixture(tmp_path))
    assert events == []


def test_runner_rejects_architecture_state_mismatch(tmp_path):
    args = _runnable_fixture(tmp_path)
    artifact = torch.load(args["checkpoint_path"], map_location="cpu", weights_only=True)
    artifact["model_state_dict"]["pred_head.0.weight"] = torch.zeros(99, 8)
    artifact["provenance"]["model_state_sha256"] = model_state_hash(artifact["model_state_dict"])
    torch.save(artifact, args["checkpoint_path"])
    with pytest.raises(ValueError, match="state|architecture|load"):
        run_astronomy_evaluation(**args)


def test_fixture_mode_cannot_be_reported_as_scientific(tmp_path):
    result = run_astronomy_evaluation(**_runnable_fixture(tmp_path))
    assert result["research_label"] == "fixture_smoke"
    assert result["scientific_result"] is False


def test_checkpoint_mutation_after_preflight_is_rejected(monkeypatch, tmp_path):
    import rls.astronomy_evaluation as runner
    real_preflight = runner.preflight_astronomy
    args = _runnable_fixture(tmp_path)

    def preflight_then_mutate(**kwargs):
        record = real_preflight(**kwargs)
        with open(kwargs["checkpoint_path"], "ab") as stream:
            stream.write(b"changed")
        return record

    monkeypatch.setattr(runner, "preflight_astronomy", preflight_then_mutate)
    with pytest.raises(ValueError, match="changed after preflight"):
        runner.run_astronomy_evaluation(**args)


def test_development_catalog_ic_group_must_be_validation_or_test_only(tmp_path):
    args = _runnable_fixture(tmp_path)
    split = json.loads(args["split_manifest_path"].read_text())
    split["splits"]["train"]["group_ids"] = ["fixture-ic"]
    split["splits"]["val"]["group_ids"] = ["other-ic"]
    split["manifest_sha256"] = hash_payload({k: v for k, v in split.items() if k != "manifest_sha256"})
    args["split_manifest_path"].write_text(json.dumps(split))
    # Rebind the artifact training split identity so preflight gets to the runner's lineage check.
    for path, key in ((args["checkpoint_path"], "model_state_dict"),
                      (args["policy_path"], "policy_state_dict")):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        artifact["provenance"]["split_manifest_sha256"] = split["manifest_sha256"]
        torch.save(artifact, path)
    with pytest.raises(ValueError, match="IC group|training"):
        run_astronomy_evaluation(**args)


def test_manifest_metadata_mutation_after_preflight_is_rejected(monkeypatch, tmp_path):
    import yaml
    import rls.astronomy_evaluation as runner
    real_preflight = runner.preflight_astronomy
    args = _runnable_fixture(tmp_path)

    def preflight_then_mutate(**kwargs):
        record = real_preflight(**kwargs)
        payload = yaml.safe_load(Path(kwargs["manifest_path"]).read_text())
        payload["catalogs"][0]["release"] = "changed after readiness"
        Path(kwargs["manifest_path"]).write_text(yaml.safe_dump(payload))
        return record

    monkeypatch.setattr(runner, "preflight_astronomy", preflight_then_mutate)
    with pytest.raises(ValueError, match="manifest changed after preflight"):
        runner.run_astronomy_evaluation(**args)


@pytest.mark.parametrize("field,value", [
    ("policy_config", {"edge_dim": 3, "node_emb_dim": 8, "hidden_dim": 99, "normalize": False}),
    ("policy_edge_feature_names", ["los_delta_v", "projected_distance", "mass_ratio"]),
])
def test_policy_architecture_and_schema_must_match_hashed_training_config(tmp_path, field, value):
    args = _runnable_fixture(tmp_path)
    artifact = torch.load(args["policy_path"], map_location="cpu", weights_only=True)
    artifact[field] = value
    torch.save(artifact, args["policy_path"])
    with pytest.raises(ValueError, match="policy.*(config|schema)|provenance|resolved"):
        run_astronomy_evaluation(**args)


def test_disjoint_method_successes_do_not_form_scientific_common_set():
    methods = ("random", "distance", "degree", "provided_rl")
    rows = [{"graph_id": f"g{i}", "method": method, "status": "success",
             "final_pair_count": 1, "available_pair_count": 3}
            for i, method in enumerate(methods)]
    counts, common = _matched_success_summary(
        rows, methods, __import__("rls.matched_benchmark", fromlist=["BudgetSpec"]).BudgetSpec(
            pair_count=1, constraint="no_isolates"))
    assert all(value == 1 for value in counts.values())
    assert common == []


def test_matched_success_summary_rejects_wrong_final_budget():
    methods = ("random", "distance", "degree", "provided_rl")
    rows = [{"graph_id": "g", "method": method, "status": "success",
             "final_pair_count": 2, "available_pair_count": 3}
            for method in methods]
    with pytest.raises(ValueError, match="requested"):
        _matched_success_summary(rows, methods,
            __import__("rls.matched_benchmark", fromlist=["BudgetSpec"]).BudgetSpec(
                pair_count=1, constraint="no_isolates"))
