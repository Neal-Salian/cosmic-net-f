"""Regression contracts for the controlled notebook campaign (tiny CPU only)."""
import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]


def graph():
    return dict(cluster_id="halo", x=torch.ones(4, 4),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2, 2, 3, 0, 1, 2, 3],
                                 [1, 0, 2, 0, 2, 1, 3, 2, 0, 1, 2, 3]]),
        edge_attr=torch.ones(12, 5), emb=torch.ones(4, 4), ctx=torch.ones(4),
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]),
        stellar_mass=torch.tensor([1., 2., 3., 4.]), y=torch.tensor([12.]))


def test_physics_statistics_use_unique_pairs_and_reject_asymmetric_masks():
    from rls.notebook_workflow import physics_diagnostics
    g = graph()
    probs = torch.arange(12).float()
    mask = torch.tensor([True, True, False, False, True, True, False, False, True, True, True, True])
    a = physics_diagnostics([g], [probs], [mask])[0]
    assert a["physical_edges"] == 4
    doubled = dict(g, edge_index=g["edge_index"].repeat(1, 2))
    b = physics_diagnostics([doubled], [probs.repeat(2)], [mask.repeat(2)])[0]
    assert a["binding_mannwhitney_p"] == b["binding_mannwhitney_p"]
    assert a["spearman_rho"] == b["spearman_rho"]
    mask[0] = False
    with pytest.raises(ValueError, match="symmetr"):
        physics_diagnostics([g], [probs], [mask])


def test_exact_notebook_masks_repeat_and_have_no_budget_inflation():
    from rls.notebook_benchmark import exact_policy_mask
    from rls.policy import EdgePolicyNet
    from rls.constraints import PhysicalPairLayout
    p = EdgePolicyNet(5, 4, hidden_dim=8)
    g = graph()
    cfg = {"seed": 3, "evaluation_constraint": "no_isolates",
           "edge_feature_names": ["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"]}
    a, _, diag = exact_policy_mask(p, g, cfg, .5)
    b, _, _ = exact_policy_mask(p, g, cfg, .5)
    layout = PhysicalPairLayout.from_edge_index(g["edge_index"], 4)
    assert torch.equal(a, b)
    assert torch.equal(layout.expand(layout.collapse(a)), a)
    assert diag["final_pair_count"] == diag["requested_pair_count"] == 2
    assert diag["physical_isolates"] == 0
    with pytest.raises(ValueError, match="infeasible"):
        exact_policy_mask(p, g, cfg, .01)


def test_notebooks_preserve_archival_outputs_and_pin_readiness_branch():
    for path in (ROOT / "notebooks").glob("notebook_[ABCD]*.ipynb"):
        new = json.loads(path.read_text())
        old = json.loads(subprocess.check_output(["git", "show", "f66bb57:" + str(path.relative_to(ROOT))], cwd=ROOT))
        new_outputs = [(c.get("execution_count"), c.get("outputs", [])) for c in new["cells"] if c["cell_type"] == "code"]
        old_outputs = [(c.get("execution_count"), c.get("outputs", [])) for c in old["cells"] if c["cell_type"] == "code"]
        assert new_outputs == old_outputs
        settings = next("".join(c["source"]) for c in new["cells"] if c.get("id") == "settings")
        assert 'REPO_BRANCH = "fix/research-pipeline-readiness"' in settings
        assert "ACKNOWLEDGE_LEGACY_INPUTS = False" in settings
        assert "historical" in "".join(new["cells"][0]["source"]).lower()


def test_context_handoff_rejects_source_and_evaluation_drift(tmp_path):
    from rls.notebook_workflow import source_artifacts, file_info
    context = dict(run_mode="smoke", seed=1, split_cluster_ids={}, backbone={},
        dataset={"tng_csv": {"sha256": "csv"}, "checkpoint": {"sha256": "ckpt"}},
        config_used={"graph": {}, "rls": {"target_sparsity_end": .4}},
        source_identity="current", evaluation_protocol={"constraint": "no_isolates"})
    (tmp_path / "policy.pt").write_bytes(b"test")
    manifest = dict(copy.deepcopy(context), status="complete",
                    artifacts={"policy.pt": file_info(tmp_path / "policy.pt")})
    manifest["source_identity"] = "old"
    (tmp_path / "provenance.json").write_text(json.dumps({"B": manifest}))
    with pytest.raises(ValueError, match="source_identity"):
        source_artifacts(tmp_path, tmp_path, "B", "policy.pt", context)


def test_existing_output_directory_is_never_reused(tmp_path):
    from rls.notebook_workflow import fresh_output_directory
    occupied = tmp_path / "run"
    occupied.mkdir()
    (occupied / "sentinel").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        fresh_output_directory(occupied)
    assert (occupied / "sentinel").read_bytes() == b"keep"


def test_negative_but_comparable_policy_is_not_hidden_from_evaluation(tmp_path):
    from rls.notebook_workflow import load_saved_policy, file_info
    from rls.policy import build_policy
    cfg = {"rls": {"sparsity_mode": "pair_pl", "policy_hidden": 8}}
    model = type("Model", (), {"output_dim": 4})()
    policy = build_policy(cfg, node_emb_dim=4)
    torch.save(policy.state_dict(), tmp_path / "policy.pt")
    manifest = {"config_used": cfg, "run_mode": "full",
        "policy_validation": {"verdict": "FAIL", "budget_comparable": True},
        "artifacts": {"policy.pt": file_info(tmp_path / "policy.pt")}}
    result = load_saved_policy(tmp_path, manifest, cfg, model, "cpu")
    assert isinstance(result, torch.nn.Module)


def test_completion_rejects_changed_inputs_before_publishing(tmp_path):
    from rls.notebook_workflow import complete_stage, file_info
    data = tmp_path / "input.csv"
    data.write_text("initial")
    context = {"dataset": {"tng_csv": file_info(data)}, "run_mode": "smoke"}
    data.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        complete_stage(tmp_path / "outputs", "A", context, [])


def test_non_smoke_checkout_requires_an_immutable_source_pin():
    import os
    import sys
    path = next((ROOT / "notebooks").glob("notebook_A*"))
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(path.read_text())["cells"]}
    namespace = {"Path": Path, "subprocess": subprocess, "os": os, "sys": sys,
        "REPO_DIR": ROOT, "REPO_URL": "unused", "REPO_BRANCH": "fix/research-pipeline-readiness",
        "SYNC_REPO": False, "SMOKE_TEST": False, "EXPECTED_REPO_COMMIT": None}
    with pytest.raises(ValueError, match="EXPECTED_REPO_COMMIT"):
        exec(compile(cells["checkout"], "checkout", "exec"), namespace)


def test_matched_notebook_boundary_moves_graphs_to_predictor_device(monkeypatch):
    import rls.notebook_benchmark as benchmark
    model = torch.nn.Linear(4, 1).to("meta")
    original = Data(**graph())
    def checked(model, graphs, *args, **kwargs):
        assert graphs[0].x.device == next(model.parameters()).device
        assert graphs[0].edge_index.device == next(model.parameters()).device
        return [], []
    monkeypatch.setattr(benchmark, "evaluate_matched_budget", checked)
    context = {"config_used": {"graph": {"edge_features": ["distance"]}, "rls": {}},
               "split_cluster_ids": {}, "dataset": {}, "seed": 1,
               "research_label": "fixture_smoke", "source_identity": "test", "run_mode": "smoke"}
    benchmark.matched_notebook_rows(model, [original], context, [.5])
    assert original.x.device.type == "cpu"  # Do not mutate caller graph storage.


def test_notebook_headline_retention_and_timing_use_common_physical_population():
    from rls.notebook_benchmark import matched_notebook_rows
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def forward(self, batch):
            return batch.edge_attr.sum() * self.weight
    cycle = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 2, 3],
                          [1, 0, 2, 1, 3, 2, 0, 3, 0, 1, 2, 3]])
    triangle = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    star = torch.tensor([[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]])
    graphs = []
    for cid, edges, nodes, target in (
            ("common-a", cycle, 4, 0.), ("common-b", cycle, 4, 2.),
            ("missing-position", triangle, 3, 100.),
            ("infeasible-budget", star, 4, 0.)):
        g = Data(x=torch.ones(nodes, 1), edge_index=edges,
            edge_attr=torch.ones(edges.shape[1], 1), stellar_mass=torch.ones(nodes),
            y=torch.tensor([target]), cluster_id=cid)
        if cid != "missing-position":
            g.pos = torch.arange(nodes, dtype=torch.float32).reshape(-1, 1)
        graphs.append(g)
    context = {"config_used": {"graph": {"edge_features": ["distance"]}, "rls": {}},
               "split_cluster_ids": {}, "dataset": {}, "seed": 1,
               "research_label": "fixture_smoke", "source_identity": "test", "run_mode": "smoke"}
    summaries, raw = matched_notebook_rows(Predictor(), graphs, context, [.5])
    for summary in summaries:
        assert summary["common_success_ids"] == ["common-a", "common-b"]
        common = [r for r in raw if r["method"] == summary["method"]
                  and r["graph_id"] in summary["common_success_ids"]]
        assert summary["comparison_population"] == "common_success"
        expected_keep = 1. if summary["method"] == "full" else .5
        assert summary["mean_keep_frac"] == summary["keep_frac"] == expected_keep
        assert summary["retention_units"] == "distinct_nonself_physical_pairs"
        assert summary["mae"] == summary["common_mae"]
        assert summary["std_residual"] == summary["common_std_residual"]
        assert summary["mean_inference_region_time_s"] == pytest.approx(
            sum(r["total_region_time_s"] for r in common) / len(common))
        assert summary["mean_inference_time_s"] == summary["mean_inference_region_time_s"]
        for field in ("scoring_time_s", "decoder_time_s", "prediction_time_s", "full_time_s"):
            assert summary["mean_" + field] == pytest.approx(
                sum(r[field] for r in common) / len(common), rel=1e-12)
    degree = next(r for r in summaries if r["method"] == "degree")
    assert degree["mean_stored_keep_frac"] == pytest.approx(2 / 3)
    assert degree["method_population_mean_keep_frac"] == pytest.approx(5 / 9)
    assert degree["method_population_mae"] != degree["mae"]
