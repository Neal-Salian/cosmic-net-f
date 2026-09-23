"""Task 3c TDD RED: truthful TTA requested/actual budget accounting.

Uses tiny real policy graphs; no meaningless mocks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def _tiny_pair_graph(seed=0):
    torch.manual_seed(seed)
    # 4 nodes, reciprocal pairs + self-loops
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 2, 3],
                               [1, 0, 2, 1, 3, 2, 0, 3, 0, 1, 2, 3]], dtype=torch.long)
    e = edge_index.shape[1]
    n = 4
    return {"x": torch.randn(n, 4), "edge_index": edge_index,
            "edge_attr": torch.randn(e, 5), "y": torch.randn(1),
            "ctx": torch.randn(4), "emb": torch.randn(n, 4),
            "stellar_mass": torch.rand(n) * 1e10 + 1e9,
            "vel_disp": torch.rand(n) * 200 + 50,
            "half_mass_r": torch.rand(n) * 0.01 + 1e-3,
            "pos": torch.randn(n, 3)}


def test_budget_helper_resolves_explicit_over_cfg():
    from rls.budget_accounting import resolve_requested_keep
    cfg = {"tta_target_sparsity": 0.5, "target_sparsity_end": 0.4}
    assert resolve_requested_keep(cfg, explicit=0.3) == 0.3
    assert resolve_requested_keep(cfg, explicit=None) == 0.5
    assert resolve_requested_keep({"target_sparsity_end": 0.4}, explicit=None) == 0.4


def test_policy_masks_accepts_explicit_target_and_preserves_two_tuple():
    from rls.policy import EdgePolicyNet
    from rls.notebook_workflow import policy_masks
    import inspect
    sig = inspect.signature(policy_masks)
    assert "target_sparsity" in sig.parameters
    torch.manual_seed(0)
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=4, hidden_dim=8)
    g = _tiny_pair_graph()
    cfg = {"sparsity_mode": "pair_pl", "target_sparsity_end": 0.4}
    out = policy_masks(policy, [g], cfg, target_sparsity=0.5)
    # existing 2-tuple compatibility
    assert isinstance(out, tuple) and len(out) == 2


def test_k0_kpositive_share_resolved_q_and_order_counts():
    from rls.policy import EdgePolicyNet
    from rls.notebook_workflow import adaptation_trial
    from rls.evaluate import evaluate_tta
    import inspect
    assert "target_sparsity" in inspect.signature(adaptation_trial).parameters
    torch.manual_seed(0)
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=4, hidden_dim=8)

    class MockGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)
            self.register_buffer("w", torch.tensor(0.1))
        def predict_with_uncertainty(self, batch, n_samples=5):
            E = batch.edge_index.shape[1]
            return {"std": torch.tensor([1.0 / max(1, E)])}
        def forward(self, batch):
            n = batch.x.shape[0]
            return torch.rand(n, 1).sum(dim=0, keepdim=True), torch.randn(n, 4)

    graphs = [_tiny_pair_graph(seed=i) for i in range(2)]
    cfg = {"tta_lr": 1e-3, "tta_steps": 2, "tta_mc_samples": 3, "tta_patience": 5,
           "min_keep_frac": 0.1, "entropy_coef": 0.01, "sparsity_mode": "pair_pl",
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0,
           "seed": 0, "tta_target_sparsity": 0.5, "target_sparsity_end": 0.4}
    row0, _ = adaptation_trial(policy, graphs, MockGNN(), cfg, "cpu", 0, target_sparsity=0.5)
    row2, hist2 = adaptation_trial(policy, graphs, MockGNN(), cfg, "cpu", 2, target_sparsity=0.5)
    assert row0["requested_keep_fraction"] == row2["requested_keep_fraction"] == 0.5
    assert row0["decoder_kind"] == row2["decoder_kind"] == "legacy_repaired"
    # sampled comes from actual order, repair may inflate final
    assert row0["sampled_pair_count"] is not None
    assert row0["final_pair_count"] >= row0["sampled_pair_count"]
    assert row0["requested_pair_count"] is not None
    # evaluate_tta shares the same q
    rows = evaluate_tta(policy, graphs, MockGNN(), cfg, "cpu", ks=(0, 2), target_sparsity=0.5)
    by_k = {r["K"]: r for r in rows}
    assert by_k[0]["requested_keep_fraction"] == by_k[2]["requested_keep_fraction"] == 0.5


def test_pair_tta_info_exposes_order_counts():
    from rls.policy import EdgePolicyNet
    from rls.pair_tta import adapt_pair_policy
    torch.manual_seed(0)
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=4, hidden_dim=8)

    class MockGNN(torch.nn.Module):
        def predict_with_uncertainty(self, batch, n_samples=3):
            E = batch.edge_index.shape[1]
            return {"mean": torch.zeros(1), "std": torch.tensor([1.0 / max(1, E)])}

    g = _tiny_pair_graph()
    cfg = {"tta_lr": 1e-3, "tta_steps": 2, "tta_mc_samples": 3, "tta_patience": 5,
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0,
           "tta_kl_coef": 0.0, "tta_lr_decay": 1.0, "reward_clip": 10.0,
           "tta_uncertainty_floor": 0.05, "seed": 0}
    mask, info = adapt_pair_policy(policy, g, MockGNN(), cfg, "cpu", 0.5)
    assert "selected_order" in info or "sampled_pair_count" in info
    assert info["requested_keep_fraction"] == 0.5
    assert info["decoder_kind"] == "legacy_repaired"
    assert info["final_pair_count"] >= info["sampled_pair_count"]


def test_low_budget_repair_excess_is_explicit():
    from rls.pair_policy import policy_action
    from rls.policy import EdgePolicyNet
    from rls.constraints import PhysicalPairLayout, pair_budget
    torch.manual_seed(0)
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=4, hidden_dim=8)
    g = _tiny_pair_graph(seed=1)
    with torch.no_grad():
        logits = policy(g["edge_attr"], g["emb"], g["edge_index"], g["ctx"]).reshape(-1)
    # Very low budget: requested ceil(0.05*P) but repair for isolates inflates final.
    mask, _, _, order = policy_action(policy, g, {"target_sparsity_end": 0.4}, keep=0.05)
    from rls.budget_accounting import legacy_repair_report
    rep = legacy_repair_report(g["edge_index"], mask, keep_fraction=0.05, order=order,
                              num_nodes=len(g["x"]))
    assert rep["requested_pair_count"] == int(pair_budget(
        PhysicalPairLayout.from_edge_index(g["edge_index"], len(g["x"])), 0.05))
    assert rep["sampled_pair_count"] == int(order.numel())
    assert rep["final_pair_count"] >= rep["sampled_pair_count"]
    assert rep["decoder_kind"] == "legacy_repaired"
    # Reciprocal symmetry: physical copies share membership.
    layout = PhysicalPairLayout.from_edge_index(g["edge_index"], len(g["x"]))
    collapsed = layout.collapse(mask)
    assert int(collapsed.sum()) == rep["final_pair_count"]


def test_run_experiment_requires_fresh_output_dir_and_matched_rows(tmp_path):
    import yaml
    from rls.run_experiment import main
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["rls"]["epochs"] = 1
    cfg["rls"]["batch_size"] = 8
    cfg["rls"]["stageb_epochs"] = 1
    cfg["data"]["source"] = "synthetic"
    out = tmp_path / "rls"
    out.mkdir()
    (out / "stale.txt").write_text("stale")
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "missing")
    import pandas as pd
    from training.full_graph_baseline import _smoke_catalog
    halos, _ = _smoke_catalog(seed=42)
    rows = []
    for halo_id, halo in enumerate(halos):
        for subhalo in halo.subhalos:
            rows.append({"subhalo_id": subhalo.subhalo_id, "halo_id": halo_id,
                         "cluster_id": halo.cluster_id, "x": subhalo.position[0],
                         "y": subhalo.position[1], "z": subhalo.position[2],
                         "vx": subhalo.velocity[0], "vy": subhalo.velocity[1],
                         "vz": subhalo.velocity[2], "stellar_mass": subhalo.stellar_mass,
                         "velocity_dispersion": subhalo.velocity_dispersion,
                         "half_mass_radius": subhalo.half_mass_radius,
                         "metallicity": subhalo.metallicity,
                         "halo_mass": halo.halo_mass, "redshift": halo.redshift})
    csv_path = tmp_path / "spd.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    cfg["data"]["synthetic"] = {"features_path": str(tmp_path / "sf.pt"), "csv_path": str(csv_path)}
    cfg["graph"]["method"] = "knn"
    cfg["graph"]["k_neighbors"] = 4
    try:
        main(cfg, checkpoint=None, output_dir=out, allow_random_backbone=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError for nonempty output_dir")


def _cpu_data():
    from torch_geometric.data import Data
    return Data(x=torch.randn(4, 4),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                edge_attr=torch.randn(4, 5),
                y=torch.tensor([1.0]),
                pos=torch.randn(4, 3),
                cluster_id="cid-001",
                metadata={"lineage_group": "g1"},
                num_nodes=4)


def test_fix1_device_transfer_helper():
    from rls.run_experiment import move_test_graphs_to_device, model_device
    graphs = [_cpu_data()]
    out = move_test_graphs_to_device(graphs, torch.device("cpu"))
    assert out[0].cluster_id == "cid-001"
    assert torch.equal(out[0].x, graphs[0].x)
    assert out[0].metadata == {"lineage_group": "g1"}

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(1, 1)
    assert model_device(M().to("cpu")) == torch.device("cpu")


def test_fix2_split_manifest_hash_and_tamper():
    from rls.run_experiment import build_legacy_split_manifest
    tr = [type("H", (), {"cluster_id": "a"})()]
    va = [type("H", (), {"cluster_id": "b"})()]
    te = [type("H", (), {"cluster_id": "c"})()]
    m1 = build_legacy_split_manifest(tr, va, te, seed=42,
                                      ratios=(0.7, 0.15, 0.15))
    assert m1["manifest_sha256"] is not None
    m2 = build_legacy_split_manifest(tr, va,
                                      [type("H", (), {"cluster_id": "CHANGED"})()],
                                      seed=42, ratios=(0.7, 0.15, 0.15))
    assert m2["manifest_sha256"] != m1["manifest_sha256"]
    assert "BaseDataLoader.split_data" in m1["method"]
    assert "grouped" not in m1["method"]


def test_fix2_strict_missing_group_fails():
    from data.splits import grouped_split
    from data.loaders.base_loader import HaloData, SubhaloData
    import numpy as np
    s = SubhaloData(subhalo_id=0, position=np.zeros(3), velocity=np.zeros(3),
                    stellar_mass=1e9, velocity_dispersion=100.,
                    half_mass_radius=0.01, metallicity=0.01)
    h = HaloData(cluster_id="h0", subhalos=[s], halo_mass=1e12, metadata={})
    import pytest
    with pytest.raises(ValueError, match="grouping key"):
        grouped_split([h], group_key="lineage_group")


def test_fix3_random_backbone_label_truthful():
    from rls.run_experiment import resolve_random_backbone_label
    lab, mode = resolve_random_backbone_label({"data": {"source": "synthetic"}}, True)
    assert lab == "synthetic_random_backbone_smoke"
    lab2, _ = resolve_random_backbone_label({"data": {"source": "tng"}}, True)
    assert lab2 != "synthetic_random_backbone_smoke"
    assert "tng" in lab2 and "untrained" in lab2 and "random" in lab2


def test_fix4_stageA_same_target_budget():
    from rls.policy import EdgePolicyNet
    from rls.notebook_workflow import policy_metrics, random_pair_reference
    torch.manual_seed(0)
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=4, hidden_dim=8)

    class MockGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d = torch.nn.Linear(1, 1)
        def forward(self, batch):
            return torch.zeros(1), None

    def g():
        n, e = 4, 8
        return {"x": torch.randn(n, 4),
                "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                                            [1, 0, 2, 1, 3, 2, 0, 3]]),
                "edge_attr": torch.randn(e, 5), "y": torch.randn(1),
                "ctx": torch.randn(4), "emb": torch.randn(n, 4),
                "stellar_mass": torch.rand(n) + 1.0,
                "vel_disp": torch.rand(n) + 50.0,
                "half_mass_r": torch.rand(n) + 1e-3,
                "pos": torch.randn(n, 3)}
    graphs = [g() for _ in range(2)]
    cfg = {"target_sparsity_end": 0.4, "tta_target_sparsity": 0.7, "seed": 0}
    m = policy_metrics(policy, graphs, MockGNN(), cfg)
    r = random_pair_reference(graphs, MockGNN(), cfg, seeds=range(2))
    assert m["requested_keep_fraction"] == r[0]["requested_keep_fraction"]
    assert m["decoder_kind"] == "legacy_repaired"


def test_fix5_source_provenance_covers_pair_training_and_augmentation():
    from rls.run_experiment import TASK3C_SOURCE_FILES
    assert "rls/pair_training.py" in TASK3C_SOURCE_FILES
    assert "data/augmentation.py" in TASK3C_SOURCE_FILES
    assert "data/loaders/base_loader.py" in TASK3C_SOURCE_FILES
    assert "graph/graph_builder.py" in TASK3C_SOURCE_FILES


def test_policy_validation_requires_known_equal_requested_budgets():
    from rls.notebook_workflow import policy_validation
    metrics = {"rmse": 1.0, "physical_isolates": 0}
    random_rows = [{"rmse": 2.0}, {"rmse": 2.2}]
    unknown = policy_validation(metrics, random_rows)
    assert unknown["budget_comparable"] is False
    assert unknown["verdict"] != "PASS"
    unequal = policy_validation({**metrics, "requested_keep_fraction": 0.4},
                                [{"rmse": 2.0, "requested_keep_fraction": 0.5},
                                 {"rmse": 2.2, "requested_keep_fraction": 0.5}])
    assert unequal["budget_comparable"] is False
    equal = policy_validation({**metrics, "requested_keep_fraction": 0.4},
                              [{"rmse": 2.0, "requested_keep_fraction": 0.4},
                               {"rmse": 2.2, "requested_keep_fraction": 0.4}])
    assert equal["budget_comparable"] is True
