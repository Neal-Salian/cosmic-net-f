"""Exact-budget notebook cohorts are structural, explicit, and mask-aligned."""
import copy

import pytest
import torch

from rls.constraints import PhysicalPairLayout, feasible_budget
from rls.notebook_benchmark import exact_policy_mask
from rls.policy import EdgePolicyNet


def make_graph(name, pairs, nodes):
    columns = [(u, v) for u, v in pairs] + [(v, u) for u, v in pairs]
    columns += [(n, n) for n in range(nodes)]
    edges = torch.tensor(columns, dtype=torch.long).T
    return dict(cluster_id=name, x=torch.ones(nodes, 4), edge_index=edges,
        edge_attr=torch.ones(len(columns), 5), emb=torch.ones(nodes, 4),
        ctx=torch.ones(4), pos=torch.zeros(nodes, 3),
        stellar_mass=torch.ones(nodes), y=torch.tensor([12.]))


def reported_graph():
    # K(2,6): matching size 2 => minimum edge cover 8 - 2 = 6.
    return make_graph("reported-12-5-6", [(u, v) for u in (0, 1) for v in range(2, 8)], 8)


def cfg():
    return dict(seed=42, evaluation_decoder="direct", evaluation_constraint="no_isolates",
        target_sparsity_end=.4,
        edge_feature_names=["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"])


def test_reported_12_available_5_requested_6_minimum_is_genuinely_infeasible():
    graph = reported_graph()
    layout = PhysicalPairLayout.from_edge_index(graph["edge_index"], 8)
    assert feasible_budget(layout, 5, "no_isolates") == dict(feasible=False,
        requested_pair_count=5, minimum_pairs=6, available_pair_count=12,
        constraint="no_isolates")
    policy = EdgePolicyNet(5, 4, hidden_dim=8)
    with pytest.raises(ValueError, match="infeasible physical-pair budget"):
        exact_policy_mask(policy, graph, cfg(), .4)


def test_cohort_records_failure_without_raising_or_enlarging_the_budget():
    from rls.notebook_budget import budget_cohort, record_budget_realizations
    good = [make_graph(name, [(0, 1), (0, 2), (1, 2)], 3) for name in ("a", "b")]
    bad = reported_graph()
    graphs = good + [bad]
    before = copy.deepcopy(graphs)
    selected, report = budget_cohort(graphs, cfg(), split="train")
    assert [g["cluster_id"] for g in selected] == ["a", "b"]
    assert report["requested_keep_fraction"] == .4
    assert report["eligible_graph_ids"] == ["a", "b"]
    rejected = report["graphs"][-1]
    assert rejected["requested_pair_count"] == 5
    assert rejected["minimum_pairs"] == 6
    assert rejected["available_pair_count"] == 12
    assert rejected["realized_pair_count"] is None
    assert rejected["reason"] == "requested_budget_below_minimum"
    assert rejected["eligible"] is False
    policy = EdgePolicyNet(5, 4, hidden_dim=8)
    masks = [exact_policy_mask(policy, g, cfg(), .4)[0] for g in selected]
    record_budget_realizations(report, selected, masks)
    assert all(row["requested_pair_count"] == row["realized_pair_count"] == 2
               for row in report["graphs"][:2])
    assert all(row["constraint_satisfied"] for row in report["graphs"][:2])
    assert report["graphs"][-1]["realized_pair_count"] is None
    for original, current in zip(before, graphs):
        assert torch.equal(original["edge_index"], current["edge_index"])


def test_cohort_recomputed_for_fraction_and_constraint():
    from rls.notebook_budget import budget_cohort
    graph = reported_graph()
    assert budget_cohort([graph], cfg(), split="val")[0] == []
    selected, report = budget_cohort([graph], cfg(), split="val", target_sparsity=.5)
    assert selected == [graph]
    assert report["graphs"][0]["requested_pair_count"] == 6
    assert budget_cohort([graph], dict(cfg(), evaluation_constraint="none"), split="val")[0] == [graph]


def test_loop_only_and_insufficient_cohorts_are_explicit():
    from rls.notebook_budget import budget_cohort, require_budget_cohort
    graph = make_graph("loops-only", [], 3)
    selected, report = budget_cohort([graph], cfg(), split="val")
    assert not selected
    assert report["graphs"][0]["minimum_pairs"] is None
    assert report["graphs"][0]["reason"] == "original_topology_cannot_satisfy_constraint"
    with pytest.raises(ValueError, match="val.*0.*2"):
        require_budget_cohort(report, minimum=2)


def test_realization_requires_aligned_symmetric_exact_masks():
    from rls.notebook_budget import budget_cohort, record_budget_realizations
    graph = make_graph("good", [(0, 1), (0, 2), (1, 2)], 3)
    selected, report = budget_cohort([graph], cfg(), split="test")
    with pytest.raises(ValueError, match="aligned"):
        record_budget_realizations(report, selected, [])
    mask = exact_policy_mask(EdgePolicyNet(5, 4, hidden_dim=8), graph, cfg(), .4)[0]
    mask[0] = ~mask[0]
    with pytest.raises(ValueError, match="symmetric"):
        record_budget_realizations(report, selected, [mask])


def test_realization_rejects_a_rewired_graph_with_the_same_id_and_pair_count():
    from rls.notebook_budget import budget_cohort, record_budget_realizations
    original = make_graph("same-id", [(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    _, report = budget_cohort([original], cfg(), split="test", target_sparsity=.5)
    rewired = make_graph("same-id", [(0, 1), (0, 2), (0, 3), (1, 2)], 4)
    mask = exact_policy_mask(EdgePolicyNet(5, 4, hidden_dim=8), rewired, cfg(), .5)[0]
    with pytest.raises(ValueError, match="topology differs"):
        record_budget_realizations(report, [rewired], [mask])
