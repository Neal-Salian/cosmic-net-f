"""Topology-only execution cohorts for an unchanged exact physical-pair budget.

This does not repair budgets or make infeasible graphs disappear from a matched
benchmark. Notebooks retain the original split manifest and benchmark failure
rows, and save these execution cohorts separately before policy/model work.
"""
from data.provenance import hash_payload
from rls.budget_accounting import resolve_requested_keep
from rls.constraints import (PhysicalPairLayout, feasible_budget, pair_budget,
                             selection_diagnostics)


def _identity(graph):
    value = graph.get("cluster_id")
    if value is None:
        raise ValueError("Budget cohorts require a stable cluster_id on every graph")
    return str(value)


def _topology_identity(layout):
    return hash_payload(dict(num_nodes=layout.num_nodes,
        physical_pairs=layout.pairs.detach().cpu().tolist()))


def budget_cohort(graphs, cfg, *, split, target_sparsity=None):
    """Return aligned eligible graphs and a report covering every original ID.

    Eligibility depends solely on topology, requested fraction and constraint.
    Labels, learned scores and prediction quality cannot affect membership.
    A feasible preflight is not a realized selection: realization starts null.
    """
    q = resolve_requested_keep(cfg, explicit=target_sparsity)
    constraint = cfg.get("evaluation_constraint", "no_isolates")
    eligible, rows, identities = [], [], set()
    for graph in graphs:
        identity = _identity(graph)
        if identity in identities:
            raise ValueError(f"Duplicate graph ID in {split} budget cohort: {identity}")
        identities.add(identity)
        layout = PhysicalPairLayout.from_edge_index(graph["edge_index"], len(graph["x"]))
        k = pair_budget(layout, q)
        feasibility = feasible_budget(layout, k, constraint)
        reason = None
        if not feasibility["feasible"]:
            reason = ("original_topology_cannot_satisfy_constraint"
                      if feasibility["minimum_pairs"] is None else "requested_budget_below_minimum")
        else:
            eligible.append(graph)
        rows.append(dict(graph_id=identity, split=split,
            requested_keep_fraction=q, **feasibility,
            topology_sha256=_topology_identity(layout),
            eligible=feasibility["feasible"], reason=reason,
            realized_pair_count=None, realized_keep_fraction=None,
            constraint_satisfied=None, budget_satisfied=None))
    return eligible, dict(split=split, requested_keep_fraction=q,
        constraint=constraint, budget_unit="distinct_nonself_physical_pairs",
        rounding="ceil(q * available_pair_count)",
        eligibility_rule="topology_only_exact_budget_no_repair",
        original_count=len(rows), eligible_count=len(eligible),
        excluded_count=len(rows) - len(eligible),
        eligible_graph_ids=[row["graph_id"] for row in rows if row["eligible"]],
        excluded_graph_ids=[row["graph_id"] for row in rows if not row["eligible"]],
        graphs=rows)


def require_budget_cohort(report, *, minimum):
    """Fail before expensive work when too few graphs support the protocol."""
    if report["eligible_count"] < minimum:
        raise ValueError(f"{report['split']} exact-budget cohort has {report['eligible_count']} "
                         f"eligible graphs; requires at least {minimum}. "
                         "Inspect the saved budget eligibility report; do not enlarge the budget silently.")


def record_budget_realizations(report, graphs, masks):
    """Validate exact symmetric selections, then record actual realized budgets."""
    if (len(graphs) != len(masks)
            or [_identity(graph) for graph in graphs] != report["eligible_graph_ids"]):
        raise ValueError("Budget realization requires aligned eligible graphs and masks")
    updates = []
    expected_rows = [row for row in report["graphs"] if row["eligible"]]
    for graph, mask, expected in zip(graphs, masks, expected_rows):
        layout = PhysicalPairLayout.from_edge_index(graph["edge_index"], len(graph["x"]))
        if _topology_identity(layout) != expected["topology_sha256"]:
            raise ValueError("Budget realization graph topology differs from its preflight cohort")
        mask = mask.to(layout.pairs.device)
        selected = layout.collapse(mask)
        if not (layout.expand(selected) == mask).all():
            raise ValueError("Budget realization requires symmetric masks and preserved architecture loops")
        k = pair_budget(layout, report["requested_keep_fraction"])
        diag = selection_diagnostics(layout, selected, k=k, constraint=report["constraint"])
        if not diag["budget_satisfied"] or not diag["constraint_satisfied"]:
            raise ValueError(f"Budget realization violates the unchanged exact protocol: {diag}")
        updates.append(dict(realized_pair_count=diag["final_pair_count"],
            realized_keep_fraction=(diag["final_pair_count"] / len(layout.pairs)
                                    if len(layout.pairs) else None),
            constraint_satisfied=True, budget_satisfied=True))
    for row, update in zip((row for row in report["graphs"] if row["eligible"]), updates):
        row.update(update)
    return report
