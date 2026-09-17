"""
Behavior-contract tests for self-loops in the RL physics/reward path.

HISTORY: the Aug 2026 deep virial audit found the original physics sum
INCLUDED self-loops (r clamped to 1e-6 Mpc), which supplied ~100% of the
pairwise PE on real graphs, drove the virial ratio to ~4e-5, and made the old
one-sided penalty identically zero. The approved repair (Aug 2026, candidate 7)
changed PHYSICS-EDGE semantics only: _graph_physics_terms now excludes
self-loops, and the reward uses the relative penalty
(log r_pruned - log r_full)^2 — see tests/test_relative_virial_reward.py.

PHYSICAL CONNECTIVITY: loops never cover a physical isolate. Coverage requires
an incident unordered non-self pair, consistently in repair and reward checks.

GNN self-loops in edge_index/edge_attr remain INTENTIONAL graph architecture
and are unchanged by the physics filter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from rls.train_policy import _graph_physics_terms, _no_isolated
from rls.rewards import virial_penalty, virial_ratio_pruned, relative_virial_penalty


def _two_node_graph():
    """2 nodes, 1 real edge (u=0,v=1), plus one self-loop per node — the same
    shape GraphBuilder produces with graph.self_loops: true. Fixed values so
    every assertion below is exact and deterministic."""
    return {
        "edge_index": torch.tensor([[0, 1, 0, 1],
                                    [1, 0, 0, 1]]),          # real pair x2, loops x2
        "stellar_mass": torch.tensor([1e11, 1e11]),
        "vel_disp": torch.tensor([200.0, 200.0]),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),  # 1 Mpc apart
    }


def test_self_loops_are_excluded_from_physics_terms():
    """Post-V1 behavior: adding self-loops to the graph leaves the physics
    terms unchanged — the dead-virial bug (loops ~100% of PE, ratio ~4e-5,
    penalty == 0) is gone by construction."""
    g = _two_node_graph()
    ke_with, pe_with = _graph_physics_terms(
        g, g["edge_index"], torch.ones(4, dtype=torch.bool))
    real = g["edge_index"][:, :2]
    ke_free, pe_free = _graph_physics_terms(
        g, real, torch.ones(2, dtype=torch.bool))
    assert torch.allclose(ke_with, ke_free)
    assert torch.allclose(pe_with, pe_free)


def test_relative_penalty_zero_on_full_mask_positive_under_pruning():
    """The approved formulation on the same toy graph: full mask -> exactly 0;
    dropping the real edge (loops only) -> finite, large (eps_ratio band)."""
    g = _two_node_graph()
    ke_f, pe_f = _graph_physics_terms(g, g["edge_index"],
                                      torch.ones(4, dtype=torch.bool))
    full_pen = relative_virial_penalty(ke_f, pe_f, ke_f, pe_f)
    assert full_pen.item() == 0.0
    loops_only = torch.tensor([False, False, True, True])
    ke_p, pe_p = _graph_physics_terms(g, g["edge_index"], loops_only)
    assert pe_p.item() == 0.0  # loop-only mask keeps no physics edge at all
    pen = relative_virial_penalty(ke_p, pe_p, ke_f, pe_f)
    assert torch.isfinite(pen) and pen.item() > 100.0


def test_legacy_one_sided_penalty_function_is_unchanged():
    """The OLD absolute penalty function is intentionally retained (used by
    tests and reporting) and keeps its historical semantics; the reward paths
    no longer call it."""
    assert virial_ratio_pruned(ke_retained=2.0, pe_retained=0.5) == 8.0
    assert virial_penalty(torch.tensor(2.0), torch.tensor(0.5)).item() == 49.0


def test_self_loop_does_not_cover_a_physical_isolate_in_rl_connectivity():
    edge_index = torch.tensor([[0, 1, 0, 2, 2], [1, 0, 2, 0, 2]])
    mask = torch.tensor([True, True, False, False, True])
    assert _no_isolated(edge_index, mask) is False
    assert _no_isolated(edge_index[:, :4], mask[:4]) is False
    from rls.sparsify import repair_connectivity
    repaired = repair_connectivity(edge_index, mask)
    assert repaired.tolist() == [True, True, True, True, True]
    assert _no_isolated(edge_index, repaired) is True
