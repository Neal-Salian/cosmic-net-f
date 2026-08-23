"""
Behavior-contract tests for self-loops in the RL physics/reward path.

These tests CHARACTERIZE the currently implemented behavior (they are not an
endorsement of it). The Aug 2026 deep virial audit established, on the real
committed TNG dataset:

  - GraphBuilder adds self-loops to edge_index (graph.self_loops: true) and
    nothing filters u == v downstream;
  - _graph_physics_terms therefore clamps self-loop separations to 1e-6 Mpc,
    which supplies ~100% of the pairwise PE and drives the virial ratio to
    ~4e-5 (median) on real graphs;
  - with the one-sided penalty max(0, ratio - 1)^2, the virial term is
    EXACTLY ZERO for every graph and mask in the real pipeline;
  - _no_isolated / repair_connectivity count a self-loop as covering a node,
    while GraphBuilder._connect_isolated_nodes deliberately excludes
    self-loops when finding isolated nodes.

Any change to this file's expectations is a reward-semantics change and
requires explicit methodology approval (see the virial audit report and the
plan's Part 3 rule 2). These tests exist so that such a change cannot happen
accidentally or silently.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from rls.train_policy import _graph_physics_terms, _no_isolated
from rls.rewards import virial_penalty, virial_ratio_pruned


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


def test_self_loops_dominate_pe_and_kill_the_penalty():
    """CURRENT behavior: self-loops (r clamped to 1e-6) dominate PE, the ratio
    collapses far below 1, and the one-sided penalty is exactly zero."""
    g = _two_node_graph()
    ke, pe = _graph_physics_terms(g, g["edge_index"],
                                  torch.ones(4, dtype=torch.bool))
    ratio = virial_ratio_pruned(ke, pe)
    assert ratio.item() < 1e-3, ratio
    assert virial_penalty(ke, pe).item() == 0.0


def test_loop_free_variant_is_active_and_large():
    """The same graph WITHOUT self-loops: the identical code path yields a
    ratio > 1 and a large positive penalty. This is the demonstration that
    self-loop presence alone flips the term between dead and dominant — the
    core finding of the virial audit. Loop removal is a METHODOLOGY GATE."""
    g = _two_node_graph()
    real = g["edge_index"][:, :2]
    ke, pe = _graph_physics_terms(g, real, torch.ones(2, dtype=torch.bool))
    ratio = virial_ratio_pruned(ke, pe)
    assert ratio.item() > 1.0
    assert virial_penalty(ke, pe).item() > 1.0


def test_self_loop_counts_as_covering_a_node_in_rl_connectivity():
    """CURRENT behavior: _no_isolated counts a kept self-loop as an incident
    edge, so a node whose ONLY kept edge is its self-loop is 'not isolated'
    on the RL side. (GraphBuilder._connect_isolated_nodes deliberately uses
    the opposite convention when building graphs.) Changing this is a reward
    change — gated. Three nodes: node 2's real edges are dropped and only its
    self-loop is kept — it still counts as covered."""
    edge_index = torch.tensor([[0, 1, 0, 2, 2],
                               [1, 0, 2, 0, 2]])  # real 0-1, real 0-2, loop on 2
    # keep the 0-1 pair and node 2's self-loop; drop the real 0-2 edges
    mask = torch.tensor([True, True, False, False, True])
    assert _no_isolated(edge_index, mask) is True
    # and without the self-loop, node 2 is isolated:
    assert _no_isolated(edge_index, mask.clone()[:4]) is False


def test_pe_is_exactly_the_clamped_selfloop_term_for_loop_only_masks():
    """Exact arithmetic contract of the clamp: for a kept self-loop the PE
    contribution is G * M^2 / 1e-6 (r == 0 clamped), i.e. the pairwise PE sum
    over a loop-only mask is fully determined by the clamp."""
    g = _two_node_graph()
    loop_only = torch.tensor([False, False, True, True])
    ke, pe = _graph_physics_terms(g, g["edge_index"], loop_only)
    G = 4.302e-9
    expected = G * (1e11 ** 2) / 1e-6 * 2  # two self-loops, identical masses
    assert torch.isclose(pe, torch.tensor(expected), rtol=1e-4), (pe, expected)
    # KE side: deg counts a self-loop twice (both index_add passes hit node i),
    # so frac_1 = deg'/deg = 2/2 = 1 for node 1; node 0 has deg 1, deg' 0.
    expected_ke = 0.5 * (1e11 * 200.0 ** 2)  # only node 1's term survives
    assert torch.isclose(ke, torch.tensor(expected_ke), rtol=1e-4)
