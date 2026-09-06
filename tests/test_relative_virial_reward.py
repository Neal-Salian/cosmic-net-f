"""Focused regression tests for the approved relative virial reward (Aug 2026):

    penalty = (log r_pruned - log r_full)^2

over LOOP-FREE physics edges, with the same graph's loop-free full-graph
reference. Deterministic toy graphs only — no real-dataset dependency.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from rls.rewards import (relative_virial_penalty, label_free_reward,
                         virial_ratio_pruned)
from rls.train_policy import _graph_physics_terms
from rls.sparsify import apply_min_keep_floor, repair_connectivity, hard_mask


def _toy_graph(with_loops=True):
    """4 nodes on a line with increasing separations + one self-loop per node
    (the builder's structure). All values fixed -> every assert is exact."""
    pos = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [4.0, 0, 0]])
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 1, 2, 3],
                       [1, 0, 2, 1, 3, 2, 0, 1, 2, 3]])  # 6 real, 4 loops
    if not with_loops:
        ei = ei[:, :6]
    return {
        "edge_index": ei,
        "stellar_mass": torch.tensor([1e11, 2e11, 1e11, 5e10]),
        "vel_disp": torch.tensor([200.0, 250.0, 180.0, 120.0]),
        "pos": pos,
    }


def _terms(g, mask):
    return _graph_physics_terms(g, g["edge_index"], mask)


def _full_reference(g):
    return _terms(g, torch.ones(g["edge_index"].shape[1], dtype=torch.bool))


def test_1_self_loops_do_not_contribute_to_physics_terms():
    g_loops, g_free = _toy_graph(True), _toy_graph(False)
    m6 = torch.ones(6, dtype=torch.bool)
    ke_a, pe_a = _graph_physics_terms(g_loops, g_loops["edge_index"], torch.ones(10, dtype=torch.bool))
    ke_b, pe_b = _graph_physics_terms(g_free, g_free["edge_index"], m6)
    assert torch.allclose(ke_a, ke_b) and torch.allclose(pe_a, pe_b)


def test_2_full_graph_vs_itself_is_exactly_zero():
    g = _toy_graph()
    ke, pe = _full_reference(g)
    assert relative_virial_penalty(ke, pe, ke, pe).item() == 0.0


def test_3_edge_removal_changes_the_penalty():
    g = _toy_graph()
    ke_f, pe_f = _full_reference(g)
    half = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
    ke_p, pe_p = _terms(g, half)
    assert relative_virial_penalty(ke_p, pe_p, ke_f, pe_f).item() > 1e-6


def test_4_structure_preserving_beats_destructive_mask():
    """Keep the three SHORT edges vs the three LONG edges of the same graph:
    the destructive (long-edge) mask must incur the larger penalty."""
    g = _toy_graph()
    ke_f, pe_f = _full_reference(g)
    short = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)  # loops + short pairs
    long_ = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)  # loops + long pairs
    ke_s, pe_s = _terms(g, short)
    ke_l, pe_l = _terms(g, long_)
    p_short = relative_virial_penalty(ke_s, pe_s, ke_f, pe_f).item()
    p_long = relative_virial_penalty(ke_l, pe_l, ke_f, pe_f).item()
    assert p_long > p_short > 0.0


def test_5_global_stellar_mass_rescaling_invariance():
    g = _toy_graph()
    ke_f, pe_f = _full_reference(g)
    half = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
    ke_p, pe_p = _terms(g, half)
    p1 = relative_virial_penalty(ke_p, pe_p, ke_f, pe_f)
    s = 37.7
    g2 = dict(g)
    g2["stellar_mass"] = g["stellar_mass"] * s
    ke_f2, pe_f2 = _full_reference(g2)
    ke_p2, pe_p2 = _terms(g2, half)
    p2 = relative_virial_penalty(ke_p2, pe_p2, ke_f2, pe_f2)
    assert torch.allclose(p1, p2, rtol=1e-5)


def test_6_zero_real_edge_mask_is_finite_and_maximal_band():
    """A mask that keeps no real physics edge (loops only or nothing) must not
    produce NaN/inf: the eps_ratio log-clamp keeps it finite and large."""
    g = _toy_graph()
    ke_f, pe_f = _full_reference(g)
    loops_only = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)
    ke_p, pe_p = _terms(g, loops_only)
    assert pe_p.item() == 0.0 and ke_p.item() == 0.0
    pen = relative_virial_penalty(ke_p, pe_p, ke_f, pe_f)
    assert torch.isfinite(pen) and pen.item() > 100.0  # clamped to log(1e-12) band
    # direct API form as well:
    assert torch.isfinite(relative_virial_penalty(0.0, 0.0, ke_f, pe_f))


def test_7_mask_dtype_contracts_hold():
    g = _toy_graph()
    probs = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    action = torch.bernoulli(probs).bool()
    m = repair_connectivity(g["edge_index"],
                            apply_min_keep_floor(action, probs, 0.1))
    assert m.dtype == torch.bool and m.numel() == g["edge_index"].shape[1]
    assert hard_mask(probs, 0.1).dtype == torch.bool


def test_8_reward_uses_exactly_the_action_mask():
    """The physics terms fed to the penalty are computed from exactly the
    floor/repaired action mask under the PIPELINE calling convention
    (full edge_index + mask; the KE degree-fraction baseline is the FULL
    degree). PE is exactly mask-slicing-equivalent; the mask itself is
    exactly the Phase-2 floor/repaired action."""
    g = _toy_graph()
    probs = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    action = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)
    m = repair_connectivity(g["edge_index"], apply_min_keep_floor(action, probs, 0.1))
    # floor is a no-op (6 kept >= ceil(0.1*10)=1) and every node is covered,
    # so the executed mask IS the sampled action:
    assert torch.equal(m, action)
    ke_m, pe_m = _terms(g, m)
    # PE depends only on the kept edge set -> exactly equal to slicing first:
    ke_slice, pe_slice = _graph_physics_terms(
        g, g["edge_index"][:, m], torch.ones(int(m.sum()), dtype=torch.bool))
    assert torch.allclose(pe_m, pe_slice)
    # KE uses the full-degree baseline (pipeline convention) and is
    # deterministic under repeated calls with the same mask:
    ke_again, pe_again = _terms(g, m)
    assert torch.equal(ke_m, ke_again) and torch.equal(pe_m, pe_again)


def test_9_full_reference_is_action_independent():
    g = _toy_graph()
    ke1, pe1 = _full_reference(g)
    _ = _terms(g, torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool))
    ke2, pe2 = _full_reference(g)
    assert torch.equal(ke1, ke2) and torch.equal(pe1, pe2)


def test_10_other_reward_components_unchanged_by_penalty_value():
    """compute_rewards/label_free_reward remain linear in the penalty argument
    with their existing coefficients."""
    cfg = {"w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 1.0}
    r1 = label_free_reward(std_pruned=0.05, std_full=0.10, keep_ratio=0.5,
                           target_sparsity=0.5, virial_penalty=0.0,
                           cfg=cfg, connectivity_ok=True)
    r2 = label_free_reward(std_pruned=0.05, std_full=0.10, keep_ratio=0.5,
                           target_sparsity=0.5, virial_penalty=2.0,
                           cfg=cfg, connectivity_ok=True)
    assert abs((r1 - r2) - 2.0) < 1e-6


def test_directed_pair_bookkeeping_is_documented_not_canonicalized():
    """DOCUMENTED BEHAVIOR (separate methodology gate, NOT changed here): the
    pairwise PE sums DIRECTED edges, so a representation storing both
    directions of every pair double-counts PE — while KE is degree-FRACTION
    normalized and therefore does NOT scale. Consistent direction-doubling
    hence HALVES the ratio, giving an exact ln(2)^2 relative penalty; a mask
    keeping exactly one direction of each pair produces the same shift in the
    opposite direction (the exact ln(2) measured for mass-ratio masks)."""
    g = _toy_graph()
    ei_one_dir = torch.tensor([[0, 1, 2], [1, 2, 3]])  # each pair once
    ei_two_dir = torch.cat([ei_one_dir, ei_one_dir.flip(0)], dim=1)
    ke1, pe1 = _graph_physics_terms(g, ei_one_dir, torch.ones(3, dtype=torch.bool))
    ke2, pe2 = _graph_physics_terms(g, ei_two_dir, torch.ones(6, dtype=torch.bool))
    assert torch.isclose(pe2, 2 * pe1, rtol=1e-5)   # PE double-counts pairs
    assert torch.isclose(ke2, ke1, rtol=1e-5)       # deg and deg' both double -> frac unchanged
    # ratio halves under consistent doubling -> exact ln(2)^2 penalty:
    pen = relative_virial_penalty(ke2, pe2, ke1, pe1).item()
    assert abs(pen - math.log(2.0) ** 2) < 1e-4


def test_11_synthetic_virialized_cluster_validates_G_units():
    """P1-physics sanity (audit Sep 2026): a synthetic two-body cluster tuned
    to exact virial equilibrium must yield ratio == 1 through the PIPELINE
    path (_graph_physics_terms + virial_ratio_pruned), validating the KE/PE
    computation and G units/dimensions end to end before trusting real-data
    Spearman/virial diagnostics.

    Setup: m1 = m2 = m, separation r, isotropic dispersion sigma with
    sigma^2 = G*m/r. Hand computation (BOTH directed edges kept, loops
    filtered): KE = m*sigma^2, |PE| = 2*G*m^2/r, so
    ratio = 2*m*sigma^2 / (2*G*m^2/r) = sigma^2*r/(G*m) = 1 exactly.
    NOTE the directed double-count is the documented pipeline convention
    (see test above); the assertion bakes it in rather than hiding it."""
    G = 4.302e-9  # Mpc (km/s)^2 / Msun — must match train_policy default
    m, r = 1e10, 1.0
    sigma = math.sqrt(G * m / r)
    g = {"edge_index": torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]]),
         "stellar_mass": torch.tensor([m, m]),
         "vel_disp": torch.tensor([sigma, sigma]),
         "pos": torch.tensor([[0.0, 0, 0], [r, 0, 0]])}
    mask = torch.ones(4, dtype=torch.bool)
    ke, pe = _graph_physics_terms(g, g["edge_index"], mask)
    assert torch.isclose(ke, torch.tensor(m * sigma ** 2), rtol=1e-4)
    assert torch.isclose(pe, torch.tensor(2 * G * m * m / r), rtol=1e-4)
    ratio = virial_ratio_pruned(ke, pe)
    assert torch.isclose(ratio, torch.tensor(1.0), rtol=1e-4)


def test_12_near_full_mask_penalty_is_small_but_not_clamped():
    """P1-physics saturation investigation (audit Sep 2026): the real-data
    median relative penalty ~0 (IQR all zero) is EXPLAINED by keep ~= 0.976
    (pruned ratio ~= full ratio), NOT by eps_ratio clamping making the term a
    no-op. A near-full mask must give a penalty that is small, strictly
    positive, finite, and computed strictly inside the clamp bounds."""
    g = _toy_graph()
    ke_f, pe_f = _full_reference(g)
    # drop a single directed pair (keep ~= 0.9 here; real data keeps 0.976,
    # which only makes the penalty smaller, same mechanism)
    near_full = torch.tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    ke_p, pe_p = _terms(g, near_full)
    pen = relative_virial_penalty(ke_p, pe_p, ke_f, pe_f).item()
    assert pen > 0.0                       # mask-sensitive: responds to the drop
    assert math.isfinite(pen) and pen < 10.0  # small, not a clamp artifact
    for ke, pe in ((ke_p, pe_p), (ke_f, pe_f)):
        r = (2.0 * float(ke)) / max(float(pe), 1e-30)
        assert 1e-12 < r < 1e12            # strictly inside eps_ratio bounds
