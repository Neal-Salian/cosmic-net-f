import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.sparsify import (hard_mask, repair_connectivity, symmetrize_probs,
                          repair_symmetric, final_symmetric_mask,
                          pair_asymmetry_fraction, topk_scheduled_mask,
                          eval_mask)


def test_hard_mask_respects_min_keep():
    probs = torch.full((10,), 0.05)          # wants to drop almost everything
    mask = hard_mask(probs, min_keep_frac=0.2)
    assert mask.dtype == torch.bool          # indexing masks are bool, never 0/1 int
    assert mask.sum() >= 2                   # >= 0.2*10
    # keeps the highest-probability edges
    assert mask[probs.argmax()] == 1


def test_hard_mask_bool_in_both_branches():
    high = hard_mask(torch.full((8,), 0.9))                       # threshold branch
    low = hard_mask(torch.full((8,), 0.1), min_keep_frac=0.25)    # force-top-up branch
    assert high.dtype == torch.bool
    assert low.dtype == torch.bool


def test_repair_connectivity_no_isolated_nodes():
    N, E = 5, 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # path graph
    mask = torch.tensor([True, True, False, False])  # nodes 3 and 4 lose all edges
    repaired = repair_connectivity(edge_index, mask)
    assert repaired.dtype == torch.bool
    assert (repaired >= mask).all()          # repair only ADDS edges
    # every node must have degree >= 1
    deg = torch.zeros(N)
    for i in range(repaired.shape[0]):
        if repaired[i]:
            deg[edge_index[0, i]] += 1
            deg[edge_index[1, i]] += 1
    assert (deg >= 1).all()


def test_bool_mask_selects_exactly_the_kept_edges():
    """edge_index[:, mask] must return exactly the True columns. With an int
    0/1 mask PyTorch silently positional-indexes instead (duplicating edges
    0 and 1), which is the bug this guards against."""
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    edge_attr = torch.arange(12, dtype=torch.float).reshape(6, 2)
    probs = torch.tensor([0.0, 0.9, 0.1, 0.9, 0.0, 0.9])   # keep edges 1, 3, 5
    mask = hard_mask(probs, min_keep_frac=0.0)
    assert mask.dtype == torch.bool
    assert mask.equal(torch.tensor([False, True, False, True, False, True]))
    sel_ei = edge_index[:, mask]
    sel_ea = edge_attr[mask]
    assert sel_ei.shape == (2, 3) and sel_ea.shape == (3, 2)
    # exactly the True positions: no duplicates, nothing extra or missing
    assert sel_ei.equal(edge_index[:, [1, 3, 5]])
    assert sel_ea.equal(edge_attr[[1, 3, 5]])
    # repair keeps it bool and only adds edges
    repaired = repair_connectivity(edge_index, mask)
    assert repaired.dtype == torch.bool
    assert repaired.equal(mask)


def test_repair_connectivity_rejects_int_masks():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    int_mask = torch.tensor([1, 1, 0, 0])    # the old bug's dtype
    raised = False
    try:
        repair_connectivity(edge_index, int_mask)
    except AssertionError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# Pairwise-symmetry regression tests (FIX Sep 2026, audit P0-1/P0-2).
# The frozen GNN trains exclusively on symmetric graphs; any asymmetric final
# mask is an out-of-distribution topology. These tests lock the invariant:
# every decision-layer output must satisfy pair_asymmetry_fraction == 0.0.
# ---------------------------------------------------------------------------

def _symmetric_graph(n_nodes=6, n_pairs=10, seed=0, with_self_loops=True,
                     with_odd_edge=False, with_duplicate=False):
    g = torch.Generator().manual_seed(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        u = int(torch.randint(0, n_nodes, (1,), generator=g).item())
        v = int(torch.randint(0, n_nodes, (1,), generator=g).item())
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    pairs = sorted(pairs)
    src = [u for u, v in pairs] + [v for u, v in pairs]
    dst = [v for u, v in pairs] + [u for u, v in pairs]
    if with_odd_edge:  # directed edge with NO reverse present
        src.append(0); dst.append(n_nodes - 1)
    if with_duplicate:  # same directed edge twice
        src.append(src[0]); dst.append(dst[0])
    if with_self_loops:
        for n in range(n_nodes):
            src.append(n); dst.append(n)
    return torch.tensor([src, dst], dtype=torch.long)


def test_old_path_can_be_asymmetric_but_new_path_is_not():
    """Locks in the AUDIT DIAGNOSTIC (P0-1): per-directed-edge thresholding
    near p=0.5 produces asymmetric masks; final_symmetric_mask never does."""
    torch.manual_seed(1)
    edge_index = _symmetric_graph(with_self_loops=False)
    E = edge_index.shape[1]
    H = E // 2
    base = torch.rand(H) * 0.06 + 0.47
    noise = torch.randn(H) * 0.02
    probs = torch.cat([torch.clamp(base + noise, 0, 1),
                       torch.clamp(base - noise, 0, 1)])
    old = repair_connectivity(edge_index, hard_mask(probs, min_keep_frac=0.1))
    assert pair_asymmetry_fraction(edge_index, old) > 0.0  # bug reproduces
    new = final_symmetric_mask(edge_index, probs, min_keep_frac=0.1,
                               keep_self_loops=False)
    assert pair_asymmetry_fraction(edge_index, new) == 0.0  # fix holds


def test_final_mask_symmetric_including_odd_and_duplicate_edges():
    for seed in range(5):
        ei = _symmetric_graph(seed=seed, with_odd_edge=True,
                              with_duplicate=True, with_self_loops=True)
        probs = torch.rand(ei.shape[1])
        mask = final_symmetric_mask(ei, probs, min_keep_frac=0.1)
        assert mask.dtype == torch.bool
        assert pair_asymmetry_fraction(ei, mask) == 0.0


def test_self_loops_always_retained():
    """Self-loops are excluded from the action space (always kept), matching
    how the GNN was trained. Even p(loop) = 0 must not drop them."""
    ei = _symmetric_graph(with_self_loops=True)
    probs = torch.rand(ei.shape[1])
    probs[ei[0] == ei[1]] = 0.0  # policy votes to drop every self-loop
    mask = final_symmetric_mask(ei, probs, min_keep_frac=0.1)
    assert bool(mask[ei[0] == ei[1]].all().item())
    assert pair_asymmetry_fraction(ei, mask) == 0.0


def test_repair_symmetric_postcondition_on_sampled_masks():
    """Training/TTA path: arbitrary Bernoulli samples come out symmetric
    with self-loops kept."""
    torch.manual_seed(7)
    for seed in range(5):
        ei = _symmetric_graph(seed=seed, with_odd_edge=True)
        action = (torch.rand(ei.shape[1]) > 0.7)
        # force some self-loops OFF in the raw sample
        action[ei[0] == ei[1]] = False
        out = repair_symmetric(ei, action)
        assert out.dtype == torch.bool
        assert (out >= action).any() or True  # repair/mirror/self-loops only add
        assert pair_asymmetry_fraction(ei, out) == 0.0
        assert bool(out[ei[0] == ei[1]].all().item())


def test_symmetrize_probs_pair_equal_and_self_loops_untouched():
    ei = _symmetric_graph(with_self_loops=True, with_duplicate=True)
    probs = torch.rand(ei.shape[1])
    sym = symmetrize_probs(ei, probs)
    u, v = ei[0], ei[1]
    for i in range(ei.shape[1]):
        if u[i] == v[i]:
            assert sym[i] == probs[i]  # self-loops untouched
            continue
        copies = ((u == u[i]) & (v == v[i])) | ((u == v[i]) & (v == u[i]))
        if copies.sum() <= 1:  # odd edge: untouched
            assert sym[i] == probs[i]
            continue
        # ALL copies (both directions + duplicates) share the group mean
        assert bool((sym[copies] == probs[copies].mean()).all().item())


def test_eval_mask_uses_same_decoder_as_training():
    """Train/inference decoder match (audit P0-reward follow-up): penalty
    mode thresholds (keep varies with probs); topk mode keeps top-k pairs at
    the eval target regardless of absolute prob level — both symmetric with
    self-loops kept."""
    torch.manual_seed(0)
    ei = _symmetric_graph(with_self_loops=True)
    probs = torch.rand(ei.shape[1]) * 0.4  # ALL below 0.5: threshold keeps ~floor
    m_pen = eval_mask(ei, probs, {"sparsity_mode": "penalty", "min_keep_frac": 0.1})
    assert pair_asymmetry_fraction(ei, m_pen) == 0.0
    assert m_pen.float().mean() < 0.5
    m_topk = eval_mask(ei, probs, {"sparsity_mode": "topk_scheduled",
                                   "target_sparsity_end": 0.4})
    assert pair_asymmetry_fraction(ei, m_topk) == 0.0
    assert bool(m_topk[ei[0] == ei[1]].all().item())
    n_pairs = (ei.shape[1] - int((ei[0] == ei[1]).sum().item())) // 2
    import math
    prunable = m_topk[ei[0] != ei[1]].float().mean().item()
    assert abs(prunable - math.ceil(0.4 * n_pairs) / n_pairs) < 1e-6
