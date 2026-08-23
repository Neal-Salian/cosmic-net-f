import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.sparsify import hard_mask, repair_connectivity


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
