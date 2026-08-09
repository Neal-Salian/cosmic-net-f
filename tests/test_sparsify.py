import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.sparsify import hard_mask, repair_connectivity


def test_hard_mask_respects_min_keep():
    probs = torch.full((10,), 0.05)          # wants to drop almost everything
    mask = hard_mask(probs, min_keep_frac=0.2)
    assert mask.sum() >= 2                   # >= 0.2*10
    # keeps the highest-probability edges
    assert mask[probs.argmax()] == 1


def test_repair_connectivity_no_isolated_nodes():
    N, E = 5, 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # path graph
    mask = torch.tensor([1, 1, 0, 0])        # nodes 3 and 4 lose all edges
    repaired = repair_connectivity(edge_index, mask)
    assert (repaired >= mask).all()          # repair only ADDS edges
    # every node must have degree >= 1
    deg = torch.zeros(N)
    for i in range(repaired.shape[0]):
        if repaired[i]:
            deg[edge_index[0, i]] += 1
            deg[edge_index[1, i]] += 1
    assert (deg >= 1).all()
