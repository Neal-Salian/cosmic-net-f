import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.baselines import (random_mask, degree_mask, distance_mask,
                           mass_ratio_mask, attention_topk_mask)

def test_masks_have_correct_fraction():
    torch.manual_seed(0)
    N, E = 10, 40
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 5)
    pos = torch.rand(N, 3)
    f = 0.5
    for mask_fn in [random_mask, degree_mask, distance_mask, mass_ratio_mask,
                    attention_topk_mask]:
        m = mask_fn(edge_index, edge_attr, pos, f)
        assert m.sum() == int(f * E), f"{mask_fn.__name__} fraction wrong"

def test_distance_mask_prefers_short_edges():
    N, E = 10, 40
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 5)
    pos = torch.rand(N, 3)
    m = distance_mask(edge_index, edge_attr, pos, 0.5)
    kept_dists = torch.norm(pos[edge_index[0][m]] - pos[edge_index[1][m]], dim=1)
    dropped_dists = torch.norm(pos[edge_index[0][~m]] - pos[edge_index[1][~m]], dim=1)
    assert kept_dists.mean() < dropped_dists.mean()

def test_mass_ratio_mask_prefers_high_mass_ratio():
    N, E = 10, 40
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 5)
    edge_attr[:, 3] = torch.linspace(0, 1, E)   # mass_ratio is feature index 3
    pos = torch.rand(N, 3)
    m = mass_ratio_mask(edge_index, edge_attr, pos, 0.5)
    assert edge_attr[m, 3].mean() > edge_attr[~m, 3].mean()
