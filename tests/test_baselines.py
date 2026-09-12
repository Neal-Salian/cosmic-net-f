import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pytest
from rls.baselines import (random_mask, degree_mask, distance_mask,
                           mass_ratio_mask, attention_topk_mask)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mask_fn", [random_mask, degree_mask, distance_mask,
                                    mass_ratio_mask, attention_topk_mask])
def test_masks_stay_on_graph_device(device, mask_fn):
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], device=device)
    edge_attr = torch.arange(30, dtype=torch.float32, device=device).reshape(6, 5)
    pos = torch.arange(9, dtype=torch.float32, device=device).reshape(3, 3)
    kwargs = {}
    if mask_fn is attention_topk_mask:
        kwargs["model"] = torch.nn.Linear(5, 1).to(device)
    mask = mask_fn(edge_index, edge_attr, pos, 0.5, **kwargs)
    assert mask.device == edge_index.device
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 3
    assert edge_index[:, mask].shape == (2, 3)
    if mask_fn is random_mask:
        expected = random_mask(edge_index.cpu(), edge_attr.cpu(), pos.cpu(), 0.5)
        assert torch.equal(mask.cpu(), expected)

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
