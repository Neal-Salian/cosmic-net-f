import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from rls.policy import EdgePolicyNet


def make_cfg():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("rls", {})
    return cfg


def test_policy_output_shapes():
    torch.manual_seed(0)
    cfg = make_cfg()
    N, E, D = 8, 20, 5
    emb_dim = cfg["model"]["output_dim"]
    net = EdgePolicyNet(edge_dim=D, node_emb_dim=emb_dim, hidden_dim=64)
    edge_attr = torch.randn(E, D)
    node_emb = torch.randn(N, emb_dim)
    edge_index = torch.randint(0, N, (2, E))
    ctx = torch.randn(emb_dim)
    logits = net(edge_attr, node_emb, edge_index, ctx)
    assert logits.shape == (E, 1)
    p = torch.sigmoid(logits)
    assert (p > 0).all() and (p < 1).all()


def test_policy_equivariant_to_node_ordering_within_edge():
    torch.manual_seed(0)
    cfg = make_cfg()
    net = EdgePolicyNet(edge_dim=5, node_emb_dim=128, hidden_dim=64)
    N, E = 8, 6
    edge_attr = torch.randn(E, 5)
    node_emb = torch.randn(N, 128)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    ctx = torch.randn(128)
    l1 = net(edge_attr, node_emb, edge_index, ctx)
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    l2 = net(edge_attr[perm], node_emb, edge_index[:, perm], ctx)
    assert torch.allclose(l1[perm], l2, atol=1e-5)
