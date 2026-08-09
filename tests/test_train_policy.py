import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml


def test_end_to_end_policy_train_smoke():
    """Trains 2 PG epochs on 8 tiny synthetic graphs; asserts no NaNs and
    finite losses."""
    from rls.policy import build_policy
    from rls.policy_gradient import PolicyGradientTrainer, ValueNet
    from rls.train_policy import train_policy

    torch.manual_seed(0)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["rls"]["epochs"] = 2
    cfg["rls"]["batch_size"] = 8
    # tiny graphs via random tensors (policy + pg only; no GNN needed)
    graphs = [{"x": torch.randn(5, 4), "edge_index": torch.randint(0, 5, (2, 12)),
               "edge_attr": torch.randn(12, 5), "y": torch.randn(1),
               "ctx": torch.randn(128), "emb": torch.randn(5, 128),
               "stellar_mass": torch.rand(5) * 1e10, "vel_disp": torch.rand(5) * 200,
               "half_mass_r": torch.rand(5) * 0.01, "pos": torch.randn(5, 3)}
              for _ in range(8)]
    policy = build_policy(cfg)
    value_net = ValueNet(128)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, cfg["rls"])
    losses = train_policy(trainer, graphs, None, cfg["rls"], device="cpu",
                          epochs=2, log_fn=None)
    assert all(torch.isfinite(l) for l in losses)


def test_prepare_graphs_real_pyg_data():
    """The adapter MUST work on real PyG Batch objects (this is the
    integration point the plan previously shipped broken)."""
    from torch_geometric.data import Data, Batch, DataLoader
    from rls.train_policy import prepare_graphs
    from rls.policy import build_policy
    from rls.policy_gradient import ValueNet

    torch.manual_seed(0)
    datalist = []
    for _ in range(4):
        n = 5
        datalist.append(Data(
            x=torch.randn(n, 4),
            edge_index=torch.randint(0, n, (2, 10)),
            edge_attr=torch.randn(10, 5),
            y=torch.randn(1),
            stellar_mass=torch.rand(n) * 1e10,
            vel_disp=torch.rand(n) * 200,
            half_mass_r=torch.rand(n) * 0.01,
            pos=torch.randn(n, 3),
        ))
    loader = DataLoader(datalist, batch_size=2)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    from model.model import build_model
    gnn = build_model(cfg)
    graphs = prepare_graphs(loader, gnn, "cpu")
    assert len(graphs) == 4
    assert all(set(g.keys()) >= {"x", "edge_index", "edge_attr", "y", "ctx", "emb",
                                 "stellar_mass", "vel_disp", "half_mass_r", "pos"}
               for g in graphs)
    assert all(g["ctx"].shape[0] == cfg["model"]["output_dim"] for g in graphs)
    assert all(g["emb"].shape[0] == g["x"].shape[0] for g in graphs)
    assert all(g["emb"].shape[1] == cfg["model"]["output_dim"] for g in graphs)
    assert all(g["emb"].device == g["x"].device for g in graphs)
    assert all(g["edge_attr"].shape[0] == g["edge_index"].shape[1] for g in graphs)
