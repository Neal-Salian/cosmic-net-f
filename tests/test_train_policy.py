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
    out_dim = cfg["model"]["output_dim"]
    # tiny graphs via random tensors (policy + pg only; no GNN needed)
    graphs = [{"x": torch.randn(5, 4), "edge_index": torch.randint(0, 5, (2, 12)),
               "edge_attr": torch.randn(12, 5), "y": torch.randn(1),
               "ctx": torch.randn(out_dim), "emb": torch.randn(5, out_dim),
               "stellar_mass": torch.rand(5) * 1e10, "vel_disp": torch.rand(5) * 200,
               "half_mass_r": torch.rand(5) * 0.01, "pos": torch.randn(5, 3)}
              for _ in range(8)]
    policy = build_policy(cfg)
    value_net = ValueNet(out_dim)
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


# ---------------------------------------------------------------------------
# Structural-sparsity mode + curriculum-divergence guard
# (FIX Sep 2026, audit P0-reward).
# ---------------------------------------------------------------------------

def _sym_graph(n=6, e_pairs=10, seed=0):
    import random
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < e_pairs:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    pairs = sorted(pairs)
    src = [u for u, v in pairs] + [v for u, v in pairs] + list(range(n))
    dst = [v for u, v in pairs] + [u for u, v in pairs] + list(range(n))
    return torch.tensor([src, dst], dtype=torch.long)


def test_topk_scheduled_mask_tracks_target_and_stays_symmetric():
    """Executed keep must equal the curriculum target on prunable edges
    (constraint, not nudge) and the mask must be pair-symmetric."""
    from rls.sparsify import topk_scheduled_mask, pair_asymmetry_fraction
    torch.manual_seed(0)
    for target in (0.4, 0.7):
        ei = _sym_graph()
        probs = torch.rand(ei.shape[1])
        mask = topk_scheduled_mask(ei, probs, target)
        assert mask.dtype == torch.bool
        assert pair_asymmetry_fraction(ei, mask) == 0.0
        loops = ei[0] == ei[1]
        assert bool(mask[loops].all().item())  # self-loops always kept
        prunable = mask[~loops].float().mean().item()
        n_pairs = (ei.shape[1] - int(loops.sum().item())) // 2
        # pair-level top-k keeps exactly ceil(target * n_pairs) pairs; with
        # no odd edges, prunable directed keep == pair keep up to rounding.
        import math
        expect = math.ceil(target * n_pairs) / n_pairs
        assert abs(prunable - expect) < 1e-6


def test_topk_scheduled_training_keeps_gradients_flowing():
    """Smoke: 2 epochs in topk_scheduled mode — finite losses AND nonzero
    policy grads (guards the no_grad logp bug class: re-pointed logp must be
    computed outside no_grad)."""
    from rls.policy import build_policy
    from rls.policy_gradient import PolicyGradientTrainer, ValueNet
    from rls.train_policy import train_policy
    import yaml
    torch.manual_seed(0)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    rls_cfg = dict(cfg["rls"], sparsity_mode="topk_scheduled",
                   target_sparsity_start=0.7, target_sparsity_end=0.4,
                   sparsity_anneal_epochs=2)
    out_dim = cfg["model"]["output_dim"]
    graphs = [{"x": torch.randn(5, 4), "edge_index": _sym_graph(n=5, e_pairs=8, seed=i),
               "edge_attr": torch.randn(_sym_graph(n=5, e_pairs=8, seed=i).shape[1], 5),
               "y": torch.randn(1), "ctx": torch.randn(out_dim),
               "emb": torch.randn(5, out_dim),
               "stellar_mass": torch.rand(5) * 1e10, "vel_disp": torch.rand(5) * 200,
               "half_mass_r": torch.rand(5) * 0.01, "pos": torch.randn(5, 3)}
              for i in range(4)]
    policy = build_policy(cfg)
    value_net = ValueNet(out_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, rls_cfg)
    losses = train_policy(trainer, graphs, None, rls_cfg, device="cpu", epochs=2)
    assert all(torch.isfinite(l) for l in losses)
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and any(bool((g != 0).any().item()) for g in grads)


def test_check_curriculum_divergence_flags_the_diagnosed_run():
    """The diagnosed failure (keep ~0.97 flat, target annealing 0.9 -> 0.4)
    must return True; a tracking run must return False; epochs before half
    the anneal schedule must not count."""
    from rls.train_policy import check_curriculum_divergence
    keeps = [0.97] * 60
    targets = [0.9 - 0.5 * min(1.0, e / 40) for e in range(60)]
    assert check_curriculum_divergence(keeps, targets, tol=0.1, patience=5,
                                       anneal_epochs=40) is True
    tracking = [t + 0.02 for t in targets]
    assert check_curriculum_divergence(tracking, targets, tol=0.1, patience=5,
                                       anneal_epochs=40) is False
    # divergence confined to the pre-half-anneal warmup must not fire
    early = [0.97] * 20 + [t for t in targets[20:]]
    early_targets = targets
    assert check_curriculum_divergence(early, early_targets, tol=0.1,
                                       patience=5, anneal_epochs=40) is False


def test_warn_fn_called_on_divergence():
    """The training loop surfaces divergence through warn_fn (or print)."""
    from rls.policy import build_policy
    from rls.policy_gradient import PolicyGradientTrainer, ValueNet
    from rls.train_policy import train_policy
    import yaml
    torch.manual_seed(0)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    # penalty mode with w_sp=0 and a stub GNN: nothing pushes keep down, so
    # keep stays wherever init puts it while the target anneals -> diverge.
    rls_cfg = dict(cfg["rls"], sparsity_mode="penalty", w_sp=0.0,
                   target_sparsity_start=0.9, target_sparsity_end=0.1,
                   sparsity_anneal_epochs=4, w_virial=0.0)
    out_dim = cfg["model"]["output_dim"]
    graphs = [{"x": torch.randn(5, 4), "edge_index": _sym_graph(n=5, e_pairs=8, seed=i),
               "edge_attr": torch.randn(_sym_graph(n=5, e_pairs=8, seed=i).shape[1], 5),
               "y": torch.zeros(1), "ctx": torch.randn(out_dim),
               "emb": torch.randn(5, out_dim),
               "stellar_mass": torch.rand(5) * 1e10, "vel_disp": torch.rand(5) * 200,
               "half_mass_r": torch.rand(5) * 0.01, "pos": torch.randn(5, 3)}
              for i in range(4)]

    def stub_gnns(graph, mask):
        y = graph["y"]
        return y.view(-1), y.view(-1)  # zero error everywhere: no prune pressure

    policy = build_policy(cfg)
    value_net = ValueNet(out_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, rls_cfg)
    calls = []
    train_policy(trainer, graphs, stub_gnns, rls_cfg, device="cpu", epochs=12,
                 warn_fn=lambda e, k, t: calls.append((e, k, t)))
    assert calls, "divergence guard never fired on a diverging run"
