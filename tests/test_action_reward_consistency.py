"""Phase 2 regression tests: the reward-time mask must derive from the SAMPLED
action — the same tensor whose log-prob feeds the policy gradient — not from a
fresh deterministic threshold of probs (the bug that broke REINFORCE credit
assignment: every rollout on a graph got an identical reward regardless of the
action actually sampled)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from rls.policy import EdgePolicyNet
from rls.policy_gradient import (PolicyGradientTrainer, ValueNet,
                                 bernoulli_logp, sample_actions)
from rls.sparsify import apply_min_keep_floor, repair_connectivity, hard_mask

# cfg keys actually consumed by train_policy/_update_batch + compute_rewards
CFG = {"batch_size": 2, "min_keep_frac": 0.1, "entropy_coef": 0.01,
       "value_coef": 0.5, "w_acc": 1.0, "w_sp": 0.5, "w_conn": 1.0,
       "w_virial": 0.0, "target_sparsity_start": 0.9,
       "target_sparsity_end": 0.9, "sparsity_anneal_epochs": 1}


def _probs(n=20, seed=0):
    torch.manual_seed(seed)
    return torch.rand(n) * 0.98 + 0.01


def _path_edges(n):
    src = torch.arange(n - 1)
    return torch.stack([src, src + 1])


def _tiny_graph(out_dim=64, n=6, e=12, seed=0):
    torch.manual_seed(seed)
    return {"x": torch.randn(n, 4), "edge_index": torch.randint(0, n, (2, e)),
            "edge_attr": torch.randn(e, 5), "y": torch.randn(1),
            "ctx": torch.randn(out_dim), "emb": torch.randn(n, out_dim),
            "stellar_mass": torch.rand(n) * 1e10, "vel_disp": torch.rand(n) * 200,
            "half_mass_r": torch.rand(n) * 0.01, "pos": torch.randn(n, 3)}


def test_two_actions_two_reward_masks():
    """Two different sampled actions on the same probs must give two different
    reward-time masks — each the floor/repair of ITS action, never the shared
    probs-threshold the old bug produced."""
    probs = _probs()
    torch.manual_seed(1); a1 = sample_actions(probs)
    torch.manual_seed(2); a2 = sample_actions(probs)
    assert not torch.equal(a1, a2)
    m1 = apply_min_keep_floor(a1.bool(), probs, 0.1)
    m2 = apply_min_keep_floor(a2.bool(), probs, 0.1)
    assert not torch.equal(m1, m2)                      # mask follows the sample
    thresh = hard_mask(probs, 0.1)                      # old (buggy) mask source
    assert not (m1.equal(thresh) and m2.equal(thresh))  # not a probs-threshold
    # what the training loop executes per rollout: repair(floor(action))
    ei = _path_edges(probs.numel())
    r1 = repair_connectivity(ei, m1)
    r2 = repair_connectivity(ei, m2)
    assert r1.dtype == torch.bool and r2.dtype == torch.bool
    assert not torch.equal(r1, r2)


def test_train_loop_mask_derives_from_the_sampled_action():
    """Spy on the live train loop: the exact action tensor whose logp enters
    the loss is the tensor passed to apply_min_keep_floor, and the mask handed
    to the GNN/reward equals repair(floor(THAT action))."""
    import importlib
    tp = importlib.import_module("rls.train_policy")  # not `import rls.train_policy as tp`:
    # the package re-exports the train_policy FUNCTION, shadowing the submodule
    torch.manual_seed(0)
    graphs = [_tiny_graph(seed=i) for i in range(4)]
    cfg = dict(CFG)
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=64, hidden_dim=16)
    value_net = ValueNet(64, hidden_dim=16)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, cfg)

    real_sample, real_floor = tp.sample_actions, tp.apply_min_keep_floor
    rollouts = []

    def spy_sample(p):
        a = real_sample(p)
        rollouts.append({"action": a.detach().clone(), "probs": p.detach().clone()})
        return a

    def spy_floor(mask, probs, min_keep_frac=0.1):
        rec = rollouts[-1]
        rec["floor_args"] = (mask.detach().clone(), probs.detach().clone())
        return real_floor(mask, probs, min_keep_frac)

    def spy_gnns(graph, mask):
        rec = rollouts[-1]
        rec["edge_index"] = graph["edge_index"].detach().clone()
        rec["reward_mask"] = mask.detach().clone()
        return graph["y"].view(-1), graph["y"].view(-1)  # constant preds: not under test

    tp.sample_actions = spy_sample
    tp.apply_min_keep_floor = spy_floor
    try:
        tp.train_policy(trainer, graphs, spy_gnns, cfg, device="cpu", epochs=1)
    finally:
        tp.sample_actions, tp.apply_min_keep_floor = real_sample, real_floor

    assert rollouts, "training loop never ran"
    for rec in rollouts:
        fa, fp = rec["floor_args"]
        assert fa.equal(rec["action"].bool()), "floor did not receive the sampled action"
        assert fp.equal(rec["probs"])
        expected = repair_connectivity(
            rec["edge_index"],
            apply_min_keep_floor(rec["action"].bool(), rec["probs"], 0.1))
        assert rec["reward_mask"].equal(expected), "reward mask is not repair(floor(action))"
    # at least one rollout's reward mask must differ from the deterministic
    # probs-threshold (the old buggy behavior gave the same mask every time)
    assert any(not rec["reward_mask"].equal(hard_mask(rec["probs"], 0.1))
               for rec in rollouts)


def test_repair_connectivity_repairs_action_derived_masks():
    """An action-derived mask that isolates a node still gets repaired: every
    node keeps >= 1 edge, and repair only ADDS edges."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # path graph
    probs = torch.tensor([0.9, 0.9, 0.1, 0.1])
    action = torch.tensor([1.0, 1.0, 0.0, 0.0])              # isolates nodes 3, 4
    m = apply_min_keep_floor(action.bool(), probs, min_keep_frac=0.0)
    assert m.equal(torch.tensor([True, True, False, False]))
    repaired = repair_connectivity(edge_index, m)
    assert repaired.dtype == torch.bool
    assert (repaired & ~m).equal(torch.tensor([False, False, True, True]))
    deg = torch.zeros(5)
    for i in range(repaired.shape[0]):
        if repaired[i]:
            deg[edge_index[0, i]] += 1
            deg[edge_index[1, i]] += 1
    assert (deg >= 1).all()


def test_policy_learns_to_drop_the_irrelevant_edge():
    """Overfit sanity check: 12 synthetic graphs, each with one clearly
    irrelevant edge (mass_ratio ~0.001, huge distance — feature idx 3 / 0) and
    a stub GNN whose pruned prediction degrades iff that edge is kept. The
    fixed credit assignment must move the policy's keep-probability for that
    edge DOWN. With the old bug (mask from re-thresholded probs) the sampled
    action never influenced the reward, so this signal could not be learned."""
    torch.manual_seed(0)
    n_graphs, n_nodes = 12, 6
    bad = 9  # last edge index is the irrelevant one
    graphs = []
    for gi in range(n_graphs):
        torch.manual_seed(100 + gi)
        path = torch.stack([torch.arange(n_nodes - 1), torch.arange(1, n_nodes)])
        extra = torch.randint(0, n_nodes, (2, 4))
        ei = torch.cat([path, extra, torch.tensor([[0], [n_nodes - 1]])], dim=1)
        e = ei.shape[1]
        ea = torch.rand(e, 5) * 0.1 + 0.8          # normal edges: high mass_ratio
        ea[:, 0] = 0.5                              # short distances
        ea[bad] = torch.tensor([50.0, 0.0, 0.0, 0.001, 50.0])  # irrelevant edge
        graphs.append({"x": torch.randn(n_nodes, 4), "edge_index": ei,
                       "edge_attr": ea, "y": torch.randn(1) * 0.1,
                       "ctx": torch.randn(64), "emb": torch.randn(n_nodes, 64),
                       "stellar_mass": torch.rand(n_nodes) * 1e10,
                       "vel_disp": torch.rand(n_nodes) * 200,
                       "half_mass_r": torch.rand(n_nodes) * 0.01,
                       "pos": torch.randn(n_nodes, 3)})

    def stub_gnns(graph, mask):
        y = graph["y"]
        kept_bad = float(mask[bad])
        pred_full = y + 0.10                        # full graph: fixed baseline error
        pred_pruned = y + 0.02 + 0.30 * kept_bad    # keeping the bad edge hurts
        return pred_full.view(-1), pred_pruned.view(-1)

    cfg = dict(CFG, batch_size=4, lr=3e-3, entropy_coef=0.001,
               target_sparsity_start=0.9, target_sparsity_end=0.9)  # keep 9/10: drop 1
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=64, hidden_dim=32)
    value_net = ValueNet(64, hidden_dim=32)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    vopt = torch.optim.Adam(value_net.parameters(), lr=cfg["lr"])
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, cfg)

    def keep_probs():
        policy.eval()
        with torch.no_grad():
            ps = [torch.sigmoid(policy(g["edge_attr"], g["emb"], g["edge_index"],
                                       g["ctx"])).squeeze(-1) for g in graphs]
        p_bad = float(torch.stack([p[bad] for p in ps]).mean())
        p_norm = float(torch.stack([torch.cat([p[:bad], p[bad + 1:]]).mean()
                                    for p in ps]).mean())
        return p_bad, p_norm

    before_bad, before_norm = keep_probs()
    train_policy = __import__("rls.train_policy", fromlist=["train_policy"]).train_policy
    losses = train_policy(trainer, graphs, stub_gnns, cfg, device="cpu",
                          epochs=15)                # 12 graphs / batch 4 = 3 steps/epoch -> 45 steps
    after_bad, after_norm = keep_probs()
    print(f"overfit sanity: keep-prob(irrelevant edge) {before_bad:.4f} -> {after_bad:.4f}; "
          f"keep-prob(normal edges) {before_norm:.4f} -> {after_norm:.4f}")
    assert all(torch.isfinite(torch.as_tensor(l)) for l in losses)
    assert after_bad < before_bad, (
        f"policy did not learn to drop the irrelevant edge: {before_bad:.4f} -> {after_bad:.4f}")


def test_tta_reward_mask_derives_from_sampled_action():
    """Same guarantee inside the TTA adaptation loop: per-step reward masks come
    from the sampled action; only the FINAL mask (after the loop) stays a
    deterministic threshold (hard_mask called exactly once)."""
    import rls.tta as tta_mod
    from rls.tta import adapt_at_test_time

    class MockGNN(torch.nn.Module):
        def predict_with_uncertainty(self, batch, n_samples=15):
            return {"mean": torch.zeros(1),
                    "std": torch.tensor([1.0 / max(1, batch.edge_index.shape[1])])}

    cfg = {"tta_lr": 1e-3, "tta_steps": 3, "tta_mc_samples": 2, "tta_patience": 10,
           "min_keep_frac": 0.1, "entropy_coef": 0.01,
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0}
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=64, hidden_dim=16)
    g = _tiny_graph()

    real_sample, real_floor, real_hard = (tta_mod.sample_actions,
                                          tta_mod.apply_min_keep_floor,
                                          tta_mod.hard_mask)
    calls = {"sample": [], "floor": [], "hard": 0}

    def spy_sample(p):
        a = real_sample(p); calls["sample"].append(a.detach().clone()); return a

    def spy_floor(mask, probs, min_keep_frac=0.1):
        calls["floor"].append(mask.detach().clone())
        return real_floor(mask, probs, min_keep_frac)

    def spy_hard(probs, min_keep_frac=0.1):
        calls["hard"] += 1
        return real_hard(probs, min_keep_frac)

    tta_mod.sample_actions, tta_mod.apply_min_keep_floor, tta_mod.hard_mask = (
        spy_sample, spy_floor, spy_hard)
    try:
        mask, info = adapt_at_test_time(policy, g, MockGNN(), cfg, device="cpu")
    finally:
        tta_mod.sample_actions = real_sample
        tta_mod.apply_min_keep_floor = real_floor
        tta_mod.hard_mask = real_hard

    assert calls["floor"], "TTA loop never called apply_min_keep_floor"
    assert len(calls["floor"]) == len(calls["sample"])
    for f, s in zip(calls["floor"], calls["sample"]):
        assert f.equal(s.bool()), "TTA reward mask did not come from the sampled action"
    assert calls["hard"] == 1, "final mask must be the single deterministic hard_mask(p)"
    assert mask.dtype == torch.bool and mask.shape[0] == g["edge_index"].shape[1]
