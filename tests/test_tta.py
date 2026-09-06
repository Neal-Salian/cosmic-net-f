import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.tta import adapt_at_test_time
from rls.policy import EdgePolicyNet


class MockGNN(torch.nn.Module):
    """Frozen stand-in with predict_with_uncertainty: std shrinks when the
    graph keeps only its shortest edges (encodes the expected direction)."""
    def predict_with_uncertainty(self, batch, n_samples=15):
        E = batch.edge_index.shape[1]
        std = torch.tensor([1.0 / max(1, E)])   # more edges -> lower std
        return {"mean": torch.zeros(1), "std": std}


def _graph():
    torch.manual_seed(0)
    n, e = 6, 12
    return {"x": torch.randn(n, 4), "edge_index": torch.randint(0, n, (2, e)),
            "edge_attr": torch.randn(e, 5), "y": torch.randn(1),
            "ctx": torch.randn(128), "emb": torch.randn(n, 128),
            "stellar_mass": torch.rand(n) * 1e10,
            "vel_disp": torch.rand(n) * 200, "half_mass_r": torch.rand(n) * 0.01,
            "pos": torch.randn(n, 3)}


def test_tta_returns_valid_mask_and_history():
    cfg = {"tta_lr": 1e-3, "tta_steps": 5, "tta_mc_samples": 5, "tta_patience": 5,
           "min_keep_frac": 0.1, "entropy_coef": 0.01,
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0}
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=128, hidden_dim=32)
    g = _graph()
    mask, info = adapt_at_test_time(policy, g, MockGNN(), cfg, device="cpu")
    assert mask.shape[0] == g["edge_index"].shape[1]
    assert mask.sum() >= 1
    assert len(info["reward_hist"]) >= 1


def test_tta_does_not_modify_input_policy():
    cfg = {"tta_lr": 1e-2, "tta_steps": 5, "tta_mc_samples": 5, "tta_patience": 5,
           "min_keep_frac": 0.1, "entropy_coef": 0.01,
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0}
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=128, hidden_dim=32)
    before = copy.deepcopy(policy.state_dict())
    adapt_at_test_time(policy, _graph(), MockGNN(), cfg, device="cpu")
    for k, v in policy.state_dict().items():
        assert torch.allclose(v, before[k]), "TTA mutated the offline policy!"


# ---------------------------------------------------------------------------
# TTA hardening (FIX Sep 2026, audit P1-TTA).
# ---------------------------------------------------------------------------

def test_edge_kl_properties():
    from rls.tta import edge_kl
    p = torch.tensor([0.2, 0.5, 0.8])
    assert float(edge_kl(p, p)) == 0.0                    # identical -> zero
    assert float(edge_kl(p, torch.full_like(p, 0.5))) > 0  # different -> positive
    kl = edge_kl(torch.tensor([0.99]), torch.tensor([0.01]))
    assert torch.isfinite(kl)                             # saturated but finite


def test_tta_should_enable_fail_closed():
    from rls.tta import tta_should_enable
    assert tta_should_enable(0.18, 0.18) is True           # tie: enabled
    assert tta_should_enable(0.18, 0.183) is True          # within tol
    assert tta_should_enable(0.18, 0.50) is False          # 2.8x worse: closed
    assert tta_should_enable(0.18, 0.19, tol=0.10) is True  # looser tol


def test_tta_step_diag_records_reward_decomposition():
    """Per-step diagnostics expose d_unc/sparsity/conn/virial separately plus
    diagnostic RMSE (labels, never in reward) so reward-hacking is visible."""
    from rls.tta import adapt_at_test_time as adapt
    cfg = {"tta_lr": 1e-3, "tta_steps": 4, "tta_mc_samples": 3, "tta_patience": 10,
           "min_keep_frac": 0.1, "entropy_coef": 0.01,
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0}
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=128, hidden_dim=32)
    g = _graph()

    class MockFullGNN(MockGNN):
        def __call__(self, batch):
            E = batch.edge_index.shape[1]
            return torch.tensor([1.0 / max(1, E)]), None

    mask, info = adapt(policy, g, MockFullGNN(), cfg, device="cpu",
                       diag_targets=g["y"])
    assert "step_diag" in info and len(info["step_diag"]) == len(info["reward_hist"])
    for d in info["step_diag"]:
        assert set(d) >= {"d_unc", "keep", "sparsity_term", "conn_ok",
                          "virial", "kl_to_offline", "rmse_pruned", "rmse_frozen"}
    # without labels: no RMSE keys, decomposition still present
    _, info2 = adapt(policy, g, MockGNN(), cfg, device="cpu")
    assert "rmse_pruned" not in info2["step_diag"][0]
    assert "d_unc" in info2["step_diag"][0]


def test_tta_trust_region_constrains_drift():
    """High tta_kl_coef must keep adapted probs closer to offline than
    unregularized adaptation (same seed => same init and samples)."""
    from rls.tta import adapt_at_test_time as adapt
    base = {"tta_lr": 5e-3, "tta_steps": 6, "tta_mc_samples": 3, "tta_patience": 10,
            "min_keep_frac": 0.1, "entropy_coef": 0.0,
            "w_unc": 1.0, "w_sp": 0.0, "w_conn": 0.0, "w_virial": 0.0,
            "tta_lr_decay": 1.0}

    def mean_kl(kl_coef):
        torch.manual_seed(11)
        policy = EdgePolicyNet(edge_dim=5, node_emb_dim=128, hidden_dim=32)
        torch.manual_seed(11)  # same action samples in both runs
        _, info = adapt(policy, _graph(), MockGNN(), dict(base, tta_kl_coef=kl_coef),
                        device="cpu")
        assert all(torch.isfinite(torch.tensor(d["kl_to_offline"]))
                   for d in info["step_diag"])
        assert info["step_diag"][0]["kl_to_offline"] == 0.0  # step 0 == offline
        return float(torch.tensor([d["kl_to_offline"]
                                   for d in info["step_diag"]]).mean())

    assert mean_kl(50.0) <= mean_kl(0.0)
