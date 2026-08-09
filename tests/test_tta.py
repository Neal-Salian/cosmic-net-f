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
