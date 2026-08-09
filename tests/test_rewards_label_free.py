import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.rewards import label_free_reward

def test_label_free_reward_prefers_uncertainty_reduction():
    cfg = {"w_unc": 1.0, "w_sp": 0.5, "w_conn": 0.0, "w_virial": 0.0}
    r_better = label_free_reward(std_pruned=0.05, std_full=0.10, keep_ratio=0.5,
                                 target_sparsity=0.5, virial_penalty=0.0,
                                 cfg=cfg, connectivity_ok=True)
    r_worse = label_free_reward(std_pruned=0.15, std_full=0.10, keep_ratio=0.5,
                                target_sparsity=0.5, virial_penalty=0.0,
                                cfg=cfg, connectivity_ok=True)
    assert r_better > 0 > r_worse

def test_label_free_reward_never_touches_labels():
    """The signature takes NO y/targets argument — that is the whole point."""
    import inspect
    params = inspect.signature(label_free_reward).parameters
    assert "y" not in params and "targets" not in params and "pred_full" not in params
