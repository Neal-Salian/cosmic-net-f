import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from rls.rewards import compute_rewards, virial_ratio_pruned, virial_penalty


def test_relative_rmse_reward_sign():
    y = torch.tensor([10.0, 11.0, 12.0])
    pred_full = torch.tensor([10.3, 10.8, 12.2])
    pred_pruned_better = torch.tensor([10.1, 10.9, 12.1])
    pred_pruned_worse = torch.tensor([10.8, 10.2, 12.7])
    r_better = compute_rewards(pred_pruned_better, pred_full, y,
                               keep_ratio=0.5, target_sparsity=0.5,
                               virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                        "w_conn": 1.0, "w_virial": 1.0},
                               connectivity_ok=True)
    r_worse = compute_rewards(pred_pruned_worse, pred_full, y,
                              keep_ratio=0.5, target_sparsity=0.5,
                              virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                       "w_conn": 1.0, "w_virial": 1.0},
                              connectivity_ok=True)
    assert r_better > 0
    assert r_worse < 0


def test_sparsity_curriculum_guides_keeps():
    # At target_sparsity=0.5, keeping 0.5 of edges should give sparsity term 0;
    # keeping 1.0 should give a negative term.
    r_at_target = compute_rewards(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                                  keep_ratio=0.5, target_sparsity=0.5,
                                  virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                           "w_conn": 1.0, "w_virial": 1.0},
                                  connectivity_ok=True)
    r_keeps_all = compute_rewards(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                                  keep_ratio=1.0, target_sparsity=0.5,
                                  virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                           "w_conn": 1.0, "w_virial": 1.0},
                                  connectivity_ok=True)
    assert r_at_target > r_keeps_all


def test_virial_penalty():
    assert virial_ratio_pruned(ke_retained=2.0, pe_retained=0.5) == 8.0  # ratio
    p_bad = virial_penalty(ke_retained=2.0, pe_retained=0.5)   # ratio 8.0 -> penalty > 0
    p_good = virial_penalty(ke_retained=1.0, pe_retained=2.0)  # ratio 1.0 -> penalty 0
    assert p_bad > 0 and p_good == 0


def test_connectivity_penalty():
    r_conn = compute_rewards(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                             keep_ratio=0.5, target_sparsity=0.5, virial_penalty=0.0,
                             cfg={"w_acc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 1.0},
                             connectivity_ok=False)
    assert r_conn < 0
