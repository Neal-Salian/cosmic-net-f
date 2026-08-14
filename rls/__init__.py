"""RL-Cosmic-Net: physics-informed RL graph sparsification + RL at inference.

Modules: policy (edge policy net), rewards (label + label-free), policy_gradient
(REINFORCE + baseline + entropy), sparsify (hard mask + connectivity repair),
train_policy (offline loop), stageb (GNN fine-tune), baselines, metrics,
evaluate (tables + plots + evaluate_tta), run_experiment (driver), cross_sim
(OOD), tta (per-instance test-time adaptation).
"""
from rls.policy import EdgePolicyNet, build_policy
from rls.policy_gradient import (PolicyGradientTrainer, ValueNet,
                                 compute_advantages, compute_pg_loss,
                                 bernoulli_logp, bernoulli_entropy,
                                 sample_actions)
from rls.sparsify import hard_mask, repair_connectivity
from rls.rewards import compute_rewards, virial_penalty, label_free_reward, virial_ratio_pruned
from rls.train_policy import prepare_graphs, train_policy
from rls.tta import adapt_at_test_time, mc_std
from rls.evaluate import build_results_table, save_paper_plots, evaluate_tta
from rls.stageb import fine_tune_gnn
from rls.baselines import (random_mask, degree_mask, distance_mask,
                           mass_ratio_mask, gradient_saliency_mask,
                           attention_topk_mask, GumbelEdgeMask)

__all__ = [
    "EdgePolicyNet", "build_policy",
    "PolicyGradientTrainer", "ValueNet", "compute_advantages", "compute_pg_loss",
    "bernoulli_logp", "bernoulli_entropy", "sample_actions",
    "hard_mask", "repair_connectivity",
    "compute_rewards", "virial_penalty", "label_free_reward", "virial_ratio_pruned",
    "prepare_graphs", "train_policy",
    "adapt_at_test_time", "mc_std",
    "build_results_table", "save_paper_plots", "evaluate_tta",
    "fine_tune_gnn",
    "random_mask", "degree_mask", "distance_mask", "mass_ratio_mask",
    "gradient_saliency_mask", "attention_topk_mask", "GumbelEdgeMask",
]
