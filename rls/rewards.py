"""Reward design for RL edge sparsification.

r = w_acc * dRMSE_rel + w_sp * (keep_ratio - target)^2 + w_conn * conn_bonus
    - w_virial * virial_penalty

All terms are per-graph scalars; the trainer averages over the batch.
"""
import torch


def virial_ratio_pruned(ke_retained, pe_retained, eps=1e-8):
    """Dimensionless virial ratio of the pruned graph: 2*KE_retained / |PE_retained|.

    KE_retained = 0.5 * sum_i (deg'(i)/deg(i)) * M_*,i * sigma_i^2
    PE_retained = G * sum_{(i,j) in E'} M_*,i * M_*,j / r_ij   (absolute value used)
    Returns the ratio (1.0 = virialized).
    """
    pe_retained = torch.as_tensor(pe_retained, dtype=torch.float32)
    ke_retained = torch.as_tensor(ke_retained, dtype=torch.float32)
    pe_retained = torch.clamp(torch.abs(pe_retained), min=eps)
    return (2.0 * ke_retained) / pe_retained


def virial_penalty(ke_retained, pe_retained):
    ratio = virial_ratio_pruned(ke_retained, pe_retained)
    return torch.clamp(ratio - 1.0, min=0.0) ** 2


def relative_virial_penalty(ke_pruned, pe_pruned, ke_full, pe_full,
                            eps_ratio=1e-12):
    """Relative virial penalty (approved formulation, Aug 2026):

        penalty = (log r_pruned - log r_full)^2

    Both ratios are computed over LOOP-FREE physics edges (self-loops are
    excluded upstream in _graph_physics_terms); r_full is the same graph's
    full-graph reference and is independent of the action mask. This replaces
    the absolute one-sided penalty max(r-1,0)^2, which was identically zero on
    the real pipeline (self-loop r=0 clamp dominated PE, measured ratio ~4e-5)
    and, without loops, would be dominated by the stellar-only mass systematic
    (measured median ratio 4.4-8.8 with IQR spanning decades).

    Properties: mask-sensitive, label-free, non-circular (no predictions),
    TTA-compatible, and exactly invariant to a global stellar-mass rescaling
    (mass factors cancel in the ratio of ratios).

    eps_ratio: a kept mask with zero real physics edges gives ke = pe = 0 and
    an undefined ratio; the ratio is clamped into [eps_ratio, 1/eps_ratio] so
    the penalty stays finite in a maximal band instead of NaN/inf. Stage-3
    sensitivity: eps in [1e-12, 1e-6] moves the zero-edge penalty only within
    ~[4e2, 1.3e3]; ordinary masks never reach the clamp (measured max |log r|
    ~8.7). This is the single documented numeric choice of the formulation.
    """
    def _ratio(ke, pe):
        ke = torch.as_tensor(ke, dtype=torch.float32)
        pe = torch.clamp(torch.abs(torch.as_tensor(pe, dtype=torch.float32)),
                         min=1e-30)
        return torch.clamp(2.0 * ke / pe, eps_ratio, 1.0 / eps_ratio)
    return (torch.log(_ratio(ke_pruned, pe_pruned))
            - torch.log(_ratio(ke_full, pe_full))) ** 2


def _rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def compute_rewards(pred_pruned, pred_full, targets, keep_ratio, target_sparsity,
                    virial_penalty, cfg, connectivity_ok=True):
    """Vectorized over a batch of graphs (one entry per graph)."""
    base_rmse = _rmse(pred_full, targets) + 1e-8
    pruned_rmse = _rmse(pred_pruned, targets)
    delta_rel = (base_rmse - pruned_rmse) / base_rmse          # positive = better

    sparsity_term = (keep_ratio - target_sparsity) ** 2        # 0 at target, + away from it
    sparsity_term = min(sparsity_term, 1.0)

    # Disconnected graphs get a hard penalty strong enough to dominate even a
    # +1.0 relative-RMSE improvement (keeps the reward strictly negative).
    conn_bonus = torch.where(torch.tensor(bool(connectivity_ok)), torch.ones_like(delta_rel),
                             -2.0 * torch.ones_like(delta_rel))

    r = (cfg["w_acc"] * delta_rel
         - cfg["w_sp"] * sparsity_term
         + cfg["w_conn"] * conn_bonus
         - cfg["w_virial"] * virial_penalty)
    return r


def label_free_reward(std_pruned, std_full, keep_ratio, target_sparsity,
                      virial_penalty, cfg, connectivity_ok=True):
    """Test-time adaptation reward — LABEL-FREE by design.

    r = w_unc * (std_full - std_pruned)/(std_full + eps)   # uncertainty reduction
        - w_sp  * (keep_ratio - target_sparsity)^2          # sparsity (fixed target)
        + w_conn * (+1/-1 connectivity)                     # no isolated nodes
        - w_virial * virial_penalty                         # physics (label-free)

    std_* are MC-dropout predictive stds of the frozen GNN on the full/pruned
    graph. Never pass ground-truth labels into this function — the signature
    above deliberately has no targets argument.
    """
    eps = 1e-8
    d_unc = (std_full - std_pruned) / (std_full + eps)
    sparsity_term = min((keep_ratio - target_sparsity) ** 2, 1.0)
    conn = 1.0 if connectivity_ok else -1.0
    return (cfg["w_unc"] * d_unc
            - cfg["w_sp"] * sparsity_term
            + cfg["w_conn"] * conn
            - cfg.get("w_virial", 0.0) * virial_penalty)
