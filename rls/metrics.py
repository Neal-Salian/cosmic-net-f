"""Evaluation metrics for the paper (interpretability + physics)."""
import torch


def fidelity(pred_pruned, pred_full):
    """Pearson correlation between pruned and full-graph predictions."""
    a, b = pred_pruned.float(), pred_full.float()
    if a.numel() < 2:
        return float("nan")
    ca, cb = a - a.mean(), b - b.mean()
    denom = (ca.norm() * cb.norm()).clamp(min=1e-8)
    return float((ca @ cb / denom).item())


def stability_jaccard(mask1, mask2):
    inter = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    return float((inter / union.clamp(min=1e-8)).item())


def sparsity_stats(masks):
    keep = [float(m.float().mean()) for m in masks]
    return {"mean_keep_frac": float(torch.tensor(keep).mean()),
            "std_keep_frac": float(torch.tensor(keep).std())}


def physics_alignment(keep_prob, u_ij):
    """Spearman rank correlation between edge keep-probability and the
    gravitational binding-energy proxy U_ij = m_i*m_j/r_ij."""
    p, u = keep_prob.detach().cpu().numpy(), u_ij.detach().cpu().numpy()
    import numpy as np
    from scipy.stats import spearmanr
    rho, pval = spearmanr(p, u)
    return float(rho), float(pval)


def calibration_coverage(pred, std, y):
    """95% CI coverage under Gaussian MC-dropout uncertainty."""
    lo, hi = pred - 1.96 * std, pred + 1.96 * std
    return float(((y >= lo) & (y <= hi)).float().mean().item())
