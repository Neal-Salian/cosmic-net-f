"""Runs the full evaluation suite and writes paper-ready artifacts:
outputs/rls/results_table.csv + outputs/rls/*.png"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _r2(a, b):
    a, b = a.float(), b.float()
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((b - b.mean()) ** 2).sum()
    return float((1 - ss_res / ss_tot.clamp(min=1e-8)).item())


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def _scatter(a, b):
    return float((a - b).std().item())


def build_results_table(preds_full, preds_policy, preds_gumbel, preds_random,
                        targets, masks_policy, masks_gumbel, masks_random,
                        preds_degree=None, preds_distance=None,
                        masks_degree=None, masks_distance=None):
    """All preds/targets are 1D tensors; masks are lists of bool tensors."""
    rows = []
    entries = [
        ("full", preds_full, None),
        ("random", preds_random, masks_random),
        ("degree", preds_degree, masks_degree),
        ("distance", preds_distance, masks_distance),
        ("gumbel", preds_gumbel, masks_gumbel),
        ("rl_policy", preds_policy, masks_policy),
    ]
    for name, preds, masks in entries:
        if preds is None:
            continue
        preds = preds.float()
        keep = None
        if masks is not None:
            keep = float(torch.stack([m.float().mean() for m in masks]).mean())
        rows.append({
            "method": name,
            "rmse": _rmse(preds, targets),
            "r2": _r2(preds, targets),
            "scatter": _scatter(preds, targets),
            "mean_keep_frac": keep if keep is not None else 1.0,
            "fidelity": float(np.corrcoef(preds.numpy(), preds_full.numpy())[0, 1]),
        })
    return rows


def save_paper_plots(rows, out_dir="outputs/rls"):
    """Sparsity-accuracy Pareto plot (RMSE vs keep-fraction) + fidelity bar."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    # Pareto: lower keep-fraction is better for sparsity; lower RMSE better.
    order = ["full", "random", "degree", "distance", "mass_ratio",
             "grad_saliency", "pgexplainer", "gumbel", "rl_policy"]
    for name in order:
        sub = df[df.method == name]
        if sub.empty:
            continue
        ax[0].plot(sub["mean_keep_frac"], sub["rmse"], "o-", label=name)
    ax[0].set_xlabel("mean keep fraction")
    ax[0].set_ylabel("RMSE (dex)")
    ax[0].set_xlim(1.05, -0.05)
    ax[0].legend()
    ax[1].bar(df.method, df.fidelity, color="steelblue")
    ax[1].axhline(0.98, color="r", ls="--", label="0.98 fidelity")
    ax[1].set_ylabel("fidelity (Pearson)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_and_fidelity.png"), dpi=200)
    plt.close(fig)
    df.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    return df
