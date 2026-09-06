"""Runs the full evaluation suite and writes paper-ready artifacts:
outputs/rls/results_table.csv + outputs/rls/*.png"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rls.sparsify import eval_mask  # mode-aware symmetric eval masks (P0-2)


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


def summarize_multiseed(rows_per_seed, n_bootstrap=1000, seed=42):
    """Multi-seed summary (audit P2, Sep 2026): rows_per_seed = list (one per
    seed) of results-row lists as produced by build_results_table (each row a
    dict with method/rmse/r2/mean_keep_frac + backbone provenance columns).
    Returns one dict per method with mean +/- std over seeds plus bootstrap
    95% CI over the pooled seed-means for rmse/r2/keep. N=82 test halos is
    small and high-variance — never report a point estimate without these."""
    import math
    rng = np.random.default_rng(seed)
    by_method = {}
    for rows in rows_per_seed:
        for r in rows:
            by_method.setdefault(r["method"], []).append(r)
    out = []
    for method, rs in sorted(by_method.items()):
        rec = {"method": method, "n_seeds": len(rs)}
        for key in ("rmse", "r2", "mean_keep_frac"):
            vals = np.array([r[key] for r in rs], dtype=float)
            rec[f"{key}_mean"] = float(vals.mean())
            rec[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            boots = rng.choice(vals, size=(n_bootstrap, len(vals)),
                               replace=True).mean(axis=1)
            rec[f"{key}_ci95_lo"] = float(np.percentile(boots, 2.5))
            rec[f"{key}_ci95_hi"] = float(np.percentile(boots, 97.5))
        # provenance: every seed's row must agree on backbone + policy artifact
        rec["backbone_stages"] = sorted({r.get("backbone_stage", "?") for r in rs})
        rec["backbone_shas"] = sorted({r.get("backbone_sha256", "?") for r in rs})
        out.append(rec)
    return out


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


def evaluate_tta(policy, graphs, gnn, cfg, device, ks=(0, 5, 10, 20),
                 inits=("offline",), target_sparsity=0.5):
    """Run TTA at several K values on every graph; labels used ONLY for the
    final RMSE (never in the reward). Returns rows for the paper table:
    {mode, K, init, rmse, fidelity, keep_frac, mean_steps, mean_time_s}."""
    import time
    from rls.tta import adapt_at_test_time
    rows = []
    for init in inits:
        for k in ks:
            preds, fulls, ys, keeps, steps, secs = [], [], [], [], [], []
            for g in graphs:
                gd = {kk: v.to(device) for kk, v in g.items()
                      if isinstance(v, torch.Tensor)}
                t0 = time.time()
                if k == 0 or init == "frozen":  # FROZEN mode: no adaptation
                    with torch.no_grad():
                        p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"],
                                                 gd["edge_index"], gd["ctx"])).squeeze(-1)
                        # FIX (audit P0-2, Sep 2026): mode-aware symmetric mask
                        # (threshold in penalty mode, top-k in topk mode).
                        mask = eval_mask(gd["edge_index"], p, cfg)
                else:
                    mask, info = adapt_at_test_time(policy, gd, gnn, cfg, device,
                                                    init=init,
                                                    target_sparsity=target_sparsity)
                    steps.append(info["steps_run"])
                # predict with frozen GNN on adapted mask
                from torch_geometric.data import Data, Batch
                d = Data(x=gd["x"], edge_index=gd["edge_index"][:, mask],
                         edge_attr=gd["edge_attr"][mask])
                with torch.no_grad():
                    pred, _ = gnn(Batch.from_data_list([d]))
                    dfull = Data(x=gd["x"], edge_index=gd["edge_index"],
                                 edge_attr=gd["edge_attr"])
                    pred_full, _ = gnn(Batch.from_data_list([dfull]))
                secs.append(time.time() - t0)
                preds.append(float(pred.view(-1)[0])); fulls.append(float(pred_full.view(-1)[0]))
                ys.append(float(gd["y"].view(-1)[0])); keeps.append(float(mask.float().mean()))
            import numpy as np
            preds, fulls, ys = np.array(preds), np.array(fulls), np.array(ys)
            rows.append({
                "mode": "frozen" if k == 0 else "tta", "K": k, "init": init,
                "rmse": float(np.sqrt(((preds - ys) ** 2).mean())),
                "fidelity": float(np.corrcoef(preds, fulls)[0, 1]) if len(preds) > 1 else float("nan"),
                "keep_frac": float(np.mean(keeps)),
                "mean_steps": float(np.mean(steps)) if steps else 0.0,
                "mean_time_s": float(np.mean(secs)),
            })
    return rows
