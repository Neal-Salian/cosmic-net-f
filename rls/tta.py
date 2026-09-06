"""RL at inference: per-instance test-time adaptation (TTA) of the edge policy.

THE core novelty of the paper. At inference, for each input graph, a COPY of
the policy takes K policy-gradient steps against the label-free reward
(rls.rewards.label_free_reward), then emits the final hard mask. Because the
reward never touches ground-truth labels, TTA works on unlabeled OOD graphs
(CAMELS) — exactly where the frozen offline policy degrades.

Always report BOTH modes side by side:
  - FROZEN: offline policy applied directly (ablation; K=0)
  - TTA:    policy adapted per-instance at inference (the method)
"""
import copy
import torch
from torch_geometric.data import Data, Batch
from rls.policy_gradient import bernoulli_logp, bernoulli_entropy, sample_actions
from rls.sparsify import (apply_min_keep_floor, symmetrize_probs,
                          repair_symmetric, final_symmetric_mask, eval_mask)
from rls.rewards import label_free_reward, relative_virial_penalty
from rls.train_policy import _graph_physics_terms, _no_isolated


def mc_std(gnn, graph, edge_index=None, edge_attr=None, n_samples=15, device="cpu"):
    """MC-dropout predictive std of the frozen GNN on a (sub)graph."""
    d = Data(x=graph["x"],
             edge_index=edge_index if edge_index is not None else graph["edge_index"],
             edge_attr=edge_attr if edge_attr is not None else graph["edge_attr"])
    b = Batch.from_data_list([d]).to(device)
    out = gnn.predict_with_uncertainty(b, n_samples=n_samples)
    return out["std"].view(-1)


def edge_kl(p, p0, eps=1e-6):
    """Mean Bernoulli KL(p || p0) per edge — the TTA trust-region distance
    between adapted and offline edge-probabilities."""
    p = torch.clamp(p, eps, 1.0 - eps)
    p0 = torch.clamp(p0, eps, 1.0 - eps)
    return (p * torch.log(p / p0) + (1 - p) * torch.log((1 - p) / (1 - p0))).mean()


def tta_should_enable(val_frozen_rmse, val_tta_rmse, tol=0.02):
    """Fail-closed guard (audit P1-TTA, Sep 2026): TTA is enabled for a run
    only if its VAL RMSE is not worse than frozen's by more than `tol`
    (relative). Mirrors the repo's ALLOW_STALE_POLICY_FALLBACK /
    synthetic-CAMELS fail-closed conventions. Unit-tested."""
    return bool(val_tta_rmse <= val_frozen_rmse * (1.0 + tol))


def adapt_at_test_time(policy, graph, gnn, cfg, device="cpu", init="offline",
                       target_sparsity=0.5, log_fn=None, diag_targets=None):
    """Adapt the policy to ONE graph via K policy-gradient steps on the
    label-free reward. Returns (final_mask, info). NEVER mutates `policy`.

    diag_targets (optional): ground-truth labels used ONLY for per-step
    DIAGNOSTIC RMSE logging (reward-hacking visibility during development) —
    never entering the reward. When None, no diagnostic RMSE is recorded.

    Trust region (FIX Sep 2026, audit P1-TTA): loss += tta_kl_coef *
    KL(p_adapted || p_offline) with step-size decay tta_lr_decay**step, so
    adaptation cannot drift far from the offline policy that was validated.
    Tune (K, tta_lr, w_unc) on the VAL split only.
    """
    if init == "offline":
        pol = copy.deepcopy(policy).to(device)
    else:  # fresh init (ablation)
        torch.manual_seed(cfg.get("seed", 42))
        pol = type(policy)(edge_dim=graph["edge_attr"].shape[1],
                           node_emb_dim=graph["emb"].shape[1]).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=cfg["tta_lr"])
    g = {k: v.to(device) for k, v in graph.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        std_full = mc_std(gnn, g, n_samples=cfg["tta_mc_samples"], device=device)
        # Relative virial (approved Aug 2026): loop-free full-graph reference,
        # computed ONCE per graph — constant with respect to the TTA actions.
        ke_full, pe_full = _graph_physics_terms(
            g, g["edge_index"],
            torch.ones(g["edge_index"].shape[1], dtype=torch.bool, device=device))

    baseline, best_r, stall = 0.0, -1e9, 0
    hist = []
    diag = []  # per-step reward diagnostics (audit P1-TTA: hacking visibility)
    p_offline = None
    lr0 = cfg["tta_lr"]
    for step in range(cfg["tta_steps"]):
        # Step-size decay (trust region, part 2): shrink the update radius.
        for pg in opt.param_groups:
            pg["lr"] = lr0 * (float(cfg.get("tta_lr_decay", 1.0)) ** step)
        probs = torch.sigmoid(pol(g["edge_attr"], g["emb"],
                                  g["edge_index"], g["ctx"])).squeeze(-1)
        # FIX (audit P0-2, Sep 2026): same symmetrization as offline training
        # — pair-averaged probs, symmetric executed mask (see train_policy).
        probs = symmetrize_probs(g["edge_index"], probs)
        if p_offline is None:
            # Step-0 adapted probs == offline probs (pol is a fresh deepcopy
            # of the offline policy); capturing here is device-safe, unlike
            # re-running the (possibly CPU-resident) offline policy on
            # device tensors.
            p_offline = probs.detach().clone()
        action = sample_actions(probs)
        logp = bernoulli_logp(probs, action).mean()
        ent = bernoulli_entropy(probs).mean()
        kl = edge_kl(probs, p_offline.to(probs.device))
        with torch.no_grad():
            mask = repair_symmetric(
                g["edge_index"],
                apply_min_keep_floor(action.bool(), probs,
                                     cfg.get("min_keep_frac", 0.1)))
            std_pr = mc_std(gnn, g, g["edge_index"][:, mask], g["edge_attr"][mask],
                            n_samples=cfg["tta_mc_samples"], device=device)
            ke, pe = _graph_physics_terms(g, g["edge_index"], mask)
            vp = (relative_virial_penalty(ke, pe, ke_full, pe_full)
                  if cfg.get("w_virial", 0) > 0
                  else torch.zeros(1).to(device))
            conn_ok = bool((mask.sum() > 0) and _no_isolated(g["edge_index"], mask))
            keep = float(mask.float().mean())
            r = label_free_reward(std_pr, std_full, keep,
                                  target_sparsity, vp, cfg, conn_ok)
            # Diagnostic decomposition (never in the reward): d_unc /
            # sparsity / conn / virial separately + RMSE vs frozen when
            # labels are supplied for development.
            step_diag = {
                "d_unc": float(((std_full - std_pr) / (std_full + 1e-8)).mean()),
                "keep": keep,
                "sparsity_term": min((keep - target_sparsity) ** 2, 1.0),
                "conn_ok": conn_ok,
                "virial": float(vp.view(-1)[0]),
                "kl_to_offline": float(kl),
            }
            if diag_targets is not None:
                from torch_geometric.data import Data as _Data, Batch as _Batch
                d = _Data(x=g["x"], edge_index=g["edge_index"][:, mask],
                          edge_attr=g["edge_attr"][mask])
                pred, _ = gnn(_Batch.from_data_list([d]).to(device))
                d0 = _Data(x=g["x"], edge_index=g["edge_index"],
                           edge_attr=g["edge_attr"])
                pred0, _ = gnn(_Batch.from_data_list([d0]).to(device))
                yt = diag_targets.view(-1).float().to(device)
                step_diag["rmse_pruned"] = float(
                    torch.sqrt(torch.mean((pred.view(-1) - yt) ** 2)))
                step_diag["rmse_frozen"] = float(
                    torch.sqrt(torch.mean((pred0.view(-1) - yt) ** 2)))
            diag.append(step_diag)
        baseline = 0.9 * baseline + 0.1 * float(r)
        adv = float(r) - baseline
        loss = (-(adv * logp) - cfg.get("entropy_coef", 0.01) * ent
                + float(cfg.get("tta_kl_coef", 0.0)) * kl)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        hist.append(float(r))
        stall = 0 if float(r) > best_r else stall + 1
        best_r = max(best_r, float(r))
        if stall >= cfg.get("tta_patience", 3):
            break
        if log_fn:
            log_fn(step, float(r))
    with torch.no_grad():
        p = torch.sigmoid(pol(g["edge_attr"], g["emb"],
                              g["edge_index"], g["ctx"])).squeeze(-1)
        # FIX (audit P0-2, Sep 2026): mode-aware symmetric final mask —
        # threshold in penalty mode, top-k at the TTA target in topk mode
        # (same train/inference decoder rule as offline training).
        final = eval_mask(g["edge_index"], p, cfg,
                          target_sparsity=target_sparsity)
    return final, {"reward_hist": hist, "steps_run": len(hist), "best_r": best_r,
                   "step_diag": diag}
