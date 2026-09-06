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
from rls.sparsify import (hard_mask, repair_connectivity, apply_min_keep_floor,
                          symmetrize_probs, repair_symmetric,
                          final_symmetric_mask)
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


def adapt_at_test_time(policy, graph, gnn, cfg, device="cpu", init="offline",
                       target_sparsity=0.5, log_fn=None):
    """Adapt the policy to ONE graph via K policy-gradient steps on the
    label-free reward. Returns (final_mask, info). NEVER mutates `policy`."""
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
    for step in range(cfg["tta_steps"]):
        probs = torch.sigmoid(pol(g["edge_attr"], g["emb"],
                                  g["edge_index"], g["ctx"])).squeeze(-1)
        # FIX (audit P0-2, Sep 2026): same symmetrization as offline training
        # — pair-averaged probs, symmetric executed mask (see train_policy).
        probs = symmetrize_probs(g["edge_index"], probs)
        action = sample_actions(probs)
        logp = bernoulli_logp(probs, action).mean()
        ent = bernoulli_entropy(probs).mean()
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
            r = label_free_reward(std_pr, std_full, float(mask.float().mean()),
                                  target_sparsity, vp, cfg, conn_ok)
        baseline = 0.9 * baseline + 0.1 * float(r)
        adv = float(r) - baseline
        loss = -(adv * logp) - cfg.get("entropy_coef", 0.01) * ent
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
        # FIX (audit P0-2, Sep 2026): deterministic eval path uses the
        # symmetric decision layer — provably pair-symmetric final mask with
        # self-loops retained.
        final = final_symmetric_mask(g["edge_index"], p,
                                     cfg.get("min_keep_frac", 0.1))
    return final, {"reward_hist": hist, "steps_run": len(hist), "best_r": best_r}
