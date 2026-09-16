"""Label-free adaptation for policies trained with physical-pair PL actions."""
import copy
import torch
from rls.pair_policy import policy_action, pair_scores, pair_stats
from rls.train_policy import _graph_physics_terms
from rls.rewards import relative_virial_penalty


def adapt_pair_policy(policy, graph, gnn, cfg, device, target_sparsity, log_fn=None):
    from rls.tta import mc_std
    g = {key: value.to(device) for key,value in graph.items() if isinstance(value,torch.Tensor) and key != "y"}
    pol = copy.deepcopy(policy).to(device).train()
    optimizer = torch.optim.Adam(pol.parameters(), lr=cfg["tta_lr"])
    full_mask = torch.ones(g["edge_index"].shape[1],dtype=torch.bool,device=device)
    with torch.no_grad():
        full_std = mc_std(gnn,g,n_samples=cfg["tta_mc_samples"],device=device)
        scale = full_std.clamp_min(float(cfg.get("tta_uncertainty_floor",.05)))
        reference_physics = _graph_physics_terms(g,g["edge_index"],full_mask)
        offline_scores = pair_scores(g["edge_index"], pol(g["edge_attr"],g["emb"],g["edge_index"],g["ctx"]).reshape(-1))
        offline_log = torch.log_softmax(offline_scores,dim=0)
    history, diagnostics = [], []
    best, stall = -float("inf"), 0
    for step in range(cfg["tta_steps"]):
        for group in optimizer.param_groups:
            group["lr"] = cfg["tta_lr"] * float(cfg.get("tta_lr_decay",1.))**step
        rewards, logps = [], []
        for _ in range(2):
            mask, logp, _, _ = policy_action(pol,g,cfg,sample=True,keep=target_sparsity)
            with torch.no_grad():
                std = mc_std(gnn,g,g["edge_index"][:,mask],g["edge_attr"][mask],n_samples=cfg["tta_mc_samples"],device=device)
                vp = relative_virial_penalty(*_graph_physics_terms(g,g["edge_index"],mask),*reference_physics)
                stats = pair_stats(g["edge_index"],mask,len(g["x"]))
                reward = (cfg.get("w_unc",.5)*((full_std-std)/scale).mean()
                    - cfg.get("w_sp",.5)*(stats["physical_pair_keep"]-target_sparsity)**2
                    + cfg.get("w_conn",1.)*(1. if stats["physical_isolates"]==0 else -2.)
                    - cfg.get("w_virial",0.)*vp).clamp(-cfg.get("reward_clip",10.),cfg.get("reward_clip",10.)).reshape(())
            rewards.append(reward); logps.append(logp)
        rewards=torch.stack(rewards)
        logits=pol(g["edge_attr"],g["emb"],g["edge_index"],g["ctx"]).reshape(-1)
        log_current=torch.log_softmax(pair_scores(g["edge_index"],logits),dim=0)
        kl=(log_current.exp()*(log_current-offline_log)).sum()
        loss=-((rewards-rewards.flip(0)).detach()*torch.stack(logps)).mean()+float(cfg.get("tta_kl_coef",0.))*kl
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite pair TTA loss.")
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        mean=float(rewards.mean());history.append(mean)
        diagnostics.append(dict(reward=mean,kl_to_offline=float(kl.detach())))
        if log_fn:log_fn(step,mean)
        stall = 0 if mean > best else stall+1
        best=max(best,mean)
        if stall>=cfg.get("tta_patience",3):break
    with torch.no_grad():mask=policy_action(pol.eval(),g,cfg,keep=target_sparsity)[0]
    return mask,dict(reward_hist=history,steps_run=len(history),best_r=best if history else None,step_diag=diagnostics)
