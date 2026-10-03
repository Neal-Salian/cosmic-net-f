"""REINFORCE over ordered physical-pair samples, with validation selection."""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from rls.pair_policy import pair_layout, pair_mask, pair_scores, pair_stats
from rls.rewards import compute_rewards, relative_virial_penalty


def train_pair_policy(trainer, graphs, gnns, cfg, device, epochs, log_fn=None,
                      validation_fn=None):
    from rls.train_policy import _graph_physics_terms
    if not graphs:
        raise ValueError("Training split is empty.")
    rollouts = int(cfg.get("pair_rollouts", 4))
    if rollouts < 2:
        raise ValueError("Leave-one-out pair training needs at least two rollouts.")
    policy, critic = trainer.policy.to(device), trainer.value_net.to(device)
    policy.fit_normalization(graphs)
    cached = []
    full_errors = []
    for graph in graphs:
        g = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in graph.items()}
        g["pair_layout"] = pair_layout(g["edge_index"])
        mask = torch.ones(g["edge_index"].shape[1], dtype=torch.bool, device=device)
        with torch.no_grad():
            full, _ = gnns(g, mask) if gnns else (g["y"] + .1, g["y"] + .1)
            full_errors.append((full.reshape(()) - g["y"].reshape(())).square())
            g["reference_prediction"] = full.detach()
            g["reference_physics"] = _graph_physics_terms(g, g["edge_index"], mask)
        cached.append(g)
    cfg["reward_error_scale"] = max(float(torch.stack(full_errors).mean().sqrt()),
                                    float(cfg.get("reward_scale_floor", 0.05)))
    history, best_state, best_value_state = [], None, None
    best_rmse, best_epoch = math.inf, None
    # Initial checkpoint competes at the SAME final inference budget.
    if validation_fn:
        initial = validation_fn(policy)
        best_rmse = initial["rmse"]
        best_state, best_value_state = copy.deepcopy(policy.state_dict()), copy.deepcopy(critic.state_dict())
        trainer.initial_validation = initial
    for epoch in range(epochs):
        target = trainer.target_sparsity(epoch)
        order = torch.randperm(len(cached)).tolist()
        records, batch_records = [], []
        policy.train(); critic.train()
        for start in range(0, len(order), int(cfg["batch_size"])):
            actor_terms, value_terms, entropy_terms = [], [], []
            for index in order[start:start + int(cfg["batch_size"])]:
                g = cached[index]
                logits = policy(g["edge_attr"], g["emb"], g["edge_index"], g["ctx"]).reshape(-1)
                scores = pair_scores(g["edge_index"], logits, g["pair_layout"])
                # A defined, differentiable regularizer on the first-draw
                # categorical distribution, independent of sampled histories.
                log_first = torch.log_softmax(scores, dim=0)
                entropy = -(log_first.exp() * log_first).sum() / max(1., math.log(max(1, scores.numel())))
                logps, rewards = [], []
                for _ in range(rollouts):
                    mask, logp, _, _ = pair_mask(g["edge_index"], logits, target, sample=True, layout=g["pair_layout"])
                    with torch.no_grad():
                        full = g["reference_prediction"]
                        _, pruned = gnns(g, mask) if gnns else (full, g["y"] + .1 * mask.float().mean())
                        ke, pe = _graph_physics_terms(g, g["edge_index"], mask)
                        vp = relative_virial_penalty(ke, pe, *g["reference_physics"])
                        stats = pair_stats(g["edge_index"], mask, len(g["x"]))
                        reward_cfg = dict(cfg)
                        anneal_start = int(cfg.get("virial_anneal_start_epoch", 0))
                        reward_cfg["w_virial"] = cfg.get("w_virial", 0.) * min(1., max(0., (epoch - anneal_start + 1) / max(1, int(cfg.get("virial_anneal_epochs", 10)))))
                        reward = compute_rewards(pruned, full, g["y"], stats["physical_pair_keep"],
                            target, vp, reward_cfg, connectivity_ok=stats["physical_isolates"] == 0).reshape(())
                        accuracy = float(((full-g["y"]).abs() - (pruned-g["y"]).abs()).mean()) / cfg["reward_error_scale"]
                        records.append(dict(reward=float(reward), accuracy=accuracy,
                            virial=-reward_cfg["w_virial"] * float(vp),
                            sparsity=-cfg["w_sp"] * (stats["physical_pair_keep"]-target)**2,
                            **stats))
                    logps.append(logp); rewards.append(reward)
                rewards = torch.stack(rewards)
                # Other IID rollouts form an action-independent baseline for
                # each draw. No batch-mean advantage normalization or biased
                # deterministic top-k pseudo-likelihood is used.
                baselines = (rewards.sum() - rewards) / (rollouts - 1)
                actor_terms.append(-((rewards - baselines).detach() * torch.stack(logps)).mean())
                context = g["ctx"]
                if policy.normalize:
                    context = ((context-policy.context_mean)/policy.context_scale).clamp(-10,10)
                value_terms.append(F.smooth_l1_loss(critic(context).reshape(()), rewards.mean()))
                entropy_terms.append(entropy)
            actor, value, entropy = (torch.stack(items).mean() for items in (actor_terms, value_terms, entropy_terms))
            loss = actor + float(cfg.get("value_coef", .5)) * value - float(cfg.get("entropy_coef", .01)) * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite pair-policy loss; aborting before checkpoint publication.")
            trainer.optimizer.zero_grad()
            if trainer.value_optimizer is not None: trainer.value_optimizer.zero_grad()
            loss.backward()
            pg_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            vf_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            trainer.optimizer.step()
            if trainer.value_optimizer is not None: trainer.value_optimizer.step()
            batch_records.append(dict(loss=float(loss.detach()), actor=float(actor.detach()),
                critic=float(value.detach()), entropy=float(entropy.detach()),
                policy_grad_norm=float(pg_norm), critic_grad_norm=float(vf_norm), n=len(actor_terms)))
        row = {key: float(np.average([b[key] for b in batch_records], weights=[b["n"] for b in batch_records]))
               for key in batch_records[0] if key != "n"}
        for key in records[0]:
            row[key + "_mean"] = float(np.mean([r[key] for r in records]))
        row.update(epoch=epoch, target_pair_keep=target, reward_min=min(r["reward"] for r in records),
                   reward_max=max(r["reward"] for r in records), reward_std=float(np.std([r["reward"] for r in records])))
        if validation_fn:
            metrics = validation_fn(policy)
            if not np.isfinite(metrics["rmse"]): raise RuntimeError("Nonfinite validation RMSE.")
            row.update({"val_"+key: value for key,value in metrics.items() if isinstance(value,(int,float))})
            if metrics["rmse"] < best_rmse:
                best_rmse, best_epoch = metrics["rmse"], epoch
                best_state, best_value_state = copy.deepcopy(policy.state_dict()), copy.deepcopy(critic.state_dict())
        history.append(row)
        if log_fn: log_fn(epoch, target, row["loss"])
    if best_state is not None:
        policy.load_state_dict(best_state); critic.load_state_dict(best_value_state)
    policy.eval()
    trainer.diagnostics = history
    trainer.selected_epoch = best_epoch
    trainer.selected_validation_rmse = best_rmse if validation_fn else None
    return [torch.tensor(row["loss"]) for row in history]
