"""End-to-end policy-gradient training of the edge policy over a frozen GNN.

Signature: train_policy(trainer, graphs, gnns, cfg, device, epochs, log_fn)

graphs: list of dicts with x, edge_index, edge_attr, y, ctx (+ physics attrs).
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from rls.policy_gradient import (compute_advantages, compute_pg_loss,
                                 bernoulli_logp, bernoulli_entropy, sample_actions)
from rls.sparsify import hard_mask, repair_connectivity
from rls.rewards import compute_rewards, virial_penalty


def _graph_physics_terms(graph, edge_index, mask, G=4.302e-9):
    """ke_retained, pe_retained per graph (see rewards.py docstring)."""
    assert mask.dtype == torch.bool, f"expected bool mask, got {mask.dtype}"
    stellar = graph["stellar_mass"]  # [N]
    vel_disp = graph["vel_disp"]     # [N]
    pos = graph["pos"]               # [N,3]
    deg = torch.zeros(stellar.numel(), device=stellar.device)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=stellar.device))
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=stellar.device))
    deg_ret = torch.zeros_like(deg)
    deg_ret.index_add_(0, edge_index[0, mask], torch.ones(mask.sum(), device=stellar.device))
    deg_ret.index_add_(0, edge_index[1, mask], torch.ones(mask.sum(), device=stellar.device))
    frac = deg_ret / deg.clamp(min=1)
    ke = 0.5 * torch.sum(frac * stellar * vel_disp ** 2)
    u, v = edge_index[0, mask], edge_index[1, mask]
    r = torch.norm(pos[u] - pos[v], dim=1)
    pe = torch.sum(G * stellar[u] * stellar[v] / r.clamp(min=1e-6))
    return ke, pe


def _no_isolated(edge_index, mask):
    """True iff every node keeps >= 1 incident edge under `mask`.
    Device-safe: the degree accumulator lives on edge_index's device."""
    device = edge_index.device
    deg = torch.zeros(int(edge_index.max().item()) + 1, dtype=torch.long, device=device)
    kept = mask.nonzero(as_tuple=False).squeeze(-1)
    if kept.numel() > 0:
        ones = torch.ones(kept.numel(), dtype=torch.long, device=device)
        deg.index_add_(0, edge_index[0, kept], ones)
        deg.index_add_(0, edge_index[1, kept], ones)
    return bool((deg >= 1).all())


def prepare_graphs(loader, gnn, device=None):
    """Adapter: PyG loader -> list of per-graph dicts for policy training.

    Uses batch.get_example(i) (the same pattern as explain/explainer.py's
    explain_batch) so per-graph tensors keep their original indices — NO
    fragile boolean-mask remapping. Precomputes frozen node embeddings and
    the graph context once (GNN stays frozen and in eval mode).

    device defaults to the GNN's own device — never trust a caller-passed
    device that mismatches the GNN (that mixes cpu graphs with cuda emb)."""
    if device is None:
        device = next(gnn.parameters()).device
    graphs = []
    gnn.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for i in range(batch.num_graphs):
                g = batch.get_example(i).to(device)
                b = Batch.from_data_list([g])
                emb = gnn.get_embeddings(b, embedding_point="pre_pooling")  # [N, out]
                ctx = global_mean_pool(emb, b.batch)                        # [1, out]
                graphs.append({
                    "x": g.x,
                    "edge_index": g.edge_index,
                    "edge_attr": g.edge_attr,
                    "y": g.y.view(1),
                    "ctx": ctx[0],
                    "emb": emb,                     # per-node [N, out] — the
                                                    # policy's node_emb input
                    "stellar_mass": getattr(g, "stellar_mass", None),
                    "vel_disp": getattr(g, "vel_disp", None),
                    "half_mass_r": getattr(g, "half_mass_r", None),
                    "pos": getattr(g, "pos", None),
                })
    return graphs


def train_policy(trainer, graphs, gnns, cfg, device="cpu", epochs=60, log_fn=None):
    """One epoch = one pass over the graphs; batch = cfg['batch_size'] graphs.

    gnns: callable (graph_dict, mask) -> (pred_full, pred_pruned) or None
    (smoke mode: random placeholder predictions).
    Returns the per-epoch mean loss list."""
    policy, value_net = trainer.policy, trainer.value_net
    policy = policy.to(device)
    value_net = value_net.to(device)
    n = max(1, len(graphs))
    batch_size = cfg["batch_size"]
    losses = []

    def _update_batch(batch_graphs, target_sp):
        """One policy-gradient step on a mini-batch of graphs.

        Everything the policy must NOT differentiate through (mask repair, GNN
        forward, rewards, baseline) runs under no_grad. Only the sampled log
        probs and the entropy term carry policy gradients."""
        total_logp = []
        total_ent = []
        total_values = []
        total_rewards = []

        for g in batch_graphs:
            g = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
            probs = torch.sigmoid(policy(g["edge_attr"], g["emb"],
                                         g["edge_index"], g["ctx"])).squeeze(-1)
            action = sample_actions(probs)
            logp = bernoulli_logp(probs, action)
            ent = bernoulli_entropy(probs).mean()
            # value-net forward must stay OUTSIDE no_grad so vf_loss.backward()
            # has a graph to flow through (the baseline is learnable).
            val = value_net(g["ctx"])
            with torch.no_grad():
                mask = hard_mask(probs, min_keep_frac=cfg.get("min_keep_frac", 0.1))
                mask = repair_connectivity(g["edge_index"], mask)
                pred_full, pred_pruned = gnns(g, mask) if gnns else (torch.randn(1), torch.randn(1))
                ke, pe = _graph_physics_terms(g, g["edge_index"], mask)
                vp = virial_penalty(ke, pe) if cfg.get("w_virial", 0) > 0 else torch.zeros(1).to(device)
                conn_ok = bool((mask.sum() > 0) and _no_isolated(g["edge_index"], mask))
                keep_ratio = mask.float().mean().item()
                rew = compute_rewards(pred_pruned, pred_full, g["y"], keep_ratio,
                                      target_sp, vp, cfg, connectivity_ok=conn_ok)
            total_logp.append(logp.mean())
            total_ent.append(ent)
            total_values.append(val)
            total_rewards.append(rew)

        logp = torch.stack(total_logp)
        entropy = torch.stack(total_ent).mean()
        values = torch.stack(total_values)
        rewards = torch.stack(total_rewards)

        adv = compute_advantages(rewards, values)
        loss, pg_loss, vf_loss, ent = compute_pg_loss(
            logp, adv, values, rewards,
            entropy=entropy,
            value_coef=cfg.get("value_coef", 0.5),
            entropy_coef=cfg.get("entropy_coef", 0.01))

        # Policy and value-net graphs are disjoint (advantages are detached);
        # ONE backward fills grads for both, then each optimizer steps its own
        # params. Two separate backward calls would re-traverse the value graph.
        trainer.optimizer.zero_grad()
        if trainer.value_optimizer is not None:
            trainer.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        trainer.optimizer.step()
        if trainer.value_optimizer is not None:
            trainer.value_optimizer.step()

        return float(loss.item())

    for epoch in range(epochs):
        target_sp = trainer.target_sparsity(epoch)
        epoch_losses = []
        for i in range(0, n, batch_size):
            epoch_losses.append(_update_batch(graphs[i:i + batch_size], target_sp))
        mean = torch.tensor(epoch_losses).mean()
        losses.append(mean)
        if log_fn:
            log_fn(epoch, target_sp, float(mean))
    return losses
