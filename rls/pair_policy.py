"""One action per physical pair; ordered Plackett–Luce sampling without replacement.

The action is the ordered sample. Its log probability is exact. Deterministic
physical-isolation repair is part of the environment, not a resampled action.
Self-loops are never stochastic actions. Connectivity means no physically
isolated nodes when the input provides incident non-loop pairs, not a spanning
connected component.
"""
import math
import torch


def pair_layout(edge_index):
    real = edge_index[0] != edge_index[1]
    pairs, inverse = torch.unique(edge_index[:, real].sort(dim=0).values.T,
                                  dim=0, sorted=True, return_inverse=True)
    return pairs, inverse, real


def pair_scores(edge_index, scores, layout=None):
    pairs, inverse, real = layout if layout is not None else pair_layout(edge_index)
    values = scores.reshape(-1)[real]
    sums = values.new_zeros(len(pairs)).index_add(0, inverse, values)
    counts = values.new_zeros(len(pairs)).index_add(0, inverse, torch.ones_like(values))
    return sums / counts.clamp_min(1)


def ordered_log_probability(scores, order):
    """Exact log p(i1,...,ik); no mean-over-edge surrogate."""
    k, n = order.numel(), scores.numel()
    if not k:
        return scores.sum() * 0, scores.sum() * 0
    excluded = torch.zeros((k, n), dtype=torch.bool, device=scores.device)
    if k > 1:
        previous = torch.arange(k, device=scores.device)[:, None] > torch.arange(k, device=scores.device)[None, :]
        excluded[:, order] = previous
    conditional = scores.expand(k, n).masked_fill(excluded, -torch.inf)
    log_probs = torch.log_softmax(conditional, dim=1)
    logp = log_probs[torch.arange(k, device=scores.device), order].sum()
    entropy = -(log_probs.exp() * log_probs.masked_fill(excluded, 0)).sum(dim=1).mean()
    return logp, entropy


def repair_pairs(pairs, selected, num_nodes):
    """Add canonical incident pairs for physical isolates, ignoring self-loops.

Repair depends only on topology and sampled membership, never on policy scores.
Unavailable connections cannot be invented; pair_stats reports those isolates.
"""
    selected = selected.clone()
    degree = torch.zeros(num_nodes, dtype=torch.long, device=pairs.device)
    if selected.any():
        degree.index_add_(0, pairs[selected].reshape(-1), torch.ones(int(selected.sum()) * 2, dtype=torch.long, device=pairs.device))
    for node in range(num_nodes):
        if degree[node] == 0:
            incident = (pairs == node).any(dim=1).nonzero().flatten()
            if incident.numel():
                index = incident[0]
                selected[index] = True
                degree[pairs[index]] += 1
    return selected


def pair_mask(edge_index, scores, keep, sample=False, layout=None, repair=True):
    if not 0 < keep <= 1:
        raise ValueError("Physical pair keep fraction must be in (0, 1].")
    pairs, inverse, real = layout if layout is not None else pair_layout(edge_index)
    scores = pair_scores(edge_index, scores, (pairs, inverse, real))
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite pair scores.")
    n = len(pairs)
    k = min(n, max(1, math.ceil(keep * n))) if n else 0
    if sample and k:
        # Gumbel top-k samples the ordered PL action; use the original scores
        # for its exact likelihood, not the perturbed ranking scores.
        uniform = torch.rand_like(scores).clamp(1e-7, 1 - 1e-7)
        order = (scores.detach() - torch.log(-torch.log(uniform))).argsort(descending=True, stable=True)[:k]
    else:
        order = scores.detach().argsort(descending=True, stable=True)[:k]
    logp, entropy = ordered_log_probability(scores, order)
    chosen = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    chosen[order] = True
    if repair:
        chosen = repair_pairs(pairs, chosen, int(edge_index.max()) + 1 if edge_index.numel() else 0)
    mask = ~real.clone()
    mask[real] = chosen[inverse]
    return mask, logp, entropy, order


def pair_stats(edge_index, mask, num_nodes=None):
    pairs, inverse, real = pair_layout(edge_index)
    chosen = torch.zeros(len(pairs), dtype=torch.long, device=edge_index.device)
    chosen.index_add_(0, inverse, mask[real].long())
    chosen = chosen > 0
    n = num_nodes if num_nodes is not None else (int(edge_index.max()) + 1 if edge_index.numel() else 0)
    degree = torch.zeros(n, dtype=torch.long, device=edge_index.device)
    if chosen.any():
        degree.index_add_(0, pairs[chosen].reshape(-1), torch.ones(int(chosen.sum()) * 2, dtype=torch.long, device=edge_index.device))
    return {"physical_pair_keep": float(chosen.float().mean()) if len(pairs) else 0.0,
            "total_edge_keep": float(mask.float().mean()) if mask.numel() else 0.0,
            "physical_isolates": int((degree == 0).sum())}


def policy_action(policy, graph, cfg, sample=False, keep=None):
    logits = policy(graph["edge_attr"], graph["emb"], graph["edge_index"], graph["ctx"]).reshape(-1)
    keep = cfg.get("target_sparsity_end", 0.4) if keep is None else keep
    return pair_mask(graph["edge_index"], logits, keep, sample=sample,
                     layout=graph.get("pair_layout"))
