"""One action per physical pair; ordered Plackett–Luce sampling without replacement.

The action is the ordered sample. Its log probability is exact. Logit-independent
physical-isolation repair is part of the environment, not a resampled action.
Self-loops are never stochastic actions. Connectivity means no physically
isolated nodes when the input provides incident non-loop pairs, not a spanning
connected component.
"""
import torch
from rls.constraints import (PhysicalPairLayout, pair_budget, pair_marks as make_pair_marks,
                             selection_diagnostics)


def pair_layout(edge_index):
    layout = PhysicalPairLayout.from_edge_index(edge_index)
    return layout.pairs, layout.inverse, layout.real


def pair_scores(edge_index, scores, layout=None):
    pairs, inverse, real = layout if layout is not None else pair_layout(edge_index)
    values = scores.reshape(-1)[real]
    sums = values.new_zeros(len(pairs)).index_add(0, inverse, values)
    counts = values.new_zeros(len(pairs)).index_add(0, inverse, torch.ones_like(values))
    return sums / counts.clamp_min(1)


def ordered_log_probability(scores, order):
    """Exact log p(i1,...,ik); no mean-over-edge surrogate."""
    if (scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all()
            or order.ndim != 1 or order.dtype != torch.long or order.device != scores.device):
        raise ValueError('finite score vector and int64 order on the same device required')
    k, n = order.numel(), scores.numel()
    if k > n or order.unique().numel() != k or bool(((order < 0) | (order >= n)).any()):
        raise ValueError('order must contain distinct available pair indices')
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


def repair_pairs(pairs, selected, num_nodes, pair_marks=None, generator=None):
    """Legacy additive no-isolate repair with exchangeable pair priorities.

    Visit pairs by independent marks, adding a pair iff either endpoint is still
    uncovered. Never use numeric node/edge order as a preference. Transport
    explicit marks under relabeling for coupled equality; omitted marks are
    exchangeable random priorities. This can exceed the sampled budget. Use
    constrained_policy.select_pairs for an exact budget or infeasibility error.
    Input-topology isolates remain visible in pair_stats, never invented away.
    """
    # Validate endpoints/uniqueness but retain caller pair-row/mark alignment.
    validated = PhysicalPairLayout.from_edge_index(pairs.T, num_nodes)
    if len(validated.pairs) != len(pairs) or not bool(validated.real.all()):
        raise ValueError('pairs must be unique non-self physical pairs')
    layout = PhysicalPairLayout(pairs, validated.inverse, validated.real, num_nodes)
    layout.validate_selection(selected)
    marks = make_pair_marks(layout, pair_marks, generator)
    selected = selected.clone()
    covered = torch.zeros(num_nodes, dtype=torch.bool, device=pairs.device)
    covered[pairs[selected].reshape(-1)] = True
    for index in marks.argsort(descending=True).tolist():
        if not bool(covered[pairs[index]].all()):
            selected[index] = True
            covered[pairs[index]] = True
    return selected


def pair_mask(edge_index, scores, keep, sample=False, layout=None, repair=True,
              pair_marks=None, generator=None, num_nodes=None):
    """Legacy ordered PL action followed by additive physical-isolate repair.

    Returns (executed_mask, ordered_logp, path_entropy, sampled_order). The
    executed mask can exceed ceil(keep * physical_pairs). Its probability is
    NOT ordered_logp: multiple orders/repair marks can yield the same mask.
    Ranking and repair use independent exchangeable marks; supply marks to
    reproduce tied inference. Deterministic inference returns zero logp/entropy
    sentinels for tuple compatibility and skips likelihood evaluation entirely.
    """
    if not 0 <= keep <= 1:
        raise ValueError('Physical pair keep fraction must be in [0, 1].')
    validated = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    if layout is not None:
        if len(layout) != 3 or any(not torch.equal(a, b) for a, b in zip(layout, (validated.pairs, validated.inverse, validated.real))):
            raise ValueError('cached pair layout does not match edge_index')
    pairs, inverse, real = validated.pairs, validated.inverse, validated.real
    if scores.numel() != edge_index.shape[1]:
        raise ValueError('one score per edge column is required')
    scores = pair_scores(edge_index, scores, (pairs, inverse, real))
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite pair scores.')
    n = len(pairs)
    k = pair_budget(validated, keep)
    marks = make_pair_marks(validated, pair_marks, generator)
    priority = marks.argsort(descending=True)
    if sample and k:
        # Fresh categorical draws preserve the exact ordered PL law. The mark
        # traversal lets a cloned generator couple node relabelings correctly.
        remaining = priority
        draws = []
        for _ in range(k):
            probabilities = torch.softmax(scores.detach()[remaining], 0)
            draw = torch.multinomial(probabilities, 1, generator=generator).item()
            draws.append(remaining[draw])
            remaining = torch.cat((remaining[:draw], remaining[draw + 1:]))
        order = torch.stack(draws)
    else:
        order = priority[scores.detach()[priority].argsort(descending=True, stable=True)[:k]]
    if sample:
        logp, entropy = ordered_log_probability(scores, order)
    else:
        logp = entropy = scores.sum() * 0
    chosen = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    chosen[order] = True
    if repair:
        chosen = repair_pairs(pairs, chosen, validated.num_nodes, pair_marks=marks)
    return validated.expand(chosen), logp, entropy, order


def pair_stats(edge_index, mask, num_nodes=None, *, requested_count=None, sampled_count=None):
    """Legacy numeric diagnostics; pass action counts to expose repair overhead.

    Existing aggregate/training callers consume only numeric entries. Full
    named constraint/likelihood diagnostics live on new PairSelection results.
    """
    layout = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    chosen = layout.collapse(mask)
    stats = selection_diagnostics(layout, chosen)
    result = {'physical_pair_keep': float(chosen.float().mean()) if len(chosen) else 0.0,
              'total_edge_keep': float(mask.float().mean()) if mask.numel() else 0.0,
              'physical_isolates': stats['physical_isolates'],
              'physical_components': stats['physical_components'],
              'final_pair_count': stats['final_pair_count'],
              'available_pair_count': len(chosen)}
    if requested_count is not None:
        result['requested_pair_count'] = int(requested_count)
        result['budget_excess_pair_count'] = int(chosen.sum()) - int(requested_count)
    if sampled_count is not None:
        result['sampled_pair_count'] = int(sampled_count)
        result['repair_added_pair_count'] = int(chosen.sum()) - int(sampled_count)
    return result


def policy_action(policy, graph, cfg, sample=False, keep=None):
    """Legacy decoder with persistent, logit-independent graph marks.

    On first use, graph['pair_marks'] is generated and retained. Validation
    repeats therefore replay the same tie/repair realization. Callers relabeling
    a graph must transport these physical-pair marks with its layout; callers
    constructing research artifacts should record them with their provenance.
    """
    if 'pair_marks' not in graph:
        layout = PhysicalPairLayout.from_edge_index(graph['edge_index'], len(graph['x']) if 'x' in graph else None)
        graph['pair_marks'] = make_pair_marks(layout)
    logits = policy(graph['edge_attr'], graph['emb'], graph['edge_index'], graph['ctx']).reshape(-1)
    keep = cfg.get('target_sparsity_end', 0.4) if keep is None else keep
    return pair_mask(graph['edge_index'], logits, keep, sample=sample,
                     layout=graph.get('pair_layout'), pair_marks=graph['pair_marks'],
                     num_nodes=len(graph['x']) if 'x' in graph else None)
