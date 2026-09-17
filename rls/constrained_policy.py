"""Exact-budget physical-pair policies, separate from the legacy pair_pl decoder.

``select_pairs(layout, scores, k, constraint='none', method='direct', ...)``
accepts one logit per physical pair. Direct sampling masks each categorical draw
by exact completion feasibility. Scaffold builds a minimum feasible set using
only independent pair marks, then samples/ranks the remaining fill pairs.

The returned log_probability is the ORDERED ACTION probability conditional on
pair marks, not final-mask probability. Scaffold edges are not sampled actions.
``exact_mask_log_probability`` sums compatible orders for small diagnostics.
No likelihood is evaluated for deterministic inference (returned as None).
These routines are reference research implementations: NetworkX feasibility
checks run on CPU and can be expensive for large/dense graphs.
"""
from dataclasses import dataclass
import itertools
import torch
from rls.constraints import (feasible_budget, pair_marks as make_pair_marks,
                             selection_diagnostics)


@dataclass
class PairSelection:
    mask: torch.Tensor
    selected: torch.Tensor
    order: torch.Tensor
    scaffold: torch.Tensor
    pair_marks: torch.Tensor
    log_probability: object
    diagnostics: dict


def _validate(layout, scores, k, constraint, method):
    if (scores.ndim != 1 or scores.shape != (len(layout.pairs),)
            or scores.device != layout.pairs.device or not scores.is_floating_point()
            or not torch.isfinite(scores).all()):
        raise ValueError('scores must be finite floating pair logits on the layout device')
    if method not in ('direct', 'scaffold'):
        raise ValueError('method must be direct or scaffold')
    report = feasible_budget(layout, k, constraint)
    if not report['feasible']:
        raise ValueError(f'infeasible physical-pair budget: {report}')
    return report


def _candidates(layout, selected, k, constraint):
    candidates = (~selected).nonzero().flatten()
    if constraint == 'none':
        return candidates
    allowed = []
    for index in candidates.tolist():
        proposed = selected.clone()
        proposed[index] = True
        if feasible_budget(layout, k, constraint, proposed)['feasible']:
            allowed.append(index)
    return torch.tensor(allowed, dtype=torch.long, device=selected.device)


def _scaffold(layout, k_min, constraint, marks, method):
    selected = torch.zeros(len(layout.pairs), dtype=torch.bool, device=layout.pairs.device)
    if method == 'scaffold':
        for _ in range(k_min):
            candidates = _candidates(layout, selected, k_min, constraint)
            selected[candidates[marks[candidates].argmax()]] = True
    return selected


def select_pairs(layout, scores, k, constraint='none', *, method='direct', sample=False,
                 pair_marks=None, generator=None, keep_self_loops=True):
    """Select exactly k pairs or raise ValueError for an infeasible budget.

    Ranking ties use independent unique marks. For stochastic draws, candidates
    are traversed in mark order before fresh categorical randomness is drawn;
    transport marks and reset/clone the generator to couple node relabelings.
    Pair marks must be independent of learned logits for scaffold likelihoods
    and policy-gradient estimates to remain valid. They are returned for replay.
    """
    report = _validate(layout, scores, k, constraint, method)
    marks = make_pair_marks(layout, pair_marks, generator)
    scaffold = _scaffold(layout, report['minimum_pairs'], constraint, marks, method)
    selected = scaffold.clone()
    order = []
    logp = scores.sum() * 0 if sample else None
    for _ in range(k - int(scaffold.sum())):
        candidates = _candidates(layout, selected, k, constraint)
        candidates = candidates[marks[candidates].argsort(descending=True)]
        if sample:
            log_probs = torch.log_softmax(scores[candidates], 0)
            draw = torch.multinomial(log_probs.detach().exp(), 1, generator=generator).item()
            index = candidates[draw]
            logp = logp + log_probs[draw]
        else:
            index = candidates[scores.detach()[candidates].argmax()]
        selected[index] = True
        order.append(int(index))
    order = torch.tensor(order, dtype=torch.long, device=scores.device)
    diagnostics = selection_diagnostics(layout, selected, k=k, sampled_count=len(order),
                                        scaffold_count=int(scaffold.sum()), constraint=constraint)
    diagnostics.update(method=method, selection_mode='sample' if sample else 'rank',
                       likelihood_kind='ordered_action_conditional_on_marks' if sample else None,
                       tie_policy='exchangeable_pair_marks')
    return PairSelection(layout.expand(selected, keep_self_loops), selected, order, scaffold,
                         marks, logp, diagnostics)


def constrained_order_log_probability(layout, scores, order, k, constraint='none', *,
                                      method='direct', pair_marks=None):
    """Exact log probability of an ordered stochastic action (not a mask).

    For scaffold replay, pass the same explicit marks as selection; omission is
    rejected. Invalid ordered actions have log probability -inf. Malformed
    tensors and infeasible requested budgets raise ValueError.
    """
    report = _validate(layout, scores, k, constraint, method)
    if order.ndim != 1 or order.dtype != torch.long or order.device != scores.device:
        raise ValueError('order must be an int64 vector on the score device')
    if method == 'scaffold' and pair_marks is None:
        raise ValueError('scaffold likelihood requires explicit pair marks for replay')
    marks = make_pair_marks(layout, pair_marks) if method == 'scaffold' else None
    selected = _scaffold(layout, report['minimum_pairs'], constraint, marks, method)
    invalid = scores.sum() * 0 + float('-inf')
    if len(order) != k - int(selected.sum()):
        return invalid
    logp = scores.sum() * 0
    for index in order.tolist():
        candidates = _candidates(layout, selected, k, constraint)
        if index not in candidates.tolist():
            return invalid
        log_probs = torch.log_softmax(scores[candidates], 0)
        logp = logp + log_probs[candidates == index].sum()
        selected[index] = True
    return logp


def exact_mask_log_probability(layout, scores, selected, constraint='none', *,
                               method='direct', pair_marks=None):
    """Small exact diagnostic: sum probabilities of orders yielding this mask.

    Restricted to <=8 available pairs and <=4 selected pairs to bound factorial
    enumeration. This is not a training/inference routine. Scaffold likelihood
    is conditional on its supplied independent marks, not marginal over them.
    """
    layout.validate_selection(selected)
    k = int(selected.sum())
    if len(layout.pairs) > 8 or k > 4:
        raise ValueError('exact mask enumeration is limited to eight pairs and k <= 4')
    report = _validate(layout, scores, k, constraint, method)
    if method == 'scaffold' and pair_marks is None:
        raise ValueError('scaffold likelihood requires explicit pair marks for replay')
    marks = make_pair_marks(layout, pair_marks) if method == 'scaffold' else None
    scaffold = _scaffold(layout, report['minimum_pairs'], constraint, marks, method)
    if bool((scaffold & ~selected).any()):
        return scores.sum() * 0 + float('-inf')
    actions = (selected & ~scaffold).nonzero().flatten().tolist()
    logps = [constrained_order_log_probability(layout, scores,
             torch.tensor(order, dtype=torch.long, device=scores.device), k, constraint,
             method=method, pair_marks=marks) for order in itertools.permutations(actions)]
    finite = [logp for logp in logps if bool(logp.isfinite())]
    return torch.logsumexp(torch.stack(finite), 0) if finite else scores.sum() * 0 + float('-inf')
