"""Bounded enumeration of feasible physical-pair masks.

Tiny-graph structured control for exact MAP and Gibbs selection over
feasible mask space. Supports transported-pair relabeling symmetry
through mask ordering by pair-mark ranks.

This is a tiny control, not L2XGNN or a scalable differentiable
relaxation claim. Larger graphs exceed the enumeration budget.
"""
from dataclasses import dataclass
import itertools
import math
import torch
from rls.constraints import (
    PhysicalPairLayout,
    feasible_budget,
    pair_marks as make_pair_marks,
    selection_diagnostics,
)
from rls.constrained_policy import PairSelection


MAX_PAIRS = 8
MAX_K = 4


def _validate_feasible_masks_args(layout, k, constraint, pair_marks, max_pairs, max_k):
    if not isinstance(layout, PhysicalPairLayout):
        raise ValueError('layout must be a PhysicalPairLayout')
    if not isinstance(k, int) or k < 0:
        raise ValueError('k must be a non-negative integer')
    if constraint not in ('none', 'no_isolates', 'connected'):
        raise ValueError('constraint must be none, no_isolates, or connected')
    n_pairs = len(layout.pairs)
    if n_pairs > max_pairs:
        raise ValueError(
            f'unsupported graph size: {n_pairs} pairs exceeds max_pairs={max_pairs}'
        )
    if k > max_k:
        raise ValueError(
            f'unsupported budget: k={k} exceeds max_k={max_k}'
        )
    report = feasible_budget(layout, k, constraint)
    if not report['feasible']:
        raise ValueError(
            f'infeasible physical-pair budget: {report}'
        )
    if pair_marks is None:
        raise ValueError('pair marks must be supplied explicitly per physical pair')
    marks = torch.as_tensor(pair_marks, device=layout.pairs.device).detach()
    if marks.shape != (n_pairs,) or not torch.isfinite(marks).all():
        raise ValueError('pair marks must be finite and match pair count')
    if marks.unique().numel() != n_pairs:
        raise ValueError('pair marks must be unique')
    return marks


def enumerate_feasible_masks(layout, k, constraint='no_isolates', *,
                             pair_marks=None, max_pairs=MAX_PAIRS,
                             max_k=MAX_K):
    """Enumerate all feasible masks of exactly k physical non-self pairs.

    Args:
        layout: PhysicalPairLayout
        k: number of physical pairs to select
        constraint: 'none', 'no_isolates', or 'connected'
        pair_marks: required unique finite marks per physical pair
        max_pairs: maximum number of pairs allowed (default 8)
        max_k: maximum k allowed (default 4)

    Returns:
        bool tensor [M, P] where M is number of feasible masks
    """
    marks = _validate_feasible_masks_args(layout, k, constraint, pair_marks,
                                          max_pairs, max_k)
    n_pairs = len(layout.pairs)
    if k == 0:
        return torch.zeros(1, n_pairs, dtype=torch.bool, device=layout.pairs.device)

    # Generate all combinations of k pairs from n_pairs
    all_masks = []
    for combo in itertools.combinations(range(n_pairs), k):
        mask = torch.zeros(n_pairs, dtype=torch.bool, device=layout.pairs.device)
        mask[list(combo)] = True
        all_masks.append(mask)

    # Filter by feasibility
    feasible_masks = []
    for mask in all_masks:
        report = feasible_budget(layout, k, constraint, mask)
        if report['feasible']:
            feasible_masks.append(mask)

    if not feasible_masks:
        return torch.zeros(0, n_pairs, dtype=torch.bool, device=layout.pairs.device)

    # Sort by transported pair-mark ranks for relabeling equivariance
    if marks is not None:
        # Compute rank for each mask: sum of marks in the mask
        # But we need to sort by the subset of transported pair-mark ranks
        # For coupled relabeling, order masks by the sorted marks they contain
        mask_ranks = []
        for mask in feasible_masks:
            # Get marks in this mask and sort them
            mask_marks = marks[mask].sort().values
            mask_ranks.append(mask_marks.tolist())
        # Sort masks lexicographically by their sorted marks
        order = sorted(range(len(feasible_masks)),
                       key=lambda i: mask_ranks[i])
        feasible_masks = [feasible_masks[i] for i in order]

    return torch.stack(feasible_masks)


def structured_gibbs(scores, feasible_masks, temperature=1.0):
    """Compute log probabilities for feasible masks using structured Gibbs.

    Args:
        scores: float tensor [P] of pair scores
        feasible_masks: bool tensor [M, P] of feasible masks
        temperature: positive finite temperature

    Returns:
        [M] log probabilities
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('temperature must be positive and finite')
    if scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError('scores must be a 1D float tensor')
    if not torch.isfinite(scores).all():
        raise ValueError('scores must be finite')
    if feasible_masks.ndim != 2 or feasible_masks.dtype != torch.bool:
        raise ValueError('feasible_masks must be a 2D bool tensor')
    if feasible_masks.shape[0] == 0:
        raise ValueError('feasible_masks must be nonempty')
    if feasible_masks.shape[1] != scores.shape[0]:
        raise ValueError('feasible_masks and scores must have same P dimension')
    if not feasible_masks.sum(dim=1).eq(feasible_masks.sum(dim=1)[0]).all():
        raise ValueError('all masks must have identical cardinality')

    # Center scores for numerical stability
    centered = scores - scores.mean()
    # Sum scores over selected pairs in each mask, preserving scores dtype/device
    mask_scores = feasible_masks.to(scores.dtype) @ centered
    # Apply temperature and compute log probabilities
    log_probs = torch.log_softmax(mask_scores / temperature, dim=0)
    return log_probs


def select_structured(layout, scores, k, constraint='no_isolates', *,
                      pair_marks=None, sample=False, generator=None,
                      temperature=1.0):
    """Select exactly k pairs using bounded enumeration.

    Args:
        layout: PhysicalPairLayout
        scores: float tensor [P] of pair scores
        k: number of physical pairs to select
        constraint: 'none', 'no_isolates', or 'connected'
        pair_marks: optional unique finite marks per physical pair
        sample: if True, sample from Gibbs distribution; else MAP
        generator: optional torch Generator for reproducibility
        temperature: temperature for Gibbs sampling

    Returns:
        PairSelection with mask, selected, order, scaffold, pair_marks,
        log_probability, diagnostics
    """
    if scores.ndim != 1 or scores.shape != (len(layout.pairs),):
        raise ValueError('scores must be a 1D float tensor matching pair count')
    if not scores.is_floating_point():
        raise ValueError('scores must be a float tensor')
    if not torch.isfinite(scores).all():
        raise ValueError('scores must be finite')
    if not isinstance(k, int) or k < 0:
        raise ValueError('k must be a non-negative integer')
    if len(layout.pairs) > MAX_PAIRS:
        raise ValueError(f'unsupported graph size: {len(layout.pairs)} pairs exceeds max_pairs={MAX_PAIRS}')
    if k > MAX_K:
        raise ValueError(f'unsupported budget: k={k} exceeds max_k={MAX_K}')

    # Create marks if not provided
    if pair_marks is None:
        marks = make_pair_marks(layout, generator=generator)
    else:
        marks = pair_marks

    # Get feasible masks
    masks = enumerate_feasible_masks(layout, k, constraint, pair_marks=marks)
    n_masks = masks.shape[0]

    if n_masks == 0:
        raise ValueError('no feasible masks found')

    if sample:
        # Gibbs sampling
        log_probs = structured_gibbs(scores, masks, temperature=temperature)
        probs = log_probs.exp()
        # Sample mask index
        idx = torch.multinomial(probs, 1, generator=generator).item()
        selected_mask = masks[idx]
        log_prob = log_probs[idx]
        method = 'enumerated_structured_gibbs'
    else:
        # MAP: center fixed-k energies in scores dtype so a common float32
        # offset cannot change the optimal selection; every feasible mask has
        # exactly k entries so centering shifts all mask scores by the same
        # k*mean amount. Ties keep mark-ordered first-max choice, no epsilon.
        centered = scores - scores.mean()
        mask_scores = masks.to(scores.dtype) @ centered
        idx = mask_scores.argmax().item()
        selected_mask = masks[idx]
        log_prob = None
        method = 'enumerated_structured_map'

    # Expand to full edge mask (including reciprocal/duplicate copies)
    mask = layout.expand(selected_mask, keep_self_loops=True)

    # Diagnostics
    diag = selection_diagnostics(
        layout, selected_mask,
        k=k,
        sampled_count=k,
        scaffold_count=0,
        constraint=constraint
    )
    diag.update(
        method=method,
        selection_mode='sample' if sample else 'map',
        likelihood_kind='unordered_feasible_mask' if sample else None,
        tie_policy='transported_pair_marks' if not sample else None,
        num_feasible_masks=n_masks,
    )

    # Empty order (bookkeeping only, not a PL likelihood)
    order = torch.tensor([], dtype=torch.long, device=scores.device)
    scaffold = torch.zeros_like(selected_mask)

    return PairSelection(
        mask=mask,
        selected=selected_mask,
        order=order,
        scaffold=scaffold,
        pair_marks=marks,
        log_probability=log_prob,
        diagnostics=diag,
    )
