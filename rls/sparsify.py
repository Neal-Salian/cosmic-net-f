"""Legacy mask decoders and additive physical-isolate repair.

All topology-aware decisions group unordered non-self physical pairs, including
one-way columns and duplicates. Exact-budget constrained selection is provided
by rls.constrained_policy; the legacy repair here can exceed a sampled budget.
"""
import math
import torch
from rls.constraints import PhysicalPairLayout, pair_budget
from rls.pair_policy import pair_mask, pair_scores, repair_pairs


def _floor_count(probs, min_keep_frac):
    if not math.isfinite(min_keep_frac) or not 0 <= min_keep_frac <= 1:
        raise ValueError('min_keep_frac must be in [0, 1]')
    return math.ceil(min_keep_frac * probs.numel())


def hard_mask(probs, min_keep_frac=0.1):
    """Threshold at 0.5 with a tie-inclusive minimum floor.

    This array-only legacy helper counts entries because it has no topology.
    For physical budgets use final_symmetric_mask or topk_scheduled_mask.
    All entries tied at the floor cutoff are retained; no index wins a tie.
    """
    return apply_min_keep_floor(probs >= 0.5, probs, min_keep_frac)


def apply_min_keep_floor(mask, probs, min_keep_frac=0.1):
    """Legacy array floor; tied cutoff values are all kept (may exceed floor)."""
    assert mask.dtype == torch.bool, f'expected bool mask, got {mask.dtype}'
    mask = mask.clone()
    k_min = _floor_count(probs, min_keep_frac)
    if int(mask.sum()) < k_min:
        cutoff = torch.topk(probs, k_min).values.min()
        mask = probs >= cutoff
    return mask


def repair_connectivity(edge_index, mask, *, pair_marks=None, num_nodes=None):
    """Add pairs for physical isolates using independent exchangeable priorities.

    All copies of each retained physical pair share membership. Loops do not
    cover isolates and retain their input mask values. Input-topology isolates
    cannot be repaired; use pair_stats(..., num_nodes=...) to report them.
    This is no-isolate repair, not a connected-component guarantee. It may
    exceed the sampled budget. Explicit unique pair marks permit coupled replay.
    """
    assert mask.dtype == torch.bool, f'expected bool mask, got {mask.dtype}'
    layout = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    chosen = repair_pairs(layout.pairs, layout.collapse(mask), layout.num_nodes, pair_marks=pair_marks)
    out = layout.expand(chosen, keep_self_loops=False)
    out[~layout.real] = mask[~layout.real]
    return out


def pair_asymmetry_fraction(edge_index, mask):
    """Fraction of physical pairs whose input copies disagree on membership.

    All copies count, even duplicated one-way columns. Singleton physical pairs
    necessarily agree. Self-loops are excluded from numerator and denominator.
    """
    assert mask.dtype == torch.bool, f'expected bool mask, got {mask.dtype}'
    layout = PhysicalPairLayout.from_edge_index(edge_index)
    any_kept = layout.collapse(mask)
    any_dropped = layout.collapse(~mask)
    return float((any_kept & any_dropped).float().mean()) if len(layout.pairs) else 0.0


def symmetrize_probs(edge_index, probs):
    """Mean over every copy of each physical pair; loops keep their values."""
    layout = PhysicalPairLayout.from_edge_index(edge_index)
    if probs.shape != layout.real.shape:
        raise ValueError('probs must be a vector matching edge_index columns')
    scores = pair_scores(edge_index, probs, (layout.pairs, layout.inverse, layout.real))
    out = probs.clone()
    out[layout.real] = scores[layout.inverse]
    return out


def _mirror_mask(edge_index, mask):
    """OR over physical copies, retaining input loop membership."""
    layout = PhysicalPairLayout.from_edge_index(edge_index)
    out = layout.expand(layout.collapse(mask), keep_self_loops=False)
    out[~layout.real] = mask[~layout.real]
    return out


def topk_scheduled_mask(edge_index, probs, target_keep_frac, keep_self_loops=True,
                        *, pair_marks=None):
    """Exactly ceil(fraction * physical_pairs) before any legacy repair.

    Includes one-way/duplicate representations. Scores are pair means; cutoff
    ties use exchangeable random marks, or transported explicit unique marks.
    """
    mask = pair_mask(edge_index, probs, target_keep_frac, repair=False, pair_marks=pair_marks)[0]
    if not keep_self_loops:
        mask[edge_index[0] == edge_index[1]] = False
    return mask


def eval_mask(edge_index, probs, cfg, target_sparsity=None):
    """Preserve the configured legacy training decoder for inference."""
    mode = cfg.get('sparsity_mode', 'penalty')
    target = cfg.get('target_sparsity_end', .4) if target_sparsity is None else target_sparsity
    if mode == 'pair_pl':
        scores = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        return pair_mask(edge_index, scores, target)[0]
    if mode == 'topk_scheduled':
        return repair_symmetric(edge_index, topk_scheduled_mask(edge_index, probs, target))
    return final_symmetric_mask(edge_index, probs, cfg.get('min_keep_frac', .1))


def repair_symmetric(edge_index, mask, keep_self_loops=True, *, pair_marks=None, num_nodes=None):
    """Legacy sampled-action repair; symmetric copies and optional loop retention."""
    mask = repair_connectivity(edge_index, mask, pair_marks=pair_marks, num_nodes=num_nodes)
    if keep_self_loops:
        mask[edge_index[0] == edge_index[1]] = True
    return mask


def final_symmetric_mask(edge_index, probs, min_keep_frac=0.1, keep_self_loops=True,
                         *, pair_marks=None, num_nodes=None):
    """Physical-pair threshold/floor, then legacy additive no-isolate repair.

    Floor ties are inclusive; repair marks are exchangeable. Both may exceed
    the floor. For an exact budget, use constrained_policy.select_pairs.
    """
    layout = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    scores = pair_scores(edge_index, probs, (layout.pairs, layout.inverse, layout.real))
    # Validate the physical fraction even for an empty graph.
    pair_budget(layout, min_keep_frac)
    selected = hard_mask(scores, min_keep_frac)
    selected = repair_pairs(layout.pairs, selected, layout.num_nodes, pair_marks=pair_marks)
    return layout.expand(selected, keep_self_loops)
