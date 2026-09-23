"""Shared requested-budget resolution + legacy-repair accounting (Task 3c).

One resolved requested keep fraction q for K=0 and K>0:
explicit argument, else cfg tta_target_sparsity, else target_sparsity_end.
Reports requested/final/sampled counts and post-repair additions/excess for
existing additive repair paths (decoder_kind='legacy_repaired'). The new
matched benchmark is the exact-budget path and lives elsewhere.
"""
import math
import numbers

import torch

from rls.constraints import PhysicalPairLayout, pair_budget, selection_diagnostics

LEGACY_DECODER_KIND = "legacy_repaired"


def _lookup(cfg, key):
    if not isinstance(cfg, dict):
        return None
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    nested = cfg.get("rls")
    if isinstance(nested, dict) and key in nested and nested[key] is not None:
        return nested[key]
    return None


def _validate_q(value, what="requested keep fraction"):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a real number in [0, 1], got {value!r}")
    q = float(value)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"{what} must be finite in [0, 1], got {value!r}")
    return q


def resolve_requested_keep(cfg, explicit=None):
    """Return one resolved q: explicit, else tta_target_sparsity, else target_sparsity_end.

    Falls back to the historical 0.5 default only when neither the explicit
    argument nor the config declares a budget, preserving public default usage.
    """
    if explicit is not None:
        return _validate_q(explicit, "explicit target_sparsity")
    candidate = _lookup(cfg, "tta_target_sparsity")
    if candidate is not None:
        return _validate_q(candidate, "tta_target_sparsity")
    candidate = _lookup(cfg, "target_sparsity_end")
    if candidate is not None:
        return _validate_q(candidate, "target_sparsity_end")
    return 0.5


def requested_pair_count(edge_index, keep_fraction, num_nodes=None):
    """ceil(q * P) for the physical-pair layout; never repairs or clamps."""
    layout = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    return int(pair_budget(layout, float(keep_fraction))), layout


def legacy_repair_report(edge_index, mask, *, keep_fraction, order=None,
                         num_nodes=None, requested_count=None):
    """Truthful diagnostics for additive legacy repair paths.

    order must be the actual returned policy-action order (or None when the
    path has no ordered sampling, e.g. penalty-mode threshold). Never infers
    sampled counts from the final mask when repair inflated it.
    """
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a bool tensor")
    layout = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    chosen = layout.collapse(mask)
    q = _validate_q(keep_fraction, "keep_fraction")
    if requested_count is None:
        requested_count = int(pair_budget(layout, q))
    else:
        if isinstance(requested_count, bool) or not isinstance(requested_count, numbers.Integral):
            raise ValueError("requested_count must be an integer")
        requested_count = int(requested_count)
    sampled = None if order is None else int(order.numel())
    diag = selection_diagnostics(layout, chosen, k=None, sampled_count=sampled,
                                 scaffold_count=0, constraint="none")
    final = int(chosen.sum())
    total_keep = float(mask.float().mean()) if mask.numel() else 0.0
    physical_keep = float(chosen.float().mean()) if len(chosen) else 0.0
    out = {
        "decoder_kind": LEGACY_DECODER_KIND,
        "requested_keep_fraction": q,
        "requested_pair_count": requested_count,
        "sampled_pair_count": sampled,
        "final_pair_count": final,
        "available_pair_count": len(layout.pairs),
        "physical_isolates": int(diag["physical_isolates"]),
        "physical_components": int(diag["physical_components"]),
        "physical_pair_keep": physical_keep,
        "total_edge_keep": total_keep,
    }
    if sampled is not None:
        out["repair_added_pair_count"] = final - sampled
    out["budget_excess_pair_count"] = final - requested_count
    return out
