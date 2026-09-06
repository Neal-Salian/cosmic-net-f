"""Hard decision layer: probabilities -> binary masks with guarantees."""
import torch


def hard_mask(probs, min_keep_frac=0.1):
    """Threshold at 0.5, then force-keep the top-k highest-prob edges so the
    kept fraction is at least min_keep_frac. Returns a BOOL mask — boolean
    indexing everywhere (edge_index[:, mask]) requires bool, not 0/1 long."""
    mask = (probs >= 0.5)
    k_min = int(torch.ceil(torch.tensor(min_keep_frac) * probs.numel()))
    if mask.sum() < k_min:
        top = torch.topk(probs, k_min).indices
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[top] = True
    return mask


def apply_min_keep_floor(mask, probs, min_keep_frac=0.1):
    """Same floor-enforcement as hard_mask, but the candidate mask need
    not be a thresholded-probs mask — e.g. a sampled Bernoulli action.
    If the candidate keeps fewer than min_keep_frac of edges, replace it
    with the top-(min_keep_frac) highest-probability edges instead."""
    assert mask.dtype == torch.bool, f"expected bool mask, got {mask.dtype}"
    mask = mask.clone()
    k_min = int(torch.ceil(torch.tensor(min_keep_frac) * probs.numel()))
    if mask.sum() < k_min:
        top = torch.topk(probs, k_min).indices
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[top] = True
    return mask


def repair_connectivity(edge_index, mask):
    """Guarantee every node has >= 1 incident kept edge.

    For each isolated node, force-keep its first incident edge (in edge_index
    order). Device-safe: the degree accumulator lives on edge_index's device,
    so this works identically on CPU and CUDA (a CPU-only accumulator crashed
    every GPU run). Requires a BOOL mask: downstream edge_index[:, mask] /
    edge_attr[mask] silently positional-index with an int 0/1 mask.

    NOTE (audit Sep 2026): repair adds a SINGLE directed edge, which can break
    pairwise (u->v == v->u) symmetry. Callers that need a symmetric final mask
    must pass the result through mirror_repair (inside final_symmetric_mask).
    """
    assert mask.dtype == torch.bool, f"expected bool mask, got {mask.dtype}"
    mask = mask.clone()
    device = edge_index.device
    num_nodes = int(edge_index.max().item()) + 1
    incident = torch.zeros(num_nodes, dtype=torch.long, device=device)
    kept_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if kept_idx.numel() > 0:
        ones = torch.ones(kept_idx.numel(), dtype=torch.long, device=device)
        incident.index_add_(0, edge_index[0, kept_idx], ones)
        incident.index_add_(0, edge_index[1, kept_idx], ones)
    isolated = (incident == 0).nonzero(as_tuple=False).squeeze(-1)
    for node in isolated.tolist():
        cand = (edge_index == node).sum(dim=0).bool()
        if cand.any():
            first = int(cand.nonzero(as_tuple=False)[0].item())
            mask[first] = True
    return mask


# ---------------------------------------------------------------------------
# Pairwise-symmetry layer (FIX Sep 2026, audit P0-1/P0-2).
#
# WHAT WAS WRONG: graphs are built symmetric (each physical neighbor pair =
# two directed rows u->v and v->u, plus self-loops), but EdgePolicyNet scores
# concat([nu, nv, ...]) which is NOT invariant to swapping u and v, so
# p(u->v) != p(v->u) in general (measured: 2/10 pairs differ by >0.01 at init;
# after thresholding near 0.5, ~97/200 pairs end up asymmetric). hard_mask /
# repair_connectivity then acted per directed edge, feeding the frozen GNN a
# directionally-asymmetric topology it never saw in training. REINFORCE
# responded by learning "keep everything" (keep stalled at 0.96-1.0 while the
# curriculum annealed to 0.40).
#
# CHOICE: symmetrize the OUTPUT (pair-averaged probabilities), not the input.
# Input symmetrization (nu+nv, |nu-nv|) would change the policy architecture
# and invalidate every trained checkpoint; output symmetrization keeps
# EdgePolicyNet untouched and enforces the invariant at the decision layer,
# where it can be unit-tested independently of the network. Combination is the
# pair MEAN: min would over-prune (drop if either direction is uncertain),
# max would under-prune; mean preserves gradient flow to both directions.
#
# SELF-LOOPS: always retained (excluded from the action space). The GNN was
# trained with self-loops on every node; dropping them is a separate,
# untested distribution shift, so the final mask forces them True.
# ---------------------------------------------------------------------------

def _reverse_index(edge_index):
    """Map each directed edge i to the index j of its reverse (v->u), or -1.

    Handles odd/duplicate edges: for duplicates, maps to the first reverse
    occurrence; edges with no reverse present (odd edges, self-loops) map
    to -1. Self-loops map to themselves.
    """
    src, dst = edge_index[0], edge_index[1]
    key_to_first = {}
    for j in range(edge_index.shape[1]):
        key = (int(src[j].item()), int(dst[j].item()))
        if key not in key_to_first:
            key_to_first[key] = j
    rev = torch.full((edge_index.shape[1],), -1, dtype=torch.long,
                     device=edge_index.device)
    for i in range(edge_index.shape[1]):
        u, v = int(src[i].item()), int(dst[i].item())
        if u == v:
            rev[i] = i
        elif (v, u) in key_to_first:
            rev[i] = key_to_first[(v, u)]
    return rev


def pair_asymmetry_fraction(edge_index, mask):
    """Diagnostic (audit P0-1): fraction of undirected pairs (u,v), u != v,
    with mask[u->v] != mask[v->u]. Self-loops and odd edges (no reverse in
    edge_index) are excluded. Returns a float in [0, 1]."""
    assert mask.dtype == torch.bool, f"expected bool mask, got {mask.dtype}"
    rev = _reverse_index(edge_index)
    src, dst = edge_index[0], edge_index[1]
    n_asym, n_pairs = 0, 0
    seen = set()
    for i in range(edge_index.shape[1]):
        u, v = int(src[i].item()), int(dst[i].item())
        if u == v or int(rev[i].item()) < 0:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        n_pairs += 1
        if bool(mask[i].item()) != bool(mask[int(rev[i].item())].item()):
            n_asym += 1
    return (n_asym / n_pairs) if n_pairs > 0 else 0.0


def symmetrize_probs(edge_index, probs):
    """Pair-consensus probabilities: every directed copy of an unordered pair
    {u,v} gets the MEAN probability over ALL copies (both directions,
    including duplicates).

    Done by pair-consensus rather than pairwise averaging so duplicate
    directed edges cannot leave the group with split values (same fixpoint
    issue as _mirror_mask). Edges with no reverse present (odd edges) and
    self-loops keep their own probability. Returns a new tensor."""
    rev = _reverse_index(edge_index)
    src, dst = edge_index[0], edge_index[1]
    out = probs.clone()
    seen = set()
    for i in range(probs.numel()):
        u, v = int(src[i].item()), int(dst[i].item())
        if u == v or int(rev[i].item()) < 0:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        copies = ((src == u) & (dst == v)) | ((src == v) & (dst == u))
        out[copies] = probs[copies].mean()
    return out


def _mirror_mask(edge_index, mask):
    """Pair-consensus OR: for every unordered pair {u,v} (u != v) present in
    edge_index, the pair is kept iff ANY directed copy (either direction,
    including duplicates) is kept; then ALL copies are set to that value.

    Single-pass mirroring (if i kept, keep reverse(i)) is NOT enough: with
    duplicate directed edges, an edge set True during the pass is never
    itself mirrored, leaving its twin asymmetric (caught by
    test_repair_symmetric_postcondition_on_sampled_masks). The consensus form
    reaches the fixpoint in one shot. Repair only ADDS edges, so consensus-OR
    preserves the no-isolated-nodes guarantee while restoring pair-symmetry.
    Self-loops are unaffected. Odd edges (no reverse present) keep their own
    value and are excluded from the asymmetry diagnostic."""
    rev = _reverse_index(edge_index)
    src, dst = edge_index[0], edge_index[1]
    out = mask.clone()
    seen = set()
    for i in range(edge_index.shape[1]):
        u, v = int(src[i].item()), int(dst[i].item())
        if u == v or int(rev[i].item()) < 0:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        copies = ((src == u) & (dst == v)) | ((src == v) & (dst == u))
        if bool(mask[copies].any().item()):
            out[copies] = True
    return out


def topk_scheduled_mask(edge_index, probs, target_keep_frac,
                        keep_self_loops=True):
    """Structural-sparsity decision layer (FIX Sep 2026, audit P0-reward).

    WHAT WAS WRONG: the soft w_sp*(keep-target)^2 penalty let the policy eat
    a fixed sparsity penalty and keep ~everything (keep 0.96-1.0 vs target
    0.40) instead of learning to prune.

    WHAT THIS DOES: the sparsity level becomes a SCHEDULED CONSTRAINT — keep
    exactly the top-k non-self-loop edges by learned probability at the
    current curriculum target (k = ceil(target * n_prunable)), so the
    probabilities only decide WHICH edges to drop, not how many. Self-loops
    are always retained on top (excluded from the k budget, matching the
    action-space decision). Pair-consensus restores symmetry afterwards, so
    the postcondition pair_asymmetry_fraction == 0.0 still holds (consensus
    can only add edges, so keep >= target afterwards — the constraint binds
    from below, which is the safe direction for the curriculum).

    Returns a BOOL mask. Switchable via rls.sparsity_mode:
    "penalty" (old behavior) | "topk_scheduled" (this).
    """
    assert probs.dtype in (torch.float32, torch.float64), \
        f"expected float probs, got {probs.dtype}"
    n = probs.numel()
    is_loop = edge_index[0] == edge_index[1]
    src, dst = edge_index[0], edge_index[1]
    mask = torch.zeros(n, dtype=torch.bool, device=probs.device)
    # Select over UNDIRECTED pairs, not directed edges: probs are pair-equal
    # after symmetrize_probs, but top-k over directed rows can still cut
    # inside a pair (keep u->v, drop v->u) and consensus-OR then inflates keep
    # far above target (measured 0.60 vs target 0.40 on 10 pairs). Pair-level
    # selection keeps exactly ceil(target * n_pairs) pairs.
    rev = _reverse_index(edge_index)
    seen, pair_idx, pair_prob = set(), [], []
    for i in range(n):
        u, v = int(src[i].item()), int(dst[i].item())
        if u == v or int(rev[i].item()) < 0:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        pair_idx.append(i)
        pair_prob.append(float(probs[i].item()))
    n_pairs = len(pair_idx)
    if n_pairs > 0:
        k = min(n_pairs, int(torch.ceil(
            torch.tensor(target_keep_frac * n_pairs)).item()))
        k = max(k, 1)
        order = torch.argsort(torch.tensor(pair_prob), descending=True)[:k]
        for j in order.tolist():
            i = pair_idx[int(j)]
            u, v = int(src[i].item()), int(dst[i].item())
            mask[((src == u) & (dst == v)) | ((src == v) & (dst == u))] = True
    mask = _mirror_mask(edge_index, mask)
    if keep_self_loops and bool(is_loop.any().item()):
        mask[is_loop] = True
    return mask


def repair_symmetric(edge_index, mask, keep_self_loops=True):
    """Repair path for SAMPLED (training/TTA) masks: floor/repair may break
    pair-symmetry, so mirror repair additions and force self-loops True.

    Postcondition: pair_asymmetry_fraction == 0.0 (and self-loops kept when
    keep_self_loops=True). Use final_symmetric_mask for the deterministic
    (eval/frozen) path; use this for the stochastic training path where the
    candidate mask is a Bernoulli sample rather than a thresholded mask."""
    assert mask.dtype == torch.bool, f"expected bool mask, got {mask.dtype}"
    mask = repair_connectivity(edge_index, mask)
    mask = _mirror_mask(edge_index, mask)
    if keep_self_loops:
        self_loop = edge_index[0] == edge_index[1]
        if bool(self_loop.any().item()):
            mask = mask.clone()
            mask[self_loop] = True
    return mask


def final_symmetric_mask(edge_index, probs, min_keep_frac=0.1,
                         keep_self_loops=True):
    """End-to-end decision layer with a PROVABLE pair-symmetry invariant.

    symmetrize probs -> threshold/floor (hard_mask) -> repair_connectivity
    -> OR-mirror repair additions -> force self-loops True.

    Postcondition (asserted in tests): pair_asymmetry_fraction == 0.0 and,
    when keep_self_loops=True, every self-loop edge is kept.
    """
    assert probs.dtype in (torch.float32, torch.float64), \
        f"expected float probs, got {probs.dtype}"
    sym = symmetrize_probs(edge_index, probs)
    mask = hard_mask(sym, min_keep_frac=min_keep_frac)
    mask = repair_connectivity(edge_index, mask)
    mask = _mirror_mask(edge_index, mask)
    if keep_self_loops:
        self_loop = edge_index[0] == edge_index[1]
        if bool(self_loop.any().item()):
            mask = mask.clone()
            mask[self_loop] = True
    return mask
