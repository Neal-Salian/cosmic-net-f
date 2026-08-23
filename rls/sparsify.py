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


def repair_connectivity(edge_index, mask):
    """Guarantee every node has >= 1 incident kept edge.

    For each isolated node, force-keep its first incident edge (in edge_index
    order). Device-safe: the degree accumulator lives on edge_index's device,
    so this works identically on CPU and CUDA (a CPU-only accumulator crashed
    every GPU run). Requires a BOOL mask: downstream edge_index[:, mask] /
    edge_attr[mask] silently positional-index with an int 0/1 mask.
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
