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

    For each isolated node, force-keep its nearest-neighbor edge (first in
    edge_index order among its incident edges is a deterministic fallback;
    nearest by distance is handled by the caller via edge reordering)."""
    mask = mask.clone()
    kept = mask.bool()
    incident = torch.zeros(edge_index.max().item() + 1, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        if kept[i]:
            incident[edge_index[0, i]] += 1
            incident[edge_index[1, i]] += 1
    isolated = (incident == 0).nonzero(as_tuple=True)[0]
    for node in isolated.tolist():
        cand = (edge_index == node).sum(dim=0).bool()
        if cand.any():
            first = cand.nonzero(as_tuple=True)[0][0]
            mask[first] = 1
    return mask
