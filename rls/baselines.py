"""Non-RL sparsification baselines (Phase 0 of the paper)."""
import torch


def random_mask(edge_index, edge_attr, pos, frac, seed=0):
    # Keep the seeded CPU permutation identical across CPU and GPU runs.
    g = torch.Generator().manual_seed(seed)
    n = edge_attr.shape[0]
    k = int(frac * n)
    idx = torch.randperm(n, generator=g, device="cpu")[:k].to(edge_index.device)
    m = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    m[idx] = True
    return m


def degree_mask(edge_index, edge_attr, pos, frac):
    """Keep the highest-degree edges (hubs are informative)."""
    n = edge_index.shape[1]
    deg = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
    counts = torch.ones(n, device=edge_index.device)
    deg.index_add_(0, edge_index[0], counts)
    deg.index_add_(0, edge_index[1], counts)
    edge_deg = (deg[edge_index[0]] + deg[edge_index[1]]) / 2
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    m[torch.topk(edge_deg, k).indices] = True
    return m


def distance_mask(edge_index, edge_attr, pos, frac):
    """Keep the shortest edges (gravity acts locally)."""
    n = edge_index.shape[1]
    dists = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    m[torch.topk(-dists, k).indices] = True
    return m


def mass_ratio_mask(edge_index, edge_attr, pos, frac, mass_ratio_idx=3):
    """Keep the highest mass_ratio edges (feature index 3) — the DEGENERATE
    shortcut policy. mass_ratio is an input edge feature, so this baseline
    bounds what a trivial 'keep massive pairs' policy achieves; RL must beat
    it, and the U_ij alignment claim is meaningless without this row."""
    n = edge_index.shape[1]
    scores = edge_attr[:, mass_ratio_idx]
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    m[torch.topk(scores, k).indices] = True
    return m


def gradient_saliency_mask(edge_index, edge_attr, pos, frac, model=None, x=None):
    """Top-k edges by gradient-saliency importance (the repo's existing
    'pgexplainer' pathway: explain/explainer.py `_explain_pgexplainer`).
    model: callable(data) -> (pred, _) used for the backward pass; falls
    back to distance when model is None (used by tests). x: real node
    features [N,4]; if omitted, ones are used (tests only). This is the
    mandatory 'why RL?' control — same frozen model, same metrics."""
    if model is None:
        return distance_mask(edge_index, edge_attr, pos, frac)
    n = edge_index.shape[1]
    num_nodes = int(edge_index.max().item()) + 1
    if x is None:
        x = edge_attr.new_ones(num_nodes, 4)
    x = x.detach().clone().requires_grad_(True)
    from torch_geometric.data import Data, Batch
    pred, _ = model(Batch.from_data_list([Data(x=x, edge_index=edge_index,
                                               edge_attr=edge_attr)]))
    pred.backward()
    g = x.grad.abs().sum(dim=1)
    scores = (g[edge_index[0]] + g[edge_index[1]]) / 2
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    m[torch.topk(scores, k).indices] = True
    return m


def attention_topk_mask(edge_index, edge_attr, pos, frac, model=None):
    """Top-k edges by a scalar edge-importance score from a light MLP.
    Falls back to distance when model is None (used by tests)."""
    if model is None:
        return distance_mask(edge_index, edge_attr, pos, frac)
    n = edge_index.shape[1]
    scores = model(edge_attr).squeeze(-1)
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    m[torch.topk(scores, k).indices] = True
    return m


class GumbelEdgeMask(torch.nn.Module):
    """Differentiable edge-mask control (the 'why not Gumbel?' baseline).

    Trained end-to-end with the GNN unfrozen: sigmoid(logits) mask with
    straight-through Gumbel-Softmax sampling. This is the direct
    differentiable alternative to the RL policy."""

    def __init__(self, edge_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr, hard=False, tau=0.5):
        logits = self.net(edge_attr).squeeze(-1)      # [E]
        logits = torch.stack([-logits, logits], dim=-1)  # [E,2] keep/drop
        u = torch.rand_like(logits) + 1e-8
        g = -torch.log(-torch.log(u))
        soft = torch.softmax((logits + g) / tau, dim=-1)
        if hard:
            idx = soft.argmax(dim=-1)
            return (torch.nn.functional.one_hot(idx, 2).float() - soft).detach() + soft
        return soft[:, 1]
