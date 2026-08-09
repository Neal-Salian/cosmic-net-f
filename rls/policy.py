"""Per-edge keep/drop policy network (inductive, graph-size agnostic)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePolicyNet(nn.Module):
    def __init__(self, edge_dim=5, node_emb_dim=128, hidden_dim=64):
        super().__init__()
        self.node_proj = nn.Linear(node_emb_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr, node_emb, edge_index, context):
        """Logits per edge. context: [emb_dim] graph-level pooled embedding.

        Edge ordering must be permutation-covariant: permuting (edge_attr,
        edge_index) together permutes logits identically.
        """
        e = F.leaky_relu(self.edge_proj(edge_attr), 0.1)          # [E,H]
        u, v = edge_index
        nu = F.leaky_relu(self.node_proj(node_emb[u]), 0.1)       # [E,H]
        nv = F.leaky_relu(self.node_proj(node_emb[v]), 0.1)       # [E,H]
        c = F.leaky_relu(self.node_proj(context.unsqueeze(0)), 0.1)  # [1,H]
        c = c.expand(e.size(0), -1)                               # [E,H]
        h = torch.cat([nu, nv, e, c], dim=-1)                     # [E,4H]
        return self.fc(h)                                          # [E,1]


def build_policy(cfg, node_emb_dim=None):
    rls_cfg = cfg.get("rls", {})
    return EdgePolicyNet(
        edge_dim=len(cfg.get("graph", {}).get("edge_features", ["distance", "delta_v",
                                                                 "cos_theta", "mass_ratio",
                                                                 "proj_sep"])),
        node_emb_dim=node_emb_dim or cfg["model"]["output_dim"],
        hidden_dim=rls_cfg.get("policy_hidden", 64),
    )
