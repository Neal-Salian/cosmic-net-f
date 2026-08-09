"""Policy-gradient training: REINFORCE with a learned baseline + entropy bonus.

WHY NOT PPO: each graph is a one-step MDP (the decision is a single mask
sample; no sequential credit assignment). At horizon 1, GAE degenerates to
reward - value and a clipped-importance ratio would be exactly 1 forever
(old and new log-probs come from the same forward pass before any update).
PPO's machinery therefore adds nothing here; REINFORCE with a baseline is
the statistically equivalent, honest choice — and the paper says exactly
this. If a multi-step MDP is ever added, swap in real PPO (rollout buffer
+ K update epochs) at this boundary."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    """Critic/baseline: graph context embedding -> scalar state value."""

    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ctx):
        return self.net(ctx).squeeze(-1)


def compute_advantages(rewards, values):
    """Advantage = reward - baseline. One-step MDP: no GAE, no bootstrap."""
    return rewards - values.detach()


def compute_pg_loss(logp, advantages, values, rewards,
                    entropy=0.0, value_coef=0.5, entropy_coef=0.01):
    """logp: [T] log-prob of sampled action per graph (mean over its edges).

    Returns (loss, pg_loss, vf_loss, entropy). advantages/values/rewards: [T].
    """
    pg_loss = -(advantages.detach() * logp).mean()
    vf_loss = F.mse_loss(values, rewards.detach())
    loss = pg_loss + value_coef * vf_loss - entropy_coef * entropy
    return loss, pg_loss, vf_loss, entropy


class PolicyGradientTrainer:
    """Trains EdgePolicyNet over a frozen GNN backbone.

    Each 'episode' is one mini-batch of graphs: policy samples masks, GNN
    predicts on pruned graphs, reward is computed, one gradient step happens
    (no replay buffer — one-step MDPs need no importance sampling)."""

    def __init__(self, policy, value_net, optimizer, value_optimizer, cfg):
        self.policy = policy
        self.value_net = value_net
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer
        self.cfg = cfg

    def target_sparsity(self, epoch):
        s, e = self.cfg["target_sparsity_start"], self.cfg["target_sparsity_end"]
        n = max(1, self.cfg["sparsity_anneal_epochs"])
        p = min(1.0, epoch / n)
        return s + (e - s) * p


def bernoulli_logp(p, action):
    """p: [E] probs; action: [E] in {0,1}. Returns [E] log probs."""
    eps = 1e-8
    return torch.where(action.bool(),
                       torch.log(torch.clamp(p, eps, 1.0)),
                       torch.log(torch.clamp(1 - p, eps, 1.0)))


def bernoulli_entropy(p):
    """H(Bernoulli(p)) per edge: -(p log p + (1-p) log(1-p)). [E]."""
    eps = 1e-8
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def sample_actions(p):
    """Straight-through-free Bernoulli sampling."""
    return torch.bernoulli(p)
