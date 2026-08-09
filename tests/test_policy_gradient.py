import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.policy_gradient import (compute_advantages, compute_pg_loss,
                                 bernoulli_entropy, bernoulli_logp,
                                 sample_actions)


def test_entropy_and_logp_finite_at_saturated_probs():
    # regression: eps=1e-8 made the clamp upper bound round to 1.0 in
    # float32 → 0*log(0) = NaN → poisoned gradients (found by the
    # end-to-end smoke test on real data).
    p = torch.tensor([0.0, 1.0, 0.999999, 0.5, 1e-7])
    assert torch.isfinite(bernoulli_entropy(p)).all()
    a = sample_actions(p)
    assert torch.isfinite(bernoulli_logp(p, a)).all()


def test_advantages_are_reward_minus_baseline():
    rewards = torch.tensor([1.0, 0.0, -0.5])
    values = torch.tensor([0.5, 0.4, 0.3])
    adv = compute_advantages(rewards, values)
    assert adv.shape == rewards.shape
    # one-step MDP: no bootstrapping, no GAE — advantage is reward - value
    assert torch.allclose(adv, rewards - values, atol=1e-5)


def test_pg_loss_encourages_positive_advantage_actions():
    # Same positive advantage; the action with HIGHER sampled logp must
    # yield a LOWER (less negative is "higher") pg loss contribution.
    adv = torch.tensor([1.0])
    logp_good = torch.tensor([-0.2])
    logp_bad = torch.tensor([-1.5])
    l_good = compute_pg_loss(logp_good, adv, torch.zeros(1), torch.ones(1),
                             entropy=torch.zeros(1))[0]
    l_bad = compute_pg_loss(logp_bad, adv, torch.zeros(1), torch.ones(1),
                            entropy=torch.zeros(1))[0]
    assert l_good.item() < l_bad.item()


def test_pg_loss_decreases_negative_advantage_action_prob():
    # Negative advantage must DECREASE the prob of the sampled action.
    # d(loss)/d(logp) = -adv > 0 → minimizing the loss pushes logp DOWN.
    adv = torch.tensor([-1.0])
    logp = torch.tensor([-0.5], requires_grad=True)
    l = compute_pg_loss(logp, adv, torch.zeros(1), torch.zeros(1),
                        entropy=torch.zeros(1))[0]
    l.backward()
    assert logp.grad.item() > 0
