"""Bounded, exact-budget selection primitives for the synthetic pilot.

This pass provides decoder/gradient/config contracts. The training/artifact
runner is intentionally deferred to the next implementation pass.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from rls.constraints import (
    PhysicalPairLayout, feasible_budget, pair_marks as make_pair_marks,
    selection_diagnostics,
)
from rls.constrained_policy import select_pairs
from rls.pair_policy import repair_pairs
from rls.structured_selection import enumerate_feasible_masks, structured_gibbs

METHODS = ('repaired_pl', 'scaffold', 'direct', 'structured')
ORACLE_MASKS_PER_BUDGET = {2: 3, 3: 16, 4: 15}


def reinforce_loss(log_probabilities, rewards):
    """Mean REINFORCE loss using detached leave-one-out reward baselines."""
    if (not isinstance(log_probabilities, torch.Tensor)
            or not isinstance(rewards, torch.Tensor)
            or log_probabilities.ndim != 1 or rewards.shape != log_probabilities.shape
            or log_probabilities.numel() < 2):
        raise ValueError('log probabilities and rewards must be matching vectors with B >= 2')
    if not log_probabilities.is_floating_point() or not rewards.is_floating_point():
        raise ValueError('log probabilities and rewards must be floating point')
    if not torch.isfinite(log_probabilities).all() or not torch.isfinite(rewards).all():
        raise ValueError('log probabilities and rewards must be finite')
    detached = rewards.detach()
    baselines = (detached.sum() - detached) / (len(detached) - 1)
    return -((detached - baselines).detach() * log_probabilities).mean()


def _ordered_mask_order(masks, marks):
    order = sorted(range(len(masks)), key=lambda i: tuple(sorted(marks[masks[i]].tolist())))
    return masks[order]


def decode(layout, scores, k, method, pair_marks=None, generator=None,
           constraint='no_isolates', temperature=1.0, sample=True):
    """Decode one sample/MAP selection and return final mask and diagnostics.

    ``log_probability`` is the actual sampled action likelihood: ordered PL
    action for repaired/scaffold/direct, and unordered feasible-mask Gibbs for
    structured. Repaired PL reports the latent likelihood, never projected
    mask probability. Deterministic calls have ``None`` likelihood.
    """
    if not isinstance(layout, PhysicalPairLayout):
        raise ValueError('layout must be a PhysicalPairLayout')
    if scores.ndim != 1 or scores.shape != (len(layout.pairs),):
        raise ValueError('scores must have one value per physical pair')
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError('scores must be finite floating point values')
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError('k must be an integer physical-pair budget')
    report = feasible_budget(layout, k, constraint)
    if not report['feasible']:
        raise ValueError(f'infeasible physical-pair budget: {report}')
    marks = make_pair_marks(layout, pair_marks, generator)
    if not isinstance(sample, bool):
        raise ValueError('sample must be boolean')
    sampled_count = k
    repaired_count = k
    projected_count = k
    original = None
    repaired = None
    projected = None
    scaffold_count = 0
    if method in ('direct', 'scaffold'):
        selected = select_pairs(layout, scores, k, constraint,
                                method=method, sample=sample,
                                pair_marks=marks, generator=generator)
        chosen, logp = selected.selected, selected.log_probability
        sampled_count = int(chosen.sum())
        scaffold_count = int(selected.scaffold.sum())
        likelihood_kind = 'ordered_action_conditional_on_marks'
        limits = {'no_residual_zero_gradient': bool(method == 'scaffold' and k == report['minimum_pairs'])}
    elif method == 'repaired_pl':
        latent = select_pairs(layout, scores, k, 'none', method='direct',
                              sample=sample, pair_marks=marks, generator=generator)
        sampled = latent.selected
        repaired = repair_pairs(layout.pairs, sampled, layout.num_nodes,
                                pair_marks=marks)
        repaired_count = int(repaired.sum())
        feasible_masks = _ordered_mask_order(
            enumerate_feasible_masks(layout, k, constraint, pair_marks=marks), marks)
        if not len(feasible_masks):
            raise ValueError('no feasible masks for projection')
        overlaps = (feasible_masks & repaired).sum(dim=1)
        # Enumeration order is based only on transported independent marks;
        # argmax therefore resolves equal-overlap projections equivariantly.
        chosen = feasible_masks[int(overlaps.argmax())]
        logp = latent.log_probability
        sampled_count = int(sampled.sum())
        projected_count = int(chosen.sum())
        original = sampled
        projected = chosen
        likelihood_kind = 'ordered_pl_latent_action_not_projected_mask_probability'
        limits = {'projection_tie_policy': 'transported_pair_mark_mask_order'}
    else:
        masks = _ordered_mask_order(
            enumerate_feasible_masks(layout, k, constraint, pair_marks=marks), marks)
        if not len(masks):
            raise ValueError('no feasible masks for structured selection')
        logps = structured_gibbs(scores, masks, temperature=temperature)
        if sample:
            draw = int(torch.multinomial(logps.detach().exp(), 1, generator=generator))
            chosen, logp = masks[draw], logps[draw]
        else:
            draw = int(logps.detach().argmax())
            chosen, logp = masks[draw], None
        sampled_count = int(chosen.sum())
        likelihood_kind = 'unordered_feasible_mask_gibbs'
        limits = {'support': 'all enumerated feasible masks', 'max_pairs': 8, 'max_k': 4}
    layout.validate_selection(chosen)
    diag = selection_diagnostics(layout, chosen, k=k,
                                 sampled_count=sampled_count,
                                 scaffold_count=scaffold_count,
                                 constraint=constraint)
    if int(chosen.sum()) != k or not diag['constraint_satisfied']:
        raise RuntimeError('decoder violated exact budget or declared constraint')
    out = {
        'selected': chosen,
        'mask': layout.expand(chosen, keep_self_loops=True),
        'log_probability': logp,
        'sampled_log_probability': logp,
        'likelihood_kind': likelihood_kind,
        'diagnostics': diag,
        'requested_count': int(k), 'sampled_count': int(sampled_count),
        'repaired_count': int(repaired_count), 'projected_count': int(projected_count),
        'final_count': int(chosen.sum()), 'scaffold_count': scaffold_count,
        'sampled_selection': original,
        'repaired_selection': repaired,
        'projected_selection': projected,
        'support_limits': limits,
    }
    if method == 'repaired_pl':
        out.update(
            repaired_added=repaired & ~original,
            repaired_removed=original & ~repaired,
            repaired_xor=original ^ repaired,
            projected_added=projected & ~repaired,
            projected_removed=repaired & ~projected,
            projected_xor=repaired ^ projected,
        )
    return out


def default_config():
    """Approved bounded pilot configuration."""
    return {
        'mode': 'capped_pilot', 'policy_seeds': [0, 1, 2],
        'data_seed': 1001, 'validation_seed': 2001,
        'n_train_add': 8, 'n_train_int': 8,
        'n_val_add': 4, 'n_val_int': 4,
        'epochs': 10, 'budgets': [2, 3, 4], 'constraint': 'no_isolates',
        'max_pairs': 6, 'max_k': 4, 'rollouts_B': 4,
        'learning_rate': 0.001, 'hidden_dim': 16,
        'structured_temperature': 1.0,
    }


def smoke_config():
    cfg = default_config()
    cfg.update(mode='smoke', policy_seeds=[0], n_train_add=1, n_train_int=1,
               n_val_add=1, n_val_int=1, epochs=1, budgets=[2, 3, 4])
    return cfg


def validate_caps(config, *, allow_smoke=False):
    """Reject unsupported pilot requests before data, output, or training work."""
    if not isinstance(config, dict):
        raise ValueError('config must be a mapping')
    budgets = config.get('budgets')
    if (not isinstance(budgets, list)
            or any(isinstance(k, bool) or not isinstance(k, int) for k in budgets)):
        raise ValueError('budgets must be a list of integer physical-pair counts')
    mode = config.get('mode')
    if mode not in ('capped_pilot', 'smoke'):
        raise ValueError('mode must be capped_pilot or smoke')
    if mode == 'smoke' and not allow_smoke:
        raise ValueError('smoke configuration requires explicit smoke execution')

    def integer(name, minimum, maximum):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
        return value

    seeds = config.get('policy_seeds')
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                   for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('policy_seeds must contain one to three unique nonnegative integers')
    if mode == 'capped_pilot' and seeds != [0, 1, 2]:
        raise ValueError('capped pilot requires exactly policy seeds [0, 1, 2]')

    for name in ('n_train_add', 'n_train_int', 'n_val_add', 'n_val_int'):
        integer(name, 1, 64 if name.startswith('n_train') else 16)
    train = config['n_train_add'] + config['n_train_int']
    val = config['n_val_add'] + config['n_val_int']
    if train > 64 or val > 16:
        raise ValueError('aggregate training/validation examples exceed 64/16 caps')
    epochs = integer('epochs', 1, 10)
    if mode == 'capped_pilot' and epochs != 10:
        raise ValueError('capped pilot requires exactly 10 epochs')
    if config.get('budgets') != [2, 3, 4]:
        raise ValueError('pilot requires integer budgets [2, 3, 4]')
    if config.get('constraint') != 'no_isolates':
        raise ValueError('capped pilot requires the no_isolates constraint')
    if config.get('max_pairs') != 6 or config.get('max_k') != 4:
        raise ValueError('capped pilot requires max_pairs=6 and max_k=4')
    rollouts = integer('rollouts_B', 2, 4)
    if mode == 'capped_pilot' and rollouts != 4:
        raise ValueError('capped pilot requires B=4 rollouts')
    data_seed = integer('data_seed', 0, 2**31 - 1)
    validation_seed = integer('validation_seed', 0, 2**31 - 1)
    if data_seed == validation_seed:
        raise ValueError('training and validation data seeds must differ')
    for key in ('learning_rate', 'structured_temperature'):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    hidden_dim = integer('hidden_dim', 1, 16)
    if mode == 'capped_pilot' and hidden_dim != 16:
        raise ValueError('capped pilot requires hidden_dim=16')
    return True


def run_pilot(config, output_dir, *, smoke=False):
    """Pass-2 guard: validate all contracts, then defer execution to pass 3."""
    validate_caps(config, allow_smoke=smoke)
    if config.get('mode') == 'smoke' and not smoke:
        raise ValueError('smoke=True is required to run smoke configuration')
    path = Path(output_dir)
    if path.exists():
        raise ValueError(f'output path already exists: {path}')
    raise NotImplementedError('pilot runner and artifact writing are deferred to pass 3')
