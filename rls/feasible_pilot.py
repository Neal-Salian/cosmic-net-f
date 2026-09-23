"""Bounded synthetic feasible-selection training runner and decoder contracts.

Runs a small CPU diagnostic on synthetic K4 predictors, with exact physical
pair budgets, metered discrete predictor calls, and JSON provenance/results.
"""
from __future__ import annotations

import math
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from rls.constraints import (
    PhysicalPairLayout, feasible_budget, pair_marks as make_pair_marks,
    selection_diagnostics,
)
from rls.constrained_policy import select_pairs
from rls.pair_policy import repair_pairs
from rls.structured_selection import enumerate_feasible_masks, structured_gibbs
from rls.pilot_synthetic import (
    build_datasets, exact_oracle, SyntheticAdditivePredictor,
    SyntheticInteractionPredictor, model_definition_identity,
)
from rls.benchmark_predictor import MeteredPredictor
from rls.raw_selector import RawBudgetPairScorer
from data.provenance import hash_payload, model_state_hash, sha256_file

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
    if mode == 'capped_pilot':
        exact = {
            'data_seed': 1001, 'validation_seed': 2001,
            'n_train_add': 8, 'n_train_int': 8,
            'n_val_add': 4, 'n_val_int': 4,
            'learning_rate': 0.001, 'structured_temperature': 1.0,
        }
        for name, expected in exact.items():
            if config.get(name) != expected:
                raise ValueError(f'capped pilot requires {name}={expected}')
    return True


def run_pilot(config, output_dir, *, smoke=False):
    """Train four budget-matched decoders and persist auditable JSON artifacts."""
    validate_caps(config, allow_smoke=smoke)
    if config.get('mode') == 'smoke' and not smoke:
        raise ValueError('smoke=True is required to run smoke configuration')
    path = Path(output_dir)
    if path.exists():
        raise ValueError(f'output path already exists: {path}')
    config = copy.deepcopy(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    started = time.perf_counter()
    train, val = build_datasets(
        config['data_seed'], config['validation_seed'], config['n_train_add'],
        config['n_train_int'], config['n_val_add'], config['n_val_int'])
    examples = train + val
    train_ids = [row['graph_id'] for row in train]
    val_ids = [row['graph_id'] for row in val]
    split_hash = hash_payload({'train_ids': train_ids, 'val_ids': val_ids})
    train_pairs = [( _scorer_graph(row), k) for row in train
                   for k in config['budgets']]

    # Fit the one shared, training-only feature transform before cloning any
    # scorer. This guarantees identical feature scaling across methods/seeds.
    with torch.random.fork_rng(devices=[]):
        normalizer = RawBudgetPairScorer(4, ['distance'], config['hidden_dim'])
    normalizer.fit_normalization(train_pairs, train_ids, val_ids, split_hash)
    normalization_hash = model_state_hash({
        'mean': normalizer.norm_mean, 'std': normalizer.norm_std})
    reward_scale = max(1.0, sum(float(row['target']) ** 2 for row in train) / len(train))

    models = {family: MeteredPredictor(model) for family, model in (
        ('additive', SyntheticAdditivePredictor()),
        ('interaction', SyntheticInteractionPredictor()))}
    full_started = time.perf_counter()
    oracle = {}
    full_predictions = {}
    for row in examples:
        family = row['family']
        meter = models[family]
        full = meter.call(row['data'], phase='shared_full')
        full_predictions[row['graph_id']] = float(full.prediction.reshape(-1)[0])
    full_elapsed = time.perf_counter() - full_started
    oracle_started = time.perf_counter()
    # Oracle diagnostics are counted once per example/budget, outside training.
    # The preceding loop only generated full-graph references.
    for row in examples:
        meter = models[row['family']]
        oracle[row['graph_id']] = {}
        for k in config['budgets']:
            oracle[row['graph_id']][k] = exact_oracle(
                row['data'], row['target'], meter, k, config['constraint'])
    oracle_elapsed = time.perf_counter() - oracle_started

    curves, val_rows, init_hashes, final_hashes = [], [], [], []
    support_limits, update_records = [], []
    phase_elapsed = {'training': 0.0, 'validation': 0.0, 'oracle': oracle_elapsed,
                     'full': full_elapsed, 'mask_preparation_and_scoring': 0.0}

    for seed in config['policy_seeds']:
        for k in config['budgets']:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                base = RawBudgetPairScorer(4, ['distance'], config['hidden_dim'])
                _copy_normalization(normalizer, base)
            initial_state = copy.deepcopy(base.state_dict())
            initial_hash = model_state_hash(initial_state)
            for method in METHODS:
                scorer = copy.deepcopy(base)
                init_hashes.append({'seed': seed, 'budget': k, 'method': method,
                                    'hash': initial_hash})
                optimizer = torch.optim.Adam(scorer.parameters(),
                                             lr=float(config['learning_rate']))
                for epoch in range(config['epochs']):
                    epoch_start = time.perf_counter()
                    order_gen = torch.Generator().manual_seed(
                        700000 + seed * 10000 + k * 100 + epoch)
                    order = torch.randperm(len(train), generator=order_gen).tolist()
                    losses = []
                    grad_total = 0.0
                    epoch_duplicates = 0
                    epoch_rollouts = 0
                    for ix in order:
                        row = train[ix]
                        layout = PhysicalPairLayout.from_edge_index(
                            row['data'].edge_index, int(row['data'].x.shape[0]))
                        graph = _scorer_graph(row)
                        scorer_start = time.perf_counter()
                        scores = scorer(graph, layout, k)
                        phase_elapsed['mask_preparation_and_scoring'] += (
                            time.perf_counter() - scorer_start)
                        rollout_generator = torch.Generator().manual_seed(
                            1000000 + seed * 100000 + k * 1000 + epoch * 100 + ix)
                        lps, rewards, masks = [], [], []
                        predictor = models[row['family']]
                        for _ in range(config['rollouts_B']):
                            decode_start = time.perf_counter()
                            decoded = decode(
                                layout, scores, k, method,
                                generator=rollout_generator,
                                constraint=config['constraint'],
                                temperature=float(config['structured_temperature']),
                                sample=True)
                            phase_elapsed['mask_preparation_and_scoring'] += (
                                time.perf_counter() - decode_start)
                            with torch.no_grad():
                                result = predictor.call(
                                    row['data'], mask=decoded['mask'], phase='training')
                                pred = float(result.prediction.reshape(-1)[0])
                            se = (pred - float(row['target'])) ** 2
                            lps.append(decoded['log_probability'])
                            rewards.append(-se / reward_scale)
                            masks.append(_mask_key(decoded['selected']))
                        epoch_duplicates += len(masks) - len(set(masks))
                        epoch_rollouts += len(masks)
                        optimizer.zero_grad(set_to_none=True)
                        loss = reinforce_loss(torch.stack(lps),
                                              torch.tensor(rewards, dtype=scores.dtype))
                        loss.backward()
                        norm_sq = sum(float(p.grad.detach().pow(2).sum())
                                      for p in scorer.parameters() if p.grad is not None)
                        grad_total += math.sqrt(norm_sq)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        if epoch == config['epochs'] - 1:
                            support_limits.append({
                                'seed': seed, 'budget': k, 'method': method,
                                **decoded['support_limits']})
                    val_start = time.perf_counter()
                    val_metrics = _validate_method(
                        scorer, method, k, val, oracle, full_predictions, models,
                        config, seed, epoch, val_rows, phase_elapsed)
                    phase_elapsed['validation'] += time.perf_counter() - val_start
                    epoch_elapsed = time.perf_counter() - epoch_start
                    phase_elapsed['training'] += epoch_elapsed - (time.perf_counter() - val_start)
                    curves.append({
                        'seed': seed, 'method': method, 'budget': k, 'epoch': epoch + 1,
                        'mean_reinforce_loss': sum(losses) / max(len(losses), 1),
                        'validation_mean_se': val_metrics['mean_se'],
                        'validation_rmse': math.sqrt(val_metrics['mean_se']),
                        'validation_mean_objective': -val_metrics['mean_se'],
                        'validation_mean_regret': val_metrics['mean_regret'],
                        'training_gradient_norm_sum': grad_total,
                        'duplicate_final_masks': epoch_duplicates,
                        'training_rollout_masks': epoch_rollouts,
                        'duplicate_final_mask_rate': epoch_duplicates / max(epoch_rollouts, 1),
                        'elapsed_seconds': epoch_elapsed,
                        'validation_convention': 'deterministic_decoder_rank_or_MAP',
                    })
                final_hashes.append({'seed': seed, 'budget': k, 'method': method,
                                     'hash': model_state_hash(scorer.state_dict())})
                update = _state_delta(initial_state, scorer.state_dict())
                update_records.append({'seed': seed, 'budget': k,
                                       'method': method, 'norm': update})

    counts = _sum_meter_counts(models)
    duplicates = [
        {'seed': row['seed'], 'budget': row['budget'], 'method': row['method'],
         'epoch': row['epoch'], 'duplicate_masks': row['duplicate_final_masks'],
         'rollout_masks': row['training_rollout_masks'],
         'duplicate_rate': row['duplicate_final_mask_rate']}
        for row in curves
    ]
    elapsed_total = time.perf_counter() - started
    source_files = ['rls/feasible_pilot.py', 'rls/pilot_synthetic.py',
                    'rls/constrained_policy.py', 'rls/structured_selection.py',
                    'rls/raw_selector.py', 'rls/benchmark_predictor.py']
    source_hashes = {name: sha256_file(name)['sha256'] for name in source_files}
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        revision = None
    dataset_hash = hash_payload([row['content_hash'] for row in examples])
    config_hash = hash_payload(config)
    provenance = {
        'source_revision': revision, 'source_hash': hash_payload(source_hashes),
        'source_files': source_hashes, 'dataset_hash': dataset_hash,
        'split_hash': split_hash, 'config_hash': config_hash,
        'config': config, 'seed_roles': {'policy': config['policy_seeds'],
                                        'data': config['data_seed'],
                                        'validation': config['validation_seed']},
        'train_ids': train_ids, 'val_ids': val_ids,
        'train_content_hashes': {r['graph_id']: r['content_hash'] for r in train},
        'val_content_hashes': {r['graph_id']: r['content_hash'] for r in val},
        'reward_scale': reward_scale, 'reward_scale_fitted_on': train_ids,
        'normalization': {'fitted_on': 'train_only', 'fit_ids': train_ids,
                          'budgets': config['budgets'], 'split_hash': split_hash,
                          'state_hash': normalization_hash},
        'predictor_identities': {family: model_definition_identity(m._model)
                                 for family, m in models.items()},
    }
    summary = {
        'mode': config['mode'], 'device': 'cpu', 'queries': counts,
        'query_accounting': _query_accounting(models),
        'normalization': {'fitted_on': 'train_only', 'fit_ids': train_ids,
                          'state_hash': normalization_hash},
        'reward_scale': reward_scale, 'oracle_elapsed_seconds': oracle_elapsed,
        'phase_elapsed_seconds': phase_elapsed, 'elapsed_seconds': elapsed_total,
        'initial_scorer_hashes': init_hashes, 'final_scorer_hashes': final_hashes,
        'param_update_norm': {m: [r for r in update_records if r['method'] == m]
                              for m in METHODS},
        'support_limits': support_limits, 'duplicate_final_masks_per_B': duplicates,
        'coverage': {'training_examples': len(train), 'validation_examples': len(val),
                     'validation_rows': len(val_rows),
                     'expected_validation_rows': len(val) * len(config['policy_seeds']) *
                     len(config['budgets']) * len(METHODS) * config['epochs']},
        'constraint_failures': 0,
        'claim_limit': 'diagnostic synthetic pilot; three seeds are descriptive only',
        'oracle_cache': 'shared once per unique example and budget; no labels used in optimization',
        'validation_convention': 'deterministic_decoder_rank_or_MAP; validation labels do not tune',
        'method_likelihoods': {
            'repaired_pl': 'ordered PL latent action; not projected-mask probability',
            'scaffold': 'ordered residual action conditional on independent scaffold marks',
            'direct': 'ordered legal-action likelihood with completion checks',
            'structured': 'exact unordered feasible-mask Gibbs likelihood',
        },
    }
    _write_json(path / 'summary.json', summary)
    _write_json(path / 'curves.json', curves)
    _write_json(path / 'val_per_example.json', val_rows)
    _write_json(path / 'provenance.json', provenance)
    return {'output_dir': str(path), 'summary': summary, 'provenance': provenance}


def _scorer_graph(row):
    data = row['data']
    return {'x': data.x, 'edge_index': data.edge_index,
            'edge_attr': data.edge_attr, 'cluster_id': row['graph_id']}


def _copy_normalization(source, target):
    """Copy train-only transform while leaving target's seeded MLP intact."""
    with torch.no_grad():
        target.norm_mean.copy_(source.norm_mean)
        target.norm_std.copy_(source.norm_std)
    _copy_normalization_metadata(source, target)


def _copy_normalization_metadata(source, target):
    for name in ('_norm_fitted', '_norm_split_hash', '_norm_ids',
                 '_norm_budgets', '_norm_schema',
                 '_norm_training_content_hash', '_norm_fit_records'):
        setattr(target, name, copy.deepcopy(getattr(source, name)))


def _mask_key(mask):
    return ''.join('1' if bool(v) else '0' for v in mask.detach().cpu().tolist())


def _state_delta(before, after):
    return math.sqrt(sum(float((after[k] - before[k]).pow(2).sum())
                         for k in before if isinstance(before[k], torch.Tensor)
                         and torch.is_floating_point(before[k])))


def _validate_method(scorer, method, k, examples, oracle, full_predictions,
                     models, config, seed, epoch, output_rows, phase_elapsed):
    squared_errors, regrets = [], []
    for row in examples:
        layout = PhysicalPairLayout.from_edge_index(row['data'].edge_index, 4)
        scorer_start = time.perf_counter()
        scores = scorer(_scorer_graph(row), layout, k)
        phase_elapsed['mask_preparation_and_scoring'] += (
            time.perf_counter() - scorer_start)
        example_seed = int(hashlib.sha256(row['graph_id'].encode()).hexdigest()[:8], 16)
        marks = make_pair_marks(layout, generator=torch.Generator().manual_seed(
            500000 + seed * 100000 + k * 1000 + epoch * 100 + example_seed))
        start = time.perf_counter()
        decoded = decode(layout, scores, k, method, pair_marks=marks,
                         constraint=config['constraint'], sample=False,
                         temperature=float(config['structured_temperature']))
        phase_elapsed['mask_preparation_and_scoring'] += time.perf_counter() - start
        result = models[row['family']].call(row['data'], mask=decoded['mask'],
                                             phase='evaluation')
        prediction = float(result.prediction.reshape(-1)[0])
        se = (prediction - float(row['target'])) ** 2
        optimum = oracle[row['graph_id']][k]['se']
        regret = se - optimum
        if regret < -1e-6:
            raise RuntimeError('validation objective beat exhaustive oracle beyond tolerance')
        squared_errors.append(se)
        regrets.append(regret)
        prediction_error = prediction - float(row['target'])
        full_error = full_predictions[row['graph_id']] - float(row['target'])
        total_stored_edges = int(row['data'].edge_index.shape[1])
        selected_stored_edges = int(decoded['mask'].sum())
        total_physical_pairs = len(layout.pairs)
        output_rows.append({
            'epoch': epoch + 1, 'seed': seed, 'method': method,
            'graph_id': row['graph_id'], 'family': row['family'], 'budget': k,
            'target': float(row['target']), 'prediction': prediction,
            'prediction_error': prediction_error,
            'absolute_prediction_error': abs(prediction_error),
            'squared_error': se, 'objective': -se,
            'oracle_se': float(optimum), 'oracle_objective': -float(optimum),
            'regret': regret,
            'full_prediction': full_predictions[row['graph_id']],
            'full_prediction_error': full_error,
            'full_prediction_absolute_error': abs(full_error),
            'full_prediction_squared_error': full_error ** 2,
            'requested_pair_count': decoded['requested_count'],
            'sampled_pair_count': decoded['sampled_count'],
            'repaired_pair_count': decoded['repaired_count'],
            'projected_pair_count': decoded['projected_count'],
            'final_pair_count': decoded['final_count'],
            'total_physical_pair_count': total_physical_pairs,
            'requested_physical_retention': decoded['requested_count'] / total_physical_pairs,
            'final_physical_retention': decoded['final_count'] / total_physical_pairs,
            'total_stored_edge_count': total_stored_edges,
            'selected_stored_edge_count': selected_stored_edges,
            'total_edge_retention': selected_stored_edges / total_stored_edges,
            'constraint_satisfied': bool(decoded['diagnostics']['constraint_satisfied']),
            'likelihood_kind': decoded['likelihood_kind'],
            'support_limits': decoded['support_limits'],
        })
    return {'mean_se': sum(squared_errors) / len(squared_errors),
            'rmse': math.sqrt(sum(squared_errors) / len(squared_errors)),
            'mean_objective': -sum(squared_errors) / len(squared_errors),
            'mean_regret': sum(regrets) / len(regrets)}


def _sum_meter_counts(models):
    total = {'training': 0, 'validation': 0, 'oracle': 0, 'full': 0,
             'forward_calls': 0, 'graph_evaluations': 0}
    for meter in models.values():
        snap = meter.snapshot()
        for phase, target in (('training', 'training'), ('evaluation', 'validation'),
                              ('oracle', 'oracle'), ('shared_full', 'full')):
            calls = int(snap[phase]['forward_calls'])
            total[target] += calls
            total['forward_calls'] += calls
            total['graph_evaluations'] += int(snap[phase]['graph_evaluations'])
    return total


def _query_accounting(models):
    phase_map = {'training': 'training', 'validation': 'evaluation',
                 'oracle': 'oracle', 'full': 'shared_full'}
    out = {}
    for label, meter_phase in phase_map.items():
        out[label] = {'forward_calls': 0, 'graph_evaluations': 0,
                      'elapsed_seconds': 0.0}
        for meter in models.values():
            phase = meter.snapshot()[meter_phase]
            for key in out[label]:
                out[label][key] += phase[key]
        out[label]['forward_calls'] = int(out[label]['forward_calls'])
        out[label]['graph_evaluations'] = int(out[label]['graph_evaluations'])
    return out


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/feasible_pilot.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit('PyYAML is required to read the pilot YAML config') from exc
    config = smoke_config() if args.smoke else yaml.safe_load(Path(args.config).read_text())
    result = run_pilot(config, args.output_dir, smoke=args.smoke)
    print(json.dumps({'output_dir': result['output_dir'],
                      'queries': result['summary']['queries']}, indent=2))


if __name__ == '__main__':
    main()
