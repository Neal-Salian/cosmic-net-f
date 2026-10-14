"""Shared execution gates and controls for explicitly labelled research runs.

The legacy intrinsic-feature policy is intentionally separate from projected
observations: a projected checkpoint needs its own supported policy/reward path.
"""
import copy
import json
import math
import re

import torch

from data.provenance import (model_state_hash, validate_checkpoint_file,
                             validate_run_provenance)


LEGACY_LABELS = {'legacy_total_radius', 'legacy_uncertified', 'synthetic'}


def inspect_backbone_checkpoint(checkpoint, cfg):
    """Validate artifact identity/schema before data loading or output creation."""
    info = validate_checkpoint_file(checkpoint)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(blob, dict):
        raise ValueError('backbone checkpoint must be a state mapping')
    ckpt_cfg = blob.get('config', {}) or {}
    compatibility = blob.get('compatibility_record', blob.get('compatibility'))
    if (compatibility or blob.get('artifact_type') == 'projected_backbone'
            or blob.get('projected_compatibility')
            or blob.get('model_type') == 'projected_backbone'
            or blob.get('checkpoint_kind') == 'projected_backbone'
            or ckpt_cfg.get('graph', {}).get('feature_mode') == 'projected_observation'):
        raise ValueError('projected checkpoint is incompatible with the legacy intrinsic-feature policy path')
    for field in ('node_features', 'edge_features'):
        saved = ckpt_cfg.get('model', {}).get(field)
        expected = cfg.get('model', {}).get(field)
        if saved is not None and expected is not None and saved != expected:
            raise ValueError(f'backbone input schema mismatch for {field}')
    saved_edges = ckpt_cfg.get('graph', {}).get('edge_features')
    if saved_edges is not None and list(saved_edges) != list(cfg.get('graph', {}).get('edge_features', [])):
        raise ValueError('backbone edge feature schema/order mismatch')
    provenance = blob.get('provenance')
    if provenance is None:
        label = cfg.get('data', {}).get('checkpoint_research_label') or cfg.get('data', {}).get('research_label')
        if label not in LEGACY_LABELS:
            raise ValueError('checkpoint has no provenance; an explicit legacy checkpoint_research_label is required')
    else:
        state = blob.get('model_state_dict', blob)
        validate_run_provenance(provenance, model_state=state)
        label = provenance.get('research_label')
        if not isinstance(label, str) or not label:
            raise ValueError('checkpoint provenance requires a research_label')
        declared = cfg.get('data', {}).get('checkpoint_research_label')
        if declared is not None and declared != label:
            raise ValueError('checkpoint research_label mismatch')
    return {**info, 'research_label': label, 'provenance': copy.deepcopy(provenance)}


def validate_backbone_population(checkpoint_info, catalog_contract, split_manifest):
    """Bind an in-domain policy run to the backbone's actual catalog and split."""
    provenance = checkpoint_info.get('provenance') if checkpoint_info else None
    if provenance is not None:
        if not isinstance(catalog_contract, dict) or not catalog_contract:
            raise ValueError('provenanced backbone requires an explicit evaluation catalog contract')
        validate_run_provenance(provenance, catalog_contract=catalog_contract,
                                split_manifest=split_manifest)


def guard_loader_fallbacks(loader, *, smoke):
    """Disallow the legacy loaders' generators before they create any data."""
    if smoke:
        return
    def reject(*args, **kwargs):
        raise ValueError('scientific execution refuses synthetic fallback; supply the declared catalog')
    for name in ('_generate_synthetic_camels', '_generate_synthetic_hf_data',
                 '_generate_synthetic_data'):
        if callable(getattr(loader, name, None)):
            setattr(loader, name, reject)


def validate_loaded_catalog(loader, halos, *, smoke, research_label):
    """Reject synthetic or mixed contracts hidden behind a real source label."""
    if not halos:
        raise ValueError('research execution requires nonempty data splits')
    contracts = [getattr(h, 'metadata', {}).get('catalog_contract') for h in halos]
    known = [c for c in contracts if isinstance(c, dict) and c]
    if smoke:
        return copy.deepcopy(known[0]) if known else None
    if getattr(loader, 'used_synthetic_fallback', False) and research_label != 'synthetic':
        raise ValueError('scientific execution refuses synthetic fallback data')
    from data.provenance import hash_payload
    if known and (len(known) != len(contracts) or len({hash_payload(c) for c in known}) != 1):
        raise ValueError('evaluation catalog contracts are missing or inconsistent across splits')
    if known and any(c.get('synthetic') or c.get('fallback') for c in known) and research_label != 'synthetic':
        raise ValueError('synthetic/fallback catalog cannot be evaluated under a scientific data label')
    if not known and research_label not in LEGACY_LABELS:
        raise ValueError('scientific execution requires an explicit catalog contract')
    return copy.deepcopy(known[0]) if known else None


def fixed_budget_validation(policy, graphs, gnns, cfg, device):
    """Validation at the final training keep request, using its declared decoder."""
    from rls.sparsify import eval_mask
    if not graphs:
        raise ValueError('policy selection requires a nonempty validation split')
    q = float(cfg.get('target_sparsity_end', .4))
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('target_sparsity_end must be finite in [0, 1]')
    flags = [(module, module.training) for module in policy.modules()]
    cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    predictions, targets = [], []
    try:
        policy.eval()
        # Freeze decoder tie/repair randomness across all validation epochs.
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(int(cfg.get('validation_seed', 0)))
            for graph in graphs:
                g = {key: val.to(device) if isinstance(val, torch.Tensor) else val
                     for key, val in graph.items()}
                probs = torch.sigmoid(policy(g['edge_attr'], g['emb'],
                    g['edge_index'], g['ctx'])).reshape(-1)
                mask = eval_mask(g['edge_index'], probs, cfg, target_sparsity=q)
                _, pruned = gnns(g, mask)
                predictions.append(pruned.detach().reshape(-1).cpu())
                targets.append(g['y'].detach().reshape(-1).cpu())
    finally:
        for module, flag in flags:
            module.training = flag
    errors = torch.cat(predictions) - torch.cat(targets)
    if not torch.isfinite(errors).all():
        raise ValueError('nonfinite validation predictions or targets')
    return {'rmse': float(errors.square().mean().sqrt()),
            'mae': float(errors.abs().mean()), 'requested_keep_fraction': q,
            'decoder_kind': 'legacy_repaired', 'n_graphs': len(graphs)}


def train_with_fixed_validation(trainer, graphs, gnns, cfg, device, *,
                                validation_fn, log_fn=None):
    """Select and restore policy/value states on held-out validation only."""
    from rls.train_policy import train_policy
    if cfg.get('sparsity_mode') == 'pair_pl':
        return train_policy(trainer, graphs, gnns, cfg, device,
            epochs=cfg['epochs'], log_fn=log_fn, validation_fn=validation_fn)
    best = {'rmse': math.inf, 'epoch': -1}
    diagnostics = []
    def consider(epoch):
        metrics = validation_fn(trainer.policy)
        value = float(metrics['rmse'])
        if not math.isfinite(value):
            raise ValueError('nonfinite validation RMSE')
        diagnostics.append({'epoch': epoch, **metrics})
        if value < best['rmse']:
            best.update(rmse=value, epoch=epoch,
                policy=copy.deepcopy(trainer.policy.state_dict()),
                value=copy.deepcopy(trainer.value_net.state_dict()))
    consider(-1)
    trainer.initial_validation = copy.deepcopy(diagnostics[0])
    def on_epoch(epoch, target, loss):
        consider(epoch)
        if log_fn:
            log_fn(epoch, target, loss)
    losses = train_policy(trainer, graphs, gnns, cfg, device,
        epochs=cfg['epochs'], log_fn=on_epoch, validation_fn=validation_fn)
    trainer.policy.load_state_dict(best['policy'])
    trainer.value_net.load_state_dict(best['value'])
    trainer.policy.eval()
    trainer.selected_epoch = best['epoch']
    trainer.selected_validation_rmse = best['rmse']
    trainer.validation_history = diagnostics
    return losses


def load_supervised_control(path, *, split_hash, edge_feature_names, device,
                             train_ids=None, heldout_ids=None):
    """Load the actual train_supervised_selector artifact; never fit on evaluation."""
    from rls.raw_selector import RawBudgetPairScorer, validate_supervised_artifact
    validate_checkpoint_file(path)
    artifact = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError('supervised control artifact must be a mapping')
    schema = artifact.get('normalization_metadata', {}).get('schema', '')
    match = re.fullmatch(r'node_dim=(\d+);features=(.+)', schema)
    if match is None or match[2].split(',') != list(edge_feature_names):
        raise ValueError('supervised control feature schema/order mismatch')
    state = artifact.get('model_state', {})
    width = state.get('mlp.0.weight')
    if not isinstance(width, torch.Tensor) or width.ndim != 2:
        raise ValueError('supervised artifact is missing trained model_state')
    scorer = RawBudgetPairScorer(int(match[1]), list(edge_feature_names),
                                hidden_dim=int(width.shape[0])).to(device)
    scorer.load_state_dict(state)
    validate_supervised_artifact(scorer, artifact, split_hash,
                                artifact.get('model_state_hash'), schema)
    fitted_ids = set(artifact['normalization_metadata']['train_ids'])
    if train_ids is not None and fitted_ids != set(map(str, train_ids)):
        raise ValueError('supervised control training IDs do not match the training split')
    if heldout_ids is not None and fitted_ids & set(map(str, heldout_ids)):
        raise ValueError('supervised control training IDs overlap held-out evaluation')
    scorer.eval()
    return scorer, artifact


def build_matched_scorers(policy, cfg, *, catalog_contract, split_hash,
                          supervised_artifact=None, device='cpu',
                          train_ids=None, heldout_ids=None,
                          historical_training_cost=None):
    """Canonical score controls plus an explicitly supplied trained artifact."""
    from rls.benchmark_scorers import PairScorer
    names = list(cfg.get('graph', {}).get('edge_features',
        ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep']))
    contract = catalog_contract or {}
    scorers = [PairScorer(name) for name in ('random', 'distance', 'degree')]
    scorers += [PairScorer('stellar_binding_proxy', distance_epsilon=1e-6,
                           distance_unit=contract.get('position_units', 'legacy_Mpc'),
                           stellar_mass_unit=contract.get('mass_units', 'legacy_M_sun')),
                PairScorer('edge_feature_grad_x_input'),
                PairScorer('provided_rl', policy=policy, edge_feature_names=names,
                           historical_training_cost=historical_training_cost)]
    controls = {'supervised_raw': {'status': 'absent',
        'reason': 'requires an explicitly supplied train-only supervised selector artifact'}}
    if supervised_artifact is not None:
        raw, artifact = load_supervised_control(supervised_artifact,
            split_hash=split_hash, edge_feature_names=names, device=device,
            train_ids=train_ids, heldout_ids=heldout_ids)
        scorers.append(PairScorer('supervised_raw', raw_scorer=raw,
                                 raw_artifact=artifact, split_hash=split_hash))
        controls['supervised_raw'] = {'status': 'provided_trained_artifact',
            **validate_checkpoint_file(supervised_artifact),
            'model_state_hash': artifact['model_state_hash'],
            'label_predictor_calls': artifact['label_provenance']['total_calls']}
    return scorers, controls


def comparison_csv_rows(summaries, matched_rows):
    """Export comparable headlines while retaining method-population evidence.

    The full reference may succeed on graphs where the requested sparse budget
    is infeasible. Every headline, including retention, therefore uses exactly
    the evaluator's common-success IDs. ID columns contain JSON for CSV readers.
    """
    records = []
    metric_fields = {'rmse': 'rmse', 'mae': 'mae', 'r2': 'r2',
                     'scatter': 'std_residual', 'fidelity': 'fidelity_full_pred'}
    reason_fields = {'rmse': 'rmse_reason', 'mae': 'mae_reason', 'r2': 'r2_reason',
                     'scatter': 'std_reason', 'fidelity': 'fidelity_reason'}
    for summary in summaries:
        stage = summary.get('backbone_stage')
        own = [row for row in matched_rows if row['method'] == summary['method']
               and row['status'] == 'success'
               and row.get('backbone_stage') == stage]
        common_ids = sorted(summary['common_success_ids'])
        common_set = set(common_ids)
        common = [row for row in own if row['graph_id'] in common_set]

        def retention(rows):
            fractions = [row['final_pair_count'] / row['available_pair_count']
                         for row in rows if row.get('available_pair_count')]
            return sum(fractions) / len(fractions) if fractions else None

        record = {
            'method': summary['method'], 'backbone_stage': stage,
            'research_label': summary.get('research_label'),
            'comparison_population': 'common_success',
            'common_n': summary['common_n'],
            'common_success_ids': json.dumps(common_ids),
            'n_success': summary['common_n'],
            'mean_keep_frac': retention(common),
            'method_population_mean_keep_frac': retention(own),
            'method_population_n_requested': summary.get('n_requested'),
            'method_population_n_success': summary.get('n_success'),
            'method_population_n_failed': summary.get('n_failed'),
            'method_population_success_ids': json.dumps(summary.get('success_graph_ids', [])),
            'headline_comparison_allowed': summary.get('headline_comparison_allowed'),
            'headline_comparison_reason': summary.get('headline_comparison_reason'),
        }
        for column, metric in metric_fields.items():
            record[column] = summary.get('common_' + metric)
            record[column + '_reason'] = summary.get('common_' + reason_fields[column])
            record['method_population_' + column] = summary.get(metric)
            record['method_population_' + column + '_reason'] = summary.get(reason_fields[column])
        records.append(record)
    return records
