"""Recover completed Notebook B policy training without repeating training.

This handoff records policy training only. It never marks Notebook B, Stage B,
matched evaluation, or any subsequent campaign complete. Source artifacts are
read-only during recovery; callers write new evaluation outputs elsewhere.
"""
import copy
from datetime import datetime, timezone
import json
import math
import numbers
from pathlib import Path
import re

import torch

from data.provenance import hash_payload, sha256_file
from rls.policy import build_policy
from rls.provenance import record_backbone


SCHEMA_VERSION = 1
MANIFEST_NAME = 'training_checkpoint.json'
ARTIFACT_NAMES = ('policy.pt', 'value_net.pt', 'training_log.csv',
                  'random_validation.csv', 'policy_validation.json')
_METADATA_STATUSES = ('recorded_at_training_completion',
                      'reconstructed_from_original_revision_and_exact_inputs')


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'training checkpoint requires a nonempty {name}')
    return value


def _revision(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', value):
        raise ValueError(f'{name} must be a full immutable Git commit hash')
    return value


def _quality(value):
    if (not isinstance(value, dict) or value.get('verdict') not in ('PASS', 'FAIL')
            or value.get('budget_comparable') is not True):
        raise ValueError('training checkpoint requires budget-comparable policy validation, including valid negative results')
    return copy.deepcopy(value)


def _inputs(context):
    records = context.get('dataset')
    if not isinstance(records, dict) or not records:
        raise ValueError('training checkpoint requires dataset input checksums')
    result = {}
    for name, record in records.items():
        if not isinstance(record, dict) or not record.get('path') or not record.get('sha256'):
            raise ValueError(f'dataset input {name} lacks a path/checksum')
        path = Path(record['path'])
        if not path.is_file():
            raise ValueError(f'dataset input is missing: {path}')
        actual = sha256_file(path)
        if actual['sha256'] != record['sha256']:
            raise ValueError(f'dataset input checksum changed: {name}')
        result[name] = actual
    return result


def _context(context, cfg):
    required = ('seed', 'split_cluster_ids', 'backbone', 'run_mode',
                'research_label', 'source_identity')
    for name in required:
        if name not in context:
            raise ValueError(f'training checkpoint context lacks {name}')
    if context['seed'] != cfg.get('seed', 42):
        raise ValueError('training seed does not match context seed')
    if not isinstance(context['split_cluster_ids'], dict) or set(context['split_cluster_ids']) != {'train', 'val', 'test'}:
        raise ValueError('training checkpoint requires exact train/val/test split IDs')
    seen = set()
    for name, ids in context['split_cluster_ids'].items():
        if (not isinstance(ids, list) or not ids or any(not isinstance(cid, str) or not cid for cid in ids)
                or len(ids) != len(set(ids)) or seen.intersection(ids)):
            raise ValueError(f'training checkpoint has invalid or overlapping split IDs: {name}')
        seen.update(ids)
    if not isinstance(context['backbone'], dict) or not context['backbone'].get('sha256') or not context['backbone'].get('stage'):
        raise ValueError('training checkpoint requires a labelled backbone checksum')
    _nonempty(context['source_identity'], 'source_identity')
    _nonempty(context['run_mode'], 'run_mode')
    _nonempty(context['research_label'], 'research_label')
    graph = context.get('config_used', {}).get('graph')
    if graph != cfg.get('graph'):
        raise ValueError('training graph configuration differs from the input context')
    result = copy.deepcopy(context)
    result['dataset'] = _inputs(context)
    # run_context precedes policy overrides and fitted reward scaling in B.
    result['config_used'] = copy.deepcopy(cfg)
    return result


def _training_contract(cfg):
    rls = cfg.get('rls')
    if not isinstance(rls, dict) or not rls.get('sparsity_mode'):
        raise ValueError('training configuration requires a sparsity_mode')
    # These settings belong to subsequent evaluation/adaptation. All remaining
    # policy/reward/curriculum/optimizer settings must match the original run.
    policy_training = {key: copy.deepcopy(value) for key, value in rls.items()
        if key != 'reward_error_scale'
        and not key.startswith(('stageb_', 'evaluation_', 'tta_'))}
    data = cfg.get('data', {})
    return {'seed': cfg.get('seed', 42), 'model': copy.deepcopy(cfg.get('model', {})),
            'graph': copy.deepcopy(cfg.get('graph', {})), 'rls': policy_training,
            'data_source': data.get('source'),
            'clustered_units': data.get('tng', {}).get('clustered_units')}


def _fitted_settings(cfg):
    rls = cfg.get('rls', {})
    if rls.get('sparsity_mode') == 'pair_pl':
        scale = rls.get('reward_error_scale')
        if (isinstance(scale, bool) or not isinstance(scale, numbers.Real)
                or not math.isfinite(float(scale)) or scale <= 0):
            raise ValueError('completed pair-policy training requires its positive finite fitted reward_error_scale')


def _policy_state(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                   for value in state.values())):
        raise ValueError('policy artifact must contain a finite tensor state_dict')
    return state


def _architecture(cfg, state):
    try:
        node = state['node_proj.weight']
        edge = state['edge_proj.weight']
        actual = {'node_emb_dim': int(node.shape[1]), 'edge_dim': int(edge.shape[1]),
                  'hidden_dim': int(node.shape[0]), 'normalize': 'edge_mean' in state}
        expected = {'node_emb_dim': int(cfg['model']['output_dim']),
                    'edge_dim': len(cfg.get('graph', {}).get('edge_features',
                        ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep'])),
                    'hidden_dim': int(cfg['rls'].get('policy_hidden', 64)),
                    'normalize': bool(cfg['rls'].get('policy_normalize', False))}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError('training configuration or policy architecture is incomplete') from exc
    if actual != expected:
        raise ValueError('policy architecture does not match its training configuration')
    return actual


def save_training_checkpoint(out, context, cfg, policy_quality, repo_head, *,
        metadata_status='recorded_at_training_completion', original_source_evidence=None):
    """Create an immutable policy-training handoff after all five files exist.

    Reconstructed metadata must be explicitly labelled and accompanied by the
    operator's original-source evidence. This function cannot infer that source.
    Returns the newly written manifest; no other artifact is modified.
    """
    out = Path(out)
    destination = out / MANIFEST_NAME
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite training checkpoint: {destination}')
    if metadata_status not in _METADATA_STATUSES:
        raise ValueError(f'unsupported training checkpoint metadata_status: {metadata_status}')
    if metadata_status == _METADATA_STATUSES[1] and (
            not isinstance(original_source_evidence, dict) or not original_source_evidence):
        raise ValueError('reconstructed metadata requires explicit nonempty original_source_evidence')
    source = {'repo_head': _revision(repo_head, 'repo_head'),
              'source_identity': _nonempty(context.get('source_identity'), 'source_identity'),
              'metadata_status': metadata_status}
    if original_source_evidence is not None:
        source['original_source_evidence'] = copy.deepcopy(original_source_evidence)
    _fitted_settings(cfg)
    saved_context = _context(context, cfg)
    quality = _quality(policy_quality)
    artifacts = {}
    for name in ARTIFACT_NAMES:
        path = out / name
        if not path.is_file():
            raise ValueError(f'training artifact is missing: {path}')
        artifacts[name] = sha256_file(path)
    try:
        stored_quality = json.loads((out / 'policy_validation.json').read_text())
    except (ValueError, OSError) as exc:
        raise ValueError('policy validation artifact is malformed') from exc
    if stored_quality != quality:
        raise ValueError('policy validation artifact differs from policy_quality')
    architecture = _architecture(cfg, _policy_state(out / 'policy.pt', 'cpu'))
    manifest = {'schema_version': SCHEMA_VERSION,
        'artifact_type': 'notebook_policy_training_checkpoint',
        'stage': 'policy_training', 'status': 'policy_training_complete',
        'notebook_status': 'incomplete',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'training_source': source, 'context': saved_context,
        'config_used': copy.deepcopy(cfg), 'policy_validation': quality,
        'policy_architecture': architecture, 'training_contract': _training_contract(cfg),
        'artifacts': artifacts}
    manifest['manifest_sha256'] = hash_payload(manifest)
    # Serialize before opening so malformed/nonfinite metadata leaves no file.
    encoded = json.dumps(manifest, indent=2, allow_nan=False)
    with destination.open('x') as stream:
        stream.write(encoded)
    return manifest


def load_training_checkpoint(folder, context, cfg, gnn, device, *,
        allow_source_revision_change=False, evaluator_repo_head=None, evaluation_campaign=None):
    """Load strict policy weights and original fitted config without training.

    Returns ``(policy.eval(), saved_cfg, recovery_record)``. A changed source
    requires explicit authorization and a distinct new evaluation campaign;
    the record retains both training and evaluation identities.
    """
    folder = Path(folder)
    path = folder / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f'training checkpoint is missing: {path}; old minimal manifests cannot recover policy training')
    try:
        manifest = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError('training checkpoint manifest is malformed') from exc
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA_VERSION
            or manifest.get('artifact_type') != 'notebook_policy_training_checkpoint'):
        raise ValueError('unsupported training checkpoint schema or artifact type')
    if (manifest.get('status') != 'policy_training_complete'
            or manifest.get('notebook_status') != 'incomplete' or manifest.get('stage') != 'policy_training'):
        raise ValueError('training checkpoint status does not certify completed policy training only')
    payload = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if manifest.get('manifest_sha256') != hash_payload(payload):
        raise ValueError('training checkpoint manifest checksum changed')
    saved_context, saved_cfg = manifest.get('context', {}), manifest.get('config_used', {})
    saved_source = manifest.get('training_source', {})
    if (saved_source.get('metadata_status') not in _METADATA_STATUSES
            or saved_source.get('source_identity') != saved_context.get('source_identity')):
        raise ValueError('training checkpoint original source metadata is incomplete')
    _revision(saved_source.get('repo_head'), 'training repo_head')
    if saved_source['metadata_status'] == _METADATA_STATUSES[1] and (
            not isinstance(saved_source.get('original_source_evidence'), dict)
            or not saved_source['original_source_evidence']):
        raise ValueError('training checkpoint lacks original_source_evidence for reconstructed metadata')
    _quality(manifest.get('policy_validation'))
    _fitted_settings(saved_cfg)
    current = _context(context, cfg)
    for name in ('seed', 'split_cluster_ids', 'backbone', 'run_mode', 'research_label'):
        if saved_context.get(name) != current[name]:
            raise ValueError(f'training checkpoint {name} differs from the recovery context')
    previous_inputs = saved_context.get('dataset', {})
    if set(previous_inputs) != set(current['dataset']) or any(
            previous_inputs[name].get('sha256') != current['dataset'][name]['sha256']
            for name in current['dataset']):
        raise ValueError('training checkpoint dataset input identity differs')
    if (saved_context.get('config_used') != saved_cfg
            or manifest.get('training_contract') != _training_contract(saved_cfg)
            or _training_contract(cfg) != manifest['training_contract']):
        raise ValueError('training configuration differs from the saved policy architecture/budget contract')
    # The supplied model must actually be the declared immutable backbone.
    if record_backbone(current['backbone']['stage'], gnn) != current['backbone']:
        raise ValueError('actual backbone weights differ from the recovery context')
    if getattr(gnn, 'output_dim', None) != manifest.get('policy_architecture', {}).get('node_emb_dim'):
        raise ValueError('backbone embedding width differs from the policy training architecture')
    if evaluator_repo_head is not None:
        _revision(evaluator_repo_head, 'evaluator_repo_head')
    source_changed = (current['source_identity'] != saved_source['source_identity']
        or (evaluator_repo_head is not None and evaluator_repo_head != saved_source['repo_head']))
    if source_changed and not allow_source_revision_change:
        raise ValueError('training source differs; explicit authorization for a new evaluation campaign is required')
    if source_changed or allow_source_revision_change:
        if evaluator_repo_head is None or evaluation_campaign is None or not str(evaluation_campaign).strip():
            raise ValueError('source revision recovery requires evaluator_repo_head and a new evaluation_campaign')
        if Path(evaluation_campaign).resolve() == folder.resolve():
            raise ValueError('recovery requires a new evaluation campaign; source artifacts are immutable')
    artifacts = manifest.get('artifacts', {})
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise ValueError('training checkpoint lacks the complete required artifact set')
    for name in ARTIFACT_NAMES:
        artifact = folder / name
        expected = artifacts[name].get('sha256') if isinstance(artifacts[name], dict) else None
        if not artifact.is_file() or not expected or sha256_file(artifact)['sha256'] != expected:
            raise ValueError(f'training artifact checksum changed or file missing: {name}')
    if json.loads((folder / 'policy_validation.json').read_text()) != manifest['policy_validation']:
        raise ValueError('training policy validation artifact differs from its manifest')
    state = _policy_state(folder / 'policy.pt', device)
    if _architecture(saved_cfg, state) != manifest['policy_architecture']:
        raise ValueError('training policy architecture differs from its manifest')
    # Instantiation consumes random numbers; preserve the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        policy = build_policy(saved_cfg, node_emb_dim=gnn.output_dim).to(device)
    policy.load_state_dict(state, strict=True)
    record = {'status': 'policy_recovered', 'notebook_status': 'incomplete',
        'policy_training_status': 'policy_training_complete',
        'training_source': copy.deepcopy(saved_source),
        'evaluation_source': {'source_identity': current['source_identity'],
                              'repo_head': evaluator_repo_head},
        'source_revision_changed': source_changed,
        'evaluation_campaign': str(evaluation_campaign) if evaluation_campaign is not None else None,
        'source_artifact_dir': str(folder.resolve()),
        'training_checkpoint_sha256': sha256_file(path)['sha256'],
        'original_training_config_sha256': hash_payload(saved_cfg),
        'requested_evaluation_config_sha256': hash_payload(cfg),
        'policy_validation': copy.deepcopy(manifest['policy_validation']),
        'restored_fitted_settings': {'reward_error_scale': saved_cfg['rls'].get('reward_error_scale')}}
    return policy.eval(), copy.deepcopy(saved_cfg), record
