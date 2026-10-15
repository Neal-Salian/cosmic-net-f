"""Recovery of completed policy training never repeats training or completes B."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from data.provenance import sha256_file
from rls.policy import build_policy
from rls.provenance import record_backbone


@pytest.fixture
def completed_training(tmp_path):
    out = tmp_path / 'interrupted-B'
    out.mkdir()
    cfg = {'seed': 7, 'model': {'output_dim': 4},
        'graph': {'method': 'knn', 'k_neighbors': 2,
                  'edge_features': ['distance', 'delta_v']},
        'rls': {'sparsity_mode': 'pair_pl', 'policy_hidden': 8,
                'policy_normalize': True, 'epochs': 60, 'lr': .001,
                'target_sparsity_start': .9, 'target_sparsity_end': .4,
                'reward_error_scale': .175, 'stageb_epochs': 10}}
    gnn = nn.Linear(4, 1)
    gnn.output_dim = 4
    backbone_path = tmp_path / 'backbone.pt'
    torch.save(gnn.state_dict(), backbone_path)
    csv = tmp_path / 'halos.csv'
    csv.write_text('halo_id,mass\na,12\nb,13\n')
    context = {'dataset': {'tng_csv': sha256_file(csv),
                           'checkpoint': sha256_file(backbone_path)},
        'seed': 7, 'config_used': copy.deepcopy(cfg),
        'split_cluster_ids': {'train': ['a'], 'val': ['b'], 'test': ['c']},
        'backbone': record_backbone('frozen', gnn), 'run_mode': 'smoke',
        'research_label': 'fixture_smoke', 'source_identity': 'b' * 64,
        'evaluation_protocol': {'decoder': 'direct', 'constraint': 'no_isolates',
                                'target': 'legacy_uncertified'}}
    # run_context is built before fitted reward scaling; save must use final cfg.
    context['config_used']['rls'].pop('reward_error_scale')
    policy = build_policy(cfg, node_emb_dim=4)
    with torch.no_grad():
        policy.edge_mean.copy_(torch.tensor([11., 12.]))
        policy.edge_scale.copy_(torch.tensor([2., 3.]))
        policy.node_mean.fill_(4.)
        policy.node_scale.fill_(5.)
        policy.context_mean.fill_(6.)
        policy.context_scale.fill_(7.)
    torch.save(policy.state_dict(), out / 'policy.pt')
    torch.save(nn.Linear(4, 1).state_dict(), out / 'value_net.pt')
    (out / 'training_log.csv').write_text('epoch,loss\n0,1\n1,.5\n')
    (out / 'random_validation.csv').write_text('seed,rmse\n0,.2\n')
    quality = {'verdict': 'FAIL', 'budget_comparable': True,
               'selected_epoch': 4, 'scope': 'smoke'}
    (out / 'policy_validation.json').write_text(json.dumps(quality))
    (out / 'provenance.json').write_text('{"B":{"status":"stage_b_failed"}}')
    return out, context, cfg, quality, gnn, policy


def save_run(fixture, **kwargs):
    from rls.notebook_recovery import save_training_checkpoint
    out, context, cfg, quality, _, _ = fixture
    return save_training_checkpoint(out, context, cfg, quality, 'a' * 40, **kwargs)


def bytes_in(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


def test_save_captures_post_training_config_and_never_claims_notebook_complete(completed_training):
    out, context, cfg, quality, _, _ = completed_training
    original = bytes_in(out)
    manifest = save_run(completed_training)
    assert manifest['status'] == 'policy_training_complete'
    assert manifest['notebook_status'] == 'incomplete'
    assert manifest['config_used']['rls']['reward_error_scale'] == .175
    assert manifest['training_source']['repo_head'] == 'a' * 40
    assert manifest['training_source']['source_identity'] == 'b' * 64
    assert manifest['policy_validation'] == quality
    assert set(manifest['artifacts']) == {'policy.pt', 'value_net.pt', 'training_log.csv',
                                         'random_validation.csv', 'policy_validation.json'}
    assert json.loads((out / 'training_checkpoint.json').read_text()) == manifest
    for name, data in original.items():
        assert (out / name).read_bytes() == data
    assert context['config_used']['rls'].get('reward_error_scale') is None
    with pytest.raises(FileExistsError):
        save_run(completed_training)


def test_recovery_restores_saved_buffers_without_training_or_input_mutation(completed_training, monkeypatch):
    from rls.notebook_recovery import load_training_checkpoint
    from rls.policy import EdgePolicyNet
    import importlib
    training = importlib.import_module('rls.train_policy')
    out, context, cfg, _, gnn, original_policy = completed_training
    save_run(completed_training)
    original = bytes_in(out)
    incoming = copy.deepcopy(cfg)
    incoming['rls'].pop('reward_error_scale')
    original_context = copy.deepcopy(context)
    def forbidden(*args, **kwargs):
        raise AssertionError('recovery must never train or refit normalization')
    monkeypatch.setattr(training, 'train_policy', forbidden)
    monkeypatch.setattr(EdgePolicyNet, 'fit_normalization', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    policy, saved, record = load_training_checkpoint(out, context, incoming, gnn, 'cpu')
    for key, expected in original_policy.state_dict().items():
        assert torch.equal(policy.state_dict()[key], expected), key
    assert not policy.training
    assert saved['rls']['reward_error_scale'] == .175
    assert record['status'] == 'policy_recovered'
    assert record['notebook_status'] == 'incomplete'
    assert record['training_source']['repo_head'] == 'a' * 40
    assert context == original_context
    assert 'reward_error_scale' not in incoming['rls']
    assert bytes_in(out) == original


@pytest.mark.parametrize('filename', ['policy.pt', 'value_net.pt', 'training_log.csv',
                                      'random_validation.csv', 'policy_validation.json'])
def test_recovery_rejects_changed_training_artifacts(completed_training, filename):
    from rls.notebook_recovery import load_training_checkpoint
    out, context, cfg, _, gnn, _ = completed_training
    save_run(completed_training)
    path = out / filename
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='artifact|checksum'):
        load_training_checkpoint(out, context, cfg, gnn, 'cpu')


@pytest.mark.parametrize('change', ['input_bytes', 'input_identity', 'split', 'backbone',
                                  'backbone_weights', 'graph', 'seed', 'mode',
                                  'policy_architecture', 'training_budget', 'training_epochs'])
def test_recovery_rejects_changed_inputs_and_training_contract(completed_training, change):
    from rls.notebook_recovery import load_training_checkpoint
    out, context, cfg, _, gnn, _ = completed_training
    save_run(completed_training)
    context, cfg = copy.deepcopy(context), copy.deepcopy(cfg)
    if change == 'input_bytes':
        Path(context['dataset']['tng_csv']['path']).write_text('changed input')
    elif change == 'input_identity':
        context['dataset']['tng_csv']['sha256'] = 'e' * 64
    elif change == 'split':
        context['split_cluster_ids']['train'] = ['different']
    elif change == 'backbone':
        context['backbone']['sha256'] = 'f' * 64
    elif change == 'backbone_weights':
        with torch.no_grad():
            gnn.weight.add_(1.)
    elif change == 'graph':
        cfg['graph']['k_neighbors'] = 9
    elif change == 'seed':
        context['seed'] = cfg['seed'] = 99
    elif change == 'mode':
        context['run_mode'] = 'full'
    elif change == 'policy_architecture':
        cfg['rls']['policy_normalize'] = False
    elif change == 'training_budget':
        cfg['rls']['target_sparsity_end'] = .5
    elif change == 'training_epochs':
        cfg['rls']['epochs'] = 61
    with pytest.raises(ValueError, match='input|dataset|split|backbone|graph|seed|run_mode|training|configuration'):
        load_training_checkpoint(out, context, cfg, gnn, 'cpu')


def test_source_revision_change_requires_explicit_new_campaign_and_preserves_both_identities(completed_training):
    from rls.notebook_recovery import load_training_checkpoint
    out, context, cfg, _, gnn, _ = completed_training
    save_run(completed_training)
    context = copy.deepcopy(context)
    context['source_identity'] = 'c' * 64
    with pytest.raises(ValueError, match='source'):
        load_training_checkpoint(out, context, cfg, gnn, 'cpu')
    with pytest.raises(ValueError, match='campaign|repo_head'):
        load_training_checkpoint(out, context, cfg, gnn, 'cpu', allow_source_revision_change=True)
    _, saved, record = load_training_checkpoint(out, context, cfg, gnn, 'cpu',
        allow_source_revision_change=True, evaluator_repo_head='d' * 40,
        evaluation_campaign=out.parent / 'fresh-evaluation')
    assert record['training_source']['source_identity'] == 'b' * 64
    assert record['training_source']['repo_head'] == 'a' * 40
    assert record['evaluation_source']['source_identity'] == 'c' * 64
    assert record['evaluation_source']['repo_head'] == 'd' * 40
    assert record['source_revision_changed'] is True
    assert record['notebook_status'] == 'incomplete'
    assert record['evaluation_campaign'].endswith('fresh-evaluation')
    assert saved['rls']['reward_error_scale'] == .175


def test_reconstructed_metadata_requires_explicit_original_source_evidence(completed_training):
    out, _, _, _, _, _ = completed_training
    with pytest.raises(ValueError, match='original_source_evidence'):
        save_run(completed_training, metadata_status='reconstructed_from_original_revision_and_exact_inputs')
    assert not (out / 'training_checkpoint.json').exists()
    evidence = {'original_checkout': 'a' * 40, 'operator_verified_exact_inputs': True}
    manifest = save_run(completed_training,
        metadata_status='reconstructed_from_original_revision_and_exact_inputs',
        original_source_evidence=evidence)
    assert manifest['training_source']['original_source_evidence'] == evidence
    assert manifest['training_source']['metadata_status'] == 'reconstructed_from_original_revision_and_exact_inputs'


@pytest.mark.parametrize('content', [None, {}, {'B': {'status': 'complete'}},
    {'schema_version': 1, 'status': 'complete'},
    {'schema_version': 0, 'status': 'policy_training_complete'}])
def test_recovery_refuses_old_or_incomplete_manifests(completed_training, content):
    from rls.notebook_recovery import load_training_checkpoint
    out, context, cfg, _, gnn, _ = completed_training
    if content is not None:
        (out / 'training_checkpoint.json').write_text(json.dumps(content))
    with pytest.raises((ValueError, FileNotFoundError), match='training|checkpoint|schema|status'):
        load_training_checkpoint(out, context, cfg, gnn, 'cpu')


@pytest.mark.parametrize('scale', [None, 0., float('nan'), float('inf'), -1., True])
def test_save_refuses_missing_or_invalid_post_training_reward_scale(completed_training, scale):
    from rls.notebook_recovery import save_training_checkpoint
    out, context, cfg, quality, _, _ = completed_training
    cfg = copy.deepcopy(cfg)
    if scale is None:
        cfg['rls'].pop('reward_error_scale')
    else:
        cfg['rls']['reward_error_scale'] = scale
    with pytest.raises(ValueError, match='fitted reward_error_scale'):
        save_training_checkpoint(out, context, cfg, quality, 'a' * 40)
    assert not (out / 'training_checkpoint.json').exists()
