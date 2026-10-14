"""CPU regression coverage for canonical execution and truthful evaluation."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data


def test_provided_rl_historical_cost_is_unknown_until_measured():
    from rls.benchmark_scorers import PairScorer
    policy = nn.Linear(1, 1)
    unknown = PairScorer('provided_rl', policy=policy).training_cost()
    assert unknown['total_calls'] is None
    assert unknown['reason'] == 'unknown_artifact_cost'
    measured = PairScorer('provided_rl', policy=policy,
                          historical_training_cost={'total_calls': 17})
    assert measured.training_cost()['total_calls'] == 17
    with pytest.raises(ValueError, match='non-negative integer'):
        PairScorer('provided_rl', policy=policy,
                   historical_training_cost={'total_calls': -1}).training_cost()


def test_matched_metrics_include_mae_for_empty_and_success_populations():
    from rls.matched_benchmark import _regression_metrics
    assert _regression_metrics([1., 5.], [2., 2.])['mae'] == 2.
    empty = _regression_metrics([], [])
    assert empty['mae'] is None
    assert empty['mae_reason'] == 'no_successes'


def test_multiseed_entrypoint_requires_checkpoint_before_loading_data(tmp_path):
    import scripts.multiseed_rl as harness
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('training: {checkpoint_dir: missing}\ndata: {source: tng}\nrls: {}\n')
    out = tmp_path / 'runs.json'
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        harness.main(['--config', str(cfg), '--seeds', '3', '--out', str(out)])
    assert not out.exists()
    assert not (tmp_path / 'runs_runs').exists()


def test_multiseed_delegates_fresh_seed_runs_and_preserves_matched_rows(tmp_path, monkeypatch):
    import scripts.multiseed_rl as harness
    import rls.run_experiment as canonical
    from data.provenance import hash_payload
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('seed: 42\ntraining: {checkpoint_dir: ignored}\ndata: {source: synthetic}\nrls: {epochs: 1}\n')
    seeds_seen = []
    def tiny_run(config, *, checkpoint, output_dir, allow_random_backbone,
                 max_graphs=None, supervised_artifact=None):
        seeds_seen.append(config['seed'])
        assert checkpoint is None and allow_random_backbone
        path = output_dir
        path.mkdir(parents=True)
        (path / 'policy.pt').write_bytes(str(config['seed']).encode())
        (path / 'provenance.json').write_text(json.dumps({
            'backbone_frozen': {'stage': 'frozen', 'sha256': hash_payload(config)},
            'research_label': 'synthetic_random_backbone_smoke',
            'run_mode': 'synthetic_random_backbone_smoke',
        }))
        (path / 'matched_summary.json').write_text(json.dumps([{
            'method': 'provided_rl', 'backbone_stage': 'frozen', 'common_rmse': 2.,
            'common_mae': 1.5, 'common_r2': None, 'common_n': 1,
            'rmse': 2., 'r2': None, 'common_success_ids': ['g'],
        }]))
        (path / 'matched_rows.json').write_text(json.dumps([{
            'graph_id': 'g', 'method': 'provided_rl', 'status': 'success',
            'backbone_stage': 'frozen', 'final_pair_count': 2,
            'available_pair_count': 4,
        }]))
        return 0
    monkeypatch.setattr(canonical, 'main', tiny_run)
    out = tmp_path / 'runs.json'
    assert harness.main(['--config', str(cfg), '--seeds', '3', '8',
                         '--allow-random-backbone', '--out', str(out)]) == 0
    report = json.loads(out.read_text())
    assert seeds_seen == [3, 8]
    assert len(report['per_seed']) == 2
    assert report['per_seed'][0]['mean_keep_frac'] == .5
    assert report['per_seed'][0]['rmse'] == 2.
    assert report['per_seed'][0]['r2'] is None
    assert report['summary'][0]['rmse_mean'] == 2.
    assert report['summary'][0]['r2_mean'] is None
    assert report['per_seed'][0]['policy_sha256'] != report['per_seed'][1]['policy_sha256']
    with pytest.raises(FileExistsError):
        harness.main(['--config', str(cfg), '--seeds', '3',
                      '--allow-random-backbone', '--out', str(out)])


def test_fixed_validation_passes_final_budget_to_mode_aware_decoder(monkeypatch):
    from rls.research_workflow import fixed_budget_validation
    import rls.sparsify as sparsify
    class Policy(nn.Module):
        def forward(self, ea, emb, ei, ctx):
            return torch.zeros(ei.shape[1])
    graph = {'edge_index': torch.tensor([[0, 1], [1, 0]]),
             'edge_attr': torch.ones(2, 1), 'emb': torch.ones(2, 1),
             'ctx': torch.ones(1), 'y': torch.tensor([1.])}
    def decoded(ei, probs, cfg, target_sparsity=None):
        assert cfg['sparsity_mode'] == 'pair_pl'
        assert target_sparsity == .25
        return torch.tensor([True, True])
    monkeypatch.setattr(sparsify, 'eval_mask', decoded)
    policy = Policy().train()
    result = fixed_budget_validation(policy, [graph],
        lambda g, mask: (torch.tensor([3.]), torch.tensor([2.])),
        {'sparsity_mode': 'pair_pl', 'target_sparsity_end': .25}, 'cpu')
    assert result['rmse'] == 1.
    assert result['mae'] == 1.
    assert result['requested_keep_fraction'] == .25
    assert policy.training


def test_legacy_validation_restores_best_policy_and_value_state(monkeypatch):
    from rls.research_workflow import train_with_fixed_validation
    import importlib
    training = importlib.import_module('rls.train_policy')
    actor, critic = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(3.); critic.weight.fill_(30.)
    trainer = SimpleNamespace(policy=actor, value_net=critic)
    def updates(trainer, graphs, gnns, cfg, device, epochs, log_fn, validation_fn):
        for epoch, value in enumerate((1., 2.)):
            with torch.no_grad():
                actor.weight.fill_(value); critic.weight.fill_(value * 10.)
            log_fn(epoch, .9 - epoch * .1, value)
        return [1., 2.]
    monkeypatch.setattr(training, 'train_policy', updates)
    train_with_fixed_validation(trainer, [], None,
        {'sparsity_mode': 'penalty', 'epochs': 2}, 'cpu',
        validation_fn=lambda policy: {'rmse': float(policy.weight.detach().item())})
    assert actor.weight.item() == 1.
    assert critic.weight.item() == 10.
    assert trainer.selected_epoch == 0
    assert trainer.selected_validation_rmse == 1.


def test_checkpoint_gate_rejects_implicit_legacy_and_projected_schema(tmp_path):
    from rls.research_workflow import inspect_backbone_checkpoint
    cfg = {'data': {'source': 'tng'}, 'model': {'node_features': 4, 'edge_features': 5},
           'graph': {'edge_features': ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep']}}
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'weight': torch.ones(1)}, checkpoint)
    with pytest.raises(ValueError, match='explicit legacy'):
        inspect_backbone_checkpoint(checkpoint, cfg)
    cfg['data']['checkpoint_research_label'] = 'legacy_total_radius'
    assert inspect_backbone_checkpoint(checkpoint, cfg)['research_label'] == 'legacy_total_radius'
    for declaration in ({'compatibility_record': {'feature_mode': 'projected_observation'}},
                        {'artifact_type': 'projected_backbone'}):
        torch.save({'model_state_dict': {'weight': torch.ones(1)}, **declaration}, checkpoint)
        with pytest.raises(ValueError, match='projected'):
            inspect_backbone_checkpoint(checkpoint, cfg)


def test_canonical_scorers_report_missing_trained_control():
    from rls.research_workflow import build_matched_scorers
    scorers, controls = build_matched_scorers(
        nn.Linear(1, 1), {'graph': {'edge_features': ['distance']}},
        catalog_contract={'position_units': 'Mpc', 'mass_units': 'M_sun'},
        split_hash='split-abc')
    assert {s.name for s in scorers} == {
        'random', 'distance', 'degree', 'stellar_binding_proxy',
        'edge_feature_grad_x_input', 'provided_rl'}
    assert controls['supervised_raw']['status'] == 'absent'
    assert next(s for s in scorers if s.name == 'stellar_binding_proxy').distance_unit == 'Mpc'


def test_scientific_loader_fallback_is_rejected_before_generation():
    from rls.research_workflow import guard_loader_fallbacks
    class Loader:
        def _generate_synthetic_camels(self, cache):
            raise AssertionError('must never generate')
    loader = Loader()
    guard_loader_fallbacks(loader, smoke=False)
    with pytest.raises(ValueError, match='synthetic fallback'):
        loader._generate_synthetic_camels('unused')


def test_inference_summary_includes_scoring_decoding_and_embedding_calls():
    from data.provenance import build_run_provenance
    from rls.benchmark_scorers import PairScorer
    from rls.matched_benchmark import BudgetSpec, evaluate_matched_budget
    class Predictor(nn.Module):
        def forward(self, batch, return_embeddings=False):
            prediction = batch.edge_attr.sum()
            return (prediction, batch.x) if return_embeddings else prediction
    class Policy(nn.Module):
        def forward(self, edge_attr, embeddings, edge_index, ctx):
            return edge_attr.sum(dim=1)
    graph = Data(x=torch.ones(2, 1), edge_attr=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0.]), cluster_id='g')
    provenance = build_run_provenance(source_files=[], catalog_contract={},
        split_manifest={'manifest_sha256': 'split'}, resolved_config={},
        source_revision='test', graph={}, augmentation_manifest=None,
        research_label='synthetic')
    scorers = [PairScorer('edge_feature_grad_x_input'),
               PairScorer('provided_rl', policy=Policy(), edge_feature_names=['distance'])]
    summaries, rows = evaluate_matched_budget(Predictor(), [graph], scorers,
        BudgetSpec(pair_count=1, constraint='none'), seed=3,
        provenance=provenance, backbone='frozen')
    for summary in summaries:
        own = [row for row in rows if row['method'] == summary['method']]
        assert summary['mean_inference_time_s'] > 0
        assert summary['inference_cost_totals']['total_region_time_s'] == own[0]['total_region_time_s']
        if summary['method'] != 'full':
            assert summary['inference_cost_totals']['scoring_forward_calls'] == 1
            assert summary['inference_cost_totals']['eval_forward_calls'] == 1
            assert summary['inference_cost_totals']['full_forward_calls'] == 0
            assert summary['mean_scoring_time_s'] > 0
            assert summary['mean_decoder_time_s'] > 0
    rl = next(s for s in summaries if s['method'] == 'provided_rl')
    assert rl['historical_training_calls'] is None
    assert rl['mae'] == 2.
