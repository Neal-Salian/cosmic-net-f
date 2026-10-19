"""Scientific contracts: a changed budget or stale resume must never look valid."""
import importlib.util
import pytest
import torch


def api():
    assert importlib.util.find_spec('experiments.paper_ablations.core') is not None, 'ablation harness missing'
    from experiments.paper_ablations import core
    return core


def graph():
    # Triangle on 0,1,2 plus a leaf 3: greedy top two pairs isolate the leaf.
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3]])
    edges = torch.cat([pairs.T, pairs.flip(1).T, torch.arange(4).repeat(2, 1)], 1)
    return {'x': torch.zeros(4, 4), 'edge_index': edges, 'cluster_id': 'toy'}


def test_exact_decoder_and_additive_repair_cannot_silently_share_a_budget():
    c = api()
    g, scores = graph(), torch.tensor([4., 3., 2., 1.])
    exact = c.decode(g, scores, .5, 'constrained_rank', 42)
    repaired = c.decode(g, scores, .5, 'rank_repair', 42)
    assert exact['diagnostics']['final_pair_count'] == 2
    assert exact['diagnostics']['physical_isolates'] == 0
    assert repaired['diagnostics']['final_pair_count'] == 3
    assert repaired['diagnostics']['budget_satisfied'] is False
    with pytest.raises(ValueError, match='budget'):
        c.require_matched([exact], [repaired])
    assert exact['mask'][-4:].all()


def test_stochastic_decode_replays_and_probability_rank_has_same_strict_order():
    c = api()
    scores = torch.tensor([1., .2, -.3, -1.])
    a = c.decode(graph(), scores, .5, 'constrained_sample', 7)
    b = c.decode(graph(), scores, .5, 'constrained_sample', 7)
    assert torch.equal(a['mask'], b['mask'])
    ranked = c.decode(graph(), scores, .5, 'constrained_rank', 7)
    probability = c.decode(graph(), scores, .5, 'probability_rank', 7)
    assert torch.equal(ranked['mask'], probability['mask'])
    assert a['diagnostics']['final_pair_count'] == 2


def test_matched_comparisons_reject_different_halo_ids_even_at_equal_cost():
    c = api()
    a = c.decode(graph(), torch.zeros(4), .5, 'constrained_rank', 42)
    b = dict(a, graph_id='other')
    with pytest.raises(ValueError, match='population'):
        c.require_matched([a], [b])


def test_prediction_only_and_single_term_removals_are_independent():
    c = api()
    base = dict(w_acc=1., w_sp=.5, w_conn=1., w_virial=1., entropy_coef=.01, value_coef=.5)
    pred = c.variant_config(base, 'prediction_only')
    assert [pred[k] for k in ['w_acc','w_sp','w_conn','w_virial']] == [1., 0., 0., 0.]
    assert pred['entropy_coef'] == .01
    for term, variant in [('w_sp','no_sparsity'), ('w_conn','no_connectivity'),
                          ('w_virial','no_virial'), ('entropy_coef','no_entropy'),
                          ('value_coef','no_value')]:
        cfg = c.variant_config(base, variant)
        assert cfg[term] == 0.
        assert all(cfg[key] == value for key,value in base.items() if key != term)
    assert base['w_virial'] == 1.


def test_resume_rejects_changed_contract_or_damaged_completed_artifact(tmp_path):
    c = api()
    folder = tmp_path/'run'
    c.start_run(folder, {'seed':42, 'variant':'full'})
    (folder/'predictions.json').write_text('[]')
    c.finish_run(folder, ['predictions.json'])
    assert c.start_run(folder, {'seed':42, 'variant':'full'}, resume=True) == 'complete'
    with pytest.raises(ValueError, match='contract'):
        c.start_run(folder, {'seed':43, 'variant':'full'}, resume=True)
    (folder/'predictions.json').write_text('[1]')
    with pytest.raises(ValueError, match='artifact'):
        c.start_run(folder, {'seed':42, 'variant':'full'}, resume=True)


def test_invalid_retention_is_not_coerced_to_a_usable_budget():
    c = api()
    for q in [float('nan'), -.1, 1.1]:
        with pytest.raises(ValueError):
            c.decode(graph(), torch.ones(4), q, 'constrained_rank', 42)


def training_fixture():
    from rls.policy import EdgePolicyNet
    from rls.policy_gradient import ValueNet, PolicyGradientTrainer
    torch.manual_seed(29)
    policy=EdgePolicyNet(edge_dim=5,node_emb_dim=3,hidden_dim=4,normalize=True)
    critic=ValueNet(3,hidden_dim=4)
    g=graph()
    g.update(edge_attr=torch.randn(12,5),emb=torch.randn(4,3),ctx=torch.randn(3),
        y=torch.tensor([12.]),stellar_mass=torch.tensor([1.,2.,3.,4.])*1e10,
        vel_disp=torch.tensor([100.,150.,180.,200.]),pos=torch.randn(4,3))
    cfg=dict(epochs=2,batch_size=1,pair_rollouts=2,reward_scale_floor=.05,reward_clip=10.,
        w_acc=1.,w_sp=.5,w_conn=1.,w_virial=.1,virial_anneal_start_epoch=0,
        virial_anneal_epochs=2,entropy_coef=.01,value_coef=.5,
        target_sparsity_start=.9,target_sparsity_end=.5,sparsity_anneal_epochs=2)
    trainer=PolicyGradientTrainer(policy,critic,torch.optim.Adam(policy.parameters(),lr=.001),
        torch.optim.Adam(critic.parameters(),lr=.001),cfg)
    def predictor(graph,mask):
        return graph['y']+.1,graph['y']+.2*mask.float().mean()
    def validation(p):
        return {'rmse':float(p.fc[-1].weight.detach().square().sum())}
    return trainer,[g],predictor,cfg,validation


def test_resumed_training_matches_uninterrupted_optimizer_and_rng(tmp_path):
    assert importlib.util.find_spec('experiments.paper_ablations.training') is not None, 'resumable trainer missing'
    from experiments.paper_ablations.training import train
    t,g,p,c,v=training_fixture()
    whole=train(t,g,p,c,'cpu',2,v,tmp_path/'whole.pt')
    expected={k:x.clone() for k,x in t.policy.state_dict().items()}
    t,g,p,c,v=training_fixture()
    part=train(t,g,p,c,'cpu',2,v,tmp_path/'resume.pt',stop_after=1)
    assert part['next_epoch']==1
    # Construction consumes random numbers; resume must restore the saved stream.
    t,g,p,c,v=training_fixture()
    resumed=train(t,g,p,c,'cpu',2,v,tmp_path/'resume.pt',resume=True)
    assert resumed['history']==whole['history']
    assert all(torch.equal(x,expected[k]) for k,x in t.policy.state_dict().items())
    assert resumed['next_epoch']==2


def test_resumable_trainer_preserves_reference_pair_objective(tmp_path):
    assert importlib.util.find_spec('experiments.paper_ablations.training') is not None, 'resumable trainer missing'
    from experiments.paper_ablations.training import train
    from rls.pair_training import train_pair_policy
    t,g,p,c,v=training_fixture()
    train_pair_policy(t,g,p,c,'cpu',2,validation_fn=v)
    expected={k:x.clone() for k,x in t.policy.state_dict().items()}
    t,g,p,c,v=training_fixture()
    train(t,g,p,c,'cpu',2,v,tmp_path/'reference.pt')
    assert all(torch.equal(x,expected[k]) for k,x in t.policy.state_dict().items())


def test_separate_value_loss_cannot_change_the_pair_actor(tmp_path):
    from experiments.paper_ablations.training import train
    t,g,p,c,v=training_fixture()
    train(t,g,p,c,'cpu',2,v,tmp_path/'value.pt')
    expected={k:x.clone() for k,x in t.policy.state_dict().items()}
    t,g,p,c,v=training_fixture(); c['value_coef']=0.
    train(t,g,p,c,'cpu',2,v,tmp_path/'no_value.pt')
    assert all(torch.equal(x,expected[k]) for k,x in t.policy.state_dict().items())


def test_recorded_diagnostics_recompute_and_invalid_budgets_have_no_accuracy():
    import json
    import math
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'experiments/paper_ablations/results/local-2026-10-16'
    summaries=json.loads((root/'summary.json').read_text())
    predictions=json.loads((root/'predictions.json').read_text())
    reference_ids=[r['graph_id'] for r in predictions['0.4_constrained_rank']]
    for summary in summaries:
        rows=predictions[summary['arm']]
        assert summary['trained_reward_ablation'] is False
        assert len(rows)==summary['n_halos']
        if summary['arm']!='original_0.4_cohort_replay':
            assert [r['graph_id'] for r in rows]==reference_ids
        valid=all(r['budget_satisfied'] for r in rows)
        if not valid:
            assert summary['rmse_dex'] is None
            continue
        actual=math.sqrt(sum((r['prediction']-r['target'])**2 for r in rows)/len(rows))
        assert summary['rmse_dex']==pytest.approx(actual,abs=1e-12)
        for r in rows:
            if 'requested_pair_count' in r:
                assert r['final_pair_count']==r['requested_pair_count']


def test_recorded_campaign_completion_checksums_are_intact():
    import hashlib
    import json
    from pathlib import Path
    folder=Path(__file__).resolve().parents[1]/'experiments/paper_ablations/results/local-2026-10-16'
    manifest=json.loads((folder/'run.json').read_text())
    assert manifest['status']=='complete'
    for name,checksum in manifest['artifacts'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==checksum


def test_substituted_input_is_rejected_before_its_own_manifest_can_authorize_it(tmp_path):
    from experiments.paper_ablations.inputs import validate_registered_inputs
    paths=[]
    for name in ['csv','backbone','archive']:
        path=tmp_path/name; path.write_bytes(b'self-consistent but different experiment')
        paths.append(path)
    with pytest.raises(ValueError,match='Registered input identity differs'):
        validate_registered_inputs(*paths)
