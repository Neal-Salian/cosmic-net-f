"""Scientific contract tests: mutations must change a numerical result or fail closed."""
import copy
import itertools
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from rls.constraints import PhysicalPairLayout, feasible_budget
from rls.constrained_policy import select_pairs
from experiments.paper_baselines.selectors import decode, alternative_saliency, rank_fraction
from experiments.paper_baselines.audit import audit_records, validate_splits
from experiments.paper_baselines.analyze import paired_bootstrap, choose_validation


def layout(edges, n):
    return PhysicalPairLayout.from_edge_index(torch.tensor(edges).T.contiguous(), n)


@pytest.mark.parametrize('ties', [False, True])
def test_lazy_decoder_matches_reference_on_sparse_graph_and_all_budgets(ties):
    # Ranking the disjoint edge first can make a later edge infeasible.
    ly = layout([(0,1),(1,2),(2,3),(3,4),(0,4),(1,4)], 5)
    scores = torch.zeros(6) if ties else torch.tensor([3.,1.,5.,2.,0.,4.])
    marks = torch.tensor([0,3,2,5,1,4])
    for k in range(7):
        if feasible_budget(ly,k,'no_isolates')['feasible']:
            expected = select_pairs(ly,scores,k,'no_isolates',pair_marks=marks).selected
            assert torch.equal(decode(ly,scores,k,marks),expected)
        else:
            with pytest.raises(ValueError,match='infeasible'):
                decode(ly,scores,k,marks)


def test_decoder_preserves_reverse_copies_and_loops_and_reproducible_sampling():
    ly=layout([(0,0),(0,1),(1,0),(0,1),(1,2),(2,1),(0,2),(2,0)],3)
    a=decode(ly,torch.tensor([1.,2.,3.]),2,torch.tensor([2,0,1]),sampling_seed=5)
    b=decode(ly,torch.tensor([1.,2.,3.]),2,torch.tensor([2,0,1]),sampling_seed=5)
    assert torch.equal(a,b)
    assert a.sum()==2
    mask=ly.expand(a)
    assert mask[0] and mask[1]==mask[2]==mask[3]
    with pytest.raises(ValueError,match='finite'):
        decode(ly,torch.tensor([1.,float('nan'),3.]),2,torch.tensor([2,0,1]))


def test_probability_ranking_agrees_without_saturation_and_rank_ties_average():
    assert rank_fraction(torch.tensor([2.,2.,8.])).tolist()==[.25,.25,1.]
    ly=layout([(0,1),(0,2),(1,2)],3)
    s=torch.tensor([-3.,.2,4.],dtype=torch.float64)
    assert torch.equal(decode(ly,s,2,torch.arange(3)),decode(ly,s.sigmoid(),2,torch.arange(3)))


class LinearEdgeModel(torch.nn.Module):
    def forward(self,batch,return_embeddings=False):
        return (batch.edge_attr * torch.tensor([2.,-3.])).sum().reshape(1),None


def test_alternative_gradients_distinguish_cancellation_and_use_no_targets():
    graph=Data(x=torch.ones(3,1),edge_index=torch.tensor([[0,1,0],[1,2,2]]),
               edge_attr=torch.tensor([[3.,2.],[1.,0.],[0.,2.]]), y=torch.tensor([999.]))
    ly=PhysicalPairLayout.from_edge_index(graph.edge_index,3)
    got=alternative_saliency(LinearEdgeModel(),graph,ly,torch.tensor([2.,1.]))
    assert got['signed_grad_x_input'].tolist()==[0.,6.,2.]
    assert got['abs_gradient_scaled'].tolist()==[7.,7.,7.]


def test_split_overlap_and_duplicate_identity_rejected():
    with pytest.raises(ValueError,match='overlap'):
        validate_splits({'train':['a'],'val':['b'],'test':['a']})
    with pytest.raises(ValueError,match='duplicate'):
        validate_splits({'train':['a','a'],'val':['b'],'test':['c']})


def records():
    shared=dict(graph_id='c',requested_keep_fraction=.6,model_state_hash='hash',
        constraint='no_isolates',decoder='direct',status='success',target=2.,
        full_prediction=2.1,available_pair_count=3,requested_pair_count=2,
        available_loop_columns=0,layout_pairs=[[0,1],[0,2],[1,2]],
        layout_inverse=[0,1,2],stored_real=[True]*3,graph_content_hash='g',
        split_manifest_sha256='s',config_sha256='cfg')
    a=dict(shared,method='full',pruned_prediction=None,final_pair_count=3,
           squared_error=.01,stored_mask=None,selected_pairs=None)
    b=dict(shared,method='random',pruned_prediction=2.2,final_pair_count=2,
           squared_error=.04,stored_mask=[True,True,False],selected_pairs=[0,1],
           budget_satisfied=True,constraint_satisfied=True)
    return [a,b]


def test_audit_rejects_budget_mask_prediction_and_population_corruption():
    splits={'train':['a'],'val':['b'],'test':['c']}
    audit_records(records(),splits,['full','random'],[.6],'hash')
    for field,bad in [('final_pair_count',3),('model_state_hash','other'),
                      ('squared_error',4.),('stored_mask',[True,False,False])]:
        r=records();r[1][field]=bad
        with pytest.raises(ValueError):audit_records(r,splits,['full','random'],[.6],'hash')
    with pytest.raises(ValueError):audit_records(records()[1:],splits,['full','random'],[.6],'hash')


def test_bootstrap_pairs_halos_and_identical_methods_have_zero_interval():
    a=np.array([1.,2.,3.,4.]); result=paired_bootstrap(a,a,17,200)
    assert result['delta_rmse']==0 and result['ci_low']==0 and result['ci_high']==0
    assert paired_bootstrap(a,a/2,17,200)==paired_bootstrap(a,a/2,17,200)


def test_hyperparameter_selection_rejects_test_records_and_preserves_losers():
    rows=[dict(split='val',method='g0',squared_error=4.),dict(split='val',method='g1',squared_error=1.)]
    assert choose_validation(rows,['g0','g1'])=='g1'
    with pytest.raises(ValueError,match='validation'):
        choose_validation([dict(rows[0],split='test')],['g0'])


def test_missing_inputs_write_explicit_not_executed_manifest(tmp_path):
    from experiments.paper_baselines.run import main
    out=tmp_path/'result'
    assert main(['--artifact-root',str(tmp_path/'absent'),'--input-dir',str(tmp_path/'absent'),
                 '--out',str(out)])==2
    import json
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='not_executed' and manifest['reason']


def test_seed_average_is_expected_single_draw_loss_not_prediction_ensemble():
    from experiments.paper_baselines.analyze import halo_losses
    import pandas as pd
    data=pd.DataFrame([dict(graph_id='a',replicate_seed=100,prediction=1.,target=0.,squared_error=1.),
                       dict(graph_id='a',replicate_seed=101,prediction=-1.,target=0.,squared_error=1.)])
    assert halo_losses(data).loc['a']==1.  # averaging predictions would incorrectly give zero


def test_analysis_rejects_missing_halo_or_duplicate_seed():
    from experiments.paper_baselines.analyze import validate_population
    import pandas as pd
    good=pd.DataFrame([dict(graph_id='a',replicate_seed=100),dict(graph_id='b',replicate_seed=100)])
    validate_population(good,['a','b'])
    with pytest.raises(ValueError):validate_population(good.iloc[:1],['a','b'])
    with pytest.raises(ValueError):validate_population(pd.concat([good,good]),['a','b'])


def test_audit_tolerates_small_cross_budget_gpu_reference_roundoff_only():
    r=records();extra=copy.deepcopy(r)
    for row in extra:
        row['requested_keep_fraction']=.5
        row['full_prediction']+=.000001
        if row['method']=='full':row['squared_error']=(row['full_prediction']-row['target'])**2
    audit_records(r+extra,{'train':['a'],'val':['b'],'test':['c']},['full','random'],[.5,.6],'hash')
    extra[0]['full_prediction']+=.01
    with pytest.raises(ValueError):
        audit_records(r+extra,{'train':['a'],'val':['b'],'test':['c']},['full','random'],[.5,.6],'hash')


def test_manifest_verification_detects_changed_result_bytes(tmp_path):
    from experiments.paper_baselines.verify import verify_hashes
    from data.provenance import sha256_file
    p=tmp_path/'predictions.csv';p.write_text('prediction\n1.0\n')
    manifest={'outputs':{'predictions.csv':sha256_file(p)}}
    verify_hashes(tmp_path,manifest)
    p.write_text('prediction\n2.0\n')
    with pytest.raises(ValueError,match='checksum'):
        verify_hashes(tmp_path,manifest)


def test_alternative_gradient_forward_is_in_the_exported_shared_meter():
    from rls.benchmark_predictor import MeteredPredictor
    graph=Data(x=torch.ones(3,1),edge_index=torch.tensor([[0,1,0],[1,2,2]]),edge_attr=torch.ones(3,2))
    ly=PhysicalPairLayout.from_edge_index(graph.edge_index,3);meter=MeteredPredictor(LinearEdgeModel())
    alternative_saliency(LinearEdgeModel(),graph,ly,torch.ones(2),meter=meter)
    assert meter.snapshot()['scoring']['forward_calls']==1


def test_recovery_checkpoint_must_match_recorded_stage_b_checksum(tmp_path):
    import json
    from data.provenance import sha256_file
    from experiments.paper_baselines.run import read_recovery
    folder=tmp_path/'recovered_training';folder.mkdir()
    p=folder/'training_checkpoint.json'
    splits={'train':['a'],'val':['b'],'test':['c']}
    d={'context':{'split_cluster_ids':splits},'training_source':{'metadata_status':'reconstructed'}}
    p.write_text(json.dumps(d));m={'artifacts':{'recovered_training/training_checkpoint.json':sha256_file(p)}}
    recovery,record=read_recovery(tmp_path,m,splits)
    assert recovery['context']['split_cluster_ids']==splits
    d['training_source']['metadata_status']='forged';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='checksum'):
        read_recovery(tmp_path,m,splits)
