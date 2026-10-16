"""Execute frozen-model robustness controls; never train or download anything.

Run from the repository root with python -m experiments.paper_baselines.run.
Required original inputs are explicit. Each output directory must be new.
"""
import argparse
import copy
import csv
import gzip
import json
import platform
import importlib.metadata
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data.provenance import sha256_file,model_state_hash,hash_payload
from data.loaders.base_loader import get_loader
from graph.graph_builder import GraphBuilder
from model.model import load_model
from rls.constraints import PhysicalPairLayout,pair_budget,feasible_budget,selection_diagnostics
from rls.benchmark_predictor import MeteredPredictor
from rls.benchmark_scorers import PairScorer
from rls.policy import build_policy
from rls.matched_benchmark import _graph_content_hash
from .selectors import decode,marks_for,halo_seed,rank_fraction,heuristic_scores,alternative_saliency
from .audit import audit_records,validate_splits
from .analyze import choose_validation

HERE=Path(__file__).resolve().parent
METHODS=['full','random','distance','degree','stellar_binding_proxy',
         'edge_feature_grad_x_input','supervised_raw','provided_rl']


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    if not rows:raise ValueError(f'no rows for {path}')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def stage(root,name):
    return root/f'{name}-v1-terminal-001/artifacts/cosmic-net-runs/notebook_{name}'


def checked(path,expected=None):
    actual=sha256_file(path)
    if expected is not None and actual['sha256']!=expected:
        raise ValueError(f'checksum mismatch: {path}')
    return actual


def read_recovery(folder,manifest,splits):
    name='recovered_training/training_checkpoint.json'
    path=Path(folder)/name
    record=checked(path,manifest['artifacts'][name]['sha256'])
    recovery=json.loads(path.read_text())
    if recovery['context']['split_cluster_ids']!=splits:
        raise ValueError('policy training split differs')
    return recovery,record


def residual_row(graph_id,method,q,pred,target,**fields):
    residual=float(pred)-float(target)
    return dict(graph_id=graph_id,method=method,retention=float(q),prediction=float(pred),
                target=float(target),residual=residual,squared_error=residual**2,
                absolute_error=abs(residual),**fields)


def execute(args,cfg,out):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    root=Path(args.artifact_root);inputs=Path(args.input_dir)
    manifests={s:json.loads((stage(root,s)/'provenance.json').read_text())[s] for s in 'ABC'}
    context=manifests['C'];splits=context['split_cluster_ids'];validate_splits(splits)
    if len(splits['test'])!=82:raise ValueError('expected all 82 original test halos')
    source_inputs=[]
    for s in 'ABC':
        m=manifests[s]
        if m['split_cluster_ids']!=splits or m['backbone']!=context['backbone']:
            raise ValueError('A/B/C split or backbone identity differs')
        if m['status']!='complete':raise ValueError('incomplete input stage')
        source_inputs.append(checked(stage(root,s)/'provenance.json'))
    for s,names in [('C',['matched_per_example.json','results_table.csv','budget_eligibility.json']),
                    ('A',['learned_control.pt']),('B',['policy.pt'])]:
        for name in names:
            source_inputs.append(checked(stage(root,s)/name,manifests[s]['artifacts'][name]['sha256']))
    for key,name in [('tng_csv','tng100_clustered.csv'),('checkpoint','best_model_augmented.pt')]:
        source_inputs.append(checked(inputs/name,context['dataset'][key]['sha256']))
    rows=json.loads((stage(root,'C')/'matched_per_example.json').read_text())
    audit=audit_records(rows,splits,METHODS,cfg['retentions'],context['backbone']['sha256'])
    index={(r['graph_id'],r['method'],r['requested_keep_fraction']):r for r in rows}
    print('Saved rows/masks audited:',len(rows),audit['verified_pruned_masks'],flush=True)
    # Exact deterministic lazy/reference equivalence on ALL historical masks.
    for r in rows:
        if r['method']=='full':continue
        ly=PhysicalPairLayout(torch.tensor(r['layout_pairs']),torch.tensor(r['layout_inverse']),
                              torch.tensor(r['stored_real']),r['available_loop_columns'])
        got=decode(ly,torch.tensor(r['scores']),r['requested_pair_count'],torch.tensor(r['pair_marks']))
        if got.nonzero().flatten().tolist()!=r['selected_pairs']:raise ValueError('lazy/reference saved-mask mismatch')
    audit['lazy_decoder_saved_mask_matches']=audit['verified_pruned_masks']
    resolved=copy.deepcopy(context['config_used'])
    resolved['data']['tng']['clustered_file']=str(inputs/'tng100_clustered.csv')
    resolved['data']['tng']['cache_dir']=str(out/'scratch_cache')
    loader=get_loader(resolved);halos=loader.load()
    regenerated=dict(zip(('train','val','test'),loader.split_data(halos)))
    if {s:[h.cluster_id for h in v] for s,v in regenerated.items()}!=splits:
        raise ValueError('reconstructed halo split differs')
    builder=GraphBuilder(resolved)
    graphs={s:[builder.build_graph(h) for h in v] for s,v in regenerated.items()}
    model=load_model(str(inputs/'best_model_augmented.pt'),resolved,torch.device('cpu'),
                     research_label='legacy_total_radius').eval().requires_grad_(False)
    frozen_hash=model_state_hash(model.state_dict())
    if frozen_hash!=context['backbone']['sha256']:raise ValueError('loaded backbone hash differs')
    policy=build_policy(manifests['B']['config_used'],node_emb_dim=model.output_dim)
    policy.load_state_dict(torch.load(stage(root,'B')/'policy.pt',map_location='cpu',weights_only=True))
    policy.eval()
    expected_policy_hash=next(r for r in rows if r['method']=='provided_rl')['scorer_metadata']['policy_state_hash']
    if model_state_hash(policy.state_dict())!=expected_policy_hash:raise ValueError('policy state hash differs')
    learned=torch.load(stage(root,'A')/'learned_control.pt',map_location='cpu',weights_only=True)
    cohort=learned['training_cohort'];trained=set(cohort['train_ids'])
    if trained-set(splits['train']) or trained.intersection(splits['val']+splits['test']):
        raise ValueError('learned selector leakage')
    if cohort['split_hash']!=hash_payload(splits):raise ValueError('learned split hash differs')
    recovery,recovery_record=read_recovery(stage(root,'B'),manifests['B'],splits)
    source_inputs.append(recovery_record)
    audit['selector_training']={
        'learned_control_train_count':len(trained),'learned_excluded_training_graphs':cohort['excluded_graphs'],
        'policy_training_seed':42,'policy_selected_epoch':manifests['B']['policy_validation']['selected_epoch'],
        'policy_selection_scope':manifests['B']['policy_validation']['scope'],
        'policy_metadata_status':recovery['training_source'].get('metadata_status'),
        'backbone_training_split_certified':False}
    catalog=pd.read_csv(inputs/'tng100_clustered.csv')
    grouped=catalog.groupby('subhalo_id')['group_id'].nunique()
    audit['catalog_subhalo_ids_in_multiple_groups']=int((grouped>1).sum())
    if (grouped>1).any():raise ValueError('subhalo identity crosses halo groups')
    meter=MeteredPredictor(model);full_by_id={};replay=[];properties=[]
    for split,items in graphs.items():
        for g in items:
            ly=PhysicalPairLayout.from_edge_index(g.edge_index,g.num_nodes)
            target=float(g.y.item())
            properties.append(dict(graph_id=g.cluster_id,split=split,nodes=g.num_nodes,pairs=len(ly.pairs),
                mean_degree=2*len(ly.pairs)/g.num_nodes,target_log_mass=target,
                velocity_dispersion=float(g.vel_disp.mean()),
                mean_stellar_log_mass=float(g.stellar_mass.double().log10().mean())))
            if split!='test':continue
            old=index[g.cluster_id,'full',.4]
            if _graph_content_hash(g,g.cluster_id)!=old['graph_content_hash']:
                raise ValueError(f'rebuilt graph hash differs: {g.cluster_id}')
            full=float(meter.call(g,phase='shared_full').prediction.item());full_by_id[g.cluster_id]=full
            for method in METHODS:
                for q in cfg['retentions']:
                    r=index[g.cluster_id,method,q]
                    if method=='full':pred=full;saved=r['full_prediction']
                    else:
                        pred=float(meter.call(g,torch.tensor(r['stored_mask']),phase='evaluation').prediction.item())
                        saved=r['pruned_prediction']
                    error=abs(pred-saved)
                    if error>cfg['prediction_replay_atol']:raise ValueError(f'prediction replay failed {g.cluster_id} {method}: {error}')
                    replay.append(residual_row(g.cluster_id,method,q,pred,target,split='test',
                        saved_prediction=saved,replay_absolute_difference=error,execution='cpu_replay',
                        policy_training_seed=42 if method=='provided_rl' else '',
                        mask_seed=42 if method=='random' else '',source='historical_masks',
                        retained_pairs=r['final_pair_count'],available_pairs=len(ly.pairs)))
            # Fresh policy scoring checks that cached GPU logits are the same artifact.
            scores=PairScorer('provided_rl',policy=policy,edge_feature_names=resolved['graph']['edge_features']).score(
                g,ly,pair_budget(ly,.4),meter=meter).scores
            saved=torch.tensor(index[g.cluster_id,'provided_rl',.4]['scores'])
            if not torch.allclose(scores,saved,atol=1e-4,rtol=1e-4):raise ValueError('policy score replay differs')
    audit['graph_hashes_rebuilt']=len(graphs['test'])
    audit['prediction_replay']={'rows':len(replay),'max_absolute_difference':max(r['replay_absolute_difference'] for r in replay),
                                'atol':cfg['prediction_replay_atol'],'device':'cpu'}
    print('All historical predictions and policy scores replayed',flush=True)
    train_attrs=torch.cat([g.edge_attr for g in graphs['train']])
    std=train_attrs.std(0).clamp_min(1e-8)
    train_dv=[]
    for g in graphs['train']:
        ly=PhysicalPairLayout.from_edge_index(g.edge_index,g.num_nodes)
        a,b=ly.pairs.T;train_dv.append((g.vel[a]-g.vel[b]).norm(dim=1))
    velocity_scale=float(torch.cat(train_dv).median().clamp_min(1.))
    controls=[];masks=[];eligibility=[];tuning=[]
    geometry={};physics={}
    for split in ['val','test']:
        for g in graphs[split]:
            ly=PhysicalPairLayout.from_edge_index(g.edge_index,g.num_nodes)
            h=heuristic_scores(g,ly,velocity_scale)
            geometry[g.cluster_id]=h;physics[g.cluster_id]=h['velocity_binding_proxy']
    def evaluate(g,split,method,q,scores,*,family='deterministic',seed=None,marks=None,scoring_s=None):
        ly=PhysicalPairLayout.from_edge_index(g.edge_index,g.num_nodes);k=pair_budget(ly,q)
        report=feasible_budget(ly,k,'no_isolates')
        if not report['feasible']:
            eligibility.append(dict(graph_id=g.cluster_id,split=split,method=method,retention=q,status='infeasible',**report))
            raise ValueError('configured experiment has an infeasible halo; no silent cohort changes')
        if marks is None:marks=marks_for(ly,42,g.cluster_id)
        start=time.perf_counter()
        sampled=halo_seed(seed,g.cluster_id,'policy_sample') if family=='policy_sampling' else None
        selected=decode(ly,scores,k,marks,sampling_seed=sampled);decode_s=time.perf_counter()-start
        mask=ly.expand(selected);start=time.perf_counter()
        pred=float(meter.call(g,mask,phase='evaluation').prediction.item());predict_s=time.perf_counter()-start
        mask_hash=hash_payload(selected.tolist())
        row=residual_row(g.cluster_id,method,q,pred,float(g.y.item()),split=split,source='new_execution',
            seed_family=family,replicate_seed=seed if seed is not None else '',
            policy_training_seed=42 if method.startswith('rl_') else '',
            random_mask_seed=seed if family=='random_mask' else '',
            policy_sampling_seed=seed if family=='policy_sampling' else '',
            tie_mark_seed=seed if family=='tie_marks' else 42,
            available_pairs=len(ly.pairs),retained_pairs=k,physical_retention=k/len(ly.pairs),
            decoder_time_s=decode_s,prediction_wrapper_time_s=predict_s,
            scoring_time_s='' if scoring_s is None else scoring_s,
            timing_scope='cached_scores; decoder+predictor wrapper only' if scoring_s is None else 'scoring+decoder+predictor wrapper',
            selected_pairs_sha256=mask_hash)
        masks.append(dict(graph_id=g.cluster_id,split=split,method=method,retention=q,
            seed_family=family,replicate_seed=seed,selected_pairs=selected.tolist(),
            scores=torch.as_tensor(scores).tolist(),pair_marks=marks.tolist(),
            selected_pairs_sha256=mask_hash,diagnostics=selection_diagnostics(ly,selected,k=k,constraint='no_isolates')))
        return row
    # Tune geometric mixing on validation q=.4, retaining every candidate/halo.
    for alpha in cfg['geometric_alpha_grid']:
        name=f'geometric_a{alpha:g}'
        for g in graphs['val']:
            h=geometry[g.cluster_id];scores=alpha*h['near3d']+(1-alpha)*h['near_projected']
            tuning.append(evaluate(g,'val',name,.4,scores))
    geom_name=choose_validation(tuning,[f'geometric_a{a:g}' for a in cfg['geometric_alpha_grid']])
    geom_alpha=float(geom_name.split('_a')[1])
    for alpha in cfg['hybrid_alpha_grid']:
        name=f'hybrid_a{alpha:g}'
        for g in graphs['val']:
            h=geometry[g.cluster_id];geom=geom_alpha*h['near3d']+(1-geom_alpha)*h['near_projected']
            scores=alpha*rank_fraction(geom)+(1-alpha)*physics[g.cluster_id]
            tuning.append(evaluate(g,'val',name,.4,scores))
    hybrid_name=choose_validation(tuning,[f'hybrid_a{a:g}' for a in cfg['hybrid_alpha_grid']])
    hybrid_alpha=float(hybrid_name.split('_a')[1])
    # Once locked above, test residuals cannot change any hyperparameter.
    hyperparameters=dict(geometric_alpha=geom_alpha,hybrid_alpha=hybrid_alpha,
                         velocity_scale_km_s=velocity_scale,train_feature_std=std.tolist(),
                         scope='train feature scales; validation q=.4 RMSE tuning; fixed for all test budgets')
    probability_checks=[]
    for i,g in enumerate(graphs['test']):
        ly=PhysicalPairLayout.from_edge_index(g.edge_index,g.num_nodes)
        start=time.perf_counter();saliency=alternative_saliency(model,g,ly,std,meter=meter);grad_seconds=time.perf_counter()-start
        start=time.perf_counter();h=heuristic_scores(g,ly,velocity_scale)
        geom=geom_alpha*h['near3d']+(1-geom_alpha)*h['near_projected']
        hybrid=hybrid_alpha*rank_fraction(geom)+(1-hybrid_alpha)*h['velocity_binding_proxy']
        heuristic_seconds=time.perf_counter()-start
        det={'geometric_tuned':geom,'velocity_binding_proxy':h['velocity_binding_proxy'],
             'hybrid_tuned':hybrid,**saliency}
        for q in cfg['retentions']:
            old=index[g.cluster_id,'provided_rl',q];marks=torch.tensor(old['pair_marks'])
            logits=torch.tensor(old['scores'],dtype=torch.float64);k=pair_budget(ly,q)
            prob=logits.sigmoid();prob_mask=decode(ly,prob,k,marks)
            logit_mask=decode(ly,logits,k,marks)
            probability_checks.append(dict(graph_id=g.cluster_id,retention=q,
                probability_unique_scores=int(prob.unique().numel()),logit_unique_scores=int(logits.unique().numel()),
                same_mask=bool(torch.equal(prob_mask,logit_mask)),
                max_abs_logit=float(logits.abs().max())))
            if not torch.equal(prob_mask,logit_mask):
                controls.append(evaluate(g,'test','rl_probability_rank',q,prob,marks=marks))
            for name,scores in det.items():
                controls.append(evaluate(g,'test',name,q,scores,marks=marks,
                    scoring_s=grad_seconds if name in saliency else heuristic_seconds))
            for seed in cfg['random_mask_seeds']:
                generator=torch.Generator().manual_seed(halo_seed(seed,g.cluster_id,'random_mask'))
                scores=torch.randn(len(ly.pairs),generator=generator)
                controls.append(evaluate(g,'test','random_repeated',q,scores,family='random_mask',seed=seed,marks=marks))
            for seed in cfg['policy_sampling_seeds']:
                controls.append(evaluate(g,'test','rl_sampled',q,logits,family='policy_sampling',seed=seed,marks=marks))
            for seed in cfg['tie_mark_seeds']:
                controls.append(evaluate(g,'test','rl_tie_marks',q,logits,family='tie_marks',seed=seed,
                                         marks=marks_for(ly,seed,g.cluster_id)))
        if (i+1)%10==0:print(f'New controls: {i+1}/82 halos',flush=True)
    if model_state_hash(model.state_dict())!=frozen_hash:raise ValueError('backbone state mutated')
    if model_state_hash(policy.state_dict())!=expected_policy_hash:raise ValueError('policy state mutated')
    audit['probability_ranking']={'checks':len(probability_checks),'same_masks':sum(r['same_mask'] for r in probability_checks),
                                'control_counted_as_new_method':False}
    write_csv(out/'historical_replay.csv',replay);write_csv(out/'new_predictions.csv',controls)
    write_csv(out/'validation_predictions.csv',tuning);write_csv(out/'halo_properties.csv',properties)
    write_csv(out/'probability_ranking_checks.csv',probability_checks)
    with gzip.open(out/'masks.jsonl.gz','wt') as stream:
        for row in masks:stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
    dump(out/'audit.json',audit);dump(out/'hyperparameters.json',hyperparameters)
    dump(out/'split_ids.json',splits);dump(out/'eligibility.json',{'excluded':eligibility,'all_test_halos_in_every_configured_run':True})
    dump(out/'config.json',cfg)
    timing={}
    for method in METHODS:
        own=[r for r in rows if r['method']==method and r['requested_keep_fraction']==.4]
        timing[method]={key:float(np.mean([r[key] for r in own])) for key in
            ['scoring_time_s','decoder_time_s','prediction_time_s','full_time_s','total_region_time_s']}
        timing[method]['mean_live_forward_calls']=float(np.mean([r['live_forward_calls'] for r in own]))
    dump(out/'historical_timing.json',timing)
    # Original checkpoint has config and weights but no certified training IDs.
    checkpoint=torch.load(inputs/'best_model_augmented.pt',map_location='cpu',weights_only=True)
    dump(out/'checkpoint_metadata.json',{'keys':sorted(checkpoint),'has_provenance':bool(checkpoint.get('provenance')),
        'training_split_certified':False,'research_label':context['research_label']})
    source={str(p.relative_to(HERE.parents[1])):sha256_file(p)['sha256'] for p in sorted(HERE.glob('*.py'))}
    result=dict(status='executed',created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=cfg['base_commit'],branch='codex/paper-baseline-robustness',
        sources=source_inputs,experiment_source_sha256=source,
        environment={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                     'numpy':np.__version__,'device':'cpu','torch_threads':1,
                     'dependencies':{name:importlib.metadata.version(name) for name in
                         ['torch-geometric','scipy','networkx','pandas','matplotlib','PyYAML','python-dotenv','requests','pytest']}},
        seeds={key:value for key,value in cfg.items() if 'seed' in key or 'bootstrap' in key},
        prediction_counts={'historical_replay':len(replay),'new_test':len(controls),'new_validation':len(tuning)},
        execution_registry=[{'experiment':'saved benchmark audit/replay','status':'executed'},
            {'experiment':'repeated random masks, ten seeds/four budgets','status':'executed'},
            {'experiment':'validation-tuned geometry/hybrid, physics proxy, alternative gradients','status':'executed'},
            {'experiment':'policy PL sampling and tie marks, same trained policy','status':'executed'},
            {'experiment':'deterministic sigmoid probability rank','status':'executed; duplicate-mask control'},
            {'experiment':'independent policy training seeds','status':'not_executed','reason':'only seed 42 trained artifact; no retraining authorized/needed'},
            {'experiment':'certified corrected-astronomy or external population','status':'not_executed','reason':'legacy checkpoint lacks target/training provenance; no matching certified weights'},
            {'experiment':'new GPU latency measurement','status':'not_executed','reason':'local CPU execution; no cross-device speedup claims'}],
        meter=meter.snapshot())
    result['outputs']={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}
    dump(out/'manifest.json',result)
    # Loader's scratch cache contains no scientific outputs.
    import shutil
    shutil.rmtree(out/'scratch_cache',ignore_errors=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',required=True)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--config',default=str(HERE/'config.json'))
    args=parser.parse_args(argv);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(args.config).read_text());start=time.perf_counter()
    try:
        result=execute(args,cfg,out)
    except (FileNotFoundError,ValueError,KeyError,RuntimeError) as exc:
        dump(out/'manifest.json',{'status':'not_executed' if isinstance(exc,FileNotFoundError) else 'failed',
            'reason':str(exc),'config':cfg,'completed_outputs_claimed':False})
        print(str(exc),file=sys.stderr);return 2
    print('Executed',result['prediction_counts'],'seconds',time.perf_counter()-start,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
