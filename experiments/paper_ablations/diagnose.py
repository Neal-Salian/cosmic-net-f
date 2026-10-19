"""Frozen-policy inference experiments. No optimization or retraining occurs.

python -m experiments.paper_ablations.diagnose --help
"""
import argparse
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from rls.constraints import PhysicalPairLayout, feasible_budget, pair_budget
from rls.notebook_workflow import predict_graph
from rls.rewards import relative_virial_penalty
from rls.train_policy import _graph_physics_terms
from .core import decode, finish_run, require_matched, start_run, write_json
from .inputs import load_inputs, policy_scores, source_record


def summarize(rows, *, publish_accuracy=True):
    errors=np.array([r['prediction']-r['target'] for r in rows])
    return dict(n_halos=len(rows), rmse_dex=float(np.sqrt(np.mean(errors**2))) if publish_accuracy else None,
        mae_dex=float(np.mean(abs(errors))) if publish_accuracy else None,
        mean_pair_keep=float(np.mean([r['final_pair_count']/r['available_pair_count'] for r in rows])),
        mean_total_edge_keep=float(np.mean([r['total_edge_keep'] for r in rows])),
        isolate_halos=sum(r['physical_isolates']>0 for r in rows),
        budget_violation_halos=sum(not r['budget_satisfied'] for r in rows),
        mean_repair_added_pairs=float(np.mean([r['repair_added_pair_count'] for r in rows])),
        mean_relative_virial_penalty=float(np.mean([r['relative_virial_penalty'] for r in rows])))


def evaluate(gnn, graphs, selections):
    rows=[]
    for g,s in zip(graphs,selections):
        mask=s['mask']; full=torch.ones_like(mask)
        ke,pe=_graph_physics_terms(g,g['edge_index'],mask)
        kf,pf=_graph_physics_terms(g,g['edge_index'],full)
        rows.append(dict(graph_id=g['cluster_id'],target=float(g['y'].reshape(())),
            prediction=float(predict_graph(gnn,g,mask).reshape(())),
            selected_pair_indices=s['selected'].nonzero().flatten().tolist(),
            total_edge_keep=float(mask.float().mean()),
            relative_virial_penalty=float(relative_virial_penalty(ke,pe,kf,pf)),**s['diagnostics']))
    return rows


def training_log_diagnostic(archive):
    with zipfile.ZipFile(archive) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('recovered_training/training_log.csv').decode())))
    keys=['reward_mean','accuracy_mean','virial_mean','sparsity_mean','physical_isolates_mean',
          'physical_pair_keep_mean','target_pair_keep','entropy']
    return dict(kind='historical_training_log_audit_not_retraining',epochs=len(rows),
        final_epoch={k:float(rows[-1][k]) for k in keys if k in rows[-1]},
        reward_min_all_epochs=min(float(r['reward_min']) for r in rows),
        reward_max_all_epochs=max(float(r['reward_max']) for r in rows),
        zero_isolate_epochs=sum(float(r['physical_isolates_mean'])==0 for r in rows),
        note='Logged accuracy omits w_acc (which is 1); logged reward includes connectivity and clipping.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('csv','backbone','recovered-archive','output'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--budgets',nargs='+',type=float,default=[.2,.4,.6,.8])
    p.add_argument('--sample-seeds',nargs='+',type=int,default=[7,19,43,71,101])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); torch.set_num_threads(1)
    source=source_record()
    cfg,gnn,policy,splits,identity=load_inputs(args.csv,args.backbone,args.recovered_archive,
        args.output.parent/(args.output.name+'-cache'))
    contract=dict(kind='inference_only_no_retraining',source=source,inputs=identity,
        budgets=args.budgets,sample_seeds=args.sample_seeds,marks_seed=42,device='cpu',
        cohort_rule='intersection of topology-feasible test halos over every requested budget',
        prediction='single deterministic frozen GNN pass; no TTA or MC dropout')
    if start_run(args.output,contract,args.resume)=='complete':
        print('Verified completed diagnostics; nothing rerun.'); return
    all_graphs=splits['test']; graphs=[]; eligibility=[]
    for g in all_graphs:
        layout=PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x']))
        reports=[dict(keep=q,**feasible_budget(layout,pair_budget(layout,q),'no_isolates')) for q in args.budgets]
        eligible=all(r['feasible'] for r in reports)
        if eligible: graphs.append(g)
        eligibility.append(dict(graph_id=g['cluster_id'],eligible=eligible,budgets=reports))
    if len(graphs)<2: raise ValueError('Too few common feasible test halos')
    write_json(args.output/'eligibility.json',eligibility)
    print(f'Common test cohort: {len(graphs)}/{len(all_graphs)}',flush=True)
    scores={g['cluster_id']:policy_scores(policy,g,cfg) for g in all_graphs}
    predictions={}; summaries=[]; mask_equivalence=[]; comparisons=[]
    def arm(name, selected_graphs, selections, q, mode, seed):
        rows=evaluate(gnn,selected_graphs,selections)
        predictions[name]=rows
        result=dict(arm=name,requested_keep=q,mode=mode,seed=seed,
            trained_reward_ablation=False,**summarize(rows,publish_accuracy=all(r['budget_satisfied'] for r in rows)))
        summaries.append(result); print(name,result['rmse_dex'],flush=True)
        return rows
    for q in args.budgets:
        selections={}
        for mode in ('constrained_rank','probability_rank','unconstrained_rank','rank_repair'):
            selections[mode]=[decode(g,scores[g['cluster_id']],q,mode,42) for g in graphs]
            arm(f'{q:g}_{mode}',graphs,selections[mode],q,mode,42)
        require_matched(selections['constrained_rank'],selections['unconstrained_rank'])
        mask_equivalence.append(dict(keep=q,n_halos=len(graphs),identical_masks=sum(torch.equal(a['mask'],b['mask'])
            for a,b in zip(selections['constrained_rank'],selections['probability_rank']))))
        for seed in args.sample_seeds:
            selected=[decode(g,scores[g['cluster_id']],q,'constrained_sample',seed) for g in graphs]
            require_matched(selections['constrained_rank'],selected)
            arm(f'{q:g}_sample_{seed}',graphs,selected,q,'constrained_sample',seed)
        # Freeze each repaired count, then compare at exactly that per-halo cost.
        repaired=selections['rank_repair']
        matched=[decode(g,scores[g['cluster_id']],q,'constrained_rank',42,
            exact_k=r['diagnostics']['final_pair_count']) for g,r in zip(graphs,repaired)]
        repaired_at_cost=[dict(r,diagnostics=dict(r['diagnostics'],budget_satisfied=True,
            requested_pair_count=r['diagnostics']['final_pair_count'])) for r in repaired]
        require_matched(matched,repaired_at_cost)
        arm(f'{q:g}_direct_at_repair_cost',graphs,matched,q,'direct_at_repair_cost',42)
        arm(f'{q:g}_repair_at_repair_cost',graphs,repaired_at_cost,q,'repair_at_repair_cost',42)
        comparisons.append(dict(keep=q,kind='matched_realized_repair_cost',
            counts=[r['diagnostics']['final_pair_count'] for r in repaired],
            note='Score/mark-dependent cost fixed by repaired arm, without labels. Not nominal-q evidence.'))
    # Replay original q=.4 topology-feasible population independently of budget-curve intersection.
    replay=[g for g in all_graphs if feasible_budget(PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x'])),
        pair_budget(PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x'])),.4),'no_isolates')['feasible']]
    arm('original_0.4_cohort_replay',replay,[decode(g,scores[g['cluster_id']],.4,'constrained_rank',42)
        for g in replay],.4,'constrained_rank',42)
    full=[dict(graph_id=g['cluster_id'],selected=torch.ones(len(PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x'])).pairs),dtype=torch.bool),
        mask=torch.ones(g['edge_index'].shape[1],dtype=torch.bool),diagnostics=dict(
            final_pair_count=len(PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x'])).pairs),
            available_pair_count=len(PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x'])).pairs),
            physical_isolates=0,budget_satisfied=True,repair_added_pair_count=0)) for g in graphs]
    arm('full_graph_common_cohort',graphs,full,1.,'reference_full_budget',42)
    write_json(args.output/'predictions.json',predictions)
    write_json(args.output/'summary.json',summaries)
    write_json(args.output/'mask_equivalence.json',mask_equivalence)
    write_json(args.output/'comparisons.json',comparisons)
    write_json(args.output/'historical_training_log.json',training_log_diagnostic(args.recovered_archive))
    finish_run(args.output,['eligibility.json','predictions.json','summary.json','mask_equivalence.json',
                            'comparisons.json','historical_training_log.json'])


if __name__=='__main__': main()
