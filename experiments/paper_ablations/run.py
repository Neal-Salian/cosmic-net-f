"""Train ONE registered ablation from scratch; resume only its own full state."""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from rls.constraints import PhysicalPairLayout, feasible_budget, pair_budget
from rls.notebook_workflow import policy_metrics, prediction_pair
from rls.policy import build_policy
from rls.policy_gradient import PolicyGradientTrainer, ValueNet
from .core import VARIANTS, decode, finish_run, start_run, variant_config, write_json
from .diagnose import evaluate, summarize
from .inputs import load_inputs, policy_scores, source_record
from .training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('csv','backbone','recovered-archive','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-commit',required=True,help='Exact committed source hash from ABLATION_SOURCE.json or git rev-parse HEAD')
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--keep-fraction',type=float,default=.4)
    p.add_argument('--cohort-budgets',type=float,nargs='+',default=[.4])
    p.add_argument('--training-decoder',choices=['legacy_repair','direct'],default='legacy_repair')
    p.add_argument('--training-cohort',choices=['original','exact_feasible'],default='original')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if args.epochs<1: p.error('--epochs must be positive')
    if args.training_decoder=='direct' and args.training_cohort!='exact_feasible':
        p.error('Direct training requires exact_feasible; use that same cohort in its repaired control')
    if args.device=='cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; choose --device cpu explicitly or enable Kaggle GPU')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    source=source_record()
    if args.expected_commit!=source['evaluator_commit']:
        raise ValueError('Source differs from --expected-commit')
    cfg,gnn,_,splits,identity=load_inputs(args.csv,args.backbone,args.recovered_archive,
        args.output.parent/(args.output.name+'-cache'),args.device)
    # Data split remains the recovered seed-42 split for EVERY policy seed.
    rls=variant_config(cfg['rls'],args.variant)
    rls.update(seed=args.seed,epochs=args.epochs,target_sparsity_end=args.keep_fraction,
        evaluation_decoder='direct',evaluation_constraint='no_isolates')
    cohort_budgets=sorted(set(args.cohort_budgets+[args.keep_fraction]))
    cohorts={}; eligibility={}
    for split,graphs in splits.items():
        rows=[]; eligible=[]
        for g in graphs:
            layout=PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x']))
            reports=[dict(q=q,**feasible_budget(layout,pair_budget(layout,q),'no_isolates')) for q in cohort_budgets]
            feasible=all(r['feasible'] for r in reports)
            rows.append(dict(graph_id=g['cluster_id'],eligible=feasible,budgets=reports))
            if feasible or (split=='train' and args.training_cohort=='original'): eligible.append(g)
        if len(eligible)<2: raise ValueError(f'Too few halos for {split}')
        cohorts[split]=eligible; eligibility[split]=rows
    contract=dict(kind='new_policy_training',source=source,inputs=identity,rls=rls,
        policy_seed=args.seed,split_seed=cfg['seed'],epochs=args.epochs,
        training_decoder=args.training_decoder,training_cohort=args.training_cohort,
        cohort_budgets=cohort_budgets,cohort_ids={k:[g['cluster_id'] for g in v] for k,v in cohorts.items()},
        device=args.device,selection='minimum validation RMSE including untrained initialization',
        evaluation_marks_seed=42,scope='planned_until_run_complete',stageB=False,TTA=False)
    if start_run(args.output,contract,args.resume)=='complete':
        print('Completed run and artifact checksums verified; no retraining.'); return
    write_json(args.output/'eligibility.json',eligibility)
    if args.preflight_only:
        print('Preflight only; no policy trained and no test predictions produced.'); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    cfg['rls']=rls
    evaluation_cfg=dict(rls,seed=42)
    policy=build_policy(cfg,node_emb_dim=gnn.output_dim).to(args.device)
    critic=ValueNet(gnn.output_dim).to(args.device)
    trainer=PolicyGradientTrainer(policy,critic,torch.optim.Adam(policy.parameters(),lr=rls['lr']),
        torch.optim.Adam(critic.parameters(),lr=rls['lr']),rls)
    state=train(trainer,cohorts['train'],prediction_pair(gnn),rls,args.device,args.epochs,
        lambda p:policy_metrics(p,cohorts['val'],gnn,evaluation_cfg,target_sparsity=args.keep_fraction),
        args.output/'resume.pt',resume=args.resume,decoder=args.training_decoder,
        log_fn=lambda row:print(json.dumps(row,allow_nan=False),flush=True))
    # No publication of test results before the complete training budget.
    if state['next_epoch']!=args.epochs: raise RuntimeError('Training incomplete')
    torch.save(policy.state_dict(),args.output/'selected_policy.pt')
    torch.save(critic.state_dict(),args.output/'selected_value.pt')
    write_json(args.output/'training_log.json',state['history'])
    write_json(args.output/'selected_validation.json',dict(best_epoch=state['best_epoch'],
        rmse=state['best_rmse'],initial_validation=state['initial_validation']))
    selections=[decode(g,policy_scores(policy,g,cfg),args.keep_fraction,'constrained_rank',42)
                for g in cohorts['test']]
    rows=evaluate(gnn,cohorts['test'],selections)
    if any(not r['budget_satisfied'] or r['physical_isolates'] for r in rows):
        raise RuntimeError('Test graph violates the exact evaluation contract')
    write_json(args.output/'per_halo_predictions.json',rows)
    write_json(args.output/'results.json',dict(variant=args.variant,seed=args.seed,
        policy_retrained=True,epochs_completed=state['next_epoch'],best_epoch=state['best_epoch'],
        fitted_reward_scale=rls['reward_error_scale'],scope='single_seed_experiment',**summarize(rows)))
    finish_run(args.output,['eligibility.json','selected_policy.pt','selected_value.pt',
        'training_log.json','selected_validation.json','per_halo_predictions.json','results.json','resume.pt'])


if __name__=='__main__': main()
