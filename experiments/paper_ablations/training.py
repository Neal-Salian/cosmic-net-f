"""Resumable reference pair-PL training, isolated from recovered artifacts.

Default objective follows rls.pair_training at BASE_COMMIT. A parity test
checks selected policy tensors against that implementation on a small graph.
The optional direct decoder is a NEW training condition, never a recovered run.
"""
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from data.provenance import hash_payload
from rls.constraints import PhysicalPairLayout, pair_budget
from rls.constrained_policy import select_pairs
from rls.pair_policy import pair_layout, pair_mask, pair_scores, pair_stats
from rls.rewards import compute_rewards, relative_virial_penalty
from rls.train_policy import _graph_physics_terms


def train(trainer, graphs, predictor, cfg, device, epochs, validation_fn, checkpoint,
          *, resume=False, stop_after=None, decoder='legacy_repair', log_fn=None):
    if not graphs or int(cfg.get('pair_rollouts',4))<2:
        raise ValueError('Nonempty training population and >=2 rollouts required')
    if decoder not in ('legacy_repair','direct'):
        raise ValueError('Unknown training decoder')
    policy,critic=trainer.policy.to(device),trainer.value_net.to(device)
    checkpoint=Path(checkpoint)
    cached=[]; full_errors=[]
    for graph in graphs:
        g={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in graph.items()}
        g['pair_layout']=pair_layout(g['edge_index'])
        mask=torch.ones(g['edge_index'].shape[1],dtype=torch.bool,device=device)
        with torch.no_grad():
            full,_=predictor(g,mask)
            full_errors.append((full.reshape(())-g['y'].reshape(())).square())
            g['reference_prediction']=full.detach()
            g['reference_physics']=_graph_physics_terms(g,g['edge_index'],mask)
        cached.append(g)
    cfg['reward_error_scale']=max(float(torch.stack(full_errors).mean().sqrt()),float(cfg.get('reward_scale_floor',.05)))
    contract=hash_payload(dict(cfg=cfg,epochs=epochs,decoder=decoder,
        graphs=[dict(id=g['cluster_id'],edges=g['edge_index'].cpu().tolist()) for g in graphs]))
    if resume and checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['contract_sha256']!=contract:
            raise ValueError('Training resume contract changed')
        policy.load_state_dict(state['policy']); critic.load_state_dict(state['critic'])
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.value_optimizer.load_state_dict(state['value_optimizer'])
        torch.set_rng_state(state['rng'])
        if torch.cuda.is_available() and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        if checkpoint.exists(): raise FileExistsError('Refusing to overwrite an existing training state')
        policy.fit_normalization(graphs)
        initial=validation_fn(policy)
        state=dict(next_epoch=0,history=[],best_rmse=initial['rmse'],best_epoch=None,
            best_policy=copy.deepcopy(policy.state_dict()),best_critic=copy.deepcopy(critic.state_dict()),
            contract_sha256=contract,initial_validation=initial)
    end=epochs if stop_after is None else min(epochs,stop_after)
    for epoch in range(state['next_epoch'],end):
        target=trainer.target_sparsity(epoch)
        order=torch.randperm(len(cached)).tolist()
        records=[]; batch_records=[]; policy.train(); critic.train()
        for start in range(0,len(order),int(cfg['batch_size'])):
            actor_terms=[]; value_terms=[]; entropy_terms=[]
            for index in order[start:start+int(cfg['batch_size'])]:
                g=cached[index]
                logits=policy(g['edge_attr'],g['emb'],g['edge_index'],g['ctx']).reshape(-1)
                scores=pair_scores(g['edge_index'],logits,g['pair_layout'])
                log_first=torch.log_softmax(scores,dim=0)
                entropy=-(log_first.exp()*log_first).sum()/max(1.,math.log(max(1,scores.numel())))
                logps=[]; rewards=[]
                for _ in range(int(cfg.get('pair_rollouts',4))):
                    if decoder=='legacy_repair':
                        mask,logp,_,_=pair_mask(g['edge_index'],logits,target,sample=True,layout=g['pair_layout'])
                    else:
                        layout=PhysicalPairLayout.from_edge_index(g['edge_index'],len(g['x']))
                        selection=select_pairs(layout,scores,pair_budget(layout,target),'no_isolates',sample=True)
                        mask,logp=selection.mask,selection.log_probability
                    with torch.no_grad():
                        full=g['reference_prediction']; _,pruned=predictor(g,mask)
                        ke,pe=_graph_physics_terms(g,g['edge_index'],mask)
                        vp=relative_virial_penalty(ke,pe,*g['reference_physics'])
                        stats=pair_stats(g['edge_index'],mask,len(g['x']))
                        reward_cfg=dict(cfg)
                        anneal_start=int(cfg.get('virial_anneal_start_epoch',0))
                        reward_cfg['w_virial']=cfg.get('w_virial',0.)*min(1.,max(0.,
                            (epoch-anneal_start+1)/max(1,int(cfg.get('virial_anneal_epochs',10)))))
                        reward=compute_rewards(pruned,full,g['y'],stats['physical_pair_keep'],target,vp,
                            reward_cfg,connectivity_ok=stats['physical_isolates']==0).reshape(())
                        records.append(dict(reward=float(reward),prediction_term=cfg['w_acc']*float(
                            ((full-g['y']).abs()-(pruned-g['y']).abs()).mean())/cfg['reward_error_scale'],
                            virial_term=-reward_cfg['w_virial']*float(vp),
                            sparsity_term=-cfg['w_sp']*(stats['physical_pair_keep']-target)**2,
                            connectivity_term=cfg['w_conn']*(1. if stats['physical_isolates']==0 else -2.),**stats))
                    logps.append(logp); rewards.append(reward)
                rewards=torch.stack(rewards)
                baselines=(rewards.sum()-rewards)/(len(rewards)-1)
                actor_terms.append(-((rewards-baselines).detach()*torch.stack(logps)).mean())
                context=g['ctx']
                if policy.normalize:
                    context=((context-policy.context_mean)/policy.context_scale).clamp(-10,10)
                value_terms.append(F.smooth_l1_loss(critic(context).reshape(()),rewards.mean()))
                entropy_terms.append(entropy)
            actor,value,entropy=(torch.stack(items).mean() for items in (actor_terms,value_terms,entropy_terms))
            loss=actor+float(cfg.get('value_coef',.5))*value-float(cfg.get('entropy_coef',.01))*entropy
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            trainer.optimizer.zero_grad(); trainer.value_optimizer.zero_grad(); loss.backward()
            pg=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            vf=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True)
            trainer.optimizer.step(); trainer.value_optimizer.step()
            batch_records.append(dict(loss=float(loss.detach()),actor=float(actor.detach()),critic=float(value.detach()),
                entropy=float(entropy.detach()),policy_grad_norm=float(pg),critic_grad_norm=float(vf),n=len(actor_terms)))
        row={key:float(np.average([b[key] for b in batch_records],weights=[b['n'] for b in batch_records]))
             for key in batch_records[0] if key!='n'}
        for key in records[0]: row[key+'_mean']=float(np.mean([r[key] for r in records]))
        metrics=validation_fn(policy)
        if not np.isfinite(metrics['rmse']): raise RuntimeError('Nonfinite validation error')
        row.update(epoch=epoch,target_pair_keep=target,val_rmse=metrics['rmse'])
        if metrics['rmse']<state['best_rmse']:
            state.update(best_rmse=metrics['rmse'],best_epoch=epoch,
                best_policy=copy.deepcopy(policy.state_dict()),best_critic=copy.deepcopy(critic.state_dict()))
        state['history'].append(row); state['next_epoch']=epoch+1
        state.update(policy=policy.state_dict(),critic=critic.state_dict(),optimizer=trainer.optimizer.state_dict(),
            value_optimizer=trainer.value_optimizer.state_dict(),rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],cfg=copy.deepcopy(cfg))
        temporary=checkpoint.with_suffix('.tmp'); torch.save(state,temporary); temporary.replace(checkpoint)
        if log_fn: log_fn(row)
    if state['next_epoch']==epochs:
        policy.load_state_dict(state['best_policy']); critic.load_state_dict(state['best_critic'])
    policy.eval()
    return state
