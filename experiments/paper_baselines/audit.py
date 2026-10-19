"""Fail-closed audit of the stored, matched per-halo benchmark records."""
import itertools
import math
import numpy as np
import torch
from rls.constraints import PhysicalPairLayout, selection_diagnostics


def validate_splits(splits):
    if set(splits)!={'train','val','test'}:raise ValueError('three exact split lists required')
    seen=set()
    for name,ids in splits.items():
        if not ids or len(set(ids))!=len(ids):raise ValueError(f'duplicate or empty {name}')
        if seen.intersection(ids):raise ValueError('split overlap')
        seen.update(ids)


def audit_records(rows,splits,methods,retentions,model_hash):
    validate_splits(splits)
    expected=set(itertools.product(splits['test'],methods,retentions))
    actual=[(r['graph_id'],r['method'],r['requested_keep_fraction']) for r in rows]
    if len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('unequal, missing, duplicate, or unexpected test population')
    identities={}
    references={}
    all_references={}
    mask_count=0
    for r in rows:
        if r['status']!='success' or r['model_state_hash']!=model_hash:
            raise ValueError('failed row or changed frozen backbone')
        if r['constraint']!='no_isolates' or r['decoder']!='direct':
            raise ValueError('different graph constraint/decoder')
        cid=r['graph_id'];p=r['available_pair_count'];q=r['requested_keep_fraction']
        k=math.ceil(q*p)
        if r['requested_pair_count']!=k:raise ValueError('incorrect ceil physical budget')
        identity=tuple(r.get(f) for f in ['target','graph_content_hash',
                     'split_manifest_sha256','config_sha256'])
        if cid in identities and identities[cid]!=identity:raise ValueError('halo identity/settings mismatch')
        identities[cid]=identity
        refkey=(cid,q)
        if refkey in references and references[refkey]!=r['full_prediction']:
            raise ValueError('nonshared full reference within budget')
        references[refkey]=r['full_prediction']
        values=all_references.setdefault(cid,[])
        values.append(r['full_prediction'])
        if max(values)-min(values)>1e-5:
            raise ValueError('full reference differs across budgets beyond GPU roundoff tolerance')
        pred=r['full_prediction'] if r['method']=='full' else r['pruned_prediction']
        if not math.isfinite(pred) or not math.isclose((pred-r['target'])**2,r['squared_error'],abs_tol=1e-9,rel_tol=1e-6):
            raise ValueError('incorrect prediction/squared error')
        if r['method']=='full':
            if r['final_pair_count']!=p:raise ValueError('full reference pruned')
            continue
        indices=r['selected_pairs']
        if (any(type(i) is not int or i<0 or i>=p for i in indices)
                or len(set(indices))!=len(indices)):
            raise ValueError('invalid selected pair indices')
        selected=torch.zeros(p,dtype=torch.bool)
        selected[indices]=True
        real=torch.tensor(r['stored_real'],dtype=torch.bool)
        pairs=torch.tensor(r['layout_pairs'],dtype=torch.long).reshape(-1,2)
        n=int(pairs.max())+1 if len(pairs) else r['available_loop_columns']
        ly=PhysicalPairLayout(pairs,torch.tensor(r['layout_inverse']),real,n)
        ly.validate_selection(selected)
        mask=torch.tensor(r['stored_mask'],dtype=torch.bool)
        if not torch.equal(mask,ly.expand(selected)):raise ValueError('mask copies/loops differ from pairs')
        diag=selection_diagnostics(ly,selected,k=k,constraint='no_isolates')
        if not diag['budget_satisfied'] or not diag['constraint_satisfied'] or r['final_pair_count']!=k:
            raise ValueError('budget or physical constraint violated')
        if r.get('budget_satisfied') is not True or r.get('constraint_satisfied') is not True:
            raise ValueError('stored diagnostics invalid')
        mask_count+=1
    return dict(status='passed',rows=len(rows),test_halos=len(splits['test']),
                methods=methods,retentions=retentions,verified_pruned_masks=mask_count,
                split_sizes={k:len(v) for k,v in splits.items()},
                backbone_state_sha256=model_hash,
                full_reference_cross_budget_max_spread=max(max(v)-min(v) for v in all_references.values()),
                available_pair_range=[min(r['available_pair_count'] for r in rows),max(r['available_pair_count'] for r in rows)],
                note='ID separation verifies selector splits; original backbone training split is not certified by this legacy checkpoint.')
