"""Verify C exports using selected-pair INDEX semantics (matched_benchmark:915)."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np


def selection(indices, count):
    ids=np.asarray(indices,dtype=int)
    if len(ids)!=len(set(ids.tolist())) or (ids<0).any() or (ids>=count).any():
        raise ValueError('Invalid or duplicate physical pair indices')
    result=np.zeros(count,dtype=bool);result[ids]=True
    return result


np.testing.assert_array_equal(selection([0,2],4),[True,False,True,False])
for bad in ([1,1],[-1],[4]):
    try:selection(bad,4)
    except ValueError:pass
    else:raise AssertionError('Invalid selection accepted')

ROOT=Path('/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015')
p=ROOT/'C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C/matched_per_example.json'
raw=json.loads(p.read_text());checked=0
for row in raw:
    if row['method']=='full':continue
    pairs=np.asarray(row['layout_pairs'],dtype=int)
    selected=selection(row['selected_pairs'],len(pairs))
    inverse=np.asarray(row['layout_inverse'],dtype=int)
    real=np.asarray(row['stored_real'],dtype=bool)
    mask=np.asarray(row['stored_mask'],dtype=bool)
    assert len(pairs)==row['available_pair_count']
    assert len(mask)==len(real) and len(inverse)==int(real.sum())
    assert np.array_equal(mask[real],selected[inverse]) and mask[~real].all()
    assert int(selected.sum())==row['requested_pair_count']==row['final_pair_count']
    assert row['final_pair_count']==math.ceil(row['requested_keep_fraction']*len(pairs))
    assert set(pairs[selected].reshape(-1))==set(pairs.reshape(-1))
    assert row['physical_isolates']==0 and row['constraint_satisfied'] and row['budget_satisfied']
    assert (np.bincount(inverse,minlength=len(pairs))==2).all()
    checked+=1
assert checked==2296
out=ROOT/'scientific-analysis-20261016/mask-verification.json'
with out.open('x') as f:
    json.dump({'C_nonfull_masks_checked':checked,'selected_pairs_serialization':'indices, not boolean vector',
        'actual_pair_expansion_loop_preservation_exact_counts_and_node_coverage':True,
        'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},f,indent=2)
print('Verified',checked,'saved C masks: exact counts, paired copies, loops and no physical isolates.')
