"""Verify manuscript sources and optionally reproduce all 21 paired contrasts."""
from pathlib import Path
import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import numpy as np

P = Path(__file__).resolve().parents[1]
manifest = json.loads((P/'evidence/source-manifest.json').read_text())
cache = {}
for record in manifest['files']:
    path = P.parent / record['snapshot']
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['snapshot_sha256'], path
    if path.suffix == '.gz': data = gzip.decompress(data)
    assert len(data) == record['size_uncompressed']
    assert hashlib.sha256(data).hexdigest() == record['sha256_uncompressed']
    cache[record['campaign_relative_path']] = data

def source(suffix):
    values = [v for k,v in cache.items() if k.endswith(suffix)]
    assert len(values)==1, suffix
    return values[0]

def rows(suffix):
    return list(csv.DictReader(source(suffix).decode().splitlines()))

parser=argparse.ArgumentParser()
parser.add_argument('--statistics',action='store_true')
args=parser.parse_args()
raw=json.loads(source('notebook_C/matched_per_example.json'))
assert len(raw)==2624 and all(r['status']=='success' for r in raw)
ids={r['graph_id'] for r in raw}
assert len(ids)==82
strata={}
for r in raw:
    key=(r['requested_keep_fraction'],r['method'])
    strata.setdefault(key,{})
    assert r['graph_id'] not in strata[key]
    strata[key][r['graph_id']]=r
assert len(strata)==32 and all(set(v)==ids for v in strata.values())
for r in rows('notebook_C/results_table.csv'):
    records=strata[float(r['frac']),r['method']]
    y=np.array([records[i]['target'] for i in sorted(ids)])
    pred=np.array([records[i]['full_prediction'] if r['method']=='full' else records[i]['pruned_prediction'] for i in sorted(ids)])
    err=pred-y
    expected=[np.sqrt(np.mean(err**2)),np.mean(np.abs(err)),1-np.sum(err**2)/np.sum((y-y.mean())**2)]
    assert np.allclose(expected,[float(r[k]) for k in ['rmse','mae','r2']],atol=1e-10,rtol=0)

d=rows('notebook_D/tta_results.csv')
assert [int(r['K']) for r in d]==[0,5,10,20]
for r in d:
    assert len(ast.literal_eval(r['graph_ids']))==82
    assert ast.literal_eval(r['graph_ids'])==ast.literal_eval(d[0]['graph_ids'])
    assert ast.literal_eval(r['final_pair_counts'])==ast.literal_eval(d[0]['final_pair_counts'])
    assert float(r['requested_keep_fraction'])==.5 and float(r['budget_excess_pair_count'])==0
    assert r['decoder_kind']=='exact_direct'
dp=json.loads(source('notebook_D/provenance.json'))['D']
assert dp['selected_K']==20
assert json.loads(source('analysis-ABCD-official-20261016-001.json'))['valid'] is True

tex='\n'.join(f.read_text() for f in [P/'main.tex',*sorted((P/'sections').glob('*.tex'))])
bib=(P/'references.bib').read_text()
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
citations={k.strip() for c in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in c.split(',')}
assert citations==keys,(citations-keys,keys-citations)
for relative in re.findall(r'\\input\{([^}]+)\}',tex):
    assert (P/(relative+'.tex')).is_file(), relative
for relative in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex):
    assert (P/relative).is_file(),relative
labels=re.findall(r'\\label\{([^}]+)\}',tex)
assert len(labels)==len(set(labels))
refs=set(re.findall(r'\\ref\{([^}]+)\}',tex))
assert refs.issubset(set(labels)),refs-set(labels)
assert not re.search(r'\b(TODO|TBD|FIXME)\b',tex)

result={'source_snapshots_sha256_verified':len(cache),'C_configurations_metrics_recomputed':32,
        'C_distinct_halos':82,'C_rows':len(raw),'D_same_budget_ids_all_K':True,
        'D_selected_K':20,'bib_entries_and_citation_keys_verified':len(keys),
        'tex_inputs_figures_labels_verified':True,'statistics_recomputed':False}
if args.statistics:
    import scipy
    from scipy import stats
    paired=json.loads(source('scientific-analysis-20261016/paired-comparisons.json'))
    def rmse_difference(a,b,axis=-1):
        return np.sqrt(np.mean(a*a,axis=axis))-np.sqrt(np.mean(b*b,axis=axis))
    def mean_difference(a,b,axis=-1):
        return np.mean(a,axis=axis)-np.mean(b,axis=axis)
    exact=stats.permutation_test((np.ones(6),np.zeros(6)),mean_difference,
        permutation_type='samples',n_resamples=np.inf,alternative='two-sided',vectorized=True)
    assert exact.pvalue==2/64
    pvalues=[]
    for index,c in enumerate(paired['contrasts']):
        q,control=c['requested_keep_fraction'],c['control']
        a,b=strata[q,'provided_rl'],strata[q,control]
        assert set(a)==set(b)==ids
        y=np.array([a[i]['target'] for i in sorted(ids)])
        assert np.array_equal(y,np.array([b[i]['target'] for i in sorted(ids)]))
        def errors(records,method):
            return np.array([records[i]['full_prediction'] if method=='full' else records[i]['pruned_prediction'] for i in sorted(ids)])-y
        ea,eb=errors(a,'provided_rl'),errors(b,control)
        bootstrap=stats.bootstrap((ea,eb),rmse_difference,paired=True,vectorized=True,axis=-1,
            n_resamples=50000,batch=2500,confidence_level=.95,method='BCa',rng=np.random.default_rng(20261016+index*2))
        permutation=stats.permutation_test((ea*ea,eb*eb),mean_difference,permutation_type='samples',
            vectorized=True,axis=-1,n_resamples=99999,batch=2500,alternative='two-sided',
            rng=np.random.default_rng(20261016+index*2+1))
        found=[rmse_difference(ea,eb),bootstrap.confidence_interval.low,bootstrap.confidence_interval.high,permutation.pvalue]
        expected=[c[k] for k in ['delta_rmse_rl_minus_control','ci95_bca_low','ci95_bca_high','paired_permutation_mse_p_two_sided']]
        np.testing.assert_allclose(found,expected,rtol=0,atol=1e-10)
        pvalues.append(permutation.pvalue)
    def holm(p):
        p=np.array(p);order=np.argsort(p,kind='stable');adjusted=np.empty_like(p)
        adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
        return adjusted
    np.testing.assert_allclose(holm(pvalues),[c['holm_p_all_21_pruned_budget_contrasts'] for c in paired['contrasts']],atol=1e-12)
    mask=[i for i,c in enumerate(paired['contrasts']) if c['requested_keep_fraction']==.4]
    np.testing.assert_allclose(holm([pvalues[i] for i in mask]),[paired['contrasts'][i]['holm_p_q40_family7'] for i in mask],atol=1e-12)
    result.update(statistics_recomputed=True,paired_contrasts_verified=21,
                  bootstrap_draws_per_contrast=50000,permutations_per_contrast=99999,
                  exact_small_sample_permutation_self_check=True,
                  holm_families_verified=[7,21],numpy=np.__version__,scipy=scipy.__version__)
(P/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
