"""Exploratory graph-paired analysis; never edits source experiment artifacts.

Sampling unit: one halo graph (82), conditional on one fitted policy, backbone,
catalog and split. This does not estimate training-seed or volume uncertainty.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import sys
import warnings

import numpy as np
import scipy
from scipy import stats

ROOT = Path('/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015')
SOURCE = ROOT / 'C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C'
OUT = ROOT / 'scientific-analysis-20261016'
SEED = 20261016
BOOTSTRAPS = 50000
PERMUTATIONS = 99999


def mean_difference(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def rmse_difference(a, b, axis=-1):
    return np.sqrt(np.mean(a*a, axis=axis)) - np.sqrt(np.mean(b*b, axis=axis))


def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def align(left, right):
    def keyed(rows):
        result = {r['graph_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate graph observations; do not pool repeated budgets/methods')
        return result
    a, b = keyed(left), keyed(right)
    if a.keys() != b.keys():
        raise ValueError('Comparisons must use identical graph IDs, without silent intersections')
    ids = sorted(a)
    y = np.array([a[k]['target'] for k in ids], float)
    assert np.array_equal(y, np.array([b[k]['target'] for k in ids], float))
    def errors(rows):
        return np.array([rows[k]['full_prediction'] if rows[k]['method'] == 'full'
                         else rows[k]['pruned_prediction'] for k in ids]) - y
    ea, eb = errors(a), errors(b)
    assert np.isfinite(ea).all() and np.isfinite(eb).all()
    return ids, y, ea, eb


def self_tests():
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    exact = stats.permutation_test((np.ones(6), np.zeros(6)), mean_difference,
        permutation_type='samples', n_resamples=np.inf, alternative='two-sided', vectorized=True)
    assert exact.pvalue == 2/64  # paired sign allocations, not independent pooling
    x = [{'graph_id': str(i), 'target': 0., 'method': 'provided_rl',
          'pruned_prediction': float(i)} for i in range(3)]
    assert np.array_equal(align(x, list(reversed(x)))[2], align(x, x)[2])
    try:
        align(x, x[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing-ID pairing was accepted')
    assert rmse_difference(np.ones(8), np.ones(8)) == 0
    print('Statistical self-checks passed: pairing, exact paired permutations, Holm adjustment.')


def main():
    self_tests()
    OUT.mkdir(exist_ok=False)
    raw_path = SOURCE / 'matched_per_example.json'
    raw = json.loads(raw_path.read_text())
    strata = defaultdict(list)
    hashes = defaultdict(set)
    for row in raw:
        assert row['status'] == 'success', 'Retain and investigate evaluation failures before inference'
        strata[(row['requested_keep_fraction'], row['method'])].append(row)
        hashes[row['graph_id']].add(row['graph_content_hash'])
    assert all(len(v) == 1 for v in hashes.values()), 'Methods/budgets changed their input graphs'
    assert len(hashes) == 82 and len(raw) == 2624
    methods = sorted({r['method'] for r in raw} - {'provided_rl'})
    contrasts = []
    for q in (.4, .6, .8):
        for method in methods:
            ids, y, er, ec = align(strata[(q, 'provided_rl')], strata[(q, method)])
            assert len(ids) == 82
            idx = len(contrasts)
            with warnings.catch_warnings(record=True) as caught:
                boot = stats.bootstrap((er, ec), rmse_difference, paired=True,
                    vectorized=True, axis=-1, n_resamples=BOOTSTRAPS, batch=2500,
                    confidence_level=.95, method='BCa', rng=np.random.default_rng(SEED+idx*2))
            assert np.isfinite([boot.confidence_interval.low, boot.confidence_interval.high]).all()
            perm = stats.permutation_test((er*er, ec*ec), mean_difference,
                permutation_type='samples', vectorized=True, axis=-1,
                n_resamples=PERMUTATIONS, batch=2500, alternative='two-sided',
                rng=np.random.default_rng(SEED+idx*2+1))
            rmse_rl, rmse_control = np.sqrt(np.mean(er*er)), np.sqrt(np.mean(ec*ec))
            loo = [float(rmse_difference(np.delete(er, i), np.delete(ec, i))) for i in range(len(er))]
            contrasts.append(dict(requested_keep_fraction=q, control=method, n_graphs=len(ids),
                rmse_rl=float(rmse_rl), rmse_control=float(rmse_control),
                delta_rmse_rl_minus_control=float(rmse_rl-rmse_control),
                relative_rmse_improvement_pct=float(100*(1-rmse_rl/rmse_control)),
                ci95_bca_low=float(boot.confidence_interval.low), ci95_bca_high=float(boot.confidence_interval.high),
                bootstrap_standard_error=float(boot.standard_error),
                paired_permutation_mse_p_two_sided=float(perm.pvalue),
                mse_difference=float(np.mean(er*er-ec*ec)),
                mae_rl=float(np.mean(np.abs(er))), mae_control=float(np.mean(np.abs(ec))),
                n_abs_error_better=int((np.abs(er)<np.abs(ec)).sum()),
                n_abs_error_worse=int((np.abs(er)>np.abs(ec)).sum()),
                n_abs_error_equal=int((np.abs(er)==np.abs(ec)).sum()),
                leave_one_graph_out_delta_min=min(loo), leave_one_graph_out_delta_max=max(loo),
                bootstrap_warnings=[type(w.message).__name__ for w in caught]))
    adjusted = holm([r['paired_permutation_mse_p_two_sided'] for r in contrasts])
    for r, p in zip(contrasts, adjusted):
        r['holm_p_all_21_pruned_budget_contrasts'] = float(p)
        r['holm_p_q40_family7'] = None
    q40 = [r for r in contrasts if r['requested_keep_fraction'] == .4]
    for r, p in zip(q40, holm([r['paired_permutation_mse_p_two_sided'] for r in q40])):
        r['holm_p_q40_family7'] = float(p)
    report = dict(analysis='exploratory post hoc graph-paired comparison',
        input_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__, scipy=scipy.__version__, rng_seed=SEED,
        bootstrap=dict(method='BCa', paired=True, resamples=BOOTSTRAPS, confidence=.95, unit='halo graph'),
        permutation=dict(statistic='mean squared error RL minus control', type='samples (within-halo swaps)',
            alternative='two-sided', resamples=PERMUTATIONS),
        multiplicity=['Holm across seven q=.4 controls', 'Holm across all 21 q=.4/.6/.8 control contrasts'],
        limitations=['One fitted policy, backbone, catalog and split; no training-seed uncertainty.',
            'Halo bootstrap assumes independent/exchangeable halos; no simulation-volume replication.',
            'Test outcomes and budgets were already inspected; these are exploratory, not preregistered tests.',
            'Only one random test-mask realization; ten random validation seeds do not supply ten test replicates.',
            'Legacy target/radius and backbone training provenance remain uncertified.'],
        contrasts=contrasts)
    with (OUT/'paired-comparisons.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    with (OUT/'paired-comparisons.csv').open('x', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(contrasts[0])); writer.writeheader(); writer.writerows(contrasts)
    for r in q40:
        print(json.dumps({k:r[k] for k in ('control','delta_rmse_rl_minus_control','ci95_bca_low',
            'ci95_bca_high','paired_permutation_mse_p_two_sided','holm_p_q40_family7',
            'holm_p_all_21_pruned_budget_contrasts','n_abs_error_better','n_abs_error_worse')}))
    print('Saved new analysis only:', OUT)


if __name__ == '__main__':
    main()
