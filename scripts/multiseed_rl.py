"""Repeat the canonical matched-budget experiment with explicit seed artifacts.

Each seed gets a fresh output directory containing policy weights, provenance,
and raw per-graph matched rows. The aggregate uses the common-success metrics
from those artifacts; undefined metrics stay null. A random backbone requires
--allow-random-backbone and every result remains labelled as smoke evidence.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _summaries(rows, *, seed=42, n_bootstrap=1000):
    import numpy as np
    rng = np.random.default_rng(seed)
    methods = sorted({row['method'] for row in rows})
    result = []
    for method in methods:
        own = [row for row in rows if row['method'] == method]
        rec = {'method': method, 'n_seeds': len(own),
               'population': 'per_seed_common_success',
               'backbone_stages': sorted({r['backbone_stage'] for r in own}),
               'backbone_shas': sorted({r['backbone_sha256'] for r in own})}
        for key in ('rmse', 'mae', 'r2', 'mean_keep_frac'):
            values = np.array([r[key] for r in own if r.get(key) is not None
                               and math.isfinite(float(r[key]))], dtype=float)
            rec[key + '_n_seeds'] = len(values)
            rec[key + '_reason'] = None if len(values) else 'no_defined_seed_metrics'
            rec[key + '_mean'] = float(values.mean()) if len(values) else None
            rec[key + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
            if len(values) > 1:
                draws = rng.choice(values, size=(n_bootstrap, len(values))).mean(axis=1)
                rec[key + '_ci95_lo'] = float(np.percentile(draws, 2.5))
                rec[key + '_ci95_hi'] = float(np.percentile(draws, 97.5))
            else:
                rec[key + '_ci95_lo'] = rec[key + '_ci95_hi'] = None
        result.append(rec)
    return result


def main(argv=None):
    import yaml
    from data.provenance import sha256_file
    from rls.research_workflow import inspect_backbone_checkpoint
    from rls.run_experiment import main as run_experiment
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--allow-random-backbone', action='store_true')
    ap.add_argument('--supervised-artifact', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--max-graphs', type=int, default=None)
    ap.add_argument('--rls-override', nargs='*', default=[], help='KEY=VALUE overrides for cfg[rls]')
    ap.add_argument('--out', default='outputs/rls/multiseed.json')
    ap.add_argument('--runs-dir', default=None)
    args = ap.parse_args(argv)
    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)
    cfg.setdefault('rls', {})
    for setting in args.rls_override:
        if '=' not in setting:
            raise ValueError('rls overrides require KEY=VALUE')
        key, value = setting.split('=', 1)
        cfg['rls'][key] = yaml.safe_load(value)
    if args.epochs is not None:
        cfg['rls']['epochs'] = args.epochs
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('seeds must be unique; repeated seeds are not independent replicates')
    out = Path(args.out)
    runs_root = Path(args.runs_dir) if args.runs_dir else out.with_name(out.stem + '_runs')
    if out.exists():
        raise FileExistsError(f'refusing to overwrite multiseed report: {out}')
    if runs_root.exists():
        raise FileExistsError(f'multiseed requires a fresh runs directory: {runs_root}')
    checkpoint = args.checkpoint
    if not (checkpoint is None and args.allow_random_backbone):
        checkpoint = checkpoint or os.path.join(cfg['training']['checkpoint_dir'], 'best_model.pt')
        try:
            inspect_backbone_checkpoint(checkpoint, cfg)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Required checkpoint does not exist: {checkpoint}') from exc
    # Model/policy replicate seeds must not silently change the data split.
    cfg.setdefault('data', {}).setdefault('split_seed', int(cfg.get('seed', 42)))
    per_seed, runs = [], []
    for seed in args.seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg['seed'] = seed
        seed_dir = runs_root / f'seed_{seed}'
        run_experiment(run_cfg, checkpoint=checkpoint, output_dir=seed_dir,
            allow_random_backbone=args.allow_random_backbone,
            max_graphs=args.max_graphs, supervised_artifact=args.supervised_artifact)
        provenance = json.loads((seed_dir / 'provenance.json').read_text())
        matched = json.loads((seed_dir / 'matched_summary.json').read_text())
        raw = json.loads((seed_dir / 'matched_rows.json').read_text())
        policy_path = seed_dir / 'policy.pt'
        backbone = provenance['backbone_frozen']
        runs.append({'seed': seed, 'output_dir': str(seed_dir.resolve()),
                     'provenance': sha256_file(seed_dir / 'provenance.json'),
                     'matched_rows': sha256_file(seed_dir / 'matched_rows.json'),
                     'matched_summary': sha256_file(seed_dir / 'matched_summary.json')})
        for summary in matched:
            if summary.get('backbone_stage') != 'frozen':
                continue
            common_ids = set(summary.get('common_success_ids', []))
            population = [r for r in raw if r.get('backbone_stage') == 'frozen'
                and r['method'] == summary['method'] and r['status'] == 'success'
                and r['graph_id'] in common_ids and r.get('available_pair_count')]
            keeps = [r['final_pair_count'] / r['available_pair_count'] for r in population]
            per_seed.append({**summary, 'seed': seed,
                'population': 'common_success',
                'rmse': summary.get('common_rmse'), 'mae': summary.get('common_mae'),
                'r2': summary.get('common_r2'),
                'mean_keep_frac': sum(keeps) / len(keeps) if keeps else None,
                'backbone_stage': backbone['stage'], 'backbone_sha256': backbone['sha256'],
                'policy_sha256': sha256_file(policy_path)['sha256'],
                'policy_md5': _md5(policy_path), 'run_mode': provenance['run_mode'],
                'research_label': provenance['research_label'],
                'output_dir': str(seed_dir.resolve())})
        print(f'[seed {seed}] wrote matched evaluation -> {seed_dir}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as stream:
        json.dump({'per_seed': per_seed, 'summary': _summaries(per_seed),
                   'runs': runs, 'split_seed': cfg['data']['split_seed'],
                   'sparsity_mode': cfg['rls'].get('sparsity_mode', 'penalty')},
                  stream, indent=2, allow_nan=False)
    print(f'WROTE {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
