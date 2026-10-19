"""Budget-safe inference diagnostics and immutable experiment contracts."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from data.provenance import hash_payload, sha256_file
from rls.constraints import PhysicalPairLayout, pair_budget, selection_diagnostics
from rls.constrained_policy import select_pairs
from rls.notebook_benchmark import fixed_marks
from rls.pair_policy import repair_pairs

BASE_COMMIT = 'c825bf57652005c80faff79cf4dfd58e3a9baa14'
VARIANTS = ('full', 'prediction_only', 'prediction_physics', 'no_virial',
            'no_sparsity', 'no_connectivity', 'no_entropy', 'no_value',
            'virial_0.1', 'virial_0.3', 'virial_3.0')


def variant_config(base, name):
    cfg = copy.deepcopy(base)
    if name not in VARIANTS:
        raise ValueError(f'Unknown trained ablation {name}')
    if name in ('prediction_only', 'prediction_physics'):
        cfg.update(w_sp=0., w_conn=0., w_virial=0. if name == 'prediction_only' else 1.)
    remove = {'no_virial':'w_virial', 'no_sparsity':'w_sp',
              'no_connectivity':'w_conn', 'no_entropy':'entropy_coef', 'no_value':'value_coef'}
    if name in remove:
        cfg[remove[name]] = 0.
    if name.startswith('virial_'):
        cfg['w_virial'] = float(name.split('_')[1])
    return cfg


def decode(graph, scores, q, mode, seed, *, exact_k=None):
    layout = PhysicalPairLayout.from_edge_index(graph['edge_index'], len(graph['x']))
    requested = pair_budget(layout, q)
    k = requested if exact_k is None else exact_k
    marks = fixed_marks(layout, seed, graph['cluster_id'])
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    if mode not in ('constrained_rank', 'constrained_sample', 'probability_rank',
                    'unconstrained_rank', 'rank_repair'):
        raise ValueError(f'Unknown inference mode {mode}')
    constraint = 'none' if mode in ('unconstrained_rank', 'rank_repair') else 'no_isolates'
    selected = select_pairs(layout, scores.sigmoid() if mode == 'probability_rank' else scores,
        k, constraint, sample=mode == 'constrained_sample', pair_marks=marks, generator=generator).selected
    if mode == 'rank_repair':
        selected = repair_pairs(layout.pairs, selected, layout.num_nodes, pair_marks=marks)
    diag = selection_diagnostics(layout, selected, k=k, constraint='no_isolates')
    diag.update(nominal_pair_count=requested, repair_added_pair_count=int(selected.sum())-k
                if mode == 'rank_repair' else 0)
    return dict(graph_id=graph['cluster_id'], mask=layout.expand(selected),
                selected=selected, diagnostics=diag, mode=mode)


def require_matched(left, right):
    if not left or [r['graph_id'] for r in left] != [r['graph_id'] for r in right]:
        raise ValueError('Comparison population differs or is empty')
    if any(not r['diagnostics']['budget_satisfied'] for r in left+right):
        raise ValueError('Comparison budget was violated')
    if [r['diagnostics']['final_pair_count'] for r in left] != [r['diagnostics']['final_pair_count'] for r in right]:
        raise ValueError('Comparison budget differs per halo')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def start_run(folder, contract, resume=False):
    folder = Path(folder)
    if folder.exists():
        if not resume:
            raise FileExistsError('Use a new output directory or explicit --resume')
        manifest = json.loads((folder/'run.json').read_text())
        if manifest['contract'] != contract:
            raise ValueError('Resume contract changed; use a new output directory')
        if manifest['status'] == 'complete':
            for name, checksum in manifest['artifacts'].items():
                if not (folder/name).is_file() or sha256_file(folder/name)['sha256'] != checksum:
                    raise ValueError(f'Completed artifact changed: {name}')
        return manifest['status']
    folder.mkdir(parents=True)
    write_json(folder/'run.json', dict(status='running', contract=contract,
        started_utc=datetime.now(timezone.utc).isoformat(),contract_sha256=hash_payload(contract)))
    return 'running'


def finish_run(folder, artifacts):
    folder = Path(folder)
    manifest = json.loads((folder/'run.json').read_text())
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts={name:sha256_file(folder/name)['sha256'] for name in artifacts})
    write_json(folder/'run.json', manifest)
