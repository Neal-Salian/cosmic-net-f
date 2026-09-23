"""Matched-budget benchmark evaluator (Task 3b framework).

Compares scorer methods under one identical physical-pair budget per graph:
every scorer sees the same final k/constraint/layout/transported pair marks,
decoded by the shared exact decoder. A shared full-graph reference is
computed once per graph/model and its cost is reported once, never
multiplied across methods.

Reuses Task1 constraints/selection, Task2 data.provenance helpers, the
metered predictor, and duck-typed PairScorer objects (the parallel scorer
worker's ``rls.benchmark_scorers.PairScorer`` satisfies the same protocol:
``.name``, ``.score(graph, layout, k, *, meter, random_scores, generator)``,
``.training_cost()``). No duplicated matching/split/hash logic; no CLI.
"""

import copy
import hashlib
import math
import numbers
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from data.provenance import (
    hash_payload,
    model_state_hash,
    sha256_file,
    validate_run_provenance,
)
from rls.benchmark_predictor import MeteredPredictor
from rls.constrained_policy import select_pairs
from rls.constraints import (
    PhysicalPairLayout,
    feasible_budget,
    pair_budget,
    pair_marks,
    selection_diagnostics,
)
from rls.structured_selection import select_structured

VALID_CONSTRAINTS = ('none', 'no_isolates', 'connected')
VALID_DECODERS = ('direct', 'scaffold', 'enumerated_structured_map')


# ---------------- budget ----------------

@dataclass(frozen=True)
class BudgetSpec:
    """Exactly one of pair_count / keep_fraction, plus constraint/decoder."""
    pair_count: Optional[int] = None
    keep_fraction: Optional[float] = None
    constraint: str = 'no_isolates'
    decoder: str = 'direct'

    def __post_init__(self):
        pc, q = self.pair_count, self.keep_fraction
        if (pc is None) == (q is None):
            raise ValueError('BudgetSpec accepts exactly one budget: '
                             'pair_count xor keep_fraction')
        if pc is not None:
            if isinstance(pc, bool) or not isinstance(pc, numbers.Integral):
                raise ValueError(f'pair_count must be an integer, got {pc!r}')
            if int(pc) < 0:
                raise ValueError(f'pair_count must be nonnegative, got {pc!r}')
            object.__setattr__(self, 'pair_count', int(pc))
        if q is not None:
            if isinstance(q, bool) or not isinstance(q, numbers.Real):
                raise ValueError(f'keep_fraction must be a real number, got {q!r}')
            qf = float(q)
            if not math.isfinite(qf) or not 0.0 <= qf <= 1.0:
                raise ValueError(f'keep_fraction must be finite in [0, 1], got {q!r}')
            object.__setattr__(self, 'keep_fraction', qf)
        if self.constraint not in VALID_CONSTRAINTS:
            raise ValueError(f'constraint must be one of {VALID_CONSTRAINTS}, '
                             f'got {self.constraint!r}')
        if self.decoder not in VALID_DECODERS:
            raise ValueError(f'decoder must be one of {VALID_DECODERS}, '
                             f'got {self.decoder!r}')

    def config_dict(self):
        return {'pair_count': self.pair_count,
                'keep_fraction': self.keep_fraction,
                'constraint': self.constraint, 'decoder': self.decoder}


# ---------------- stable hashing / RNG ----------------

def _stable_int(*parts: str) -> int:
    digest = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def _sync_cuda(meter=None, graph=None):
    if not torch.cuda.is_available():
        return
    devices = set()
    try:
        model = getattr(meter, '_model', None) if meter is not None else None
        if model is not None:
            for param in model.parameters():
                if isinstance(param, torch.Tensor) and param.is_cuda:
                    devices.add(param.device)
            for buf in model.buffers():
                if isinstance(buf, torch.Tensor) and buf.is_cuda:
                    devices.add(buf.device)
    except Exception:
        pass
    try:
        if graph is not None:
            for key in ('x', 'edge_index', 'edge_attr', 'pos'):
                tensor = getattr(graph, key, None)
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    devices.add(tensor.device)
    except Exception:
        pass
    try:
        if not devices:
            torch.cuda.synchronize()
        else:
            for device in devices:
                torch.cuda.synchronize(device)
    except Exception:
        pass


def _snapshot_delta(before, after):
    out = {}
    for phase in after:
        out[phase] = {k: after[phase][k] - before[phase][k]
                      for k in ('forward_calls', 'graph_evaluations')}
        out[phase]['elapsed_seconds'] = (
            after[phase]['elapsed_seconds'] - before[phase]['elapsed_seconds'])
    return out


def _sum_calls(delta):
    return sum(v['forward_calls'] for v in delta.values())


def _graph_content_hash(graph, cluster_id: str) -> str:
    payload = {'cluster_id': cluster_id}
    for key in ('x', 'edge_index', 'edge_attr', 'y', 'pos', 'stellar_mass'):
        tensor = getattr(graph, key, None)
        if isinstance(tensor, torch.Tensor):
            payload[key] = tensor.detach().cpu()
    lineage = getattr(graph, 'lineage_group', None)
    payload['lineage_group'] = lineage if isinstance(
        lineage, str) else (None if lineage is None else str(lineage))
    return model_state_hash(payload)


def _dataset_content_hash(content_by_id: Dict[str, str]) -> str:
    return hash_payload([{'graph_id': gid, 'content_hash': content_by_id[gid]}
                         for gid in sorted(content_by_id)])


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() <= 4096:
            return _json_safe(value.detach().cpu().tolist())
        return {'tensor_shape': list(value.shape),
                'tensor_dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _retention(kept, total: int):
    if total <= 0:
        return None, 'zero_denominator'
    return kept / total, None


def _finite_scalar(value, name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f'{name} must be a single-value tensor')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return float(value.detach().cpu().reshape(()).item())


# ---------------- metrics ----------------

def _rmse(preds: List[float], targets: List[float]) -> float:
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets))
                     / len(preds))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _pearson(xs: List[float], ys: List[float]):
    if len(xs) < 2:
        return None, 'n_lt_2'
    mx, my = _mean(xs), _mean(ys)
    vx = sum((v - mx) ** 2 for v in xs)
    vy = sum((v - my) ** 2 for v in ys)
    if vx == 0.0 or vy == 0.0:
        return None, 'zero_variance'
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    return cov / math.sqrt(vx * vy), None


def _regression_metrics(preds: List[float], targets: List[float]) -> Dict[str, Any]:
    n = len(preds)
    if n == 0:
        return {'rmse': None, 'rmse_reason': 'no_successes', 'r2': None,
                'r2_reason': 'no_successes', 'bias': None,
                'bias_reason': 'no_successes', 'std_residual': None,
                'std_reason': 'no_successes'}
    rmse = _rmse(preds, targets)
    bias = _mean([p - t for p, t in zip(preds, targets)])
    out = {'rmse': rmse, 'rmse_reason': None, 'bias': bias, 'bias_reason': None,
           'std_residual': None, 'std_reason': None,
           'r2': None, 'r2_reason': None}
    if n < 2:
        out['std_reason'] = 'n_lt_2'
        out['r2_reason'] = 'n_lt_2'
        return out
    resid = [p - t for p, t in zip(preds, targets)]
    mean_resid = _mean(resid)
    out['std_residual'] = math.sqrt(sum((r - mean_resid) ** 2 for r in resid) / n)
    mean_t = _mean(targets)
    sstot = sum((t - mean_t) ** 2 for t in targets)
    if sstot == 0.0:
        out['r2_reason'] = 'zero_target_variance'
        return out
    ssres = sum((p - t) ** 2 for p, t in zip(preds, targets))
    out['r2'] = 1.0 - ssres / sstot
    return out


# ---------------- validation ----------------

def _validate_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f'seed must be an integer, got {seed!r}')
    return int(seed)


def _validate_backbone(backbone) -> str:
    if not isinstance(backbone, str) or not backbone.strip():
        raise ValueError('backbone must be an explicit nonempty stage string')
    return backbone


def _validate_provenance(provenance) -> Dict[str, Any]:
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('provenance must be a nonempty mapping')
    if provenance.get('schema_version') != 1:
        raise ValueError('provenance schema_version must be 1 '
                         '(build_run_provenance schema)')
    label = provenance.get('research_label')
    if not isinstance(label, str) or not label.strip():
        raise ValueError('provenance must carry an explicit research_label; '
                         'refusing to manufacture one')
    try:
        validate_run_provenance(provenance)
    except ValueError as exc:
        raise ValueError(f'provenance validation failed: {exc}')
    return copy.deepcopy(provenance)


def _preflight(predictor, graphs, scorers, seed, backbone, provenance,
               marks_by_id, scores_by_id):
    if not isinstance(predictor, nn.Module):
        raise ValueError('predictor must be an actual nn.Module')
    if not isinstance(graphs, (list, tuple)) or not graphs:
        raise ValueError('graphs must be a nonempty list')
    if not isinstance(scorers, (list, tuple)) or not scorers:
        raise ValueError('scorers must be a nonempty list')
    names = []
    for scorer in scorers:
        name = getattr(scorer, 'name', None)
        if not isinstance(name, str) or not name:
            raise ValueError('every scorer must expose a nonempty .name')
        if not callable(getattr(scorer, 'score', None)):
            raise ValueError(f'scorer {name!r} must expose .score()')
        if not callable(getattr(scorer, 'training_cost', None)):
            raise ValueError(f'scorer {name!r} must expose .training_cost()')
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f'duplicate scorer names: {names}')
    seen_ids = set()
    for graph in graphs:
        if not isinstance(graph, Data):
            raise ValueError('graphs must be single PyG Data objects')
        cid = getattr(graph, 'cluster_id', None)
        if not isinstance(cid, str) or not cid:
            raise ValueError('every graph must carry a stable nonempty '
                             'string cluster_id')
        if cid in seen_ids:
            raise ValueError(f'duplicate graph IDs: {cid!r}')
        seen_ids.add(cid)
    for label, mapping in (('pair_marks_by_id', marks_by_id),
                           ('random_scores_by_id', scores_by_id)):
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            raise ValueError(f'{label} must be a mapping keyed by graph ID')
        unknown = set(mapping) - seen_ids
        if unknown:
            raise ValueError(f'{label} has unknown graph IDs: {sorted(unknown)}')


def _lineage_group(graph) -> Optional[str]:
    value = getattr(graph, 'lineage_group', None)
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


# ---------------- row builders ----------------

def _base_row(*, graph_id, method, seed, budget, comparison_axis, backbone,
               model_hash, provenance, budget_hash, evaluator_hash,
               content_hash, lineage):
    return {
        'graph_id': graph_id, 'method': method, 'seed': seed,
        'constraint': budget.constraint, 'decoder': budget.decoder,
        'comparison_axis': comparison_axis,
        'requested_keep_fraction': budget.keep_fraction,
        'requested_pair_count_config': budget.pair_count,
        'backbone_stage': backbone, 'backbone': backbone,
        'model_state_hash': model_hash,
        'research_label': provenance.get('research_label'),
        'provenance': copy.deepcopy(provenance),
        'catalog_contract_sha256': provenance.get('catalog_contract_sha256'),
        'split_manifest_sha256': provenance.get('split_manifest_sha256'),
        'config_sha256': provenance.get('config_sha256'),
        'source_revision': provenance.get('source_revision'),
        'budget_config_hash': budget_hash,
        'evaluator_file_hash': evaluator_hash,
        'graph_content_hash': content_hash,
        'dataset_content_hash': None,
        'lineage_group': lineage,
    }


def _failed_row(base, *, requested_k, reason, costs=None, extra=None):
    row = dict(base)
    row.update({
        'status': 'failed', 'reason': str(reason),
        'requested_pair_count': requested_k,
        'requested_stored_count': None,
        'requested_physical_retention': None,
        'requested_physical_retention_reason': 'unavailable_on_failure',
        'physical_retention': None,
        'physical_retention_reason': 'no_successful_selection',
        'stored_retention': None,
        'stored_retention_reason': 'no_successful_selection',
        'available_nonself_columns': None, 'available_loop_columns': None,
        'scorer_metadata': None,
        'target': None, 'full_prediction': None, 'pruned_prediction': None,
        'squared_error': None, 'absolute_error': None,
        'deviation_from_full': None,
        'available_pair_count': None, 'sampled_pair_count': None,
        'scaffold_pair_count': None, 'repair_additions': None,
        'final_pair_count': None, 'constraint_satisfied': None,
        'budget_satisfied': None, 'available_stored_columns': None,
        'retained_nonself_columns': None, 'loops_kept': None,
        'total_kept_columns': None, 'physical_isolates': None,
        'physical_components': None, 'layout_pairs': None,
        'layout_inverse': None, 'selected_pairs': None, 'stored_mask': None,
        'stored_real': None, 'pair_marks': None, 'scores': None,
        'random_scores': None, 'diagnostics': None,
        'scoring_time_s': 0.0, 'decoder_time_s': 0.0,
        'prediction_time_s': 0.0, 'full_time_s': 0.0,
        'total_region_time_s': 0.0,
        'scoring_forward_calls': 0, 'eval_forward_calls': 0,
        'full_forward_calls': 0, 'live_forward_calls': 0,
        'scoring_graph_evaluations': 0, 'eval_graph_evaluations': 0,
        'full_graph_evaluations': 0, 'live_graph_evaluations': 0,
        'historical_training_calls': None,
        'historical_artifact_identity': None,
    })
    if costs:
        row.update(costs)
    if extra:
        row.update(extra)
    return row


# ---------------- main evaluation ----------------

def evaluate_matched_budget(predictor, graphs, scorers, budget, *, seed,
                            provenance, backbone, pair_marks_by_id=None,
                            random_scores_by_id=None):
    """Run one matched-budget comparison; see module docstring."""
    if not isinstance(budget, BudgetSpec):
        raise ValueError('budget must be a BudgetSpec')
    seed = _validate_seed(seed)
    backbone = _validate_backbone(backbone)
    provenance = _validate_provenance(provenance)
    _preflight(predictor, graphs, scorers, seed, backbone, provenance,
               pair_marks_by_id, random_scores_by_id)

    meter = MeteredPredictor(predictor)
    model_hash = meter.model_state_hash()
    budget_hash = hash_payload(budget.config_dict())
    try:
        evaluator_hash = sha256_file(__file__)['sha256']
    except Exception:
        evaluator_hash = 'unavailable'

    per_example_rows: List[Dict[str, Any]] = []
    content_by_id: Dict[str, str] = {}
    for graph in graphs:
        content_by_id[graph.cluster_id] = _graph_content_hash(
            graph, graph.cluster_id)
    dataset_hash = _dataset_content_hash(content_by_id)
    for graph in graphs:
        _evaluate_one_graph(
            graph, scorers, budget, seed=seed, backbone=backbone,
            provenance=provenance, meter=meter, model_hash=model_hash,
            budget_hash=budget_hash, evaluator_hash=evaluator_hash,
            marks_by_id=pair_marks_by_id or {},
            scores_by_id=random_scores_by_id or {},
            per_example_rows=per_example_rows)
    for row in per_example_rows:
        row['dataset_content_hash'] = dataset_hash

    summary_rows = _summarize(per_example_rows, scorers, budget, seed,
                              backbone, model_hash, provenance, budget_hash,
                              evaluator_hash)
    return summary_rows, per_example_rows


def _evaluate_one_graph(graph, scorers, budget, *, seed, backbone,
                          provenance, meter, model_hash, budget_hash,
                          evaluator_hash, marks_by_id, scores_by_id,
                          per_example_rows):
    cid = graph.cluster_id
    lineage = _lineage_group(graph)
    content_hash = _graph_content_hash(graph, cid)
    edge_index = getattr(graph, 'edge_index', None)
    try:
        num_nodes = int(graph.x.shape[0]) if isinstance(
            getattr(graph, 'x', None), torch.Tensor) else int(graph.num_nodes)
        layout = PhysicalPairLayout.from_edge_index(edge_index, num_nodes)
    except Exception as exc:
        for scorer in scorers:
            base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                             budget=budget, comparison_axis='scorer',
                             backbone=backbone, model_hash=model_hash,
                             provenance=provenance, budget_hash=budget_hash,
                             evaluator_hash=evaluator_hash,
                             content_hash=content_hash, lineage=lineage)
            per_example_rows.append(_failed_row(
                base, requested_k=None,
                reason=f'layout construction failed: {exc}'))
        base = _base_row(graph_id=cid, method='full', seed=seed,
                         budget=budget, comparison_axis='reference',
                         backbone=backbone, model_hash=model_hash,
                         provenance=provenance, budget_hash=budget_hash,
                         evaluator_hash=evaluator_hash,
                         content_hash=content_hash, lineage=lineage)
        per_example_rows.append(_failed_row(
            base, requested_k=None,
            reason=f'layout construction failed: {exc}'))
        return

    P = len(layout.pairs)
    try:
        if budget.pair_count is not None:
            k = int(budget.pair_count)
        else:
            k = int(pair_budget(layout, budget.keep_fraction))
    except Exception as exc:
        for scorer in scorers:
            base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                             budget=budget, comparison_axis='scorer',
                             backbone=backbone, model_hash=model_hash,
                             provenance=provenance, budget_hash=budget_hash,
                             evaluator_hash=evaluator_hash,
                             content_hash=content_hash, lineage=lineage)
            per_example_rows.append(_failed_row(
                base, requested_k=None,
                reason=f'budget resolution failed: {exc}'))
        base = _base_row(graph_id=cid, method='full', seed=seed,
                         budget=budget, comparison_axis='reference',
                         backbone=backbone, model_hash=model_hash,
                         provenance=provenance, budget_hash=budget_hash,
                         evaluator_hash=evaluator_hash,
                         content_hash=content_hash, lineage=lineage)
        per_example_rows.append(_failed_row(
            base, requested_k=None, reason=f'budget resolution failed: {exc}'))
        return

    try:
        if cid in marks_by_id:
            marks = pair_marks(
                layout, marks=torch.as_tensor(
                    marks_by_id[cid],
                    device=layout.pairs.device).detach())
        else:
            # Device-matching RNG convention: draw the exchangeable
            # permutation on CPU with a per-graph seeded generator, then
            # transport to the layout device. Passing a CPU generator
            # directly for a CUDA layout raises inside randperm, so every
            # CUDA graph would fail; CPU behaviour is unchanged.
            gen_marks = torch.Generator()
            gen_marks.manual_seed(_stable_int('marks', seed, cid))
            cpu_marks = torch.randperm(P, generator=gen_marks)
            marks = pair_marks(layout, marks=cpu_marks)
    except Exception as exc:
        for scorer in scorers:
            base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                             budget=budget, comparison_axis='scorer',
                             backbone=backbone, model_hash=model_hash,
                             provenance=provenance, budget_hash=budget_hash,
                             evaluator_hash=evaluator_hash,
                             content_hash=content_hash, lineage=lineage)
            per_example_rows.append(_failed_row(
                base, requested_k=k, reason=f'invalid pair marks: {exc}'))
        base = _base_row(graph_id=cid, method='full', seed=seed,
                         budget=budget, comparison_axis='reference',
                         backbone=backbone, model_hash=model_hash,
                         provenance=provenance, budget_hash=budget_hash,
                         evaluator_hash=evaluator_hash,
                         content_hash=content_hash, lineage=lineage)
        per_example_rows.append(_failed_row(
            base, requested_k=k, reason=f'invalid pair marks: {exc}'))
        return

    try:
        if cid in scores_by_id:
            rand_vec = torch.as_tensor(
                scores_by_id[cid], dtype=torch.float32,
                device=layout.pairs.device).detach().clone()
            if rand_vec.shape != (P,) or not torch.isfinite(rand_vec).all():
                raise ValueError('transported random_scores must be a finite '
                                 f'[{P}] vector')
        else:
            # Same CPU-draw-then-transfer convention as pair marks: a CPU
            # torch.Generator cannot drive randn(device='cuda'), so draw on
            # CPU and transport to the layout device.
            gen_rand = torch.Generator()
            gen_rand.manual_seed(_stable_int('random-scores', seed, cid))
            rand_vec = torch.randn(P, generator=gen_rand,
                                   dtype=torch.float32).to(
                                       layout.pairs.device)
    except Exception as exc:
        for scorer in scorers:
            base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                             budget=budget, comparison_axis='scorer',
                             backbone=backbone, model_hash=model_hash,
                             provenance=provenance, budget_hash=budget_hash,
                             evaluator_hash=evaluator_hash,
                             content_hash=content_hash, lineage=lineage)
            per_example_rows.append(_failed_row(
                base, requested_k=k, reason=f'invalid random scores: {exc}'))
        base = _base_row(graph_id=cid, method='full', seed=seed,
                         budget=budget, comparison_axis='reference',
                         backbone=backbone, model_hash=model_hash,
                         provenance=provenance, budget_hash=budget_hash,
                         evaluator_hash=evaluator_hash,
                         content_hash=content_hash, lineage=lineage)
        per_example_rows.append(_failed_row(
            base, requested_k=k, reason=f'invalid random scores: {exc}'))
        return

    try:
        target = _finite_scalar(getattr(graph, 'y', None), 'graph.y target')
    except Exception as exc:
        for scorer in scorers:
            base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                             budget=budget, comparison_axis='scorer',
                             backbone=backbone, model_hash=model_hash,
                             provenance=provenance, budget_hash=budget_hash,
                             evaluator_hash=evaluator_hash,
                             content_hash=content_hash, lineage=lineage)
            per_example_rows.append(_failed_row(
                base, requested_k=k,
                reason=f'explicit data error: invalid target: {exc}',
                extra={'available_pair_count': P,
                       'pair_marks': marks.detach().cpu().tolist(),
                       'random_scores': rand_vec.detach().cpu().tolist(),
                       'layout_pairs': layout.pairs.detach().cpu().tolist(),
                       'layout_inverse': layout.inverse.detach().cpu().tolist()}))
        base = _base_row(graph_id=cid, method='full', seed=seed,
                         budget=budget, comparison_axis='reference',
                         backbone=backbone, model_hash=model_hash,
                         provenance=provenance, budget_hash=budget_hash,
                         evaluator_hash=evaluator_hash,
                         content_hash=content_hash, lineage=lineage)
        per_example_rows.append(_failed_row(
            base, requested_k=k,
            reason=f'explicit data error: invalid target: {exc}',
            extra={'available_pair_count': P}))
        return

    # Shared full reference: exactly one call per graph.
    full_base = _base_row(graph_id=cid, method='full', seed=seed,
                          budget=budget, comparison_axis='reference',
                          backbone=backbone, model_hash=model_hash,
                          provenance=provenance, budget_hash=budget_hash,
                          evaluator_hash=evaluator_hash,
                          content_hash=content_hash, lineage=lineage)
    snap0 = meter.snapshot()
    _sync_cuda(meter, graph)
    t0 = time.perf_counter()
    try:
        full_out = meter.call(graph, mask=None, phase='shared_full')
        full_pred = _finite_scalar(full_out.prediction, 'full prediction')
        full_ok = True
        full_err = None
    except Exception as exc:
        full_pred = None
        full_ok = False
        full_err = str(exc)
    _sync_cuda(meter, graph)
    full_time = time.perf_counter() - t0
    full_delta = _snapshot_delta(snap0, meter.snapshot())

    if full_ok:
        per_example_rows.append(_full_row(
            full_base, graph=graph, layout=layout, requested_k=k,
            budget=budget, target=target, full_pred=full_pred,
            full_time=full_time, full_delta=full_delta))
    else:
        per_example_rows.append(_failed_row(
            full_base, requested_k=k,
            reason=f'full reference failed: {full_err}',
            costs={'full_time_s': full_time,
                   'total_region_time_s': full_time,
                   'full_forward_calls': full_delta['shared_full']['forward_calls'],
                   'full_graph_evaluations': full_delta['shared_full'].get(
                       'graph_evaluations', 0),
                   'live_forward_calls': _sum_calls(full_delta),
                   'live_graph_evaluations': sum(
                       v.get('graph_evaluations', 0)
                       for v in full_delta.values())},
            extra={'available_pair_count': P, 'target': target}))
        for scorer in scorers:
            base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                             budget=budget, comparison_axis='scorer',
                             backbone=backbone, model_hash=model_hash,
                             provenance=provenance, budget_hash=budget_hash,
                             evaluator_hash=evaluator_hash,
                             content_hash=content_hash, lineage=lineage)
            per_example_rows.append(_failed_row(
                base, requested_k=k,
                reason=f'dependent row failed: full reference failed: {full_err}',
                extra={'available_pair_count': P, 'target': target,
                       'pair_marks': marks.detach().cpu().tolist(),
                       'random_scores': rand_vec.detach().cpu().tolist()}))
        return

    for scorer in scorers:
        base = _base_row(graph_id=cid, method=scorer.name, seed=seed,
                         budget=budget, comparison_axis='scorer',
                         backbone=backbone, model_hash=model_hash,
                         provenance=provenance, budget_hash=budget_hash,
                         evaluator_hash=evaluator_hash,
                         content_hash=content_hash, lineage=lineage)
        try:
            cost = scorer.training_cost()
            hist_calls = cost.get('total_calls')
            hist_ident = _artifact_identity(cost)
            if isinstance(hist_calls, bool) or not isinstance(
                    hist_calls, numbers.Integral) or int(hist_calls) < 0:
                hist_calls = None
            else:
                hist_calls = int(hist_calls)
            if hist_ident is not None and not isinstance(hist_ident, str):
                hist_ident = str(hist_ident)
        except Exception:
            hist_calls, hist_ident = None, None
        snap_before = meter.snapshot()
        t_before = time.perf_counter()
        try:
            row = _evaluate_one_scorer(
                graph, layout, k, scorer, budget, seed=seed, target=target,
                full_pred=full_pred, marks=marks, rand_vec=rand_vec,
                meter=meter, base=base,
                hist=(hist_calls, hist_ident))
        except Exception as exc:
            snap_now = meter.snapshot()
            delta = _snapshot_delta(snap_before, snap_now)
            elapsed = time.perf_counter() - t_before
            scoring_calls = delta.get('scoring', {}).get('forward_calls', 0)
            eval_calls = delta.get('evaluation', {}).get('forward_calls', 0)
            scoring_evals = delta.get('scoring', {}).get(
                'graph_evaluations', 0)
            eval_evals = delta.get('evaluation', {}).get(
                'graph_evaluations', 0)
            row = _failed_row(
                base, requested_k=k, reason=str(exc) or type(exc).__name__,
                costs={'scoring_forward_calls': scoring_calls,
                       'eval_forward_calls': eval_calls,
                       'scoring_graph_evaluations': scoring_evals,
                       'eval_graph_evaluations': eval_evals,
                       'live_forward_calls': _sum_calls(delta),
                       'live_graph_evaluations': sum(
                           v.get('graph_evaluations', 0)
                           for v in delta.values()),
                       'total_region_time_s': elapsed},
                extra={'available_pair_count': P, 'target': target,
                       'full_prediction': full_pred,
                       'pair_marks': marks.detach().cpu().tolist(),
                       'random_scores': rand_vec.detach().cpu().tolist(),
                       'layout_pairs': layout.pairs.detach().cpu().tolist(),
                       'layout_inverse': layout.inverse.detach().cpu().tolist(),
                       'historical_training_calls': hist_calls,
                       'historical_artifact_identity': hist_ident})
        per_example_rows.append(row)


def _full_row(base, *, graph, layout, requested_k, budget, target, full_pred,
              full_time, full_delta):
    row = dict(base)
    P = len(layout.pairs)
    real = layout.real
    E = int(real.shape[0])
    nonself = int(real.sum())
    loops = int((~real).sum())
    requested_ret, requested_reason = _retention(requested_k, P)
    physical_ret, physical_reason = _retention(P, P)
    stored_ret, stored_reason = _retention(E, E)
    try:
        selected_all = torch.ones(P, dtype=torch.bool,
                                  device=layout.pairs.device)
        full_diag = selection_diagnostics(
            layout, selected_all, k=P, sampled_count=None,
            scaffold_count=0, constraint='none')
        isolates = int(full_diag['physical_isolates'])
        components = int(full_diag['physical_components'])
    except Exception:
        isolates, components = None, None
    row.update({
        'status': 'success', 'reason': None,
        'requested_pair_count': requested_k,
        'requested_stored_count': None,
        'requested_physical_retention': requested_ret,
        'requested_physical_retention_reason': requested_reason,
        'physical_retention': physical_ret,
        'physical_retention_reason': physical_reason,
        'stored_retention': stored_ret,
        'stored_retention_reason': stored_reason,
        'available_nonself_columns': nonself,
        'available_loop_columns': loops,
        'scorer_metadata': {},
        'target': target, 'full_prediction': full_pred,
        'pruned_prediction': None,
        'squared_error': (full_pred - target) ** 2,
        'absolute_error': abs(full_pred - target),
        'deviation_from_full': 0.0,
        'available_pair_count': P, 'sampled_pair_count': None,
        'scaffold_pair_count': None, 'repair_additions': None,
        'final_pair_count': P, 'constraint_satisfied': None,
        'budget_satisfied': None,
        'available_stored_columns': E,
        'retained_nonself_columns': int(real.sum()),
        'loops_kept': int((~real).sum()),
        'total_kept_columns': E,
        'physical_isolates': isolates, 'physical_components': components,
        'layout_pairs': layout.pairs.detach().cpu().tolist(),
        'layout_inverse': layout.inverse.detach().cpu().tolist(),
        'selected_pairs': None, 'stored_mask': None,
        'stored_real': real.detach().cpu().tolist(),
        'pair_marks': None, 'scores': None, 'random_scores': None,
        'diagnostics': {'kind': 'full_reference',
                        'constraint_applies': False},
        'scoring_time_s': 0.0, 'decoder_time_s': 0.0,
        'prediction_time_s': 0.0, 'full_time_s': full_time,
        'total_region_time_s': full_time,
        'scoring_forward_calls': 0, 'eval_forward_calls': 0,
        'full_forward_calls': full_delta['shared_full']['forward_calls'],
        'full_graph_evaluations': full_delta['shared_full'].get(
            'graph_evaluations', 0),
        'scoring_graph_evaluations': 0, 'eval_graph_evaluations': 0,
        'live_forward_calls': _sum_calls(full_delta),
        'live_graph_evaluations': sum(
            v.get('graph_evaluations', 0) for v in full_delta.values()),
        'historical_training_calls': None,
        'historical_artifact_identity': None,
    })
    return row


def _decode(layout, scores, k, budget, marks):
    if budget.decoder == 'enumerated_structured_map':
        return select_structured(layout, scores, k, budget.constraint,
                                 pair_marks=marks, sample=False)
    method = 'direct' if budget.decoder == 'direct' else 'scaffold'
    return select_pairs(layout, scores, k, budget.constraint, method=method,
                        sample=False, pair_marks=marks)


def _evaluate_one_scorer(graph, layout, k, scorer, budget, *, seed, target,
                         full_pred, marks, rand_vec, meter, base, hist):
    P = len(layout.pairs)
    hist_calls, hist_ident = hist
    region_t0 = time.perf_counter()
    snap0 = meter.snapshot()
    _sync_cuda(meter, graph)
    t0 = time.perf_counter()
    try:
        out = scorer.score(graph, layout, k, meter=meter,
                           random_scores=rand_vec.detach().clone(),
                           generator=None)
        scores = out.scores
        metadata = dict(getattr(out, 'metadata', {}) or {})
    finally:
        _sync_cuda(meter, graph)
        scoring_time = time.perf_counter() - t0
    snap1 = meter.snapshot()
    scoring_delta = _snapshot_delta(snap0, snap1)
    if not isinstance(scores, torch.Tensor):
        raise ValueError(f'scorer {scorer.name!r} returned non-tensor scores')
    scores = scores.detach().to(dtype=torch.float32,
                                device=layout.pairs.device)
    if scores.shape != (P,) or not torch.isfinite(scores).all():
        raise ValueError(f'scorer {scorer.name!r} must return finite [{P}] '
                         'scores')

    _sync_cuda(meter, graph)
    t1 = time.perf_counter()
    try:
        selection = _decode(layout, scores, k, budget, marks)
    finally:
        _sync_cuda(meter, graph)
        decoder_time = time.perf_counter() - t1
    snap2 = meter.snapshot()
    selected = selection.selected.detach().cpu()
    stored_mask = selection.mask.detach().cpu()
    if stored_mask.shape != layout.real.shape:
        raise ValueError('decoder returned a mask of wrong stored length')
    if int(selected.sum()) != k:
        raise ValueError('decoder final count does not match requested k')

    _sync_cuda(meter, graph)
    t2 = time.perf_counter()
    snap3 = meter.snapshot()
    try:
        pred_out = meter.call(graph, mask=stored_mask.to(
            graph.edge_index.device), phase='evaluation')
        pruned_pred = _finite_scalar(pred_out.prediction, 'pruned prediction')
    finally:
        _sync_cuda(meter, graph)
        prediction_time = time.perf_counter() - t2
    snap4 = meter.snapshot()
    pred_delta = _snapshot_delta(snap3, snap4)

    diag = dict(getattr(selection, 'diagnostics', {}) or {})
    scaffold = int(diag.get('scaffold_pair_count', 0))
    repair = 0 if budget.decoder in ('direct',
                                     'enumerated_structured_map') else None
    if budget.decoder == 'scaffold' and scaffold:
        repair = 0
    real = layout.real.detach().cpu()
    E = int(real.shape[0])
    nonself = int(real.sum())
    loops = int((~real).sum())
    kept_nonself = int((stored_mask & real).sum())
    loops_kept = int((stored_mask & ~real).sum())
    diag_counts = selection_diagnostics(
        layout, selection.selected.detach().to(layout.pairs.device), k=k,
        sampled_count=int(selected.sum()) - scaffold,
        scaffold_count=scaffold, constraint=budget.constraint)
    requested_ret, requested_reason = _retention(k, P)
    physical_ret, physical_reason = _retention(int(selected.sum()), P)
    stored_ret, stored_reason = _retention(int(stored_mask.sum()), E)
    region_elapsed = time.perf_counter() - region_t0

    row = dict(base)
    row.update({
        'status': 'success', 'reason': None,
        'requested_pair_count': k, 'requested_stored_count': None,
        'requested_physical_retention': requested_ret,
        'requested_physical_retention_reason': requested_reason,
        'physical_retention': physical_ret,
        'physical_retention_reason': physical_reason,
        'stored_retention': stored_ret,
        'stored_retention_reason': stored_reason,
        'available_nonself_columns': nonself,
        'available_loop_columns': loops,
        'scorer_metadata': _json_safe(metadata),
        'target': target, 'full_prediction': full_pred,
        'pruned_prediction': pruned_pred,
        'squared_error': (pruned_pred - target) ** 2,
        'absolute_error': abs(pruned_pred - target),
        'deviation_from_full': pruned_pred - full_pred,
        'available_pair_count': P,
        'sampled_pair_count': int(selected.sum()) - scaffold,
        'scaffold_pair_count': scaffold, 'repair_additions': repair,
        'final_pair_count': int(selected.sum()),
        'constraint_satisfied': bool(diag_counts['constraint_satisfied']),
        'budget_satisfied': bool(diag_counts['budget_satisfied']),
        'available_stored_columns': E,
        'retained_nonself_columns': kept_nonself,
        'loops_kept': loops_kept,
        'total_kept_columns': int(stored_mask.sum()),
        'physical_isolates': int(diag_counts['physical_isolates']),
        'physical_components': int(diag_counts['physical_components']),
        'layout_pairs': layout.pairs.detach().cpu().tolist(),
        'layout_inverse': layout.inverse.detach().cpu().tolist(),
        'selected_pairs': selected.nonzero().flatten().tolist(),
        'stored_mask': stored_mask.tolist(),
        'stored_real': real.tolist(),
        'pair_marks': marks.detach().cpu().tolist(),
        'scores': scores.detach().cpu().tolist(),
        'random_scores': rand_vec.detach().cpu().tolist(),
        'diagnostics': _json_safe(diag),
        'scoring_time_s': scoring_time, 'decoder_time_s': decoder_time,
        'prediction_time_s': prediction_time, 'full_time_s': 0.0,
        'total_region_time_s': region_elapsed,
        'scoring_forward_calls': scoring_delta['scoring']['forward_calls'],
        'scoring_graph_evaluations': scoring_delta['scoring'].get(
            'graph_evaluations', 0),
        'eval_forward_calls': pred_delta['evaluation']['forward_calls'],
        'eval_graph_evaluations': pred_delta['evaluation'].get(
            'graph_evaluations', 0),
        'full_forward_calls': 0,
        'full_graph_evaluations': 0,
        'live_forward_calls': (_sum_calls(scoring_delta)
                               + _sum_calls(_snapshot_delta(snap2, snap3))
                               + _sum_calls(pred_delta)),
        'live_graph_evaluations': sum(
            v.get('graph_evaluations', 0) for v in scoring_delta.values()) + sum(
            v.get('graph_evaluations', 0)
            for v in _snapshot_delta(snap2, snap3).values()) + sum(
            v.get('graph_evaluations', 0) for v in pred_delta.values()),
        'historical_training_calls': hist_calls,
        'historical_artifact_identity': hist_ident,
    })
    return row


def _artifact_identity(cost: Dict[str, Any]) -> Optional[str]:
    for key in ('model_state_hash', 'artifact_hash', 'training_content_hash',
                'label_hash', 'identity'):
        value = cost.get(key)
        if isinstance(value, str) and value:
            return value
    try:
        return hash_payload({k: (v if isinstance(v, (str, int, float, bool,
                                                     type(None))) else str(v))
                             for k, v in sorted(cost.items())})
    except Exception:
        return None


def _method_summary(rows, method, *, budget, seed, backbone, model_hash,
                    provenance, budget_hash, evaluator_hash,
                    common_ids, unqualified, unqualified_reason):
    own = [r for r in rows if r['method'] == method
           and r['status'] == 'success']
    failed = [r for r in rows if r['method'] == method
              and r['status'] != 'success']
    own_ids = sorted(r['graph_id'] for r in own)
    failed_ids = sorted(r['graph_id'] for r in failed)
    is_full = (method == 'full')
    preds = [(r['full_prediction'] if is_full else r['pruned_prediction'])
             for r in own]
    targets = [r['target'] for r in own]
    metrics = _regression_metrics(list(preds), list(targets))
    if is_full:
        if len(own) >= 2 and len(set(preds)) > 1:
            fidelity, fidelity_reason = 1.0, None
        elif len(own) < 2:
            fidelity, fidelity_reason = None, 'n_lt_2'
        else:
            fidelity, fidelity_reason = None, 'constant_reference_vector'
    else:
        fulls = [r['full_prediction'] for r in own]
        fidelity, fidelity_reason = _pearson(list(preds), list(fulls))
    common_rows = [r for r in own if r['graph_id'] in common_ids]
    common_preds = [(r['full_prediction'] if is_full
                     else r['pruned_prediction']) for r in common_rows]
    common_targets = [r['target'] for r in common_rows]
    common_metrics = _regression_metrics(list(common_preds),
                                         list(common_targets))
    if is_full:
        if len(common_rows) >= 2 and len(set(common_preds)) > 1:
            common_fidelity, common_fidelity_reason = 1.0, None
        elif len(common_rows) < 2:
            common_fidelity, common_fidelity_reason = (
                None, 'n_lt_2' if not common_rows
                and len(common_ids) == 0 else 'n_lt_2')
            if not common_rows:
                common_fidelity_reason = 'no_common_successes'
        else:
            common_fidelity, common_fidelity_reason = (
                None, 'constant_reference_vector')
    else:
        common_fulls = [r['full_prediction'] for r in common_rows]
        if not common_rows:
            common_fidelity, common_fidelity_reason = (
                None, 'no_common_successes')
        else:
            common_fidelity, common_fidelity_reason = _pearson(
                list(common_preds), list(common_fulls))
            if common_fidelity is None and common_fidelity_reason is None:
                common_fidelity_reason = 'unavailable'
    # Run-level dedup artifact costs, separate from live counts. Unknown
    # costs stay null with a reason; never a silent false zero.
    cost_map: Dict[str, Any] = {}
    for r in own + failed:
        calls = r.get('historical_training_calls')
        ident = r.get('historical_artifact_identity')
        if isinstance(calls, bool) or not isinstance(
                calls, numbers.Integral) or int(calls) < 0:
            continue
        key = ident if isinstance(ident, str) and ident else (
            f"unidentified:{r.get('graph_id')}")
        known = cost_map.get(key)
        if known is None or int(calls) > int(known):
            cost_map[key] = int(calls)
    if cost_map:
        hist_total = sum(cost_map.values())
        hist_calls_val: Optional[int] = hist_total
        hist_reason = None
    else:
        hist_total = None
        hist_calls_val = None
        hist_reason = 'unknown_artifact_cost'
    hist_idents = sorted({r.get('historical_artifact_identity') for r in own
                          if r.get('historical_artifact_identity')})
    return {
        'method': method, 'seed': seed, 'constraint': budget.constraint,
        'decoder': budget.decoder,
        'comparison_axis': 'reference' if is_full else 'scorer',
        'backbone_stage': backbone, 'model_state_hash': model_hash,
        'research_label': provenance.get('research_label'),
        'budget_config_hash': budget_hash,
        'evaluator_file_hash': evaluator_hash,
        'population': 'method_success',
        'n_requested': len(own) + len(failed),
        'n_success': len(own), 'n_failed': len(failed),
        'success_graph_ids': own_ids, 'failed_graph_ids': failed_ids,
        'failures': [{'graph_id': r['graph_id'], 'reason': r.get('reason')}
                     for r in failed],
        'rmse': metrics['rmse'], 'rmse_reason': metrics['rmse_reason'],
        'r2': metrics['r2'], 'r2_reason': metrics['r2_reason'],
        'bias': metrics['bias'], 'bias_reason': metrics['bias_reason'],
        'std_residual': metrics['std_residual'],
        'std_reason': metrics['std_reason'],
        'fidelity_full_pred': fidelity, 'fidelity_reason': fidelity_reason,
        'common_success_ids': sorted(common_ids), 'common_n': len(common_ids),
        'common_rmse': common_metrics['rmse'],
        'common_rmse_reason': common_metrics['rmse_reason'],
        'common_r2': common_metrics['r2'],
        'common_r2_reason': common_metrics['r2_reason'],
        'common_bias': common_metrics['bias'],
        'common_bias_reason': common_metrics['bias_reason'],
        'common_std_residual': common_metrics['std_residual'],
        'common_std_reason': common_metrics['std_reason'],
        'common_fidelity_full_pred': common_fidelity,
        'common_fidelity_reason': common_fidelity_reason,
        'comparison_population': 'common_success',
        'unqualified_comparison': unqualified,
        'unqualified_reason': unqualified_reason,
        'historical_training_calls': hist_calls_val,
        'historical_training_calls_reason': hist_reason,
        'historical_artifact_cost_map': dict(cost_map),
        'historical_training_total': hist_total,
        'historical_artifact_identities': hist_idents,
    }


def _summarize(rows, scorers, budget, seed, backbone, model_hash,
               provenance, budget_hash, evaluator_hash):
    methods = [s.name for s in scorers] + ['full']
    success_sets = {}
    for method in methods:
        success_sets[method] = {r['graph_id'] for r in rows
                                if r['method'] == method
                                and r['status'] == 'success'}
    pruned = [s.name for s in scorers]
    common_ids = set.intersection(
        *(success_sets[m] for m in pruned)) if pruned else set()
    unqualified = len({frozenset(success_sets[m]) for m in pruned}) > 1
    unqualified_reason = ('success sets differ across methods; headline '
                          'comparison requires the common-success population'
                          if unqualified else None)
    if not common_ids:
        headline_allowed: bool = False
        headline_reason: Optional[str] = ('no common successes; refusing '
                                          'headline comparison')
    elif unqualified:
        headline_allowed = False
        headline_reason = ('success sets differ across methods; headline '
                           'comparison requires the common-success population')
    else:
        headline_allowed = True
        headline_reason = None
    summaries = [_method_summary(rows, method, budget=budget, seed=seed,
                                 backbone=backbone, model_hash=model_hash,
                                 provenance=provenance,
                                 budget_hash=budget_hash,
                                 evaluator_hash=evaluator_hash,
                                 common_ids=common_ids,
                                 unqualified=unqualified,
                                 unqualified_reason=unqualified_reason)
                 for method in methods]
    for summary in summaries:
        summary['headline_comparison_allowed'] = headline_allowed
        summary['headline_comparison_reason'] = headline_reason
    return summaries


# ---------------- paired comparison ----------------

def _stratum_key(row):
    # Budget-CONFIG strata: the same keep_fraction config groups graphs of
    # differing richness together; exact integer budgets stay exact via the
    # shared budget_config_hash. Per-graph requested k is validated within
    # the stratum, not used to split it.
    return (row.get('seed'), row.get('budget_config_hash'),
            row.get('requested_keep_fraction'), row.get('constraint'),
            row.get('decoder'), row.get('comparison_axis'),
            row.get('model_state_hash'), row.get('backbone_stage'))


def summarize_paired(rows, left, right, *, group_by='lineage_group', seed=0,
                     n_bootstrap=1000, confidence=0.95):
    """Paired left-minus-right RMSE with seeded block bootstrap.

    Full-reference comparisons across differing comparison_axis values are
    explicitly unsupported (left/right must share one axis); strata follow
    the budget config, not per-graph realized k under a shared fraction.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError('left/right must be method-name strings')
    if left == right:
        raise ValueError('left and right must be distinct methods')
    seed = _validate_seed(seed)
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap,
                                                       numbers.Integral):
        raise ValueError('n_bootstrap must be an integer')
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    if not isinstance(confidence, numbers.Real) or not 0.0 < float(confidence) < 1.0:
        raise ValueError('confidence must be in (0, 1)')
    confidence = float(confidence)
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('rows must be a nonempty list of per-example rows')

    relevant = [r for r in rows if r.get('method') in (left, right)]
    if not relevant:
        raise ValueError(f'no rows for methods {left!r}/{right!r}')
    axes = {r.get('comparison_axis') for r in relevant}
    if len(axes) > 1:
        raise ValueError(
            'full-reference comparisons are unsupported when '
            f'comparison_axis differs across methods: {sorted(axes, key=repr)}; '
            'pair scorers only with scorers, references only with references')

    eligible = [r for r in relevant if r.get('status') == 'success']
    seen = set()
    for r in eligible:
        key = (_stratum_key(r), r.get('method'), r.get('graph_id'))
        if key in seen:
            raise ValueError(f'duplicate eligible row for {key}')
        seen.add(key)

    # Preserve failure-only strata: keys come from every relevant row, not
    # just eligible successes, so all-failed comparisons stay visible.
    strata_keys: Dict[Any, None] = {}
    for r in relevant:
        strata_keys.setdefault(_stratum_key(r))
    strata: Dict[Any, List[Dict[str, Any]]] = {k: [] for k in strata_keys}
    for r in eligible:
        strata.setdefault(_stratum_key(r), []).append(r)
    out = []
    for key in sorted(strata, key=repr):
        out.append(_paired_stratum(
            strata[key], key, left, right, group_by=group_by, seed=seed,
            n_bootstrap=n_bootstrap, confidence=confidence, all_rows=rows))
    return out


def _paired_stratum(stratum_rows, key, left, right, *, group_by, seed,
                    n_bootstrap, confidence, all_rows):
    (seed_v, budget_hash, req_q, constraint, decoder, axis, model_hash,
     backbone_stage) = key
    by_method: Dict[str, Dict[str, Dict[str, Any]]] = {left: {}, right: {}}
    for r in stratum_rows:
        by_method[r['method']][r['graph_id']] = r
    paired_ids = sorted(set(by_method[left]) & set(by_method[right]))
    # Joined-identity validation: never combine different dataset/split
    # identities, targets, graph content, or lineage for one graph ID; the
    # per-graph requested k must also match between paired left/right.
    for gid in paired_ids:
        lrow, rrow = by_method[left][gid], by_method[right][gid]
        if lrow.get('target') != rrow.get('target'):
            raise ValueError(
                f'paired target mismatch for graph {gid!r}: '
                f'{lrow.get("target")!r} != {rrow.get("target")!r}')
        if lrow.get('graph_content_hash') != rrow.get('graph_content_hash'):
            raise ValueError(
                f'paired graph-content mismatch for graph {gid!r}')
        if lrow.get('lineage_group') != rrow.get('lineage_group'):
            raise ValueError(
                f'paired lineage mismatch for graph {gid!r}: '
                f'{lrow.get("lineage_group")!r} != {rrow.get("lineage_group")!r}')
        for field in ('dataset_content_hash', 'budget_config_hash',
                      'split_manifest_sha256', 'config_sha256',
                      'catalog_contract_sha256', 'requested_keep_fraction'):
            if lrow.get(field) != rrow.get(field):
                raise ValueError(
                    f'paired {field} mismatch for graph {gid!r}: refusing to '
                    'combine different dataset/split identities')
        if lrow.get('requested_pair_count') != rrow.get('requested_pair_count'):
            raise ValueError(
                f'paired requested_pair_count mismatch for graph {gid!r}: '
                f'{lrow.get("requested_pair_count")!r} != '
                f'{rrow.get("requested_pair_count")!r}')
    left_fail = sorted({r.get('graph_id') for r in all_rows
                        if r.get('method') == left
                        and r.get('status') != 'success'
                        and _stratum_key(r) == key})
    right_fail = sorted({r.get('graph_id') for r in all_rows
                         if r.get('method') == right
                         and r.get('status') != 'success'
                         and _stratum_key(r) == key})
    base = {
        'left': left, 'right': right, 'seed': seed_v,
        'requested_pair_count': None, 'requested_keep_fraction': req_q,
        'budget_config_hash': budget_hash,
        'constraint': constraint, 'decoder': decoder,
        'comparison_axis': axis, 'model_state_hash': model_hash,
        'backbone_stage': backbone_stage,
        'paired_graph_ids': paired_ids, 'n_paired': len(paired_ids),
        'left_failures': left_fail, 'right_failures': right_fail,
        'n_left_failed': len(left_fail), 'n_right_failed': len(right_fail),
    }
    if paired_ids:
        ks = sorted({by_method[left][gid].get('requested_pair_count')
                     for gid in paired_ids})
        base['requested_pair_counts'] = ks
        base['requested_pair_count'] = ks[0] if len(ks) == 1 else None
    else:
        base['requested_pair_counts'] = []
    if not paired_ids:
        base.update({'metric_rmse_diff': None, 'metric_reason': 'no_paired_ids',
                     'ci_low': None, 'ci_high': None, 'ci_reason': 'no_paired_ids',
                     'confidence': confidence, 'n_bootstrap': n_bootstrap,
                     'n_blocks': 0, 'bootstrap_unit': group_by})
        return base
    left_res = [by_method[left][gid]['pruned_prediction']
                if by_method[left][gid]['pruned_prediction'] is not None
                else by_method[left][gid]['full_prediction']
                for gid in paired_ids]
    right_res = [by_method[right][gid]['pruned_prediction']
                 if by_method[right][gid]['pruned_prediction'] is not None
                 else by_method[right][gid]['full_prediction']
                 for gid in paired_ids]
    targets = [by_method[left][gid]['target'] for gid in paired_ids]
    metric = _rmse(left_res, targets) - _rmse(right_res, targets)
    base.update({'metric_rmse_diff': metric, 'metric_reason': None})

    blocks: Dict[Any, List[str]] = {}
    missing_groups: List[str] = []
    conflict_groups: List[str] = []
    for gid in paired_ids:
        lgrp = by_method[left][gid].get(group_by)
        rgrp = by_method[right][gid].get(group_by)
        if lgrp is None or rgrp is None:
            missing_groups.append(gid)
            continue
        if lgrp != rgrp:
            conflict_groups.append(gid)
            continue
        blocks.setdefault(lgrp, []).append(gid)
    if missing_groups or conflict_groups:
        base.update({'ci_low': None, 'ci_high': None,
                     'ci_reason': ('interval unavailable: '
                                   f'{len(missing_groups)} paired unit(s) lack '
                                   f'a declared {group_by} and '
                                   f'{len(conflict_groups)} conflict between '
                                   'methods; refusing to silently exclude '
                                   'units from the interval while the point '
                                   'estimate includes them'),
                      'missing_group_ids': sorted(missing_groups),
                      'conflict_group_ids': sorted(conflict_groups),
                      'confidence': confidence, 'n_bootstrap': n_bootstrap,
                      'n_blocks': len(blocks),
                      'bootstrap_unit': group_by})
        return base
    if len(blocks) < 2:
        base.update({'ci_low': None, 'ci_high': None,
                     'ci_reason': ('interval unavailable: fewer than two '
                                   'declared group blocks; refusing a '
                                   'spuriously precise single-block interval'),
                     'confidence': confidence, 'n_bootstrap': n_bootstrap,
                     'n_blocks': len(blocks),
                     'bootstrap_unit': group_by})
        return base
    block_names = sorted(blocks, key=repr)
    per_graph = {gid: (by_method[left][gid]['pruned_prediction']
                       if by_method[left][gid]['pruned_prediction'] is not None
                       else by_method[left][gid]['full_prediction'],
                       by_method[right][gid]['pruned_prediction']
                       if by_method[right][gid]['pruned_prediction'] is not None
                       else by_method[right][gid]['full_prediction'],
                       by_method[left][gid]['target']) for gid in paired_ids}
    gen = torch.Generator()
    gen.manual_seed(_stable_int('paired-bootstrap', seed, left, right,
                                repr(key)))
    estimates = []
    for _ in range(n_bootstrap):
        draw = torch.randint(len(block_names), (len(block_names),),
                             generator=gen).tolist()
        sample_gids = []
        for idx in draw:
            sample_gids.extend(blocks[block_names[idx]])
        lp = [per_graph[g][0] for g in sample_gids]
        rp = [per_graph[g][1] for g in sample_gids]
        tg = [per_graph[g][2] for g in sample_gids]
        estimates.append(_rmse(lp, tg) - _rmse(rp, tg))
    estimates.sort()
    alpha = 1.0 - confidence
    lo_idx = min(max(int(math.floor(alpha / 2 * n_bootstrap)), 0),
                 n_bootstrap - 1)
    hi_idx = min(max(int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1, 0),
                 n_bootstrap - 1)
    base.update({'ci_low': estimates[lo_idx], 'ci_high': estimates[hi_idx],
                 'ci_reason': None, 'confidence': confidence,
                 'n_bootstrap': n_bootstrap, 'n_blocks': len(blocks),
                 'bootstrap_unit': group_by})
    return base
