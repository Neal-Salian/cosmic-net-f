"""TESTS for rls.matched_benchmark (Task 3b framework dispatch, TDD).

Contract under test (see task-3b-framework-dispatch.md):
- BudgetSpec(pair_count xor keep_fraction, constraint, decoder) frozen.
- evaluate_matched_budget(predictor, graphs, scorers, budget, *, seed,
    provenance, backbone, pair_marks_by_id=None, random_scores_by_id=None)
    -> (summary_rows, per_example_rows). Predictor is nn.Module (meter owned
    locally); scorers are duck-typed PairScorer (name/score/training_cost).
- summarize_paired(rows, left, right, *, group_by, seed, n_bootstrap,
    confidence) with seeded block bootstrap over declared groups.

Stub scorers below implement the duck-typed scorer protocol so the framework
is testable without the parallel scorer worker's file. Real PairScorer
objects satisfy the same protocol.
"""
import copy
import hashlib
import importlib.util
import json
import math

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from rls.constraints import PhysicalPairLayout

MODULE = 'rls.matched_benchmark'


def _api():
    assert importlib.util.find_spec(MODULE) is not None, \
        'matched_benchmark API missing'
    from rls import matched_benchmark as mb
    for name in ('BudgetSpec', 'evaluate_matched_budget', 'summarize_paired'):
        assert hasattr(mb, name), f'{name} API missing'
    return mb


# ---------------- fixtures ----------------

class ToyPredictor(nn.Module):
    """prediction = w * edge_attr.sum() + 0.5 * x.sum(); analytic."""

    def __init__(self, weight=2.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(weight)))
        self.lin = nn.Linear(2, 2)
        self.forward_calls = 0

    def forward(self, batch, return_embeddings=False):
        self.forward_calls += 1
        if return_embeddings:
            raise ValueError('toy has no embeddings')
        return self.w * batch.edge_attr.sum() + 0.5 * batch.x.sum()


class _ScoreOut:
    def __init__(self, scores, metadata):
        self.scores = scores
        self.metadata = metadata


class StubScorer:
    """Duck-typed PairScorer: fixed scores_fn(layout) -> [P] tensor."""

    def __init__(self, name, scores_fn=None, cost=None):
        self.name = name
        self._fn = scores_fn or (lambda layout: torch.arange(
            len(layout.pairs), dtype=torch.float32))
        self._cost = dict(cost) if cost is not None else {'total_calls': 0}

    def score(self, graph, layout, k, *, meter=None, random_scores=None,
              generator=None):
        scores = self._fn(layout).to(dtype=torch.float32)
        assert scores.shape == (len(layout.pairs),)
        assert torch.isfinite(scores).all()
        return _ScoreOut(scores, {'scorer': self.name})

    def training_cost(self):
        return dict(self._cost)


def _graph(ei, n_nodes=3, cid='g', y_val=1.0, extra=None):
    E = ei.shape[1]
    g = Data(x=torch.ones(n_nodes, 2),
             edge_index=ei.clone(),
             edge_attr=torch.ones(E, 2))
    g.cluster_id = cid
    g.y = torch.tensor([float(y_val)])
    if extra:
        for k, v in extra.items():
            setattr(g, k, v)
    return g


def _triangle(cid='tri'):
    ei = torch.tensor([[0, 1, 0, 2, 1, 2],
                       [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    return _graph(ei, 3, cid)


def _star4():
    # center 0 with leaves 1,2,3 reciprocal; P=3
    ei = torch.tensor([[0, 1, 0, 2, 0, 3],
                       [1, 0, 2, 0, 3, 0]], dtype=torch.long)
    return _graph(ei, 4, 'star')


def _provenance(label='test-label'):
    from data.provenance import build_run_provenance
    return build_run_provenance(
        source_files=[{'path': 'synthetic', 'sha256': 'abc'}],
        catalog_contract={'target_definition': 't'},
        split_manifest={'manifest_sha256': 'split-abc'},
        resolved_config={'k': 1},
        source_revision='rev-1',
        graph={'n': 1},
        augmentation_manifest=None,
        research_label=label,
    )


# ---------------- BudgetSpec ----------------

def test_budget_spec_exactly_one_and_rejects_bool():
    mb = _api()
    b = mb.BudgetSpec(pair_count=2)
    assert b.pair_count == 2 and b.keep_fraction is None
    b2 = mb.BudgetSpec(keep_fraction=0.5)
    assert b2.keep_fraction == 0.5 and b2.pair_count is None
    with pytest.raises(ValueError, match='exactly one|one budget'):
        mb.BudgetSpec()
    with pytest.raises(ValueError, match='exactly one|one budget'):
        mb.BudgetSpec(pair_count=2, keep_fraction=0.5)
    with pytest.raises(ValueError, match='bool|integer|budget'):
        mb.BudgetSpec(pair_count=True)
    with pytest.raises(ValueError, match='bool|fraction|budget'):
        mb.BudgetSpec(keep_fraction=True)


def test_budget_spec_invalid_range_constraint_decoder():
    mb = _api()
    with pytest.raises(ValueError, match='nonneg|negativ|budget|count'):
        mb.BudgetSpec(pair_count=-1)
    for bad in (float('nan'), float('inf'), -0.1, 1.5):
        with pytest.raises(ValueError, match='fraction|range|finite'):
            mb.BudgetSpec(keep_fraction=bad)
    with pytest.raises(ValueError, match='constraint'):
        mb.BudgetSpec(pair_count=1, constraint='nope')
    with pytest.raises(ValueError, match='decoder'):
        mb.BudgetSpec(pair_count=1, decoder='nope')


# ---------------- preflight ----------------

def test_duplicate_scorer_names_rejected_before_predictor_calls():
    mb = _api()
    toy = ToyPredictor()
    g = _triangle('a')
    s1 = StubScorer('dup')
    s2 = StubScorer('dup')
    with pytest.raises(ValueError, match='duplicate.*scorer|scorer.*duplicate'):
        mb.evaluate_matched_budget(
            toy, [g], [s1, s2], mb.BudgetSpec(pair_count=2),
            seed=0, provenance=_provenance(), backbone='frozen')
    assert toy.forward_calls == 0


def test_duplicate_missing_ids_rejected_before_calls():
    mb = _api()
    toy = ToyPredictor()
    g1 = _triangle('same')
    g2 = _triangle('same')
    with pytest.raises(ValueError, match='duplicate.*ID|ID.*duplicate'):
        mb.evaluate_matched_budget(
            toy, [g1, g2], [StubScorer('a')],
            mb.BudgetSpec(pair_count=2),
            seed=0, provenance=_provenance(), backbone='frozen')
    assert toy.forward_calls == 0
    g3 = _triangle('ok')
    del g3.cluster_id
    with pytest.raises(ValueError, match='cluster_id|missing.*ID|ID.*missing'):
        mb.evaluate_matched_budget(
            toy, [g3], [StubScorer('a')], mb.BudgetSpec(pair_count=2),
            seed=0, provenance=_provenance(), backbone='frozen')
    assert toy.forward_calls == 0


# ---------------- core evaluation ----------------

def test_exact_k_symmetric_masks_and_shared_full_once():
    mb = _api()
    toy = ToyPredictor(weight=2.0)
    graphs = [_triangle('g1'), _triangle('g2')]
    scorers = [StubScorer('lo'), StubScorer(
        'hi', lambda layout: -torch.arange(len(layout.pairs),
                                           dtype=torch.float32))]
    summary, rows = mb.evaluate_matched_budget(
        toy, graphs, scorers, mb.BudgetSpec(pair_count=2),
        seed=7, provenance=_provenance(), backbone='frozen')
    # shared full: one row/call per graph, cost not multiplied
    full_rows = [r for r in rows if r['method'] == 'full']
    assert len(full_rows) == 2
    assert toy.forward_calls == 2 + 2 * 2  # 2 full + 4 pruned
    pruned = [r for r in rows if r['method'] in ('lo', 'hi')]
    assert len(pruned) == 4
    for r in pruned:
        assert r['status'] == 'success'
        assert r['final_pair_count'] == 2
        assert r['requested_pair_count'] == 2
        assert r['constraint_satisfied'] is True
        # symmetric expansion: reciprocal copies share membership.
        # layout.inverse indexes nonself columns only (length excludes
        # loops), so map pair rows back to stored columns via the nonself
        # positions rather than &-ing against the full stored vector.
        mask = torch.tensor(r['stored_mask'])
        inv = torch.tensor(r['layout_inverse'])
        real = torch.tensor(r['stored_real'])
        sel = torch.tensor(r['selected_pairs'])
        real_idx = real.nonzero().flatten()
        assert len(inv) == int(real.sum())
        for p in sel.tolist():
            cols = real_idx[(inv == p).nonzero().flatten()]
            assert len(cols) > 0
            assert bool(mask[cols].all())
        # loops are always kept by the shared decoder
        assert bool(mask[~real].all())
        assert r['prediction_time_s'] is not None
        assert r['scoring_time_s'] is not None
    # full reference describes original graph (no constraint masking)
    for r in full_rows:
        assert r['status'] == 'success'
        assert r['total_kept_columns'] == 6
    # summary counts
    by_method = {s['method']: s for s in summary}
    assert by_method['lo']['n_success'] == 2
    assert by_method['lo']['n_requested'] == 2


def test_infeasible_star_k_too_low_explicit_failed():
    mb = _api()
    toy = ToyPredictor()
    graphs = [_star4(), _triangle('tri-ok')]
    summary, rows = mb.evaluate_matched_budget(
        toy, graphs, [StubScorer('a')], mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    star_rows = [r for r in rows
                 if r['graph_id'] == 'star' and r['method'] == 'a']
    assert len(star_rows) == 1
    assert star_rows[0]['status'] == 'failed'
    assert 'feasib' in star_rows[0]['reason'].lower() or \
        'isolat' in star_rows[0]['reason'].lower()
    assert star_rows[0]['requested_pair_count'] == 2
    ok_rows = [r for r in rows
               if r['graph_id'] == 'tri-ok' and r['method'] == 'a']
    assert ok_rows[0]['status'] == 'success'
    by_method = {s['method']: s for s in summary}
    assert by_method['a']['n_failed'] == 1
    assert by_method['a']['n_success'] == 1


def test_structured_bound_unsupported_fails_explicitly():
    mb = _api()
    toy = ToyPredictor()
    # P=3 triangle but k=5 > P and > MAX_K=4 -> unsupported/infeasible
    g = _triangle('big-k')
    summary, rows = mb.evaluate_matched_budget(
        toy, [g], [StubScorer('a')],
        mb.BudgetSpec(pair_count=5, decoder='enumerated_structured_map'),
        seed=0, provenance=_provenance(), backbone='frozen')
    pruned = [r for r in rows if r['method'] == 'a']
    assert len(pruned) == 1
    assert pruned[0]['status'] == 'failed'
    assert pruned[0]['requested_pair_count'] == 5


def test_same_marks_transport_relabel_equivariance():
    mb = _api()
    ei = torch.tensor([[0, 1, 0, 2, 1, 2],
                       [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    g1 = _graph(ei, 3, 'relabel')
    perm = torch.tensor([2, 0, 1])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(3)
    g2 = Data(x=g1.x[perm].clone(),
              edge_index=inv[g1.edge_index].clone(),
              edge_attr=g1.edge_attr.clone())
    g2.cluster_id = 'relabel'
    g2.y = g1.y.clone()
    # tied scores: marks decide; transport marks by physical endpoints
    tied = StubScorer('tied', lambda layout: torch.zeros(len(layout.pairs)))
    layout1 = PhysicalPairLayout.from_edge_index(g1.edge_index, 3)
    marks1 = torch.tensor([0.1, 0.9, 0.5])
    # map layout1 pairs -> layout2 pairs through perm
    layout2 = PhysicalPairLayout.from_edge_index(g2.edge_index, 3)
    key_to_i = {tuple(sorted(map(int, layout1.pairs[i].tolist()))): i
                for i in range(len(layout1.pairs))}
    fwd = []
    for i in range(len(layout2.pairs)):
        a, b = (int(v) for v in layout2.pairs[i].tolist())
        fwd.append(key_to_i[tuple(sorted((int(perm[a]), int(perm[b]))))])
    fwd = torch.tensor(fwd)
    marks2 = marks1[fwd]
    _, rows1 = mb.evaluate_matched_budget(
        ToyPredictor(), [g1], [tied], mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen',
        pair_marks_by_id={'relabel': marks1})
    _, rows2 = mb.evaluate_matched_budget(
        ToyPredictor(), [g2], [tied], mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen',
        pair_marks_by_id={'relabel': marks2})
    r1 = [r for r in rows1 if r['method'] == 'tied'][0]
    r2 = [r for r in rows2 if r['method'] == 'tied'][0]
    assert r1['status'] == r2['status'] == 'success'
    # selected physical endpoints must correspond through the relabeling.
    # Compare in the old label frame: r2 pairs (new labels) map back via
    # new[i] = old[perm[i]], i.e. new (a,b) -> old (perm[a],perm[b]).
    s1 = {tuple(sorted(map(int, layout1.pairs[i].tolist())))
          for i in r1['selected_pairs']}
    s2 = {tuple(sorted(map(int, layout2.pairs[i].tolist())))
          for i in r2['selected_pairs']}
    s2_as_old = {tuple(sorted((int(perm[a]), int(perm[b])))) for a, b in s2}
    assert s1 == s2_as_old


def test_rng_order_invariance_by_graph_ids():
    mb = _api()
    gs = [_triangle('id1'), _triangle('id2'), _triangle('id3')]
    scorers = [StubScorer('a')]
    _, rows_fwd = mb.evaluate_matched_budget(
        ToyPredictor(), gs, scorers, mb.BudgetSpec(pair_count=2),
        seed=123, provenance=_provenance(), backbone='frozen')
    _, rows_rev = mb.evaluate_matched_budget(
        ToyPredictor(), list(reversed(gs)), scorers,
        mb.BudgetSpec(pair_count=2),
        seed=123, provenance=_provenance(), backbone='frozen')
    by_id_fwd = {r['graph_id']: r for r in rows_fwd if r['method'] == 'a'}
    by_id_rev = {r['graph_id']: r for r in rows_rev if r['method'] == 'a'}
    for gid in ('id1', 'id2', 'id3'):
        assert by_id_fwd[gid]['selected_pairs'] == by_id_rev[gid]['selected_pairs']
        assert by_id_fwd[gid]['pair_marks'] == by_id_rev[gid]['pair_marks']
        assert by_id_fwd[gid]['pruned_prediction'] == \
            by_id_rev[gid]['pruned_prediction']


def test_model_change_gives_fresh_predictions():
    mb = _api()
    g = _triangle('fresh')
    _, rows1 = mb.evaluate_matched_budget(
        ToyPredictor(weight=2.0), [g], [StubScorer('a')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    _, rows2 = mb.evaluate_matched_budget(
        ToyPredictor(weight=5.0), [g], [StubScorer('a')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    f1 = [r for r in rows1 if r['method'] == 'full'][0]['full_prediction']
    f2 = [r for r in rows2 if r['method'] == 'full'][0]['full_prediction']
    assert f1 != f2
    # changed graph content also changes prediction (no stale cache)
    g3 = _triangle('fresh')
    g3.edge_attr = g3.edge_attr * 3.0
    _, rows3 = mb.evaluate_matched_budget(
        ToyPredictor(weight=2.0), [g3], [StubScorer('a')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    f3 = [r for r in rows3 if r['method'] == 'full'][0]['full_prediction']
    assert f3 != f1


def test_missing_target_explicit_failed_rows_with_costs():
    mb = _api()
    toy = ToyPredictor()
    g = _triangle('no-target')
    g.y = torch.tensor([float('nan')])
    summary, rows = mb.evaluate_matched_budget(
        toy, [g], [StubScorer('a')], mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    assert len(rows) >= 2  # full + pruned preserved
    for r in rows:
        assert r['status'] == 'failed'
        assert r['graph_id'] == 'no-target'
        assert isinstance(r['reason'], str) and r['reason']
    by_method = {s['method']: s for s in summary}
    assert by_method['a']['n_failed'] == 1


def test_metrics_recomputed_and_common_gating():
    mb = _api()
    graphs = [_triangle('m1'), _triangle('m2'), _triangle('m3')]
    graphs[0].y = torch.tensor([10.0])
    graphs[1].y = torch.tensor([20.0])
    graphs[2].y = torch.tensor([30.0])

    class FailOne(StubScorer):
        def score(self, graph, layout, k, *, meter=None,
                  random_scores=None, generator=None):
            if graph.cluster_id == 'm3':
                raise ValueError('controlled scorer failure')
            return super().score(graph, layout, k, meter=meter,
                                 random_scores=random_scores,
                                 generator=generator)

    summary, rows = mb.evaluate_matched_budget(
        ToyPredictor(), graphs, [StubScorer('good'), FailOne('flaky')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    by_method = {s['method']: s for s in summary}
    assert by_method['flaky']['n_success'] == 2
    assert by_method['flaky']['n_failed'] == 1
    assert by_method['good']['n_success'] == 3
    # success sets differ -> unqualified comparison flagged
    assert by_method['good'].get('unqualified_comparison') is True or \
        by_method['flaky'].get('unqualified_comparison') is True
    # independently recompute RMSE for 'good' from emitted predictions
    good_rows = [r for r in rows if r['method'] == 'good'
                 and r['status'] == 'success']
    se = [(r['pruned_prediction'] - {'m1': 10.0, 'm2': 20.0, 'm3': 30.0}[
        r['graph_id']]) ** 2 for r in good_rows]
    expected_rmse = math.sqrt(sum(se) / len(se))
    assert by_method['good']['rmse'] == pytest.approx(expected_rmse, rel=1e-6)
    # zero-variance target slice -> R2 null with reason (check helper path:
    # single-graph summary has null R2)
    s1, _ = mb.evaluate_matched_budget(
        ToyPredictor(), [_triangle('solo')], [StubScorer('a')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    solo = {s['method']: s for s in s1}['a']
    assert solo['r2'] is None
    assert solo['r2_reason']


def test_paired_grouped_bootstrap_and_single_block_unavailable():
    mb = _api()
    # two groups A (3 graphs incl. replicated view) and B (2 graphs)
    graphs = []
    for i, grp in (('a1', 'A'), ('a2', 'A'), ('a3', 'A'), ('b1', 'B'),
                   ('b2', 'B')):
        g = _triangle(i)
        g.lineage_group = grp
        g.y = torch.tensor([float(ord(i[0]) + len(i))])
        graphs.append(g)
    scorers = [StubScorer('L', lambda layout: torch.arange(
        len(layout.pairs), dtype=torch.float32)),
        StubScorer('R', lambda layout: -torch.arange(
            len(layout.pairs), dtype=torch.float32))]
    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), graphs, scorers, mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    out = mb.summarize_paired(rows, 'L', 'R', group_by='lineage_group',
                              seed=0, n_bootstrap=200, confidence=0.95)
    assert isinstance(out, list) and len(out) >= 1
    pooled = out[0]
    assert pooled['n_paired'] == 5
    assert pooled['ci_low'] is not None and pooled['ci_high'] is not None
    assert pooled['n_blocks'] == 2
    # within-group replicated views travel together: bootstrap resamples
    # whole groups, so ci must differ from a naive per-graph bootstrap;
    # at minimum the method records block treatment honestly.
    assert pooled['bootstrap_unit'] == 'lineage_group'
    # single block -> unavailable, not spuriously precise
    solo_graphs = []
    for i in ('s1', 's2'):
        g = _triangle(i)
        g.lineage_group = 'only'
        g.y = torch.tensor([1.0])
        solo_graphs.append(g)
    _, rows_solo = mb.evaluate_matched_budget(
        ToyPredictor(), solo_graphs, scorers, mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    out_solo = mb.summarize_paired(rows_solo, 'L', 'R',
                                   group_by='lineage_group', seed=0,
                                   n_bootstrap=200)
    assert out_solo[0]['ci_low'] is None
    assert out_solo[0]['ci_reason']


def test_json_safe_truthful_provenance():
    mb = _api()
    toy = ToyPredictor()
    graphs = [_triangle('j1'), _triangle('j2')]
    summary, rows = mb.evaluate_matched_budget(
        toy, graphs, [StubScorer('a')], mb.BudgetSpec(pair_count=2),
        seed=42, provenance=_provenance('real-label'), backbone='frozen')
    blob = json.dumps({'summary': summary, 'rows': rows}, allow_nan=False)
    assert json.loads(blob)['rows']
    for r in rows:
        assert r['research_label'] == 'real-label'
        assert r['backbone_stage'] == 'frozen'
        assert r['model_state_hash'] == hashlib.sha256(b'').hexdigest() \
            or isinstance(r['model_state_hash'], str)
        assert isinstance(r['seed'], int) and r['seed'] == 42
        assert 'budget_config_hash' in r and r['budget_config_hash']
        assert 'evaluator_file_hash' in r and r['evaluator_file_hash']
    assert 'real-label' in blob


def test_tiny_cosmicnet_integration():
    mb = _api()
    from model.model import CosmicNetGNN
    config = {
        'model': {'node_features': 2, 'edge_features': 2, 'hidden_dim': 8,
                  'output_dim': 8, 'num_layers': 1, 'dropout': 0.0,
                  'pooling': 'mean', 'residual': False,
                  'activation': 'relu', 'mc_dropout': False},
        'graph': {'edge_features': ['distance', 'delta_v']},
    }
    model = CosmicNetGNN(config)
    model.eval()
    graphs = [_triangle('c1')]
    summary, rows = mb.evaluate_matched_budget(
        model, graphs, [StubScorer('a')], mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    ok = [r for r in rows if r['method'] == 'a']
    assert ok and ok[0]['status'] == 'success'
    assert math.isfinite(ok[0]['pruned_prediction'])
    json.dumps({'s': summary, 'r': rows}, allow_nan=False)


def test_independent_seeded_random_draws_stable():
    mb = _api()
    graphs = [_triangle('r1'), _triangle('r2')]

    class RandScorer(StubScorer):
        def score(self, graph, layout, k, *, meter=None,
                  random_scores=None, generator=None):
            assert random_scores is not None
            assert random_scores.shape == (len(layout.pairs),)
            return _ScoreOut(random_scores.clone(),
                             {'scorer': 'rand',
                              'random_scores': random_scores.clone()})

    _, rows1 = mb.evaluate_matched_budget(
        ToyPredictor(), graphs, [RandScorer('rand')],
        mb.BudgetSpec(pair_count=2),
        seed=99, provenance=_provenance(), backbone='frozen')
    _, rows2 = mb.evaluate_matched_budget(
        ToyPredictor(), graphs, [RandScorer('rand')],
        mb.BudgetSpec(pair_count=2),
        seed=99, provenance=_provenance(), backbone='frozen')
    v1 = {r['graph_id']: r['scores'] for r in rows1 if r['method'] == 'rand'}
    v2 = {r['graph_id']: r['scores'] for r in rows2 if r['method'] == 'rand'}
    assert v1['r1'] == v2['r1'] and v1['r2'] == v2['r2']
    assert v1['r1'] != v1['r2']  # independent per-graph draws
    # explicit transport overrides the draw
    vec = torch.tensor([0.7, -0.4, 0.1])
    _, rows3 = mb.evaluate_matched_budget(
        ToyPredictor(), [_triangle('r1')], [RandScorer('rand')],
        mb.BudgetSpec(pair_count=2),
        seed=99, provenance=_provenance(), backbone='frozen',
        random_scores_by_id={'r1': vec})
    got = [r for r in rows3 if r['method'] == 'rand'][0]
    assert got['scores'] == pytest.approx([0.7, -0.4, 0.1])


# ---------------- task-3b review regressions (RED before fix) ----------------

def test_red_failed_scoring_preserves_meter_costs():
    mb = _api()
    g = _triangle('cost-fail')

    class FailAfterCall(StubScorer):
        def score(self, graph, layout, k, *, meter=None,
                  random_scores=None, generator=None):
            meter.call(graph, phase='scoring')
            raise ValueError('controlled scoring failure after meter call')

    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), [g], [FailAfterCall('boom')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    row = [r for r in rows if r['method'] == 'boom'][0]
    assert row['status'] == 'failed'
    assert row['live_forward_calls'] >= 1
    assert row['scoring_forward_calls'] >= 1
    assert row['total_region_time_s'] > 0


def test_red_success_total_region_time_covers_full_region():
    mb = _api()
    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), [_triangle('region')],
        [StubScorer('a')], mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    row = [r for r in rows if r['method'] == 'a'][0]
    frag = row['scoring_time_s'] + row['decoder_time_s'] + row['prediction_time_s']
    assert row['total_region_time_s'] >= frag


def test_red_graph_content_hash_covers_stellar_mass():
    mb = _api()
    g1 = _triangle('hash-stellar')
    g1.stellar_mass = torch.tensor([1.0, 2.0, 3.0])
    g2 = _triangle('hash-stellar')
    g2.stellar_mass = torch.tensor([9.0, 2.0, 3.0])
    h1 = mb._graph_content_hash(g1, 'hash-stellar')
    h2 = mb._graph_content_hash(g2, 'hash-stellar')
    assert h1 != h2


def test_red_required_outputs_present():
    mb = _api()
    g = _triangle('outputs')
    g.pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    g.stellar_mass = torch.tensor([1.0, 2.0, 3.0])
    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), [g], [StubScorer('a')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    row = [r for r in rows if r['method'] == 'a'][0]
    assert row.get('scorer_metadata') == {'scorer': 'a'}
    assert row['requested_physical_retention'] == pytest.approx(2 / 3)
    assert row['physical_retention'] == pytest.approx(2 / 3)
    assert row['available_nonself_columns'] == 6
    assert row['dataset_content_hash']
    assert row['catalog_contract_sha256']
    assert row['split_manifest_sha256']
    assert row['config_sha256']
    full = [r for r in rows if r['method'] == 'full'][0]
    assert full['physical_retention'] == pytest.approx(1.0)
    assert full['physical_isolates'] is not None


def test_red_common_metrics_and_headline_flag():
    mb = _api()
    graphs = [_triangle('m1'), _triangle('m2'), _triangle('m3')]
    graphs[0].y = torch.tensor([10.0])
    graphs[1].y = torch.tensor([20.0])
    graphs[2].y = torch.tensor([30.0])

    class FailOne(StubScorer):
        def score(self, graph, layout, k, *, meter=None,
                  random_scores=None, generator=None):
            if graph.cluster_id == 'm3':
                raise ValueError('controlled scorer failure')
            return super().score(graph, layout, k, meter=meter,
                                 random_scores=random_scores,
                                 generator=generator)

    summary, _ = mb.evaluate_matched_budget(
        ToyPredictor(), graphs, [StubScorer('good'), FailOne('flaky')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    by_method = {s['method']: s for s in summary}
    assert by_method['good']['headline_comparison_allowed'] is False
    assert by_method['good']['headline_comparison_reason']
    assert by_method['good']['common_r2'] is not None \
        or by_method['good']['common_r2_reason']
    assert by_method['good']['common_bias'] is not None \
        or by_method['good']['common_bias_reason']
    # explicit zero artifact cost stays zero with a dedup map; unknown costs
    # must not silently read as zero
    assert by_method['good']['historical_training_calls'] == 0
    assert 'historical_artifact_cost_map' in by_method['good']

    class UnknownCost(StubScorer):
        def training_cost(self):
            raise ValueError('no artifact')

    summary2, _ = mb.evaluate_matched_budget(
        ToyPredictor(), [_triangle('u1')], [UnknownCost('unknown')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    by2 = {s['method']: s for s in summary2}
    assert by2['unknown']['historical_training_calls'] is None


def test_red_paired_missing_group_gives_unavailable_ci():
    mb = _api()
    graphs = []
    for i, grp in (('p1', 'A'), ('p2', 'B'), ('p3', None)):
        g = _triangle(i)
        if grp is not None:
            g.lineage_group = grp
        graphs.append(g)
    scorers = [StubScorer('L'), StubScorer('R')]
    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), graphs, scorers, mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    out = mb.summarize_paired(rows, 'L', 'R', group_by='lineage_group',
                              seed=0, n_bootstrap=50)
    assert out[0]['ci_low'] is None and out[0]['ci_reason']


def test_red_paired_rejects_mismatched_targets():
    mb = _api()
    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), [_triangle('q1'), _triangle('q2')],
        [StubScorer('L'), StubScorer('R')],
        mb.BudgetSpec(pair_count=2),
        seed=0, provenance=_provenance(), backbone='frozen')
    tampered = [dict(r) for r in rows]
    for r in tampered:
        if r['method'] == 'R' and r['graph_id'] == 'q1':
            r['target'] = float(r['target']) + 5.0
    with pytest.raises(ValueError, match='target|lineage|content|identity'):
        mb.summarize_paired(tampered, 'L', 'R', group_by='lineage_group',
                            seed=0, n_bootstrap=10)


def test_red_paired_q_config_single_stratum():
    mb = _api()
    # P=3 triangle vs P=6 larger graph under same keep_fraction -> one stratum
    ei6 = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                        [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    big = _graph(ei6, 4, 'big')
    small = _triangle('small')
    _, rows = mb.evaluate_matched_budget(
        ToyPredictor(), [small, big], [StubScorer('L'), StubScorer('R')],
        mb.BudgetSpec(keep_fraction=0.5, constraint='none'),
        seed=0, provenance=_provenance(), backbone='frozen')
    out = mb.summarize_paired(rows, 'L', 'R', group_by='lineage_group',
                              seed=0, n_bootstrap=10)
    assert len(out) == 1
    assert out[0]['n_paired'] == 2


# ---------------- real-adapter integration (task-3b review item 6) ----------------

class AnalyticPredictor(nn.Module):
    """Literal analytic predictor with embeddings + edge grads.

    prediction = w * edge_attr.sum() + 0.5 * x.sum();
    embeddings = Linear(x) per node (for provided_rl).
    """

    def __init__(self, emb_dim=4):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(2.0))
        self.proj = nn.Linear(2, emb_dim)

    def forward(self, batch, return_embeddings=False):
        pred = self.w * batch.edge_attr.sum() + 0.5 * batch.x.sum()
        if return_embeddings:
            return pred, self.proj(batch.x)
        return pred


def _rich_graph(ei, n_nodes, cid, y_val=1.0):
    E = ei.shape[1]
    g = Data(x=torch.ones(n_nodes, 2),
             edge_index=ei.clone(),
             edge_attr=torch.ones(E, 2))
    g.cluster_id = cid
    g.y = torch.tensor([float(y_val)])
    g.pos = torch.arange(n_nodes * 2, dtype=torch.float32).reshape(n_nodes, 2)
    g.stellar_mass = torch.tensor([1.0 + i for i in range(n_nodes)])
    g.lineage_group = 'synthetic-lineage'
    return g


def _real_fixtures():
    tri = torch.tensor([[0, 1, 0, 2, 1, 2],
                        [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    oneway = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    dup = torch.tensor([[0, 1, 0, 2, 1, 2, 0],
                        [1, 0, 2, 0, 2, 1, 1]], dtype=torch.long)
    loop = torch.tensor([[0, 1, 0, 2, 1, 2, 0],
                         [1, 0, 2, 0, 2, 1, 0]], dtype=torch.long)
    return [
        _rich_graph(tri, 3, 'reciprocal'),
        _rich_graph(oneway, 3, 'oneway'),
        _rich_graph(dup, 3, 'duplicate'),
        _rich_graph(loop, 3, 'loops'),
    ]


def _real_scorers():
    from rls.benchmark_scorers import PairScorer
    from rls.policy import EdgePolicyNet
    policy = EdgePolicyNet(edge_dim=2, node_emb_dim=4, hidden_dim=8)
    return [
        PairScorer('random'),
        PairScorer('distance'),
        PairScorer('degree'),
        PairScorer('stellar_binding_proxy', distance_epsilon=1e-3,
                   distance_unit='kpc', stellar_mass_unit='Msun'),
        PairScorer('edge_feature_grad_x_input'),
        PairScorer('provided_rl', policy=policy,
                   edge_feature_names=['distance', 'delta_v']),
    ]


def _trained_supervised_scorer(graphs, k=2):
    from rls.benchmark_scorers import PairScorer
    from rls.raw_selector import (
        RawBudgetPairScorer, train_supervised_selector)
    raw = RawBudgetPairScorer(node_dim=2,
                              edge_feature_names=['distance', 'delta_v'])
    split_hash = 'synthetic-split-hash'
    norm_examples = []
    train_examples = []
    for g in graphs[:2]:
        layout = PhysicalPairLayout.from_edge_index(g.edge_index, 3)
        P = len(layout.pairs)
        gdict = {'edge_index': g.edge_index, 'edge_attr': g.edge_attr,
                 'x': g.x, 'cluster_id': g.cluster_id}
        norm_examples.append((dict(gdict), k))
        train_examples.append({
            'graph': dict(gdict), 'k': k,
            'labels': torch.zeros(P, dtype=torch.float32),
            'label_definition': 'synthetic-zero-labels',
            'label_predictor_calls': 1,
            'label_provenance': {'synthetic': True, 'calls': 1},
        })
    raw.fit_normalization(norm_examples,
                          [graphs[0].cluster_id, graphs[1].cluster_id], [],
                          split_hash)
    artifact = train_supervised_selector(
        raw, train_examples,
        [graphs[0].cluster_id, graphs[1].cluster_id], [],
        split_hash, seed=0, epochs=1, lr=1e-2)
    return (PairScorer('supervised_raw', raw_scorer=raw,
                       raw_artifact=artifact, split_hash=split_hash),
            artifact)


def test_real_adapters_all_fixtures_costs_and_metadata():
    mb = _api()
    graphs = _real_fixtures()
    scorers = _real_scorers()
    sup, artifact = _trained_supervised_scorer(graphs)
    scorers.append(sup)
    label_cost = int(artifact['label_provenance']['total_calls'])
    assert label_cost == 2
    summary, rows = mb.evaluate_matched_budget(
        AnalyticPredictor(), graphs, scorers,
        mb.BudgetSpec(pair_count=2, constraint='none'),
        seed=0, provenance=_provenance(), backbone='frozen')
    for g in graphs:
        full = [r for r in rows
                if r['graph_id'] == g.cluster_id and r['method'] == 'full'][0]
        assert full['status'] == 'success'
        assert full['full_forward_calls'] == 1
    by_method = {(r['graph_id'], r['method']): r for r in rows}
    for scorer in scorers:
        for g in graphs:
            row = by_method[(g.cluster_id, scorer.name)]
            assert row['status'] == 'success', (scorer.name, row.get('reason'))
            assert row['final_pair_count'] == 2
            assert row['eval_forward_calls'] == 1
            assert row['physical_retention'] == pytest.approx(2 / 3)
            assert row['scorer_metadata']
            assert row['dataset_content_hash']
    # live-call truth: saliency 1, provided_rl embedding 1, rest 0
    expected_scoring = {'random': 0, 'distance': 0, 'degree': 0,
                        'stellar_binding_proxy': 0,
                        'edge_feature_grad_x_input': 1, 'provided_rl': 1,
                        'supervised_raw': 0}
    for name, want in expected_scoring.items():
        got = by_method[('reciprocal', name)]['scoring_forward_calls']
        assert got == want, (name, got)
    rl_meta = by_method[('reciprocal', 'provided_rl')]['scorer_metadata']
    assert rl_meta['policy_passes'] == 2
    assert rl_meta['predictor_embedding_calls'] == 1
    assert rl_meta['edge_feature_names'] == ['distance', 'delta_v']
    bind_meta = by_method[('reciprocal',
                           'stellar_binding_proxy')]['scorer_metadata']
    assert bind_meta['distance_epsilon'] == pytest.approx(1e-3)
    assert bind_meta['distance_unit'] == 'kpc'
    sal_meta = by_method[('reciprocal',
                          'edge_feature_grad_x_input')]['scorer_metadata']
    assert 'edge_saliency_status' in sal_meta or \
        'unused_edge_input' in sal_meta
    # mask expansion on the loop fixture uses nonself indices
    loop_row = by_method[('loops', 'degree')]
    mask = torch.tensor(loop_row['stored_mask'])
    inv = torch.tensor(loop_row['layout_inverse'])
    real = torch.tensor(loop_row['stored_real'])
    real_idx = real.nonzero().flatten()
    assert len(inv) == int(real.sum())
    for p in torch.tensor(loop_row['selected_pairs']).tolist():
        cols = real_idx[(inv == p).nonzero().flatten()]
        assert bool(mask[cols].all())
    assert bool(mask[~real].all())
    blob = json.dumps({'summary': summary, 'rows': rows}, allow_nan=False)
    assert 'synthetic-lineage' in blob


def test_real_structured_map_scaffold_and_unsupported_row():
    mb = _api()
    graphs = _real_fixtures()[:2]
    scorers = _real_scorers()[:3]
    for decoder in ('enumerated_structured_map', 'scaffold'):
        summary, rows = mb.evaluate_matched_budget(
            AnalyticPredictor(), graphs, scorers,
            mb.BudgetSpec(pair_count=2, constraint='none', decoder=decoder),
            seed=0, provenance=_provenance(), backbone='frozen')
        for r in rows:
            if r['method'] in ('random', 'distance', 'degree'):
                assert r['status'] == 'success', (decoder, r.get('reason'))
                assert r['final_pair_count'] == 2
    # unsupported bounded budget: k=5 exceeds structured MAX_K=4
    _, rows = mb.evaluate_matched_budget(
        AnalyticPredictor(), graphs[:1], scorers[:1],
        mb.BudgetSpec(pair_count=5, constraint='none',
                      decoder='enumerated_structured_map'),
        seed=0, provenance=_provenance(), backbone='frozen')
    row = [r for r in rows if r['method'] == 'random'][0]
    assert row['status'] == 'failed'
    assert row['requested_pair_count'] == 5
    assert 'unsupported' in row['reason'].lower() or \
        'exceed' in row['reason'].lower() or 'max_k' in row['reason'].lower()


def test_real_feasible_no_isolates_exact_k():
    mb = _api()
    graphs = _real_fixtures()[:2]
    summary, rows = mb.evaluate_matched_budget(
        AnalyticPredictor(), graphs, _real_scorers()[:2],
        mb.BudgetSpec(pair_count=2, constraint='no_isolates'),
        seed=0, provenance=_provenance(), backbone='frozen')
    for r in rows:
        if r['method'] in ('random', 'distance'):
            assert r['status'] == 'success'
            assert r['constraint_satisfied'] is True
            assert r['budget_satisfied'] is True


def test_cuda_end_to_end_or_honest_skip():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device; honest skip of CUDA end-to-end')
    mb = _api()
    g = _rich_graph(torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
        dtype=torch.long), 3, 'cuda-tiny')
    for key in ('x', 'edge_index', 'edge_attr', 'pos', 'stellar_mass'):
        t = getattr(g, key)
        setattr(g, key, t.cuda())
    _, rows = mb.evaluate_matched_budget(
        AnalyticPredictor().cuda(), [g], _real_scorers()[:2],
        mb.BudgetSpec(pair_count=2, constraint='none'),
        seed=0, provenance=_provenance(), backbone='frozen')
    assert [r for r in rows if r['method'] == 'random'][0]['status'] == \
        'success'
