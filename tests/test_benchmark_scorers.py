"""TESTS ONLY for rls.benchmark_scorers.PairScorer (Task 3b2 red phase).

Contract under test (see task-3b-scorers-handoff.md):
- PairScoreResult(scores: Tensor[P], metadata: dict).
- PairScorer(name, *, policy=None, raw_scorer=None, raw_artifact=None,
             split_hash=None, edge_feature_names=None, distance_epsilon=None,
             distance_unit=None, stellar_mass_unit=None).
- .score(graph, layout, k, *, meter=None, random_scores=None, generator=None)
  -> PairScoreResult.
- .training_cost(): artifact-level label cost for supervised_raw, zero otherwise.

Graphs here are single PyG Data objects (meter-compatible) carrying x,
edge_index, edge_attr, pos, stellar_mass and cluster_id. Layouts are
rls.constraints.PhysicalPairLayout. These tests run red (missing-API
failure) until rls/benchmark_scorers.py exists. No production code here.
"""
import copy
import importlib.util

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from rls.constraints import PhysicalPairLayout

SCORER_MODULE = 'rls.benchmark_scorers'
VALID_NAMES = {'random', 'distance', 'degree', 'stellar_binding_proxy',
               'edge_feature_grad_x_input', 'provided_rl', 'supervised_raw'}


def _api():
    assert importlib.util.find_spec(SCORER_MODULE) is not None, \
        'benchmark_scorers API missing'
    from rls import benchmark_scorers as bs
    assert hasattr(bs, 'PairScorer'), 'PairScorer API missing'
    assert hasattr(bs, 'PairScoreResult'), 'PairScoreResult API missing'
    return bs


# ---------------- literal graph fixtures ----------------

def _reciprocal_graph():
    """3 nodes, reciprocal pairs (0,1),(0,2),(1,2); P=3, sorted pairs."""
    pos = torch.tensor([[0., 0.], [3., 4.], [6., 0.]])
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    stellar = torch.tensor([1e10, 2e10, 4e10])
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2],
                               [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[5.0, 0.5], [5.0, -0.5],
                              [6.0, 0.25], [6.0, -0.25],
                              [5.0, 0.1], [5.0, -0.1]])
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             pos=pos, stellar_mass=stellar)
    g.cluster_id = 'recip-g'
    layout = PhysicalPairLayout.from_edge_index(edge_index, 3)
    return g, layout


def _messy_graph():
    """Duplicates + self-loops + one-way: cols (0,1),(0,1),(1,1),(2,2),(1,2)."""
    pos = torch.tensor([[0., 0.], [3., 4.], [6., 0.]])
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    stellar = torch.tensor([1e10, 2e10, 4e10])
    edge_index = torch.tensor([[0, 0, 1, 2, 1],
                               [1, 1, 1, 2, 2]], dtype=torch.long)
    edge_attr = torch.tensor([[5.0, 0.5], [5.0, 0.5],
                              [0.0, 0.0], [0.0, 0.0],
                              [5.0, 0.1]])
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             pos=pos, stellar_mass=stellar)
    g.cluster_id = 'messy-g'
    layout = PhysicalPairLayout.from_edge_index(edge_index, 3)
    assert len(layout.pairs) == 2  # (0,1),(1,2); loops excluded
    return g, layout


def _clone_data(g):
    return copy.deepcopy(g)


def _pair_index_map(pairs_old, pairs_new, perm):
    """Map each row of pairs_new (new node IDs) to the matching row of
    pairs_old (old node IDs). perm[i] is the old node now labeled i, so a
    new pair (a, b) corresponds to old endpoints (perm[a], perm[b])."""
    perm = [int(v) for v in perm]
    key_to_i = {tuple(sorted(map(int, pairs_old[i].tolist()))): i
                for i in range(len(pairs_old))}
    fwd = []
    for i in range(len(pairs_new)):
        a, b = (int(v) for v in pairs_new[i].tolist())
        fwd.append(key_to_i[tuple(sorted((perm[a], perm[b])))])
    return torch.tensor(fwd, dtype=torch.long)


def _permuted_graph(g, perm):
    """Relabel nodes by perm (new[i] = old[perm[i]]); transport all tensors."""
    perm = torch.tensor(list(perm), dtype=torch.long)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    g2 = Data(x=g.x[perm].clone(),
              edge_index=inv[g.edge_index].clone(),
              edge_attr=g.edge_attr.clone(),
              pos=g.pos[perm].clone(),
              stellar_mass=g.stellar_mass[perm].clone())
    g2.cluster_id = g.cluster_id
    return g2


# ---------------- API shape ----------------

def test_api_importable():
    _api()


def test_unsupported_name_raises():
    bs = _api()
    with pytest.raises(ValueError, match='unknown|unsupported|name'):
        bs.PairScorer('not_a_scorer')


def test_result_fields_scores_and_metadata():
    bs = _api()
    g, layout = _reciprocal_graph()
    out = bs.PairScorer('distance').score(g, layout, 1)
    assert hasattr(out, 'scores') and hasattr(out, 'metadata')
    assert isinstance(out.scores, torch.Tensor)
    assert isinstance(out.metadata, dict)
    assert out.scores.shape == (len(layout.pairs),)
    assert torch.is_floating_point(out.scores)
    assert torch.isfinite(out.scores).all()


def test_training_cost_zero_for_non_supervised():
    bs = _api()
    for name in ['random', 'distance', 'degree']:
        cost = bs.PairScorer(name).training_cost()
        assert isinstance(cost, dict), 'training_cost must return a dict'
        assert cost['total_calls'] == 0


# ---------------- random ----------------

def test_random_replays_provided_vector_and_metadata():
    bs = _api()
    g, layout = _reciprocal_graph()
    vec = torch.tensor([0.3, -1.2, 2.0])
    out = bs.PairScorer('random').score(g, layout, 2, random_scores=vec)
    torch.testing.assert_close(out.scores, vec)
    assert out.scores.shape == (3,)
    blob = ' '.join(map(str, out.metadata.keys())) + ' ' + \
        ' '.join(map(str, out.metadata.values()))
    assert 'random' in blob.lower()
    found = False
    for v in out.metadata.values():
        if isinstance(v, torch.Tensor) and v.shape == vec.shape:
            torch.testing.assert_close(v.detach().cpu(), vec)
            found = True
    assert found, 'metadata must persist the actual returned random vector'


def test_random_requires_explicit_source_no_global_draw():
    bs = _api()
    g, layout = _reciprocal_graph()
    with pytest.raises(ValueError, match='random|generator|explicit|provided'):
        bs.PairScorer('random').score(g, layout, 1)
    with pytest.raises(ValueError, match='random|shape|length|finite'):
        bs.PairScorer('random').score(
            g, layout, 1, random_scores=torch.tensor([0.1]))


def test_random_generator_seeded_replay_without_global_rng():
    bs = _api()
    g, layout = _reciprocal_graph()
    rng_before = torch.get_rng_state().clone()
    gen1 = torch.Generator()
    gen1.manual_seed(1234)
    out1 = bs.PairScorer('random').score(g, layout, 1, generator=gen1)
    gen2 = torch.Generator()
    gen2.manual_seed(1234)
    out2 = bs.PairScorer('random').score(g, layout, 1, generator=gen2)
    torch.testing.assert_close(out1.scores, out2.scores)
    assert torch.isfinite(out1.scores).all()
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_random_rejects_nonfinite_vector():
    bs = _api()
    g, layout = _reciprocal_graph()
    with pytest.raises(ValueError, match='finite'):
        bs.PairScorer('random').score(
            g, layout, 1,
            random_scores=torch.tensor([1.0, float('nan'), 0.5]))


# ---------------- distance ----------------

def test_distance_exact_values_reciprocal():
    bs = _api()
    g, layout = _reciprocal_graph()
    out = bs.PairScorer('distance').score(g, layout, 2)
    # pairs sorted: (0,1) d=5, (0,2) d=6, (1,2) d=5
    torch.testing.assert_close(out.scores,
                               torch.tensor([-5., -6., -5.]),
                               atol=1e-6, rtol=1e-6)


def test_distance_messy_graph_duplicates_and_loops_ignored():
    bs = _api()
    g, layout = _messy_graph()
    out = bs.PairScorer('distance').score(g, layout, 1)
    # pairs (0,1) d=5, (1,2) d=5
    torch.testing.assert_close(out.scores, torch.tensor([-5., -5.]),
                               atol=1e-6, rtol=1e-6)


def test_distance_requires_pos_no_fallback():
    bs = _api()
    g, layout = _reciprocal_graph()
    g_no_pos = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
    g_no_pos.cluster_id = 'no-pos'
    with pytest.raises(ValueError, match='pos'):
        bs.PairScorer('distance').score(g_no_pos, layout, 1)


# ---------------- degree ----------------

def test_degree_exact_reciprocal():
    bs = _api()
    g, layout = _reciprocal_graph()
    out = bs.PairScorer('degree').score(g, layout, 1)
    # triangle: every node degree 2 -> every pair mean 2.0
    torch.testing.assert_close(out.scores, torch.tensor([2., 2., 2.]),
                               atol=1e-6, rtol=1e-6)


def test_degree_duplicates_and_selfloops_do_not_inflate():
    bs = _api()
    g, layout = _messy_graph()
    out = bs.PairScorer('degree').score(g, layout, 1)
    # physical degrees 0:1, 1:2, 2:1 -> pairs (0,1):1.5, (1,2):1.5
    torch.testing.assert_close(out.scores, torch.tensor([1.5, 1.5]),
                               atol=1e-6, rtol=1e-6)


# ---------------- stellar_binding_proxy ----------------

def _binding_expected(stellar, pos, eps):
    out = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for u, v in pairs:
        d = float(torch.linalg.vector_norm(pos[u] - pos[v]))
        out.append(float(torch.log(stellar[u]) + torch.log(stellar[v])
                         - torch.log(torch.tensor(max(d, eps)))))
    return torch.tensor(out)


def test_binding_exact_values_and_metadata():
    bs = _api()
    g, layout = _reciprocal_graph()
    scorer = bs.PairScorer('stellar_binding_proxy', distance_epsilon=1e-3,
                           distance_unit='kpc', stellar_mass_unit='Msun')
    out = scorer.score(g, layout, 1)
    torch.testing.assert_close(
        out.scores, _binding_expected(g.stellar_mass, g.pos, 1e-3),
        atol=1e-6, rtol=1e-6)
    blob_keys = ' '.join(map(str, out.metadata.keys())).lower()
    blob_vals = ' '.join(map(str, out.metadata.values()))
    assert 'kpc' in blob_vals and 'Msun' in blob_vals
    assert 'epsilon' in blob_keys
    assert 'halo' not in blob_keys + ' '.join(
        map(str, out.metadata.values())).lower().replace('total-halo', ''), \
        'must never label the proxy total-halo binding'


def test_binding_rejects_missing_or_invalid_inputs():
    bs = _api()
    g, layout = _reciprocal_graph()
    good = dict(distance_epsilon=1e-3, distance_unit='kpc',
                stellar_mass_unit='Msun')
    # missing epsilon / units
    with pytest.raises(ValueError, match='epsilon|unit'):
        bs.PairScorer('stellar_binding_proxy', distance_unit='kpc',
                      stellar_mass_unit='Msun').score(g, layout, 1)
    with pytest.raises(ValueError, match='unit|epsilon'):
        bs.PairScorer('stellar_binding_proxy', distance_epsilon=1e-3,
                      stellar_mass_unit='Msun').score(g, layout, 1)
    with pytest.raises(ValueError, match='epsilon|positive|finite'):
        bs.PairScorer('stellar_binding_proxy', distance_epsilon=0.0,
                      **{k: v for k, v in good.items()
                         if k != 'distance_epsilon'}).score(g, layout, 1)
    # non-positive / nonfinite mass
    bad = _clone_data(g)
    bad.stellar_mass = torch.tensor([1e10, 0.0, 4e10])
    with pytest.raises(ValueError, match='stellar_mass|positive|finite'):
        bs.PairScorer('stellar_binding_proxy', **good).score(bad, layout, 1)
    bad2 = _clone_data(g)
    bad2.stellar_mass = torch.tensor([1e10, float('nan'), 4e10])
    with pytest.raises(ValueError, match='stellar_mass|finite|positive'):
        bs.PairScorer('stellar_binding_proxy', **good).score(bad2, layout, 1)
    # missing positions
    bad3 = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                stellar_mass=g.stellar_mass)
    bad3.cluster_id = 'no-pos'
    with pytest.raises(ValueError, match='pos'):
        bs.PairScorer('stellar_binding_proxy', **good).score(bad3, layout, 1)


def test_binding_clamps_tiny_distance_at_epsilon():
    bs = _api()
    pos = torch.tensor([[0., 0.], [1e-9, 0.]])
    x = torch.zeros(2, 2)
    stellar = torch.tensor([1e10, 1e10])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.zeros(1, 2)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             pos=pos, stellar_mass=stellar)
    g.cluster_id = 'close-g'
    layout = PhysicalPairLayout.from_edge_index(edge_index, 2)
    eps = 1e-3
    out = bs.PairScorer('stellar_binding_proxy', distance_epsilon=eps,
                        distance_unit='kpc',
                        stellar_mass_unit='Msun').score(g, layout, 1)
    expected = (torch.log(stellar[0]) + torch.log(stellar[1])
                - torch.log(torch.tensor(eps)))
    torch.testing.assert_close(out.scores, expected.reshape(1),
                               atol=1e-6, rtol=1e-6)


# ---------------- shared validation ----------------

def test_shared_k_and_layout_validation():
    bs = _api()
    g, layout = _reciprocal_graph()
    scorer = bs.PairScorer('distance')
    for bad_k in (-1, 4, 1.5, True):
        with pytest.raises(ValueError, match='k|budget'):
            scorer.score(g, layout, bad_k)
    foreign = PhysicalPairLayout.from_edge_index(
        torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long), 3)
    with pytest.raises(ValueError, match='layout|topology|match'):
        scorer.score(g, foreign, 1)


def test_layout_rejects_missing_isolated_node():
    bs = _api()
    g, layout = _reciprocal_graph()
    # actual graph has 4 nodes (node 3 isolated); layout only covers 3
    g4 = Data(x=torch.zeros(4, 2), edge_index=g.edge_index,
              edge_attr=g.edge_attr,
              pos=torch.zeros(4, 2),
              stellar_mass=torch.ones(4))
    g4.cluster_id = 'four-node'
    with pytest.raises(ValueError, match='num_nodes|isolated|N|topology|match'):
        bs.PairScorer('distance').score(g4, layout, 1)


def test_layout_rejects_mixed_tensor_devices():
    bs = _api()
    g, layout = _reciprocal_graph()
    bad = PhysicalPairLayout(pairs=layout.pairs,
                             inverse=layout.inverse.to('meta'),
                             real=layout.real, num_nodes=3)
    with pytest.raises(ValueError, match='device|layout'):
        bs.PairScorer('distance').score(g, bad, 1)


def test_no_input_mutation_heuristic_free_scores():
    bs = _api()
    g, layout = _reciprocal_graph()
    before = {k: getattr(g, k).clone()
              for k in ('x', 'edge_index', 'edge_attr', 'pos',
                        'stellar_mass')}
    vec = torch.tensor([0.1, 0.2, 0.3])
    bs.PairScorer('random').score(g, layout, 1, random_scores=vec)
    bs.PairScorer('distance').score(g, layout, 1)
    bs.PairScorer('degree').score(g, layout, 1)
    bs.PairScorer('stellar_binding_proxy', distance_epsilon=1e-3,
                  distance_unit='kpc',
                  stellar_mass_unit='Msun').score(g, layout, 1)
    for k, v in before.items():
        torch.testing.assert_close(getattr(g, k), v)


# ---------------- coupled relabeling ----------------

def test_coupled_relabel_transports_scores_by_endpoints():
    bs = _api()
    g, layout = _reciprocal_graph()
    perm = [2, 0, 1]
    g2 = _permuted_graph(g, perm)
    layout2 = PhysicalPairLayout.from_edge_index(g2.edge_index, 3)
    fwd = _pair_index_map(layout.pairs, layout2.pairs, perm)
    for name, kwargs in [
            ('distance', {}),
            ('degree', {}),
            ('stellar_binding_proxy',
             dict(distance_epsilon=1e-3, distance_unit='kpc',
                  stellar_mass_unit='Msun'))]:
        s1 = bs.PairScorer(name, **kwargs).score(g, layout, 2).scores
        s2 = bs.PairScorer(name, **kwargs).score(g2, layout2, 2).scores
        torch.testing.assert_close(s2, s1[fwd], atol=1e-6, rtol=1e-6)


def test_coupled_relabel_one_way_storage_and_random_transport():
    bs = _api()
    g, layout = _messy_graph()
    perm = [2, 0, 1]
    g2 = _permuted_graph(g, perm)
    layout2 = PhysicalPairLayout.from_edge_index(g2.edge_index, 3)
    fwd = _pair_index_map(layout.pairs, layout2.pairs, perm)
    s1 = bs.PairScorer('distance').score(g, layout, 1).scores
    s2 = bs.PairScorer('distance').score(g2, layout2, 1).scores
    torch.testing.assert_close(s2, s1[fwd], atol=1e-6, rtol=1e-6)
    vec = torch.tensor([0.7, -0.4])
    r1 = bs.PairScorer('random').score(g, layout, 1,
                                       random_scores=vec).scores
    r2 = bs.PairScorer('random').score(g2, layout2, 1,
                                       random_scores=vec[fwd]).scores
    torch.testing.assert_close(r2, r1[fwd], atol=1e-6, rtol=1e-6)


# ---------------- edge_feature_grad_x_input ----------------

class _SumEdgeToy(nn.Module):
    """prediction = w * calib[0] * sum(edge_attr); analytic grad w."""

    def __init__(self, weight=2.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(weight)))
        self.proj = nn.Linear(2, 2)
        self.register_buffer('calibration', torch.tensor([1.0, 2.0]))
        self.forward_calls = 0

    def forward(self, batch, return_embeddings=False):
        self.forward_calls += 1
        if return_embeddings:
            raise ValueError('toy has no embeddings')
        return self.w * self.calibration[0] * batch.edge_attr.sum()


class _NodeOnlyToy(nn.Module):
    """Prediction ignores edge_attr: scorer must flag unsupported/zero."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(2, 2)

    def forward(self, batch, return_embeddings=False):
        if return_embeddings:
            raise ValueError('toy has no embeddings')
        return self.w * batch.x.sum()


def test_grad_x_input_exact_copy_means_one_scoring_forward():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _reciprocal_graph()
    toy = _SumEdgeToy(weight=2.0)
    toy.eval()
    for p in toy.parameters():
        p.grad = torch.randn_like(p)
    grads_before = {n: p.grad.clone() for n, p in toy.named_parameters()}
    buffers_before = {n: b.clone() for n, b in toy.named_buffers()}
    modes_before = {n: m.training for n, m in toy.named_modules()}
    x_before, ea_before, ei_before = (g.x.clone(), g.edge_attr.clone(),
                                      g.edge_index.clone())
    meter = MeteredPredictor(toy)
    snap_before = meter.snapshot()
    out = bs.PairScorer('edge_feature_grad_x_input').score(
        g, layout, 2, meter=meter)
    snap_after = meter.snapshot()
    # exact copy-mean values: |2.0 * e|.sum(-1) per stored col, mean per pair
    per_col = (2.0 * g.edge_attr).abs().sum(dim=1)
    expected = torch.zeros(len(layout.pairs))
    counts = torch.zeros(len(layout.pairs))
    expected.index_add_(0, layout.inverse, per_col)
    counts.index_add_(0, layout.inverse,
                      torch.ones_like(per_col))
    expected = expected / counts.clamp_min(1)
    torch.testing.assert_close(out.scores, expected, atol=1e-5, rtol=1e-5)
    assert (out.scores > 0).all()
    # exactly ONE scoring forward, nothing else
    assert (snap_after['scoring']['forward_calls']
            - snap_before['scoring']['forward_calls']) == 1
    total_delta = sum(snap_after[p]['forward_calls']
                      - snap_before[p]['forward_calls']
                      for p in snap_after)
    assert total_delta == 1
    # preservation: params/grads/buffers/modes/caller graph untouched
    for n, p in toy.named_parameters():
        torch.testing.assert_close(p.grad, grads_before[n])
    for n, b in toy.named_buffers():
        torch.testing.assert_close(b, buffers_before[n])
    assert {n: m.training for n, m in toy.named_modules()} == modes_before
    assert torch.equal(g.x, x_before) and torch.equal(g.edge_attr, ea_before)
    assert torch.equal(g.edge_index, ei_before)
    assert g.x.grad is None and g.edge_attr.grad is None


def test_grad_x_input_messy_duplicates_and_loops_pair_means():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _messy_graph()
    meter = MeteredPredictor(_SumEdgeToy(weight=2.0))
    snap_before = meter.snapshot()
    out = bs.PairScorer('edge_feature_grad_x_input').score(
        g, layout, 1, meter=meter)
    # stored cols: [5,.5]->11, [5,.5]->11, loop->excluded, loop->excluded,
    # [5,.1]->10.2 ; pair means over nonself copies only
    torch.testing.assert_close(out.scores, torch.tensor([11.0, 10.2]),
                               atol=1e-5, rtol=1e-5)
    snap_after = meter.snapshot()
    assert (snap_after['scoring']['forward_calls']
            - snap_before['scoring']['forward_calls']) == 1
    total_delta = sum(snap_after[p]['forward_calls']
                      - snap_before[p]['forward_calls']
                      for p in snap_after)
    assert total_delta == 1


def test_grad_x_input_float64_dtype_preserved():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _reciprocal_graph()
    g64 = Data(x=g.x.double(), edge_index=g.edge_index,
               edge_attr=g.edge_attr.double(), pos=g.pos.double(),
               stellar_mass=g.stellar_mass.double())
    g64.cluster_id = g.cluster_id
    meter = MeteredPredictor(_SumEdgeToy(weight=2.0).double())
    out = bs.PairScorer('edge_feature_grad_x_input').score(
        g64, layout, 2, meter=meter)
    assert out.scores.dtype == torch.float64
    per_col = (2.0 * g64.edge_attr).abs().sum(dim=1)
    expected = torch.zeros(len(layout.pairs), dtype=torch.float64)
    counts = torch.zeros(len(layout.pairs), dtype=torch.float64)
    expected.index_add_(0, layout.inverse, per_col)
    counts.index_add_(0, layout.inverse,
                      torch.ones_like(per_col))
    expected = expected / counts.clamp_min(1)
    torch.testing.assert_close(out.scores, expected, atol=1e-9, rtol=1e-9)


def test_grad_x_input_works_under_outer_no_grad():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _reciprocal_graph()
    meter = MeteredPredictor(_SumEdgeToy(weight=2.0))
    with torch.no_grad():
        out = bs.PairScorer('edge_feature_grad_x_input').score(
            g, layout, 2, meter=meter)
    per_col = (2.0 * g.edge_attr).abs().sum(dim=1)
    expected = torch.zeros(len(layout.pairs))
    counts = torch.zeros(len(layout.pairs))
    expected.index_add_(0, layout.inverse, per_col)
    counts.index_add_(0, layout.inverse, torch.ones_like(per_col))
    expected = expected / counts.clamp_min(1)
    torch.testing.assert_close(out.scores, expected, atol=1e-5, rtol=1e-5)


class _FrozenNodeToy(nn.Module):
    """Parameter-free model whose prediction has no derivative graph."""

    def forward(self, batch, return_embeddings=False):
        if return_embeddings:
            raise ValueError('toy has no embeddings')
        return batch.x.sum()


def test_grad_x_input_frozen_noderivative_yields_explicit_zeros():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _reciprocal_graph()
    assert len(list(_FrozenNodeToy().parameters())) == 0
    out = bs.PairScorer('edge_feature_grad_x_input').score(
        g, layout, 1, meter=MeteredPredictor(_FrozenNodeToy()))
    torch.testing.assert_close(out.scores, torch.zeros(3),
                               atol=1e-6, rtol=1e-6)
    assert out.metadata.get('unused_edge_input') is True


def test_grad_x_input_requires_meter_and_flags_unused_edge():
    bs = _api()
    g, layout = _reciprocal_graph()
    with pytest.raises(ValueError, match='meter|predictor'):
        bs.PairScorer('edge_feature_grad_x_input').score(g, layout, 1)
    from rls.benchmark_predictor import MeteredPredictor
    meter = MeteredPredictor(_NodeOnlyToy())
    out = bs.PairScorer('edge_feature_grad_x_input').score(
        g, layout, 1, meter=meter)
    # unused edge input is explicitly zero, never node-feature saliency
    torch.testing.assert_close(out.scores, torch.zeros(3),
                               atol=1e-6, rtol=1e-6)
    blob = (' '.join(map(str, out.metadata.keys()))
            + ' '.join(map(str, out.metadata.values()))).lower()
    assert 'unused' in blob


# ---------------- provided_rl ----------------

class _OrientedPolicy(nn.Module):
    """logit = emb[src].sum() + 10 * signed mass_ratio; counts calls."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.calls = 0
        self.last_inputs = []

    def forward(self, edge_attr, node_emb, edge_index, ctx):
        self.calls += 1
        self.last_inputs.append((edge_attr.detach().clone(),
                                 edge_index.clone()))
        src = edge_index[0]
        mr = edge_attr[:, 1:2]
        return node_emb[src].sum(dim=1, keepdim=True) + 10.0 * mr


class _EmbedToy(nn.Module):
    """Deterministic embeddings = x; scalar prediction = sum."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(2, 2)

    def forward(self, batch, return_embeddings=False):
        pred = self.w * (batch.x.sum() + batch.edge_attr.sum() * 0.0)
        if return_embeddings:
            return pred, batch.x.detach().clone()
        return pred


def _one_way_rl_graph():
    # single one-way stored column (0 -> 1); P=1
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    pos = torch.zeros(3, 2)
    stellar = torch.ones(3)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.tensor([[2.0, 0.5]])  # distance, signed mass_ratio
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             pos=pos, stellar_mass=stellar)
    g.cluster_id = 'oneway-rl'
    layout = PhysicalPairLayout.from_edge_index(edge_index, 3)
    return g, layout


def test_provided_rl_exact_symmetrized_one_way_counts_and_restore():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    from data.provenance import model_state_hash
    g, layout = _one_way_rl_graph()
    policy = _OrientedPolicy()
    policy.train()
    meter = MeteredPredictor(_EmbedToy())
    snap_before = meter.snapshot()
    scorer = bs.PairScorer('provided_rl', policy=policy,
                           edge_feature_names=['distance', 'mass_ratio'])
    out = scorer.score(g, layout, 1, meter=meter)
    # orientation 1: S0 + 10m ; orientation 2: S1 - 10m ; avg = (S0+S1)/2
    s0, s1, m = 3.0, 7.0, 0.5
    expected = torch.tensor([(s0 + 10 * m + s1 - 10 * m) / 2.0])
    torch.testing.assert_close(out.scores, expected, atol=1e-5, rtol=1e-5)
    assert policy.calls == 2, 'two orientation policy passes required'
    snap_after = meter.snapshot()
    total_delta = sum(snap_after[p]['forward_calls']
                      - snap_before[p]['forward_calls']
                      for p in snap_after)
    assert total_delta == 1, 'exactly one predictor embedding call'
    assert policy.training, 'policy mode must be restored to train'
    # orientation-2 input: swapped endpoints + negated signed mass_ratio
    _, ei2 = policy.last_inputs[1]
    assert torch.equal(ei2, torch.tensor([[1], [0]]))
    ea1, _ = policy.last_inputs[0]
    ea2, _ = policy.last_inputs[1]
    torch.testing.assert_close(ea2[:, 1], -ea1[:, 1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ea2[:, 0], ea1[:, 0], atol=1e-6, rtol=1e-6)
    # metadata records actual policy state hash + method
    assert model_state_hash(policy.state_dict()) in list(
        out.metadata.values())
    blob = (' '.join(map(str, out.metadata.keys()))
            + ' '.join(map(str, out.metadata.values()))).lower()
    assert 'provided_rl' in blob or 'policy' in blob


def test_provided_rl_rejects_duplicate_feature_names():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _one_way_rl_graph()
    with pytest.raises(ValueError, match='duplicate'):
        bs.PairScorer('provided_rl', policy=_OrientedPolicy(),
                      edge_feature_names=['mass_ratio',
                                          'mass_ratio']).score(
            g, layout, 1, meter=MeteredPredictor(_EmbedToy()))


def test_provided_rl_rejects_missing_schema_or_policy_or_unknown():
    bs = _api()
    g, layout = _one_way_rl_graph()
    from rls.benchmark_predictor import MeteredPredictor
    meter = MeteredPredictor(_EmbedToy())
    with pytest.raises(ValueError, match='policy'):
        bs.PairScorer('provided_rl',
                      edge_feature_names=['distance',
                                          'mass_ratio']).score(
            g, layout, 1, meter=meter)
    with pytest.raises(ValueError, match='schema|edge_feature_names'):
        bs.PairScorer('provided_rl', policy=_OrientedPolicy()).score(
            g, layout, 1, meter=meter)
    with pytest.raises(ValueError, match='unknown|schema|feature'):
        bs.PairScorer('provided_rl', policy=_OrientedPolicy(),
            edge_feature_names=['distance',
                                          'nope']).score(
            g, layout, 1, meter=meter)


def _projected_rl_graph():
    from data.projected_graph import ProjectedGraphConfig, checkpoint_compatibility_record
    g, layout = _two_pair_rl_graph()
    config = ProjectedGraphConfig()
    names = list(config.edge_features)
    # Three columns carry the actual projected semantics in the declared order.
    g.edge_attr = torch.tensor([[2., .5, .1], [2., .5, -.1],
                                [3., .25, .2], [3., .25, -.2]])
    g.feature_mode = "projected_observation"
    g.graph_schema = "projected_graph_v1"
    g.observer_axis = "z"
    g.compatibility_record = checkpoint_compatibility_record(config, "z")
    g.schema_identity = g.compatibility_record["schema_identity"]
    return g, layout, names


def test_provided_rl_accepts_projected_names_only_with_exact_projected_schema():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout, names = _projected_rl_graph()
    scorer = bs.PairScorer("provided_rl", policy=_OrientedPolicy(), edge_feature_names=names)
    result = scorer.score(g, layout, 1, meter=MeteredPredictor(_EmbedToy()))
    assert result.scores.shape == (2,)
    assert result.metadata["edge_feature_names"] == names
    reverse = copy.deepcopy(g)
    reverse.edge_index = reverse.edge_index.flip(0)
    reverse.edge_attr[:, 2] *= -1  # signed mass_ratio reverses with endpoints
    result_reversed = scorer.score(reverse, layout, 1, meter=MeteredPredictor(_EmbedToy()))
    torch.testing.assert_close(result.scores, result_reversed.scores)


def test_projected_policy_schema_rejects_legacy_graph_and_wrong_order():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout, names = _projected_rl_graph()
    meter = MeteredPredictor(_EmbedToy())
    g.feature_mode = "legacy_3d"
    with pytest.raises(ValueError, match="projected|feature mode"):
        bs.PairScorer("provided_rl", policy=_OrientedPolicy(), edge_feature_names=names).score(
            g, layout, 1, meter=meter)
    g.feature_mode = "projected_observation"
    with pytest.raises(ValueError, match="exactly match|schema"):
        bs.PairScorer("provided_rl", policy=_OrientedPolicy(), edge_feature_names=names[::-1]).score(
            g, layout, 1, meter=meter)

    scorer = bs.PairScorer("provided_rl", policy=_OrientedPolicy(), edge_feature_names=names)
    g.schema_identity = "0" * 64
    with pytest.raises(ValueError, match="schema|compatibility"):
        scorer.score(g, layout, 1, meter=meter)
    g.schema_identity = g.compatibility_record["schema_identity"]
    g.observer_axis = "x"
    with pytest.raises(ValueError, match="exactly match|schema"):
        scorer.score(g, layout, 1, meter=meter)


def _two_pair_rl_graph():
    # reciprocal pairs (0,1),(1,2); P=2 with distinct node embedding sums
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])  # row sums 3, 7, 11
    pos = torch.zeros(3, 2)
    stellar = torch.ones(3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[2.0, 0.5], [2.0, -0.5],
                              [3.0, 0.25], [3.0, -0.25]])
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             pos=pos, stellar_mass=stellar)
    g.cluster_id = 'twopair-rl'
    layout = PhysicalPairLayout.from_edge_index(edge_index, 3)
    return g, layout


def test_provided_rl_scores_transport_under_nontrivial_relabel():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    g, layout = _two_pair_rl_graph()
    perm = [2, 0, 1]
    g2 = _permuted_graph(g, perm)
    layout2 = PhysicalPairLayout.from_edge_index(g2.edge_index, 3)
    fwd = _pair_index_map(layout.pairs, layout2.pairs, perm)
    assert not torch.equal(fwd, torch.arange(len(fwd))), \
        'relabeling must permute pair rows for a real transport check'
    kwargs = dict(policy=_OrientedPolicy(),
                  edge_feature_names=['distance', 'mass_ratio'])
    s1 = bs.PairScorer('provided_rl', **kwargs).score(
        g, layout, 1, meter=MeteredPredictor(_EmbedToy())).scores
    # symmetrized exact values: (S_u + S_v) / 2 with S = x row sums
    torch.testing.assert_close(s1, torch.tensor([5.0, 9.0]),
                               atol=1e-5, rtol=1e-5)
    s2 = bs.PairScorer('provided_rl', **kwargs).score(
        g2, layout2, 1, meter=MeteredPredictor(_EmbedToy())).scores
    torch.testing.assert_close(s2, s1[fwd], atol=1e-5, rtol=1e-5)


def test_provided_rl_policy_failure_restores_all_flags():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor

    class _FailPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.calls = 0

        def forward(self, edge_attr, node_emb, edge_index, ctx):
            self.calls += 1
            raise RuntimeError('controlled policy failure')

    g, layout = _one_way_rl_graph()
    policy = _FailPolicy()
    policy.lin.train()
    modes_before = {n: m.training for n, m in policy.named_modules()}
    meter = MeteredPredictor(_EmbedToy())
    with pytest.raises(RuntimeError, match='controlled policy failure'):
        bs.PairScorer('provided_rl', policy=policy,
                      edge_feature_names=['distance',
                                          'mass_ratio']).score(
            g, layout, 1, meter=meter)
    assert policy.calls >= 1
    assert {n: m.training for n, m in policy.named_modules()} == modes_before


def test_provided_rl_tiny_edgepolicynet_integration():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    from rls.policy import EdgePolicyNet
    torch.manual_seed(0)
    g, layout = _reciprocal_graph()
    g2 = Data(x=torch.randn(3, 4), edge_index=g.edge_index,
              edge_attr=torch.randn(6, 5), pos=g.pos,
              stellar_mass=g.stellar_mass)
    g2.cluster_id = 'rl-int'
    layout2 = PhysicalPairLayout.from_edge_index(g2.edge_index, 3)

    class _TinyEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch, return_embeddings=False):
            pred = self.w * batch.x.sum()
            if return_embeddings:
                emb = torch.tanh(batch.x @ torch.ones(4, 8))
                return pred, emb
            return pred

    names = ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep']
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=8, hidden_dim=8)
    policy.train()
    meter = MeteredPredictor(_TinyEmbed())
    snap_before = meter.snapshot()
    out = bs.PairScorer('provided_rl', policy=policy,
                        edge_feature_names=names).score(
        g2, layout2, 2, meter=meter)
    assert out.scores.shape == (3,)
    assert torch.isfinite(out.scores).all()
    assert policy.training, 'EdgePolicyNet mode must be restored'
    snap_after = meter.snapshot()
    total_delta = sum(snap_after[p]['forward_calls']
                      - snap_before[p]['forward_calls']
                      for p in snap_after)
    assert total_delta == 1


# ---------------- supervised_raw ----------------

def _tiny_raw_artifact():
    from rls import raw_selector as rs
    torch.manual_seed(0)
    scorer = rs.RawBudgetPairScorer(node_dim=2,
                                    edge_feature_names=['distance'],
                                    hidden_dim=8)
    ei = torch.tensor([[0, 1], [1, 0]])
    ea = torch.tensor([[1.0], [2.0]])
    x = torch.tensor([[0.5, -0.5], [1.0, 0.0]])
    graph = {'edge_index': ei, 'edge_attr': ea, 'x': x,
             'cluster_id': 'g-train'}
    labels = torch.tensor([1.5])  # P=1
    scorer.fit_normalization(examples=[(graph, 1)], train_ids=['g-train'],
                             heldout_ids=[], split_hash='split-abc')
    ex = {'graph': graph, 'k': 1, 'labels': labels,
          'label_definition': 'test-def', 'label_predictor_calls': 3,
          'label_provenance': {'source': 'unit'}}
    artifact = rs.train_supervised_selector(
        scorer, [ex], train_ids=['g-train'], heldout_ids=[],
        split_hash='split-abc', seed=0, epochs=3, lr=1e-2)
    return scorer, artifact


def _heldout_data_graph():
    ei = torch.tensor([[0, 1], [1, 0]])
    ea = torch.tensor([[1.5], [2.5]])
    x = torch.tensor([[0.25, 0.75], [-1.0, 0.5]])
    pos = torch.tensor([[0., 0.], [3., 4.]])
    stellar = torch.tensor([1e10, 2e10])
    g = Data(x=x, edge_index=ei, edge_attr=ea, pos=pos,
             stellar_mass=stellar)
    g.cluster_id = 'g-heldout'
    layout = PhysicalPairLayout.from_edge_index(ei, 2)
    return g, layout


def test_supervised_raw_matches_scorer_zero_predictor_calls_cost_once():
    bs = _api()
    from rls.benchmark_predictor import MeteredPredictor
    raw_scorer, artifact = _tiny_raw_artifact()
    g, layout = _heldout_data_graph()
    scorer = bs.PairScorer('supervised_raw', raw_scorer=raw_scorer,
                           raw_artifact=artifact, split_hash='split-abc')
    toy = _SumEdgeToy()
    meter = MeteredPredictor(toy)
    snap_before = meter.snapshot()
    out = scorer.score(g, layout, 1, meter=meter)
    snap_after = meter.snapshot()
    for phase in snap_after:
        assert snap_after[phase]['forward_calls'] == \
            snap_before[phase]['forward_calls'], \
            'selection must use zero predictor calls'
    expected = raw_scorer(
        {'edge_index': g.edge_index, 'edge_attr': g.edge_attr, 'x': g.x},
        layout, 1).detach()
    torch.testing.assert_close(out.scores, expected, atol=1e-6, rtol=1e-6)
    blob = (' '.join(map(str, out.metadata.keys()))
            + ' '.join(map(str, out.metadata.values()))).lower()
    assert 'artifact' in blob or 'state' in blob or 'split' in blob
    cost1 = scorer.training_cost()
    cost2 = scorer.training_cost()
    assert cost1 == cost2, 'historical cost counted once per artifact'
    assert isinstance(cost1, dict), 'training_cost must return a dict'
    assert cost1['total_calls'] == 3, \
        'exact supplied label creation cost surfaced once'
    assert cost1.get('model_state_hash') == artifact['model_state_hash'], \
        'training_cost must carry artifact identity'
    cost1['total_calls'] = 999
    assert scorer.training_cost()['total_calls'] == 3, \
        'training_cost must return copied metadata'


def test_training_cost_rejects_tampered_weights():
    bs = _api()
    raw_scorer, artifact = _tiny_raw_artifact()
    scorer = bs.PairScorer('supervised_raw', raw_scorer=raw_scorer,
                           raw_artifact=artifact, split_hash='split-abc')
    assert scorer.training_cost()['total_calls'] == 3
    with torch.no_grad():
        for p in raw_scorer.parameters():
            p.add_(1.0)
            break
    with pytest.raises(ValueError, match='state|hash|artifact|split'):
        scorer.training_cost()


def test_training_cost_rejects_negative_cost():
    bs = _api()
    raw_scorer, artifact = _tiny_raw_artifact()
    bad = copy.deepcopy(artifact)
    bad['label_provenance']['total_calls'] = -1
    scorer = bs.PairScorer('supervised_raw', raw_scorer=raw_scorer,
                           raw_artifact=bad, split_hash='split-abc')
    with pytest.raises(ValueError, match='cost|calls|negative|malformed|provenance'):
        scorer.training_cost()


def test_training_cost_rejects_malformed_artifact():
    bs = _api()
    raw_scorer, artifact = _tiny_raw_artifact()
    bad = copy.deepcopy(artifact)
    del bad['label_provenance']['total_calls']
    scorer = bs.PairScorer('supervised_raw', raw_scorer=raw_scorer,
                           raw_artifact=bad, split_hash='split-abc')
    with pytest.raises(ValueError, match='cost|calls|malformed|provenance'):
        scorer.training_cost()


def test_supervised_raw_rejects_wrong_split_state_or_missing():
    bs = _api()
    from rls import raw_selector as rs
    raw_scorer, artifact = _tiny_raw_artifact()
    g, layout = _heldout_data_graph()
    with pytest.raises(ValueError, match='split'):
        bs.PairScorer('supervised_raw', raw_scorer=raw_scorer,
                      raw_artifact=artifact,
                      split_hash='wrong-split').score(g, layout, 1)
    tampered = copy.deepcopy(raw_scorer)
    with torch.no_grad():
        for p in tampered.parameters():
            p.add_(1.0)
            break
    with pytest.raises(ValueError, match='state|hash|artifact|split'):
        bs.PairScorer('supervised_raw', raw_scorer=tampered,
                      raw_artifact=artifact,
                      split_hash='split-abc').score(g, layout, 1)
    with pytest.raises(ValueError, match='artifact|trained|missing|split'):
        bs.PairScorer('supervised_raw',
                      split_hash='split-abc').score(g, layout, 1)
    fresh = rs.RawBudgetPairScorer(node_dim=2,
                                   edge_feature_names=['distance'],
                                   hidden_dim=8)
    with pytest.raises(ValueError, match='norm|trained|artifact|fit|split'):
        bs.PairScorer('supervised_raw', raw_scorer=fresh,
                      raw_artifact=artifact,
                      split_hash='split-abc').score(g, layout, 1)


def test_supervised_raw_missing_prereqs_raise_clearly():
    bs = _api()
    g, layout = _heldout_data_graph()
    with pytest.raises(ValueError, match='artifact|raw_scorer|split|trained'):
        bs.PairScorer('supervised_raw').score(g, layout, 1)
