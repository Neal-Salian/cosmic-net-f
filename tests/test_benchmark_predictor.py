"""Substantive tests for rls.benchmark_predictor.MeteredPredictor (TESTS ONLY).

Contract under test (see task-3b-meter-handoff.md):
- MeteredPredictor(model).call(graph, mask=None, *, phase='evaluation',
    return_embeddings=False, grad_edge_attr=False) -> PredictorResult
  where `graph` is a single PyG Data (meter builds the Batch internally).
- PredictorResult fields: prediction (scalar tensor), embeddings (optional
  [N, D]), edge_attr (batch leaf when grad_edge_attr=True), elapsed_seconds,
  kept_edge_count.
- Physical slicing of edge_index/edge_attr; no zeroed-feature pseudo-masks.
- Exact per-phase counts; failed forwards charged, pre-invocation validation
  errors not charged; elapsed recorded even on failure.
- Frozen eval with full training-flag restoration (incl. mixed modes and
  exceptions); params/buffers/grads/requires_grad preserved.
- grad_edge_attr=True yields a differentiable leaf under outer no_grad.

Production module rls/benchmark_predictor.py implements the contract;
these tests run red (missing-API failure) until it exists.
"""
import copy
import math
import warnings

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from data.provenance import model_state_hash
from model.model import CosmicNetGNN

PHASES = ['scoring', 'training_label', 'shared_full',
          'evaluation', 'training', 'oracle']
WEIGHT = 2.5


def _api():
    import importlib.util
    assert importlib.util.find_spec('rls.benchmark_predictor') is not None, \
        'benchmark_predictor API missing'
    from rls import benchmark_predictor
    assert hasattr(benchmark_predictor, 'MeteredPredictor'), \
        'MeteredPredictor API missing'
    assert hasattr(benchmark_predictor, 'PredictorResult'), \
        'PredictorResult API missing'
    return benchmark_predictor


def _tiny_config():
    return {
        'model': {'node_features': 4, 'edge_features': 5,
                  'hidden_dim': 8, 'output_dim': 8,
                  'num_layers': 1, 'dropout': 0.0,
                  'pooling': 'mean', 'residual': False,
                  'activation': 'relu', 'mc_dropout': False},
        'graph': {'edge_features': ['distance', 'delta_v', 'cos_theta',
                                    'mass_ratio', 'proj_sep']},
    }


def _analytic_graph(edge_feature_dim=5):
    """Single Data with duplicates + self-loops and deterministic values."""
    edge_pairs = [[0, 1], [1, 0], [0, 0], [1, 1], [1, 2], [2, 1]]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    n_edges = edge_index.shape[1]
    edge_attr = (torch.arange(n_edges * edge_feature_dim,
                              dtype=torch.float32)
                 .reshape(n_edges, edge_feature_dim) + 1.0)
    x = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) + 0.5
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.cluster_id = 'analytic-g1'
    return g


class AnalyticToy(nn.Module):
    """REAL differentiable toy: prediction = w * calib[0] * sum(edge_attr).

    calibration[0] is 1.0, so the expected value/derivative equal the known
    weight 2.5 while the buffer participates in the autograd graph. This
    catches unsafe unconditional in-place buffer restoration in the meter,
    which would bump tensor version counters before caller autograd.grad.
    Records the actual edge_index/edge_attr the meter delivered so tests can
    verify physical slicing. Plain-tensor return path (no embeddings).
    """

    def __init__(self, weight=WEIGHT):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(weight)))
        self.proj = nn.Linear(4, 4)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.register_buffer('calibration',
                             torch.tensor([1.0, 2.0, 3.0]))
        self.forward_calls = 0
        self.seen_edge_index = None
        self.seen_edge_attr = None
        self.seen_n_nodes = None

    def forward(self, batch, return_embeddings=False):
        self.forward_calls += 1
        self.seen_edge_index = batch.edge_index.detach().clone()
        self.seen_edge_attr = batch.edge_attr.detach().clone()
        self.seen_n_nodes = batch.x.shape[0]
        if return_embeddings:
            raise ValueError('AnalyticToy does not provide embeddings')
        return self.w * self.calibration[0] * batch.edge_attr.sum()


class FailingMixedToy(nn.Module):
    """Increments a real counter then raises; keeps submodules + buffer."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(WEIGHT))
        self.proj = nn.Linear(4, 4)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.register_buffer('calibration',
                             torch.tensor([1.0, 2.0, 3.0]))
        self.forward_calls = 0

    def forward(self, batch, return_embeddings=False):
        self.forward_calls += 1
        raise RuntimeError('controlled forward failure')


def _mode_map(model):
    return {name: mod.training for name, mod in model.named_modules()}


# ---------------- API + snapshot shape ----------------

def test_api_importable():
    _api()


def test_snapshot_initial_phases_zero():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy())
    snap = meter.snapshot()
    for phase in PHASES:
        assert phase in snap, f'phase {phase} missing'
        assert snap[phase]['forward_calls'] == 0
        assert snap[phase]['graph_evaluations'] == 0
        assert snap[phase]['elapsed_seconds'] == 0


# ---------------- analytic toy: slicing + exact values ----------------

def test_toy_partial_mask_exact_prediction_and_columns():
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    mask = torch.tensor([True, False, True, False, True, False],
                        dtype=torch.bool)
    before = copy.deepcopy(graph)
    result = meter.call(graph, mask=mask, phase='evaluation',
                        return_embeddings=False)
    expected_attr = before.edge_attr[mask]
    expected_index = before.edge_index[:, mask]
    assert result.kept_edge_count == int(mask.sum())
    torch.testing.assert_close(
        result.prediction.detach().reshape(()),
        torch.tensor(WEIGHT * expected_attr.sum().item(),
                     dtype=result.prediction.dtype).reshape(()))
    assert result.prediction.numel() == 1
    assert torch.equal(toy.seen_edge_attr, expected_attr)
    assert torch.equal(toy.seen_edge_index, expected_index)
    assert toy.seen_n_nodes == graph.x.shape[0]
    assert result.embeddings is None
    assert result.edge_attr is None
    # input Data not mutated
    assert torch.equal(graph.x, before.x)
    assert torch.equal(graph.edge_index, before.edge_index)
    assert torch.equal(graph.edge_attr, before.edge_attr)


def test_toy_empty_mask_zero_prediction():
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    mask = torch.zeros(graph.edge_index.shape[1], dtype=torch.bool)
    result = meter.call(graph, mask=mask, phase='evaluation',
                        return_embeddings=False)
    assert result.kept_edge_count == 0
    torch.testing.assert_close(result.prediction.detach().reshape(()),
                               torch.tensor(0.0).reshape(()))
    assert toy.seen_edge_index.shape[1] == 0
    assert toy.seen_edge_attr.shape[0] == 0


# ---------------- exact counts ----------------

def test_exact_counts_one_call_one_forward_one_graph():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy())
    graph = _analytic_graph()
    meter.call(graph, phase='evaluation', return_embeddings=False)
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 1
    assert snap['evaluation']['graph_evaluations'] == 1
    for phase in PHASES:
        if phase == 'evaluation':
            continue
        assert snap[phase]['forward_calls'] == 0
        assert snap[phase]['graph_evaluations'] == 0


def test_exact_counts_isolated_per_phase():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy())
    graph = _analytic_graph()
    meter.call(graph, phase='scoring', return_embeddings=False)
    meter.call(graph, phase='scoring', return_embeddings=False)
    meter.call(graph, phase='shared_full', return_embeddings=False)
    snap = meter.snapshot()
    assert snap['scoring']['forward_calls'] == 2
    assert snap['scoring']['graph_evaluations'] == 2
    assert snap['shared_full']['forward_calls'] == 1
    assert snap['shared_full']['graph_evaluations'] == 1
    assert snap['evaluation']['forward_calls'] == 0


def test_failing_forward_charged_and_timed():
    r = _api()
    toy = FailingMixedToy()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    with pytest.raises(RuntimeError, match='controlled forward failure'):
        meter.call(graph, phase='evaluation', return_embeddings=False)
    assert toy.forward_calls == 1
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 1
    assert snap['evaluation']['graph_evaluations'] == 1
    assert math.isfinite(snap['evaluation']['elapsed_seconds'])
    assert snap['evaluation']['elapsed_seconds'] >= 0


@pytest.mark.parametrize('kind', ['mask_length', 'mask_dtype',
                                  'nonfinite_x', 'edge_row_mismatch'])
def test_invalid_input_zero_calls(kind):
    r = _api()
    toy = AnalyticToy()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    if kind == 'mask_length':
        mask = torch.ones(graph.edge_index.shape[1] - 1, dtype=torch.bool)
        ctx = pytest.raises(ValueError, match='shape|length|mask')
    elif kind == 'mask_dtype':
        mask = torch.ones(graph.edge_index.shape[1], dtype=torch.float32)
        ctx = pytest.raises(ValueError, match='dtype|bool|mask')
    elif kind == 'nonfinite_x':
        mask = None
        graph.x[0, 0] = float('nan')
        ctx = pytest.raises(ValueError, match='finite|nonfinite|nan|inf')
    else:
        mask = None
        graph.edge_attr = graph.edge_attr[:-1]
        ctx = pytest.raises(ValueError, match='shape|row|edge_attr|edge')
    with ctx:
        meter.call(graph, mask=mask, phase='evaluation',
                   return_embeddings=False)
    assert toy.forward_calls == 0
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 0
    assert snap['evaluation']['graph_evaluations'] == 0


def test_snapshot_independence_mutation():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy())
    meter.call(_analytic_graph(), phase='evaluation',
               return_embeddings=False)
    snap1 = meter.snapshot()
    snap1['evaluation']['forward_calls'] = 999
    snap1['evaluation']['elapsed_seconds'] = 999.0
    snap2 = meter.snapshot()
    assert snap2['evaluation']['forward_calls'] == 1
    assert snap2['evaluation']['graph_evaluations'] == 1
    assert snap2['evaluation']['elapsed_seconds'] != 999.0


# ---------------- analytic gradients + preservation ----------------

def test_analytic_edge_gradient_exact_and_state_preserved():
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    toy.eval()
    for p in toy.parameters():
        p.grad = torch.randn_like(p)
    grads_before = {n: p.grad.clone() for n, p in toy.named_parameters()}
    req_before = {n: p.requires_grad for n, p in toy.named_parameters()}
    params_before = {n: p.detach().clone()
                     for n, p in toy.named_parameters()}
    buffers_before = {n: b.clone() for n, b in toy.named_buffers()}
    assert torch.is_grad_enabled()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    x_before = graph.x.clone()
    ea_before = graph.edge_attr.clone()
    ei_before = graph.edge_index.clone()
    mask = torch.tensor([True, False, True, False, True, False],
                        dtype=torch.bool)
    result = meter.call(graph, mask=mask, phase='evaluation',
                        return_embeddings=False, grad_edge_attr=True)
    assert result.edge_attr is not None
    assert result.edge_attr.requires_grad
    assert result.edge_attr.is_leaf
    grad = torch.autograd.grad(result.prediction.sum(), result.edge_attr)[0]
    assert grad.shape == result.edge_attr.shape
    torch.testing.assert_close(
        grad, torch.full_like(grad, WEIGHT))
    # frozen state preserved
    for n, p in toy.named_parameters():
        torch.testing.assert_close(p.grad, grads_before[n])
        assert p.requires_grad == req_before[n]
        torch.testing.assert_close(p.detach(), params_before[n])
    for n, b in toy.named_buffers():
        torch.testing.assert_close(b, buffers_before[n])
    assert torch.equal(graph.x, x_before)
    assert torch.equal(graph.edge_attr, ea_before)
    assert torch.equal(graph.edge_index, ei_before)
    assert not graph.x.requires_grad
    assert not graph.edge_attr.requires_grad
    assert graph.x.grad is None and graph.edge_attr.grad is None
    assert torch.is_grad_enabled()


def test_grad_callable_under_outer_no_grad_and_restored():
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        result = meter.call(graph, phase='evaluation',
                            return_embeddings=False, grad_edge_attr=True)
        assert result.edge_attr.requires_grad
        assert result.edge_attr.is_leaf
        grad = torch.autograd.grad(result.prediction,
                                   result.edge_attr)[0]
        torch.testing.assert_close(grad,
                                   torch.full_like(grad, WEIGHT))
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_mixed_mode_mapping_restored_normal_path():
    r = _api()
    toy = AnalyticToy()
    toy.proj.train()
    toy.head.eval()
    buffers_before = {n: b.clone() for n, b in toy.named_buffers()}
    expected_modes = _mode_map(toy)
    assert len(set(expected_modes.values())) > 1  # genuinely mixed
    meter = r.MeteredPredictor(toy)
    meter.call(_analytic_graph(), phase='evaluation',
               return_embeddings=False)
    assert _mode_map(toy) == expected_modes
    for n, b in toy.named_buffers():
        torch.testing.assert_close(b, buffers_before[n])


def test_mixed_mode_mapping_restored_failing_path():
    r = _api()
    toy = FailingMixedToy()
    toy.proj.train()
    toy.head.eval()
    expected_modes = _mode_map(toy)
    assert len(set(expected_modes.values())) > 1
    meter = r.MeteredPredictor(toy)
    with pytest.raises(RuntimeError, match='controlled forward failure'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=False)
    assert _mode_map(toy) == expected_modes


# ---------------- no stale cache; state hash ----------------

def test_same_id_changed_content_changes_prediction():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy(weight=WEIGHT))
    g1 = _analytic_graph()
    g1.cluster_id = 'same-id'
    g2 = _analytic_graph()
    g2.cluster_id = 'same-id'
    g2.edge_attr = g2.edge_attr + 1.0
    r1 = meter.call(g1, phase='evaluation', return_embeddings=False)
    r2 = meter.call(g2, phase='evaluation', return_embeddings=False)
    delta = WEIGHT * float((g2.edge_attr - g1.edge_attr).sum())
    torch.testing.assert_close(
        (r2.prediction - r1.prediction).detach().reshape(()),
        torch.tensor(delta).reshape(()))
    assert r1.prediction.item() != r2.prediction.item()
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 2


def test_model_state_hash_tracks_weight_change():
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    meter = r.MeteredPredictor(toy)
    assert meter.model_state_hash() == model_state_hash(toy.state_dict())
    h_before = meter.model_state_hash()
    with torch.no_grad():
        toy.w.add_(1.0)
    assert meter.model_state_hash() == model_state_hash(toy.state_dict())
    assert meter.model_state_hash() != h_before


# ---------------- controlled output failures (charged) ----------------

def test_nonfinite_prediction_rejected_and_charged():
    r = _api()

    class NanToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))
            self.forward_calls = 0

        def forward(self, batch, return_embeddings=False):
            self.forward_calls += 1
            return torch.tensor(float('nan'))

    toy = NanToy()
    meter = r.MeteredPredictor(toy)
    with pytest.raises(ValueError, match='finite|nonfinite'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=False)
    assert toy.forward_calls == 1
    assert meter.snapshot()['evaluation']['forward_calls'] == 1


def test_wrong_scalar_shape_rejected_and_charged():
    r = _api()

    class VecToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))
            self.forward_calls = 0

        def forward(self, batch, return_embeddings=False):
            self.forward_calls += 1
            return torch.tensor([1.0, 2.0])

    toy = VecToy()
    meter = r.MeteredPredictor(toy)
    with pytest.raises(ValueError, match='scalar|shape'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=False)
    assert toy.forward_calls == 1
    assert meter.snapshot()['evaluation']['forward_calls'] == 1


def test_bad_embeddings_rejected_and_charged():
    r = _api()

    class NanEmbedToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))
            self.forward_calls = 0

        def forward(self, batch, return_embeddings=False):
            self.forward_calls += 1
            n = batch.x.shape[0]
            return torch.tensor(1.0), torch.full((n, 4), float('nan'))

    toy = NanEmbedToy()
    meter = r.MeteredPredictor(toy)
    with pytest.raises(ValueError, match='finite|nonfinite|embedding'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=True)
    assert toy.forward_calls == 1
    assert meter.snapshot()['evaluation']['forward_calls'] == 1


def test_plain_toy_rejects_embeddings_request():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy())
    with pytest.raises(ValueError, match='embedding'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=True)


# ---------------- tiny real CosmicNet integration (only one) ----------------

def test_tiny_cosmicnet_sliced_graph_with_embeddings():
    r = _api()
    model = CosmicNetGNN(_tiny_config())
    model.eval()
    meter = r.MeteredPredictor(model)
    graph = _analytic_graph()  # node_dim 4 / edge_dim 5 match tiny config
    mask = torch.tensor([True, False, True, False, True, False],
                        dtype=torch.bool)
    result = meter.call(graph, mask=mask, phase='evaluation',
                        return_embeddings=True)
    assert result.prediction.numel() == 1
    assert torch.isfinite(result.prediction).all()
    assert result.embeddings is not None
    assert result.embeddings.shape[0] == graph.x.shape[0]
    assert result.embeddings.shape[1] == 8
    assert torch.isfinite(result.embeddings).all()
    assert result.kept_edge_count == int(mask.sum())
    assert math.isfinite(result.elapsed_seconds)
    assert result.elapsed_seconds >= 0


# ---------------- review regressions: nested buffers ----------------

class _ChildWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer('running', torch.tensor([1.0]))


class NestedMutatingToy(nn.Module):
    """Toy with a NESTED registered buffer mutated in-place each forward."""

    def __init__(self, fail=False):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(WEIGHT))
        self.register_buffer('calibration',
                             torch.tensor([1.0, 2.0, 3.0]))
        self.child = _ChildWithBuffer()
        self.forward_calls = 0
        self._fail = fail

    def forward(self, batch, return_embeddings=False):
        self.forward_calls += 1
        self.child.running.add_(1)
        if self._fail:
            raise RuntimeError('controlled forward failure')
        if return_embeddings:
            raise ValueError(
                'NestedMutatingToy does not provide embeddings')
        return self.w * self.calibration[0] * batch.edge_attr.sum()


def test_nested_buffer_restored_success_with_grad():
    r = _api()
    toy = NestedMutatingToy()
    meter = r.MeteredPredictor(toy)
    h_before = meter.model_state_hash()
    graph = _analytic_graph()
    result = meter.call(graph, phase='evaluation', return_embeddings=False,
                        grad_edge_attr=True)
    assert toy.child.running.item() == 1.0
    assert 'running' in dict(toy.child.named_buffers())
    assert 'child.running' not in toy._buffers
    assert toy.forward_calls == 1
    assert meter.model_state_hash() == h_before
    grad = torch.autograd.grad(result.prediction, result.edge_attr)[0]
    torch.testing.assert_close(grad, torch.full_like(grad, WEIGHT))


def test_nested_buffer_restored_failure():
    r = _api()
    toy = NestedMutatingToy(fail=True)
    meter = r.MeteredPredictor(toy)
    h_before = meter.model_state_hash()
    with pytest.raises(RuntimeError, match='controlled forward failure'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=False)
    assert toy.child.running.item() == 1.0
    assert 'running' in dict(toy.child.named_buffers())
    assert 'child.running' not in toy._buffers
    assert toy.forward_calls == 1
    assert meter.model_state_hash() == h_before
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 1
    assert snap['evaluation']['graph_evaluations'] == 1


# ---------------- review regressions: signature charging ----------------

class PlainToy(nn.Module):
    """forward(batch) without a return_embeddings kwarg."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(WEIGHT))
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        return self.w * batch.edge_attr.sum()


class PlainFailingToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        raise RuntimeError('controlled plain failure')


def test_unsupported_signature_zero_charge():
    r = _api()
    toy = PlainToy()
    meter = r.MeteredPredictor(toy)
    with pytest.raises(ValueError, match='embedding'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=True)
    assert toy.forward_calls == 0
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 0
    assert snap['evaluation']['graph_evaluations'] == 0


def test_plain_signature_works_and_counts():
    r = _api()
    toy = PlainToy()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    result = meter.call(graph, phase='evaluation', return_embeddings=False)
    torch.testing.assert_close(
        result.prediction.detach().reshape(()),
        torch.tensor(WEIGHT * graph.edge_attr.sum().item()).reshape(()))
    assert toy.forward_calls == 1
    assert meter.snapshot()['evaluation']['forward_calls'] == 1


def test_attempted_plain_forward_raises_charged():
    r = _api()
    toy = PlainFailingToy()
    meter = r.MeteredPredictor(toy)
    with pytest.raises(RuntimeError, match='controlled plain failure'):
        meter.call(_analytic_graph(), phase='evaluation',
                   return_embeddings=False)
    assert toy.forward_calls == 1
    assert meter.snapshot()['evaluation']['forward_calls'] == 1


# ---------------- review regressions: non-leaf input edge_attr ----------------

@pytest.mark.parametrize('grad_edge_attr', [True, False])
def test_nonleaf_edge_attr_no_caller_grads(grad_edge_attr):
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    n = graph.edge_index.shape[1]
    f = graph.edge_attr.shape[1]
    graph.edge_attr = torch.ones(n, f, requires_grad=True) * 2
    assert not graph.edge_attr.is_leaf
    mask = torch.tensor([True, False, True, False, True, False],
                        dtype=torch.bool)
    result = meter.call(graph, mask=mask, phase='evaluation',
                        return_embeddings=False,
                        grad_edge_attr=grad_edge_attr)
    expected = WEIGHT * 2.0 * int(mask.sum()) * f
    torch.testing.assert_close(result.prediction.detach().reshape(()),
                               torch.tensor(expected).reshape(()))
    assert result.kept_edge_count == int(mask.sum())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        assert graph.edge_attr.grad is None
    assert torch.equal(graph.edge_attr.detach(),
                       torch.full((n, f), 2.0))
    if grad_edge_attr:
        assert result.edge_attr is not None
        assert result.edge_attr.requires_grad
        assert result.edge_attr.is_leaf
        grad = torch.autograd.grad(result.prediction, result.edge_attr)[0]
        torch.testing.assert_close(grad, torch.full_like(grad, WEIGHT))
    else:
        assert result.edge_attr is None


# ---------------- review regressions: graph validation ----------------

@pytest.mark.parametrize('kind', ['x_1d', 'negative_endpoint',
                                  'oob_endpoint'])
def test_malformed_graph_zero_calls(kind):
    r = _api()
    toy = AnalyticToy()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    if kind == 'x_1d':
        graph.x = torch.randn(4)
        match = 'x|shape'
    elif kind == 'negative_endpoint':
        graph.edge_index[0, 0] = -1
        match = 'endpoint|bound'
    else:
        graph.edge_index[0, 0] = 99
        match = 'endpoint|bound'
    with pytest.raises(ValueError, match=match):
        meter.call(graph, phase='evaluation', return_embeddings=False)
    assert toy.forward_calls == 0
    snap = meter.snapshot()
    assert snap['evaluation']['forward_calls'] == 0
    assert snap['evaluation']['graph_evaluations'] == 0


def test_zero_edge_graph_supported():
    r = _api()
    toy = AnalyticToy(weight=WEIGHT)
    meter = r.MeteredPredictor(toy)
    x = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) + 0.5
    graph = Data(x=x,
                 edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, 5)))
    result = meter.call(graph, phase='evaluation', return_embeddings=False)
    assert result.kept_edge_count == 0
    torch.testing.assert_close(result.prediction.detach().reshape(()),
                               torch.tensor(0.0).reshape(()))
    assert meter.snapshot()['evaluation']['forward_calls'] == 1


def test_cuda_mixed_device_rejected():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    r = _api()
    toy = AnalyticToy()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph()
    graph.x = graph.x.cuda()
    with pytest.raises(ValueError, match='device'):
        meter.call(graph, phase='evaluation', return_embeddings=False)
    assert toy.forward_calls == 0
    assert meter.snapshot()['evaluation']['forward_calls'] == 0


# ---------------- timing ----------------

def test_timing_finite_nonnegative():
    r = _api()
    meter = r.MeteredPredictor(AnalyticToy())
    result = meter.call(_analytic_graph(), phase='evaluation',
                        return_embeddings=False)
    assert math.isfinite(result.elapsed_seconds)
    assert result.elapsed_seconds >= 0
    snap = meter.snapshot()
    assert math.isfinite(snap['evaluation']['elapsed_seconds'])
    assert snap['evaluation']['elapsed_seconds'] >= 0


def test_cuda_timing_conditional():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    r = _api()
    toy = AnalyticToy().cuda()
    meter = r.MeteredPredictor(toy)
    graph = _analytic_graph().cuda()
    mask = torch.tensor([True, False, True, False, True, False],
                        dtype=torch.bool, device='cuda')
    result = meter.call(graph, mask=mask, phase='evaluation',
                        return_embeddings=False, grad_edge_attr=True)
    assert result.prediction.device.type == 'cuda'
    assert result.edge_attr is not None
    assert result.edge_attr.device.type == 'cuda'
    assert toy.seen_edge_attr.device.type == 'cuda'
    assert result.kept_edge_count == int(mask.sum().item())
    assert math.isfinite(result.elapsed_seconds)
    assert result.elapsed_seconds >= 0
