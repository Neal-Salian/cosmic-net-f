"""Common pair-score adapters for matched-budget benchmarking.

Each PairScorer returns SCORES ONLY (one finite float per physical pair);
scorers never choose/repair masks or change budgets. The shared decoder is
a later module.
"""

import copy
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from data.provenance import model_state_hash
from rls.constraints import PhysicalPairLayout
from rls.raw_selector import (
    RawBudgetPairScorer,
    VALID_EDGE_FEATURE_NAMES,
    validate_supervised_artifact,
)

METHOD_NAMES = (
    'random',
    'distance',
    'degree',
    'stellar_binding_proxy',
    'edge_feature_grad_x_input',
    'provided_rl',
    'supervised_raw',
)


@dataclass
class PairScoreResult:
    scores: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


def _validate_k(k: Any, P: int) -> int:
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError(f'k must be an integer budget, got {k!r}')
    k = int(k)
    if k < 0:
        raise ValueError(f'k must be non-negative, got {k}')
    if k > P:
        raise ValueError(f'k={k} exceeds P={P}')
    return k


def _graph_edge_index(graph: Any) -> torch.Tensor:
    ei = graph.edge_index if not isinstance(graph, dict) else graph.get('edge_index')
    if not isinstance(ei, torch.Tensor) or ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError('graph edge_index must be a [2, E] integer tensor')
    if ei.dtype not in (torch.int32, torch.int64):
        raise ValueError('graph edge_index must be an integer tensor')
    if int(ei.numel()) > 0 and bool((ei < 0).any()):
        raise ValueError('negative node indices are invalid')
    return ei


def _check_device(tensor: torch.Tensor, device: torch.device, what: str) -> None:
    if tensor.device != device:
        raise ValueError(f'{what} device {tensor.device} != layout device {device}')


def _validate_layout_alignment(ei: torch.Tensor, n_nodes: int,
                               layout: PhysicalPairLayout) -> None:
    if not isinstance(layout, PhysicalPairLayout):
        raise ValueError('layout must be a PhysicalPairLayout')
    try:
        expected = PhysicalPairLayout.from_edge_index(ei, n_nodes)
    except ValueError as e:
        raise ValueError(str(e))
    if expected.num_nodes != layout.num_nodes:
        raise ValueError(
            f'layout num_nodes {layout.num_nodes} != graph N {expected.num_nodes}')
    if len(expected.pairs) != len(layout.pairs):
        raise ValueError('layout P does not match graph topology')
    if int(expected.real.numel()) != int(layout.real.numel()) or not torch.equal(
            expected.real.cpu(), layout.real.cpu()):
        raise ValueError('layout stored-edge mask does not match graph')
    if not torch.equal(expected.pairs.cpu(), layout.pairs.cpu()):
        raise ValueError('layout pairs do not match graph topology')
    if not torch.equal(expected.inverse.cpu(), layout.inverse.cpu()):
        raise ValueError('layout inverse does not match graph topology')


def _finish(scores: torch.Tensor, layout: PhysicalPairLayout,
            metadata: Dict[str, Any]) -> PairScoreResult:
    if not isinstance(scores, torch.Tensor) or not torch.is_floating_point(scores):
        raise ValueError('scorer must produce a floating score tensor')
    scores = scores.to(device=layout.pairs.device).reshape(len(layout.pairs))
    if not torch.isfinite(scores).all():
        raise ValueError('scorer produced non-finite scores')
    return PairScoreResult(scores=scores, metadata=dict(metadata))


def _mean_over_real_copies(stored: torch.Tensor,
                           layout: PhysicalPairLayout) -> torch.Tensor:
    """Mean stored nonself-copy value per physical pair via layout.inverse.

    The reduction preserves the stored dtype (e.g. float64 saliency stays
    float64); counts use the same dtype so index_add_ never mixes types.
    """
    P = len(layout.pairs)
    device = layout.pairs.device
    dtype = stored.dtype if torch.is_floating_point(stored) else torch.float32
    inv = layout.inverse.to(device)
    stored_real = stored.to(device=device, dtype=dtype)[layout.real.to(device)]
    sums = torch.zeros(P, dtype=dtype, device=device)
    sums.index_add_(0, inv, stored_real)
    counts = torch.zeros(P, dtype=dtype, device=device)
    counts.index_add_(0, inv, torch.ones(len(stored_real), dtype=dtype,
                                         device=device))
    return sums / counts.clamp_min(1)


class PairScorer:
    """Score every physical pair; scores only, never masks or budgets."""

    def __init__(
        self,
        name: str,
        *,
        policy: Optional[nn.Module] = None,
        raw_scorer: Optional[RawBudgetPairScorer] = None,
        raw_artifact: Optional[Dict[str, Any]] = None,
        split_hash: Optional[str] = None,
        edge_feature_names: Optional[List[str]] = None,
        distance_epsilon: Optional[float] = None,
        distance_unit: Optional[str] = None,
        stellar_mass_unit: Optional[str] = None,
    ):
        if name not in METHOD_NAMES:
            raise ValueError(
                f'unknown scorer name {name!r}; expected one of {list(METHOD_NAMES)}')
        self.name = name
        self.policy = policy
        self.raw_scorer = raw_scorer
        self.raw_artifact = raw_artifact
        self.split_hash = split_hash
        self.edge_feature_names = list(edge_feature_names) \
            if edge_feature_names is not None else None
        self.distance_epsilon = distance_epsilon
        self.distance_unit = distance_unit
        self.stellar_mass_unit = stellar_mass_unit

    # ---------------- shared validation ----------------

    def _base(self, graph: Any, layout: PhysicalPairLayout, k: Any):
        if not isinstance(layout, PhysicalPairLayout):
            raise ValueError('layout must be a PhysicalPairLayout')
        for tensor_name in ('pairs', 'inverse', 'real'):
            tensor = getattr(layout, tensor_name, None)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f'layout.{tensor_name} must be a torch tensor')
        if not (layout.pairs.device == layout.inverse.device
                == layout.real.device):
            raise ValueError('layout pairs/inverse/real must share one device')
        ei = _graph_edge_index(graph)
        _check_device(ei, layout.pairs.device, 'edge_index')
        x = graph.x if not isinstance(graph, dict) else graph.get('x')
        if isinstance(x, torch.Tensor):
            n_nodes = int(x.shape[0])
            if n_nodes != int(layout.num_nodes):
                raise ValueError(
                    f'graph x has {n_nodes} nodes but layout num_nodes is '
                    f'{layout.num_nodes}; an isolated node may be missing '
                    'from the layout')
        else:
            n_nodes = int(layout.num_nodes)
        _validate_layout_alignment(ei, n_nodes, layout)
        P = len(layout.pairs)
        k = _validate_k(k, P)
        return ei, n_nodes, P, k

    # ---------------- entry point ----------------

    def score(self, graph: Any, layout: PhysicalPairLayout, k: int, *,
              meter: Any = None, random_scores: Any = None,
              generator: Any = None) -> PairScoreResult:
        if self.name == 'random':
            return self._score_random(graph, layout, k, random_scores, generator)
        if self.name == 'distance':
            return self._score_distance(graph, layout, k)
        if self.name == 'degree':
            return self._score_degree(graph, layout, k)
        if self.name == 'stellar_binding_proxy':
            return self._score_binding(graph, layout, k)
        if self.name == 'edge_feature_grad_x_input':
            return self._score_grad_x_input(graph, layout, k, meter)
        if self.name == 'provided_rl':
            return self._score_provided_rl(graph, layout, k, meter)
        if self.name == 'supervised_raw':
            return self._score_supervised_raw(graph, layout, k)
        raise ValueError(f'unknown scorer name {self.name!r}')

    def training_cost(self) -> Dict[str, Any]:
        if self.name != 'supervised_raw':
            return {'total_calls': 0}
        _, artifact = self._validated_supervised()
        prov = artifact.get('label_provenance')
        if not isinstance(prov, dict):
            raise ValueError(
                'malformed artifact: label_provenance must be a mapping')
        total = prov.get('total_calls')
        if isinstance(total, bool) or not isinstance(total, numbers.Integral) \
                or int(total) < 0:
            raise ValueError(
                'malformed artifact: label_provenance total_calls must be a '
                f'non-negative integer, got {total!r}; no fallback cost sums '
                'are reported')
        return copy.deepcopy({
            'total_calls': int(total),
            'model_state_hash': artifact.get('model_state_hash'),
            'split_hash': artifact.get('split_hash'),
            'label_definition': artifact.get('label_definition'),
        })

    # ---------------- random ----------------

    def _score_random(self, graph, layout, k, random_scores, generator):
        self._base(graph, layout, k)
        P = len(layout.pairs)
        device = layout.pairs.device
        if random_scores is not None:
            vec = torch.as_tensor(random_scores, dtype=torch.float32,
                                  device=device)
            if vec.shape != (P,) or not torch.isfinite(vec).all():
                raise ValueError(
                    'random_scores must be a finite floating vector of length P')
            role = 'provided_vector'
        elif generator is not None:
            if not isinstance(generator, torch.Generator):
                raise ValueError('generator must be a torch.Generator')
            vec = torch.rand(P, generator=generator, device=device,
                             dtype=torch.float32)
            role = 'seeded_generator'
        else:
            raise ValueError(
                'random scorer requires provided finite random_scores or an '
                'explicit seeded generator; no implicit global RNG draw')
        return _finish(vec, layout, {
            'method': 'random',
            'randomness_role': role,
            'random_scores': vec.detach().clone(),
        })

    # ---------------- distance ----------------

    def _graph_pos(self, graph, n_nodes, device):
        pos = graph.pos if not isinstance(graph, dict) else graph.get('pos')
        if not isinstance(pos, torch.Tensor) or not torch.is_floating_point(pos) \
                or pos.ndim != 2 or pos.shape[0] != n_nodes:
            raise ValueError('graph.pos must be a finite floating [N, d] tensor')
        _check_device(pos, device, 'pos')
        if not torch.isfinite(pos).all():
            raise ValueError('graph.pos must be finite (got nan/inf)')
        return pos

    def _score_distance(self, graph, layout, k):
        self._base(graph, layout, k)
        pos = self._graph_pos(graph, int(layout.num_nodes), layout.pairs.device)
        pairs = layout.pairs.to(pos.device)
        dist = torch.linalg.vector_norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], dim=1)
        return _finish(-dist, layout, {'method': 'distance'})

    # ---------------- degree ----------------

    def _score_degree(self, graph, layout, k):
        self._base(graph, layout, k)
        n_nodes = int(layout.num_nodes)
        device = layout.pairs.device
        deg = torch.zeros(n_nodes, dtype=torch.float32, device=device)
        pairs = layout.pairs.to(device)
        deg.index_add_(0, pairs[:, 0], torch.ones(len(pairs), device=device))
        deg.index_add_(0, pairs[:, 1], torch.ones(len(pairs), device=device))
        scores = (deg[pairs[:, 0]] + deg[pairs[:, 1]]) / 2.0
        return _finish(scores, layout, {'method': 'degree'})

    # ---------------- stellar_binding_proxy ----------------

    def _score_binding(self, graph, layout, k):
        self._base(graph, layout, k)
        eps = self.distance_epsilon
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real) \
                or not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError(
                f'distance_epsilon must be positive finite, got {eps!r}')
        eps = float(eps)
        if not isinstance(self.distance_unit, str) or not self.distance_unit.strip():
            raise ValueError('distance_unit must be a nonempty declaration')
        if not isinstance(self.stellar_mass_unit, str) \
                or not self.stellar_mass_unit.strip():
            raise ValueError('stellar_mass_unit must be a nonempty declaration')
        device = layout.pairs.device
        n_nodes = int(layout.num_nodes)
        stellar = graph.stellar_mass if not isinstance(graph, dict) \
            else graph.get('stellar_mass')
        if not isinstance(stellar, torch.Tensor) \
                or not torch.is_floating_point(stellar) \
                or stellar.ndim != 1 or stellar.shape[0] != n_nodes:
            raise ValueError('graph.stellar_mass must be a floating [N] tensor')
        _check_device(stellar, device, 'stellar_mass')
        if not torch.isfinite(stellar).all() or not bool((stellar > 0).all()):
            raise ValueError('graph.stellar_mass must be finite positive')
        pos = self._graph_pos(graph, n_nodes, device)
        pairs = layout.pairs.to(device)
        dist = torch.linalg.vector_norm(
            pos[pairs[:, 0]] - pos[pairs[:, 1]], dim=1).clamp_min(eps)
        scores = torch.log(stellar[pairs[:, 0]]) \
            + torch.log(stellar[pairs[:, 1]]) - torch.log(dist)
        return _finish(scores, layout, {
            'method': 'stellar_binding_proxy',
            'distance_epsilon': eps,
            'distance_unit': self.distance_unit,
            'stellar_mass_unit': self.stellar_mass_unit,
        })

    # ---------------- edge_feature_grad_x_input ----------------

    def _score_grad_x_input(self, graph, layout, k, meter):
        self._base(graph, layout, k)
        if meter is None:
            raise ValueError(
                'edge_feature_grad_x_input requires a MeteredPredictor meter')
        result = meter.call(graph, phase='scoring', grad_edge_attr=True)
        leaf = result.edge_attr
        if leaf is None or not isinstance(leaf, torch.Tensor):
            raise ValueError('meter did not return an edge_attr leaf')
        if not result.prediction.requires_grad:
            # Frozen/parameter-free model: the prediction has no derivative
            # graph at all, so there is no edge saliency. Report explicit
            # zeros rather than raising inside autograd.
            stored = torch.zeros(leaf.shape[0], dtype=leaf.dtype,
                                 device=leaf.device)
            metadata = {'method': 'edge_feature_grad_x_input',
                        'unused_edge_input': True,
                        'edge_saliency_status': 'unused_edge_zero'}
        else:
            grads = torch.autograd.grad(result.prediction, leaf,
                                        allow_unused=True)[0]
            if grads is None:
                stored = torch.zeros(leaf.shape[0], dtype=leaf.dtype,
                                     device=leaf.device)
                metadata = {'method': 'edge_feature_grad_x_input',
                            'unused_edge_input': True,
                            'edge_saliency_status': 'unused_edge_zero'}
            else:
                stored = (grads.detach() * leaf.detach()).abs().sum(dim=1)
                metadata = {'method': 'edge_feature_grad_x_input',
                            'unused_edge_input': False}
        scores = _mean_over_real_copies(stored, layout)
        return _finish(scores, layout, metadata)

    # ---------------- provided_rl ----------------

    def _score_provided_rl(self, graph, layout, k, meter):
        self._base(graph, layout, k)
        if self.policy is None or not isinstance(self.policy, nn.Module):
            raise ValueError('provided_rl requires a provided policy module')
        names = self.edge_feature_names
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError(
                'provided_rl requires a declared nonempty edge_feature_names schema')
        for name in names:
            if name not in VALID_EDGE_FEATURE_NAMES:
                raise ValueError(f'unknown edge feature name: {name}. '
                                 f'Valid: {VALID_EDGE_FEATURE_NAMES}')
        if len(set(names)) != len(names):
            raise ValueError(
                f'duplicate edge feature name in provided_rl schema: {list(names)}')
        if meter is None:
            raise ValueError('provided_rl requires a MeteredPredictor meter')
        device = layout.pairs.device
        result = meter.call(graph, phase='scoring', return_embeddings=True)
        emb = result.embeddings
        if emb is None:
            raise ValueError('meter did not return node embeddings')
        ctx = emb.mean(dim=0)
        ea = graph.edge_attr if not isinstance(graph, dict) else graph.get('edge_attr')
        ei = _graph_edge_index(graph)
        if not isinstance(ea, torch.Tensor) or ea.ndim != 2 \
                or ea.shape[0] != ei.shape[1] or ea.shape[1] != len(names):
            raise ValueError(
                f'graph edge_attr shape {tuple(ea.shape) if isinstance(ea, torch.Tensor) else ea} '
                f'must be (E, F={len(names)}) matching the declared schema')
        if not torch.is_floating_point(ea) or not torch.isfinite(ea).all():
            raise ValueError('graph edge_attr must be a finite floating tensor')
        _check_device(ea, device, 'edge_attr')
        # Second orientation: cloned attrs with signed mass_ratio reversed,
        # endpoints swapped. Other supported names are symmetric.
        ea2 = ea.detach().clone()
        if 'mass_ratio' in names:
            ea2[:, names.index('mass_ratio')] = \
                -ea2[:, names.index('mass_ratio')]
        ei2 = ei[[1, 0]].clone()
        policy = self.policy
        flags_before = [(n, m.training) for n, m in policy.named_modules()]
        try:
            policy.eval()
            with torch.no_grad():
                logits1 = policy(ea.detach().clone(), emb, ei, ctx)
                logits2 = policy(ea2, emb, ei2, ctx)
        finally:
            modules = list(policy.named_modules())
            for (_, flag), (_, module) in zip(flags_before, modules):
                module.training = flag
        for tag, logits in (('orientation_0', logits1), ('orientation_1', logits2)):
            if not isinstance(logits, torch.Tensor) \
                    or not torch.is_floating_point(logits) \
                    or logits.reshape(-1).numel() != ei.shape[1] \
                    or not torch.isfinite(logits).all():
                raise ValueError(
                    f'policy {tag} logits must be finite floating with one value per stored edge')
        avg = (logits1.reshape(-1) + logits2.reshape(-1)) / 2
        scores = _mean_over_real_copies(avg, layout)
        return _finish(scores, layout, {
            'method': 'provided_rl',
            'policy_state_hash': model_state_hash(policy.state_dict()),
            'edge_feature_names': list(names),
            'policy_passes': 2,
            'predictor_embedding_calls': 1,
        })

    # ---------------- supervised_raw ----------------

    def _validated_supervised(self):
        """Shared prerequisites + current state/split/schema validation."""
        scorer = self.raw_scorer
        artifact = self.raw_artifact
        if scorer is None or not isinstance(scorer, RawBudgetPairScorer):
            raise ValueError(
                'supervised_raw missing prerequisite: a RawBudgetPairScorer '
                'raw_scorer and validated trained artifact are required')
        if not isinstance(artifact, dict) or not artifact:
            raise ValueError(
                'supervised_raw requires a validated trained artifact')
        if not isinstance(self.split_hash, str) or not self.split_hash.strip():
            raise ValueError('supervised_raw requires an expected split_hash')
        validate_supervised_artifact(
            scorer, artifact,
            expected_split=self.split_hash,
            expected_state_hash=artifact.get('model_state_hash'),
            expected_schema=scorer.declared_schema,
        )
        return scorer, artifact

    def _score_supervised_raw(self, graph, layout, k):
        self._base(graph, layout, k)
        scorer, artifact = self._validated_supervised()
        x = graph.x if not isinstance(graph, dict) else graph.get('x')
        ea = graph.edge_attr if not isinstance(graph, dict) else graph.get('edge_attr')
        ei = _graph_edge_index(graph)
        graph_dict = {'edge_index': ei, 'edge_attr': ea, 'x': x}
        flags_before = [(n, m.training) for n, m in scorer.named_modules()]
        try:
            scorer.eval()
            with torch.no_grad():
                scores = scorer(graph_dict, layout, int(k))
        finally:
            modules = list(scorer.named_modules())
            for (_, flag), (_, module) in zip(flags_before, modules):
                module.training = flag
        return _finish(scores.detach().reshape(len(layout.pairs)), layout, {
            'method': 'supervised_raw',
            'model_state_hash': artifact.get('model_state_hash'),
            'split_hash': artifact.get('split_hash'),
        })
