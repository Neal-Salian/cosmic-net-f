"""Metered frozen predictor for matched-budget benchmarking.

Wraps actual model forward calls with exact per-phase accounting. Builds a
cloned single-graph Batch internally, physically slices stored edge columns
for masks, and preserves model/input state across frozen evaluation.
"""

import copy
import inspect
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch_geometric.data import Batch, Data

from data.provenance import model_state_hash

PHASES = ('scoring', 'training_label', 'shared_full',
          'evaluation', 'training', 'oracle')


@dataclass
class PredictorResult:
    prediction: torch.Tensor
    embeddings: Optional[torch.Tensor]
    edge_attr: Optional[torch.Tensor]
    elapsed_seconds: float
    kept_edge_count: int


def _cuda_devices_for(model, batch):
    devices = set()
    for tensor in (batch.x, batch.edge_index, batch.edge_attr):
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            devices.add(tensor.device)
    for param in model.parameters():
        if param.is_cuda:
            devices.add(param.device)
    for buf in model.buffers():
        if isinstance(buf, torch.Tensor) and buf.is_cuda:
            devices.add(buf.device)
    return devices


def _synchronize(model, batch):
    if not torch.cuda.is_available():
        return
    for device in _cuda_devices_for(model, batch):
        torch.cuda.synchronize(device)


def _forward_accepts_return_embeddings(model):
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return True
    params = sig.parameters.values()
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params) or \
        'return_embeddings' in sig.parameters


def _clone_graph(graph):
    """Clone a single-graph Data with detached tensor payload.

    copy.deepcopy fails on non-leaf tensors requiring grad; PyG clone plus a
    detached clone of each tensor payload keeps caller grads/storage
    untouched while giving the meter fresh tensors to slice.
    """
    cloned = graph.clone()
    for key in ('x', 'edge_index', 'edge_attr'):
        tensor = getattr(cloned, key, None)
        if isinstance(tensor, torch.Tensor):
            setattr(cloned, key, tensor.detach().clone())
    return cloned


class MeteredPredictor:
    """Frozen-model forward meter with exact per-phase call accounting."""

    def __init__(self, model):
        self._model = model
        self._counts: Dict[str, Dict[str, float]] = {
            phase: {'forward_calls': 0, 'graph_evaluations': 0,
                    'elapsed_seconds': 0.0}
            for phase in PHASES
        }

    def snapshot(self):
        return copy.deepcopy(self._counts)

    def model_state_hash(self):
        return model_state_hash(self._model.state_dict())

    def call(self, graph, mask=None, *, phase='evaluation',
             return_embeddings=False, grad_edge_attr=False):
        if phase not in self._counts:
            raise ValueError(f'unknown phase {phase!r}; '
                             f'expected one of {list(PHASES)}')
        n_edges, device = self._validate_inputs(graph, mask)
        if return_embeddings and \
                not _forward_accepts_return_embeddings(self._model):
            raise ValueError('model does not provide embeddings, '
                             'but return_embeddings=True was requested')

        cloned = _clone_graph(graph)
        if mask is not None:
            mask = mask.to(device)
            cloned.edge_index = cloned.edge_index[:, mask]
            cloned.edge_attr = cloned.edge_attr[mask]
        kept_edge_count = cloned.edge_index.shape[1]
        batch = Batch.from_data_list([cloned])
        n_nodes = cloned.x.shape[0]

        entry = self._counts[phase]
        entry['forward_calls'] += 1
        entry['graph_evaluations'] += 1

        model = self._model
        training_before = {name: mod.training
                           for name, mod in model.named_modules()}
        buffers_before = {name: buf.detach().clone()
                          for name, buf in model.named_buffers()}
        grad_enabled_before = torch.is_grad_enabled()

        _synchronize(model, batch)
        start = time.perf_counter()
        result = None
        try:
            model.eval()
            if grad_edge_attr:
                leaf = batch.edge_attr.detach().clone().requires_grad_(True)
                batch.edge_attr = leaf
                with torch.enable_grad():
                    raw = self._invoke(batch, return_embeddings)
                edge_attr_out = leaf
            else:
                with torch.no_grad():
                    raw = self._invoke(batch, return_embeddings)
                edge_attr_out = None
            prediction, embeddings = self._parse_output(
                raw, return_embeddings, n_nodes)
            result = PredictorResult(
                prediction=prediction,
                embeddings=embeddings,
                edge_attr=edge_attr_out,
                elapsed_seconds=0.0,  # stamped in finally after stop timer
                kept_edge_count=kept_edge_count,
            )
            return result
        finally:
            _synchronize(model, batch)
            elapsed = time.perf_counter() - start
            entry['elapsed_seconds'] += elapsed
            for name, mod in model.named_modules():
                mod.training = training_before[name]
            named_modules = dict(model.named_modules())
            for dotted, saved in buffers_before.items():
                if '.' in dotted:
                    parent_name, leaf = dotted.rsplit('.', 1)
                    owner = named_modules[parent_name]
                else:
                    owner, leaf = model, dotted
                current = owner._buffers.get(leaf)
                if current is None or current.shape != saved.shape or \
                        current.dtype != saved.dtype or \
                        not torch.equal(current, saved):
                    owner._buffers[leaf] = saved if current is None \
                        else saved.to(current.device)
            torch.set_grad_enabled(grad_enabled_before)
            if result is not None:
                result.elapsed_seconds = elapsed

    def _invoke(self, batch, return_embeddings):
        if _forward_accepts_return_embeddings(self._model):
            return self._model(batch, return_embeddings=return_embeddings)
        if return_embeddings:
            raise ValueError('model does not provide embeddings, '
                             'but return_embeddings=True was requested')
        return self._model(batch)

    @staticmethod
    def _parse_output(raw, return_embeddings, n_nodes):
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            prediction, embeddings = raw[0], raw[1]
        else:
            prediction, embeddings = raw, None
        if not isinstance(prediction, torch.Tensor):
            raise ValueError('model prediction must be a scalar tensor, '
                             f'got {type(prediction).__name__}')
        if prediction.numel() != 1:
            raise ValueError('model prediction must be a scalar tensor '
                             f'(single element), got shape {tuple(prediction.shape)}')
        if not torch.isfinite(prediction).all():
            raise ValueError('model returned a nonfinite prediction')
        if return_embeddings:
            if embeddings is None:
                raise ValueError('model did not provide embeddings, '
                                 'but return_embeddings=True was requested')
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError('model embeddings must be a tensor, '
                                 f'got {type(embeddings).__name__}')
            if embeddings.dim() != 2 or embeddings.shape[0] != n_nodes:
                raise ValueError('model embeddings must have shape [N, D] '
                                 f'matching {n_nodes} nodes, got '
                                 f'{tuple(embeddings.shape)}')
            if not torch.isfinite(embeddings).all():
                raise ValueError('model returned nonfinite embeddings')
        else:
            embeddings = None
        return prediction, embeddings

    @staticmethod
    def _validate_inputs(graph, mask):
        if not isinstance(graph, Data):
            raise ValueError('graph must be a single PyG Data object, '
                             f'got {type(graph).__name__}')
        if getattr(graph, 'num_graphs', 1) != 1:
            raise ValueError('graph must be a single graph, not a batch')
        x = getattr(graph, 'x', None)
        edge_index = getattr(graph, 'edge_index', None)
        edge_attr = getattr(graph, 'edge_attr', None)
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError('graph.x must be a finite floating tensor')
        if not torch.isfinite(x).all():
            raise ValueError('graph.x must be finite (got nan/inf)')
        if x.dim() != 2:
            raise ValueError('graph.x must have shape [N, D]')
        n_nodes = x.shape[0]
        if not isinstance(edge_index, torch.Tensor) or \
                edge_index.dtype != torch.long or edge_index.dim() != 2 or \
                edge_index.shape[0] != 2:
            raise ValueError('graph.edge_index must be a LongTensor of '
                             'shape [2, E]')
        n_edges = edge_index.shape[1]
        if n_edges > 0:
            endpoint_lo = int(edge_index.min())
            endpoint_hi = int(edge_index.max())
            if endpoint_lo < 0 or endpoint_hi >= n_nodes:
                raise ValueError('graph.edge_index endpoints out of bounds '
                                 f'for {n_nodes} nodes')
        if not isinstance(edge_attr, torch.Tensor) or \
                not edge_attr.is_floating_point() or edge_attr.dim() != 2 or \
                edge_attr.shape[0] != n_edges:
            raise ValueError('graph.edge_attr must be a floating tensor '
                             'with one row per stored edge column')
        if not torch.isfinite(edge_attr).all():
            raise ValueError('graph.edge_attr must be finite (got nan/inf)')
        device = edge_index.device
        if x.device != device or edge_attr.device != device:
            raise ValueError('graph.x, graph.edge_index and graph.edge_attr '
                             'must live on the same device')
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError('mask must be a bool tensor of stored-edge '
                                 'length on the graph device')
            if mask.dtype != torch.bool:
                raise ValueError('mask dtype must be bool')
            if mask.dim() != 1 or mask.shape[0] != n_edges:
                raise ValueError('mask shape must match stored-edge length')
            if mask.device != device:
                raise ValueError('mask must live on the graph device')
        return n_edges, device
