"""Raw budget pair scorer for physics-informed graph sparsification."""

from typing import Dict, List, Optional, Tuple, Any
import copy
import math
import numbers

import torch
import torch.nn as nn
from rls.constraints import PhysicalPairLayout
from data.provenance import (
    model_state_hash as _model_state_hash,
    hash_payload as _hash_payload,
    canonical_json as _canonical_json,
)


VALID_EDGE_FEATURE_NAMES = ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep']


def _declared_schema(node_dim: int, edge_feature_names: List[str]) -> str:
    return f"node_dim={node_dim};features={','.join(edge_feature_names)}"


class RawBudgetPairScorer(nn.Module):
    """Scorer producing [P] scores from raw pair features (4*D+F+1)."""

    def __init__(
        self,
        node_dim: int,
        edge_feature_names: List[str],
        hidden_dim: int = 16,
    ):
        super().__init__()
        if isinstance(node_dim, bool) or not isinstance(node_dim, numbers.Integral) or node_dim <= 0:
            raise ValueError(f'node_dim must be a positive integer, got {node_dim}')
        if not isinstance(edge_feature_names, (list, tuple)) or not edge_feature_names:
            raise ValueError('edge_feature_names must be a nonempty list')
        seen = set()
        for name in edge_feature_names:
            if name not in VALID_EDGE_FEATURE_NAMES:
                raise ValueError(f'unknown edge feature name: {name}. Valid: {VALID_EDGE_FEATURE_NAMES}')
            if name in seen:
                raise ValueError(f'duplicate edge feature name: {name}')
            seen.add(name)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, numbers.Integral) or hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be a positive integer, got {hidden_dim}')
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.edge_feature_names = list(edge_feature_names)
        self.num_edge_features = len(self.edge_feature_names)
        self.total_input_width = 4 * self.node_dim + self.num_edge_features + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.total_input_width, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        width = self.total_input_width
        self.register_buffer('norm_mean', torch.zeros(width))
        self.register_buffer('norm_std', torch.ones(width))
        self._norm_fitted = False
        self._norm_split_hash: Optional[str] = None
        self._norm_ids: Optional[List[str]] = None
        self._norm_budgets: Optional[List[int]] = None
        self._norm_schema: Optional[str] = None
        self._norm_training_content_hash: Optional[str] = None
        self._norm_fit_records: Optional[List[Dict[str, Any]]] = None

    @property
    def declared_schema(self) -> str:
        return _declared_schema(self.node_dim, self.edge_feature_names)

    def _scorer_dtype(self):
        return self.mlp[0].weight.dtype

    def _scorer_device(self):
        return self.mlp[0].weight.device

    # ---------------- feature construction ----------------

    def _construct_pair_features(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        layout: PhysicalPairLayout,
        k: int,
    ) -> torch.Tensor:
        P = len(layout.pairs)
        D = self.node_dim
        F = self.num_edge_features
        if P == 0:
            return torch.zeros(0, self.total_input_width, device=x.device, dtype=x.dtype)
        src = layout.pairs[:, 0]
        dst = layout.pairs[:, 1]
        endpoint_sum = x[src] + x[dst]
        endpoint_diff = torch.abs(x[src] - x[dst])
        sym_edge_feats = self._sym_mean_edge_features(edge_attr, layout, F)
        global_mean = x.mean(dim=0)
        global_mean_expanded = global_mean.view(1, -1).expand(P, D)
        global_max = x.amax(dim=0)
        global_max_expanded = global_max.view(1, -1).expand(P, D)
        k_norm = torch.full((P, 1), float(k) / max(float(P), 1.0), dtype=x.dtype, device=x.device)
        return torch.cat([
            endpoint_sum, endpoint_diff, sym_edge_feats,
            global_mean_expanded, global_max_expanded, k_norm,
        ], dim=1)

    def _sym_mean_edge_features(self, edge_attr, layout, F: int) -> torch.Tensor:
        P = len(layout.pairs)
        if edge_attr.numel() == 0 or P == 0:
            return torch.zeros(P, F, device=edge_attr.device, dtype=edge_attr.dtype
                               if torch.is_floating_point(edge_attr) else torch.float32)
        real_indices = torch.where(layout.real)[0]
        n_real = len(real_indices)
        if n_real == 0:
            return torch.zeros(P, F, device=edge_attr.device, dtype=edge_attr.dtype)
        pair_lists: List[List[torch.Tensor]] = [[] for _ in range(P)]
        for idx in range(n_real):
            edge_col = real_indices[idx].item()
            p = layout.inverse[idx].item()
            if 0 <= p < P:
                feat = edge_attr[edge_col].clone()
                if 'mass_ratio' in self.edge_feature_names:
                    mr_idx = self.edge_feature_names.index('mass_ratio')
                    feat[mr_idx] = torch.abs(feat[mr_idx])
                pair_lists[p].append(feat)
        out = []
        for p in range(P):
            if pair_lists[p]:
                out.append(torch.stack(pair_lists[p], dim=0).mean(dim=0))
            else:
                out.append(torch.zeros(F, device=edge_attr.device, dtype=edge_attr.dtype))
        return torch.stack(out, dim=0)

    # ---------------- shared validation ----------------

    def _validate_k(self, k: Any, P: int) -> None:
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ValueError(f'k must be an integer budget, got {k!r}')
        k = int(k)
        if k < 0:
            raise ValueError(f'k must be non-negative, got {k}')
        if k > P:
            raise ValueError(f'k={k} exceeds P={P}')

    def _validate_graph_tensors(self, graph: Dict, need_cluster: bool = True) -> Tuple[Any, Any, Any, int, int]:
        try:
            ei = graph['edge_index']
            ea = graph['edge_attr']
            x = graph['x']
        except KeyError as e:
            raise ValueError(f'graph missing key: {e}')
        if need_cluster and graph.get('cluster_id') is None:
            raise ValueError('graph must have stable cluster_id')
        if not isinstance(ei, torch.Tensor) or ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError('edge_index must be [2, E] integer tensor')
        if ei.dtype not in (torch.int32, torch.int64):
            raise ValueError('edge_index must be integer dtype')
        if ei.device != self._scorer_device():
            raise ValueError(f'edge_index device {ei.device} != scorer device {self._scorer_device()}')
        if int(ei.numel()) > 0 and bool((ei < 0).any()):
            raise ValueError('negative node indices are invalid')
        E = int(ei.shape[1])
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError('x must be a [N, D] floating tensor')
        if not torch.is_floating_point(x):
            raise ValueError('x must be a floating-point tensor')
        if x.shape[1] != self.node_dim:
            raise ValueError(f'x width {x.shape[1]} != node_dim {self.node_dim}')
        if x.dtype != self._scorer_dtype():
            raise ValueError(f'x dtype {x.dtype} != scorer dtype {self._scorer_dtype()}')
        if x.device != self._scorer_device():
            raise ValueError(f'x device {x.device} != scorer device {self._scorer_device()}')
        if not torch.isfinite(x).all():
            raise ValueError('x contains non-finite values')
        if not isinstance(ea, torch.Tensor) or ea.ndim != 2:
            raise ValueError('edge_attr must be [E, F] floating tensor')
        if not torch.is_floating_point(ea):
            raise ValueError('edge_attr must be a floating-point tensor')
        if ea.shape[0] != E or ea.shape[1] != self.num_edge_features:
            raise ValueError(f'edge_attr shape {tuple(ea.shape)} != (E={E}, F={self.num_edge_features})')
        if ea.dtype != self._scorer_dtype():
            raise ValueError(f'edge_attr dtype {ea.dtype} != scorer dtype {self._scorer_dtype()}')
        if ea.device != self._scorer_device():
            raise ValueError(f'edge_attr device {ea.device} != scorer device {self._scorer_device()}')
        if not torch.isfinite(ea).all():
            raise ValueError('edge_attr contains non-finite values')
        n_nodes = int(x.shape[0])
        return ei, ea, x, n_nodes, E

    def _expected_layout(self, ei, n_nodes):
        return PhysicalPairLayout.from_edge_index(ei, n_nodes)

    def _validate_layout_alignment(self, ei, n_nodes: int, layout: PhysicalPairLayout) -> None:
        if not isinstance(layout, PhysicalPairLayout):
            raise ValueError('layout must be a PhysicalPairLayout')
        for t, name in ((layout.pairs, 'layout pairs'), (layout.inverse, 'layout inverse'), (layout.real, 'layout real')):
            if not isinstance(t, torch.Tensor):
                raise ValueError(f'{name} must be a tensor')
            if t.device != ei.device:
                raise ValueError(f'{name} device {t.device} != edge_index device {ei.device}')
        try:
            expected = PhysicalPairLayout.from_edge_index(ei, n_nodes)
        except ValueError as e:
            raise ValueError(str(e))
        if expected.num_nodes != layout.num_nodes:
            raise ValueError(f'layout num_nodes {layout.num_nodes} != graph N {expected.num_nodes}')
        if len(expected.pairs) != len(layout.pairs):
            raise ValueError('layout P does not match graph topology')
        if int(expected.real.numel()) != int(layout.real.numel()) or not torch.equal(
                expected.real.cpu(), layout.real.cpu()):
            raise ValueError('layout stored-edge mask does not match graph')
        if not torch.equal(expected.pairs.cpu(), layout.pairs.cpu()):
            raise ValueError('layout pairs do not match graph topology')
        if not torch.equal(expected.inverse.cpu(), layout.inverse.cpu()):
            raise ValueError('layout inverse does not match graph topology')

    @staticmethod
    def _validate_id_lists(train_ids, heldout_ids, split_hash) -> None:
        if not isinstance(train_ids, (list, tuple)) or not train_ids:
            raise ValueError('train_ids must be a nonempty list')
        if not isinstance(heldout_ids, (list, tuple)):
            raise ValueError('heldout_ids must be a list')
        for v in list(train_ids) + list(heldout_ids):
            if not isinstance(v, str) or not v:
                raise ValueError('IDs must be nonempty strings')
        if len(set(train_ids)) != len(list(train_ids)):
            raise ValueError('duplicate declared train IDs')
        if len(set(heldout_ids)) != len(list(heldout_ids)):
            raise ValueError('duplicate declared heldout IDs')
        if set(heldout_ids) & set(train_ids):
            raise ValueError('overlap between train_ids and heldout_ids')
        if not isinstance(split_hash, str) or not split_hash.strip():
            raise ValueError('split hash must be nonempty')

    # ---------------- normalization ----------------

    def fit_normalization(self, examples, train_ids, heldout_ids, split_hash) -> None:
        self._validate_id_lists(train_ids, heldout_ids, split_hash)
        if not examples:
            raise ValueError('empty examples list in fit_normalization')
        # Per-example validation (no state change)
        parsed = []
        seen_pairs = set()
        for item in examples:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError('examples must be (graph, k) pairs')
            graph, k = item
            ei, ea, x, n_nodes, _E = self._validate_graph_tensors(graph, need_cluster=True)
            cid = graph.get('cluster_id')
            if cid not in train_ids:
                raise ValueError(f'unknown graph ID: {cid}')
            if cid in (heldout_ids or []):
                raise ValueError(f'unknown graph ID (heldout in fit): {cid}')
            if isinstance(k, bool) or not isinstance(k, numbers.Integral):
                raise ValueError(f'k must be an integer budget, got {k!r}')
            k = int(k)
            if k < 0:
                raise ValueError(f'k must be non-negative, got {k}')
            if n_nodes == 0:
                raise ValueError('zero-node normalization input rejected')
            layout = self._expected_layout(ei, n_nodes)
            P = len(layout.pairs)
            if P == 0:
                raise ValueError('zero-pair normalization input rejected')
            if k > P:
                raise ValueError(f'k={k} exceeds P={P} for graph')
            key = (cid, k)
            if key in seen_pairs:
                raise ValueError(f'duplicate (ID, k) pair: {key}')
            seen_pairs.add(key)
            parsed.append((graph, k, ei, ea, x, n_nodes, layout, P, cid))
        # Exact cohort: every declared train ID covered, no extras
        example_ids = {cid for (_, _, _, _, _, _, _, _, cid) in parsed}
        if set(train_ids) != example_ids:
            missing = set(train_ids) - example_ids
            extra = example_ids - set(train_ids)
            raise ValueError(f'declared train cohort mismatch: missing={sorted(missing)} extra={sorted(extra)}')
        # Canonicalize by (ID, k)
        order = sorted(range(len(parsed)), key=lambda i: (parsed[i][8], parsed[i][1]))
        parsed = [parsed[i] for i in order]
        schema = self.declared_schema
        # Per-record content hashes over actual data
        records = []
        for (graph, k, ei, ea, x, n_nodes, layout, P, cid) in parsed:
            ch = _model_state_hash({
                'cluster_id': cid,
                'k': int(k),
                'schema': schema,
                'edge_index': ei.detach().cpu(),
                'edge_attr': ea.detach().cpu(),
                'x': x.detach().cpu(),
            })
            records.append({'cluster_id': cid, 'k': int(k), 'content_hash': ch, 'P': int(P)})
        overall = _hash_payload({
            'schema': schema,
            'split_hash': split_hash,
            'train_ids': sorted(train_ids),
            'heldout_ids': sorted(heldout_ids or []),
            'records': [{'cluster_id': r['cluster_id'], 'k': r['k'], 'content_hash': r['content_hash']} for r in records],
        })
        # Features (detached)
        feats = []
        with torch.no_grad():
            for (graph, k, ei, ea, x, n_nodes, layout, P, cid) in parsed:
                f = self._construct_pair_features(x.detach(), ea.detach(), layout, int(k))
                feats.append(f.detach())
        X = torch.cat(feats, dim=0)
        mean = X.mean(dim=0).detach()
        std = X.std(dim=0, unbiased=False).detach()
        std = torch.where(std == 0, torch.ones_like(std), std).detach()
        # Commit state only now
        if tuple(self.norm_mean.shape) != tuple(mean.shape):
            raise ValueError('schema width mismatch on fit')
        with torch.no_grad():
            self.norm_mean.copy_(mean)
            self.norm_std.copy_(std)
        self._norm_fitted = True
        self._norm_split_hash = split_hash
        self._norm_ids = sorted(train_ids)
        self._norm_budgets = [r['k'] for r in records]
        self._norm_schema = schema
        self._norm_training_content_hash = overall
        self._norm_fit_records = copy.deepcopy(records)

    def apply_normalization(self, features: torch.Tensor) -> torch.Tensor:
        if not self._norm_fitted:
            return features
        return (features - self.norm_mean.to(dtype=features.dtype, device=features.device)) / \
            self.norm_std.to(dtype=features.dtype, device=features.device)

    # ---------------- forward ----------------

    def forward(self, graph: Dict, layout: PhysicalPairLayout, k: int) -> torch.Tensor:
        P = len(layout.pairs) if isinstance(layout, PhysicalPairLayout) else 0
        self._validate_k(k, P)
        ei, ea, x, n_nodes, _E = self._validate_graph_tensors(graph, need_cluster=False)
        self._validate_layout_alignment(ei, n_nodes, layout)
        P = len(layout.pairs)
        self._validate_k(k, P)
        if P == 0:
            return torch.zeros(0, device=x.device, dtype=x.dtype)
        if n_nodes == 0 and layout.num_nodes > 0:
            raise ValueError('zero-node graph rejected')
        pair_features = self._construct_pair_features(x, ea, layout, int(k))
        if self._norm_fitted:
            pair_features = self.apply_normalization(pair_features)
        return self.mlp(pair_features).squeeze(-1)

    # ---------------- persistence ----------------

    def _norm_meta_dict(self) -> Dict[str, Any]:
        return {
            'fitted': bool(self._norm_fitted),
            'split_hash': self._norm_split_hash,
            'train_ids': copy.deepcopy(self._norm_ids),
            'budgets': copy.deepcopy(self._norm_budgets),
            'schema': self._norm_schema,
            'content_hash': self._norm_training_content_hash,
            'records': copy.deepcopy(self._norm_fit_records),
        }

    def _restore_norm_meta(self, meta: Dict[str, Any]) -> None:
        if not isinstance(meta, dict):
            raise ValueError('invalid normalization metadata')
        schema = meta.get('schema')
        if meta.get('fitted'):
            if schema != self.declared_schema:
                raise ValueError(f'incompatible schema restore: {schema} != {self.declared_schema}')
        self._norm_fitted = bool(meta.get('fitted', False))
        self._norm_split_hash = meta.get('split_hash')
        self._norm_ids = copy.deepcopy(meta.get('train_ids'))
        self._norm_budgets = copy.deepcopy(meta.get('budgets'))
        self._norm_schema = schema
        self._norm_training_content_hash = meta.get('content_hash')
        self._norm_fit_records = copy.deepcopy(meta.get('records'))

    def get_extra_state(self) -> Dict[str, Any]:
        return self._norm_meta_dict()

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self._restore_norm_meta(state)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Pre-validate schema for a clear ValueError (not RuntimeError)
        incoming = None
        try:
            incoming = state_dict.get('_extra_state', None)
        except Exception:
            incoming = None
        if isinstance(incoming, dict) and incoming.get('fitted'):
            if incoming.get('schema') != self.declared_schema:
                raise ValueError(
                    f"incompatible schema restore: {incoming.get('schema')} != {self.declared_schema}")
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as e:
            raise ValueError(str(e))


# ---------------- label + training helpers ----------------

def _validate_label_calls(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError(f'label_predictor_calls must be a non-negative integer, got {v!r}')
    v = int(v)
    if v < 0:
        raise ValueError('label_predictor_calls must be non-negative')
    return v


def _validate_provenance(v: Any) -> Dict:
    if not isinstance(v, dict) or not v:
        raise ValueError('label_provenance must be a nonempty mapping')
    return v


def _validate_definition(v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError('label_definition must be a nonempty string')
    return v


def _label_hash(labels: torch.Tensor, definition: str, calls: int, prov: Dict) -> str:
    return _model_state_hash({
        'labels': labels.detach().cpu(),
        'definition': definition,
        'calls': int(calls),
        'provenance_json': _canonical_json(prov),
    })


def train_supervised_selector(
    scorer: RawBudgetPairScorer,
    examples: List[Dict[str, Any]],
    train_ids: List[str],
    heldout_ids: List[str],
    split_hash: str,
    seed: int,
    epochs: int,
    lr: float,
) -> Dict[str, Any]:
    # Validate ID lists / split
    if not isinstance(train_ids, (list, tuple)) or not train_ids:
        raise ValueError('train_ids must be a nonempty list')
    if not isinstance(heldout_ids, (list, tuple)):
        raise ValueError('heldout_ids must be a list')
    for v in list(train_ids) + list(heldout_ids):
        if not isinstance(v, str) or not v:
            raise ValueError('IDs must be nonempty strings')
    if len(set(train_ids)) != len(list(train_ids)):
        raise ValueError('duplicate declared train IDs')
    if len(set(heldout_ids)) != len(list(heldout_ids)):
        raise ValueError('duplicate declared heldout IDs')
    if set(heldout_ids) & set(train_ids):
        raise ValueError('overlap between train_ids and heldout_ids')
    if not isinstance(split_hash, str) or not split_hash.strip():
        raise ValueError('split hash must be nonempty')
    if not examples:
        raise ValueError('empty examples list')
    if isinstance(epochs, bool) or not isinstance(epochs, numbers.Integral) or int(epochs) < 1:
        raise ValueError(f'epochs must be a positive integer, got {epochs!r}')
    epochs = int(epochs)
    if isinstance(lr, bool) or not isinstance(lr, numbers.Real) or not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError(f'lr must be positive finite, got {lr!r}')
    lr = float(lr)
    if not isinstance(seed, numbers.Integral) or isinstance(seed, bool):
        raise ValueError(f'seed must be an integer, got {seed!r}')

    # Duplicate (ID, k)
    seen = set()
    for ex in examples:
        if not isinstance(ex, dict):
            raise ValueError('example must be a mapping')
        for key in ('graph', 'k', 'labels', 'label_definition', 'label_predictor_calls', 'label_provenance'):
            if key not in ex:
                raise ValueError(f'example missing key: {key}')
        ck = (ex['graph'].get('cluster_id'), ex['k'])
        if ck in seen:
            raise ValueError(f'duplicate (ID, k) pair: {ck}')
        seen.add(ck)

    # Require fitted normalization (no implicit fit)
    if not getattr(scorer, '_norm_fitted', False):
        raise ValueError('supervised training requires fitted normalization')
    if scorer._norm_split_hash != split_hash:
        raise ValueError('split hash mismatch with fitted normalization')
    if set(scorer._norm_ids or []) != set(train_ids):
        raise ValueError('train IDs do not match normalization metadata')

    # Validate each example fully (graphs, k, labels, metadata)
    s_dtype = scorer.mlp[0].weight.dtype
    s_device = scorer.mlp[0].weight.device
    parsed = []
    for ex in examples:
        graph, k, labels = ex['graph'], ex['k'], ex['labels']
        ei, ea, x, n_nodes, _E = scorer._validate_graph_tensors(graph, need_cluster=True)
        cid = graph.get('cluster_id')
        if cid not in train_ids or cid in (heldout_ids or []):
            raise ValueError(f'unknown graph ID in training: {cid}')
        layout = scorer._expected_layout(ei, n_nodes)
        P = len(layout.pairs)
        if P == 0:
            raise ValueError('zero-pair training input rejected')
        if n_nodes == 0:
            raise ValueError('zero-node training input rejected')
        scorer._validate_k(k, P)
        if not isinstance(labels, torch.Tensor):
            raise ValueError('labels must be a tensor')
        if labels.ndim != 1 or labels.shape[0] != P:
            raise ValueError(f'labels shape {tuple(labels.shape)} != [P={P}]')
        if not torch.is_floating_point(labels):
            raise ValueError('labels must be floating-point')
        if labels.dtype != s_dtype:
            raise ValueError(f'labels dtype {labels.dtype} != scorer dtype {s_dtype}')
        if labels.device != s_device:
            raise ValueError(f'labels device {labels.device} != scorer device {s_device}')
        if not torch.isfinite(labels).all():
            raise ValueError('labels contain non-finite values')
        definition = _validate_definition(ex['label_definition'])
        calls = _validate_label_calls(ex['label_predictor_calls'])
        prov = _validate_provenance(ex['label_provenance'])
        parsed.append((graph, int(k), labels, definition, calls, prov, ei, ea, x, layout, P, cid))

    # Exact match to fit records (membership/budget/content)
    fit_recs = {(r['cluster_id'], r['k']): r['content_hash'] for r in (scorer._norm_fit_records or [])}
    if set(fit_recs) != {(cid, k) for (graph, k, labels, d, c, p, ei, ea, x, layout, P, cid) in parsed}:
        raise ValueError('training (ID, k) set does not match fitted normalization records')
    for (graph, k, labels, d, c, p, ei, ea, x, layout, P, cid) in parsed:
        ch = _model_state_hash({
            'cluster_id': cid, 'k': int(k), 'schema': scorer._norm_schema,
            'edge_index': ei.detach().cpu(), 'edge_attr': ea.detach().cpu(), 'x': x.detach().cpu(),
        })
        if ch != fit_recs[(cid, k)]:
            raise ValueError(f'training content mismatch for {(cid, k)}')

    # Canonicalize by (ID, k)
    parsed = sorted(parsed, key=lambda t: (t[11], t[1]))

    caller_rng = torch.get_rng_state()
    flags_before = [(n, m.training) for n, m in scorer.named_modules()]
    try:
        local_gen = torch.Generator()
        local_gen.manual_seed(int(seed))
        # Build detached features
        with torch.no_grad():
            feats, labs, defs, calls_l, provs = [], [], [], [], []
            for (graph, k, labels, d, c, p, ei, ea, x, layout, P, cid) in parsed:
                f = scorer._construct_pair_features(x.detach(), ea.detach(), layout, int(k)).detach()
                f = scorer.apply_normalization(f).detach()
                feats.append(f)
                labs.append(labels.detach())
                defs.append(d)
                calls_l.append(int(c))
                provs.append(copy.deepcopy(p))
        X = torch.cat(feats, dim=0).detach()
        y = torch.cat(labs, dim=0).detach()
        label_hashes = [_label_hash(l, d, c, p) for (l, d, c, p) in zip(labs, defs, calls_l, provs)]

        with torch.enable_grad():
            scorer.eval()
            with torch.no_grad():
                loss_before = torch.nn.functional.mse_loss(scorer.mlp(X).squeeze(-1), y).item()
            # Train only mlp params; locally enable grad
            scorer.train()
            opt = torch.optim.Adam(scorer.mlp.parameters(), lr=lr)
            mse = torch.nn.MSELoss()
            for _ in range(epochs):
                perm = torch.randperm(X.shape[0], generator=local_gen)
                Xs, ys = X[perm], y[perm]
                opt.zero_grad()
                pred = scorer.mlp(Xs).squeeze(-1)
                loss = mse(pred, ys)
                loss.backward()
                opt.step()
            scorer.eval()
            with torch.no_grad():
                loss_after = torch.nn.functional.mse_loss(scorer.mlp(X).squeeze(-1), y).item()
    finally:
        torch.set_rng_state(caller_rng)
        # Restore exact flags
        mod_list = list(scorer.named_modules())
        for (n, flag), (_, m) in zip(flags_before, mod_list):
            m.training = flag

    state = scorer.state_dict()
    state_hash = _model_state_hash(state)
    artifact: Dict[str, Any] = {
        'model_state': copy.deepcopy(state),
        'model_state_hash': state_hash,
        'training_error_before': loss_before,
        'training_error_after': loss_after,
        'training_settings': {'epochs': epochs, 'lr': lr, 'seed': int(seed), 'objective': 'mse'},
        'label_definition': list(defs),
        'label_provenance': {
            'label_definitions': list(defs),
            'label_predictor_calls': list(calls_l),
            'label_provenances': copy.deepcopy(provs),
            'total_calls': int(sum(calls_l)),
        },
        'label_hashes': list(label_hashes),
        'normalization_metadata': {
            'split_hash': scorer._norm_split_hash,
            'train_ids': copy.deepcopy(scorer._norm_ids),
            'schema': scorer._norm_schema,
            'budgets': copy.deepcopy(scorer._norm_budgets),
            'training_content_hash': scorer._norm_training_content_hash,
        },
        'split_hash': split_hash,
    }
    return artifact


def validate_supervised_artifact(
    scorer: RawBudgetPairScorer,
    artifact: Dict[str, Any],
    expected_split: str,
    expected_state_hash: str,
    expected_schema: str,
) -> bool:
    if not getattr(scorer, '_norm_fitted', False):
        raise ValueError('scorer normalization not fitted')
    if artifact.get('split_hash') != expected_split:
        raise ValueError(f"artifact split_hash {artifact.get('split_hash')} != expected {expected_split}")
    if artifact.get('model_state_hash') != expected_state_hash:
        raise ValueError('artifact model_state_hash mismatch')
    current_hash = _model_state_hash(scorer.state_dict())
    if artifact.get('model_state_hash') != current_hash:
        raise ValueError(
            f"artifact state_hash {artifact.get('model_state_hash')} != current scorer state_hash {current_hash}")
    norm_meta = artifact.get('normalization_metadata', {}) or {}
    if norm_meta.get('schema') != expected_schema:
        raise ValueError(f"artifact schema {norm_meta.get('schema')} != expected {expected_schema}")
    ts = artifact.get('training_settings', {}) or {}
    ep = ts.get('epochs')
    if isinstance(ep, bool) or not isinstance(ep, numbers.Integral) or int(ep) < 1:
        raise ValueError(f'artifact epochs invalid: {ep!r}')
    if scorer._norm_split_hash != expected_split or norm_meta.get('split_hash') != expected_split:
        raise ValueError('split hash metadata mismatch')
    if scorer._norm_schema != expected_schema or norm_meta.get('schema') != scorer._norm_schema:
        raise ValueError('schema metadata mismatch')
    if scorer._norm_training_content_hash != norm_meta.get('training_content_hash'):
        raise ValueError('training content metadata mismatch')
    if sorted(scorer._norm_ids or []) != sorted(norm_meta.get('train_ids') or []):
        raise ValueError('train IDs metadata mismatch')
    if sorted(scorer._norm_budgets or []) != sorted(norm_meta.get('budgets') or []):
        raise ValueError('budgets metadata mismatch')
    if 'label_hashes' not in artifact or not isinstance(artifact['label_hashes'], list):
        raise ValueError('artifact missing label_hashes')
    # Saved-state integrity: hash the actual saved tensors, not only the declared string
    saved = artifact.get('model_state')
    if not isinstance(saved, dict) or not saved:
        raise ValueError('artifact missing model_state')
    try:
        saved_hash = _model_state_hash(saved)
    except Exception:
        raise ValueError('artifact model_state unhashable')
    if saved_hash != artifact.get('model_state_hash'):
        raise ValueError('artifact saved model_state hash mismatch')
    # Historical cost / label metadata consistency (present saved metadata only)
    lp = artifact.get('label_provenance', {}) or {}
    defs = lp.get('label_definitions')
    calls = lp.get('label_predictor_calls')
    provs = lp.get('label_provenances')
    total = lp.get('total_calls')
    hashes = artifact.get('label_hashes')
    n_fit = len(scorer._norm_fit_records or [])
    for arr, name in ((hashes, 'label_hashes'), (defs, 'label_definitions'),
                      (calls, 'label_predictor_calls'), (provs, 'label_provenances')):
        if not isinstance(arr, list) or not arr:
            raise ValueError(f'artifact {name} must be a nonempty list')
    if not (len(hashes) == len(defs) == len(calls) == len(provs) == n_fit):
        raise ValueError('artifact label arrays misaligned with fit records')
    for h in hashes:
        if not isinstance(h, str) or not h:
            raise ValueError('artifact label hashes invalid')
    for d in defs:
        if not isinstance(d, str) or not d.strip():
            raise ValueError('artifact label definitions invalid')
    for c in calls:
        if isinstance(c, bool) or not isinstance(c, numbers.Integral) or int(c) < 0:
            raise ValueError(f'artifact label cost invalid: {c!r}')
    for p in provs:
        if not isinstance(p, dict) or not p:
            raise ValueError('artifact label provenance invalid')
    if isinstance(total, bool) or not isinstance(total, numbers.Integral) or int(total) < 0:
        raise ValueError(f'artifact total_calls invalid: {total!r}')
    if int(total) != int(sum(int(c) for c in calls)):
        raise ValueError('artifact total_calls sum mismatch')
    if 'label_definition' in artifact and list(artifact['label_definition']) != list(defs):
        raise ValueError('artifact label_definition mismatch')
    lr_v = ts.get('lr')
    if isinstance(lr_v, bool) or not isinstance(lr_v, numbers.Real) or not math.isfinite(float(lr_v)) \
            or float(lr_v) <= 0:
        raise ValueError(f'artifact lr invalid: {lr_v!r}')
    for key in ('training_error_before', 'training_error_after'):
        v = artifact.get(key)
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            raise ValueError(f'artifact {key} invalid: {v!r}')
    return True
