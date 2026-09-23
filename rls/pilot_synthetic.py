"""Synthetic K4 diagnostic data, predictors, and exact oracle (Task 4 pilot).

Diagnostic blackboxes only; no astrophysical claim. Learning-family raw x
columns are [strength, pos_x, pos_y, family_code]; their EA schema is
['distance'] holding actual Euclidean position distances. Literal correctness
fixtures use separate toy additive weights and interaction factors in edge_attr.
For learning-family graphs, w_uv = (s_u + s_v) / (1 + d_uv).

- Additive family: prediction = sum of selected physical w; target = full sum.
- Interaction family: prediction = sum of w_e*w_f over disjoint selected
  physical pairs; target = strongest full-graph perfect-matching product.
- Literal fixture weights and interaction factors travel in edge_attr, so their
  meaning follows the stored edges under node relabeling.

Learning-family predictors read only endpoint strengths/positions and the
stored Euclidean distances; literal fixture predictors read stored edge_attr.
No predictor reads graph.y.
"""
import inspect
import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from data.provenance import hash_payload, model_state_hash, sha256_file

from rls.constraints import PhysicalPairLayout
from rls.structured_selection import enumerate_feasible_masks

RAW_X_COLUMNS = ['strength', 'pos_x', 'pos_y', 'family_code']
EA_FEATURE_NAMES = ['distance']

K4_NODES = 4
K4_PAIRS = 6
FEASIBLE_COUNTS = {2: 3, 3: 16, 4: 15}

LITERAL_ADDITIVE_W = [8.0, 5.0, 1.0, 2.0, 4.0, 3.0]
LITERAL_ADDITIVE_TARGET = 23.0
LITERAL_INTERACTION_TARGET = 8.0
# Constructor lookup for canonical fixture pair order a=01,b=02,c=03,d=12,e=13,f=23.
_LITERAL_W_BY_ENDPOINT = {
    (0, 1): 8.0, (0, 2): 5.0, (0, 3): 1.0,
    (1, 2): 2.0, (1, 3): 4.0, (2, 3): 3.0,
}

FAMILY_CODE = {'additive': 0.0, 'interaction': 1.0}


def _layout_of(data):
    return PhysicalPairLayout.from_edge_index(data.edge_index, int(data.x.shape[0]))


def _unique_pairs(edge_index):
    pairs = set()
    for u, v in edge_index.t().tolist():
        if u == v:
            continue
        pairs.add((min(u, v), max(u, v)))
    return sorted(pairs)


def _pair_weights_from_tensors(x, edge_index, edge_attr):
    """w per unique unordered non-self pair, sorted endpoint order."""
    pairs = _unique_pairs(edge_index)
    strengths = x[:, 0]
    dist_by_pair = {}
    for col in range(edge_index.shape[1]):
        u, v = int(edge_index[0, col]), int(edge_index[1, col])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        dist_by_pair.setdefault(key, []).append(float(edge_attr[col, 0]))
    weights = []
    for (u, v) in pairs:
        d = sum(dist_by_pair[(u, v)]) / len(dist_by_pair[(u, v)])
        w = (float(strengths[u]) + float(strengths[v])) / (1.0 + d)
        weights.append(w)
    return pairs, torch.tensor(weights, dtype=torch.float32)


def _single_graphs(batch):
    if hasattr(batch, 'num_graphs') and int(batch.num_graphs) != 1:
        raise ValueError('synthetic predictors accept one graph per call')
    if hasattr(batch, 'to_data_list'):
        datas = batch.to_data_list()
        if len(datas) != 1:
            raise ValueError('synthetic predictors accept one graph per call')
        return datas
    return [batch]


def _literal_pair_attributes(data):
    """Return physical pairs and mean stored toy attributes per pair."""
    pairs = _unique_pairs(data.edge_index)
    values = {}
    for col in range(data.edge_index.shape[1]):
        u, v = map(int, data.edge_index[:, col])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        values.setdefault(key, []).append(data.edge_attr[col].to(torch.float64))
    return pairs, torch.stack([
        torch.stack(values[p]).mean(0) for p in pairs
    ])


def physical_weights(data):
    """[P] weights in canonical layout order."""
    layout = _layout_of(data)
    _, w = _pair_weights_from_tensors(data.x, data.edge_index, data.edge_attr)
    # _unique_pairs returns sorted order == layout.pairs order for K4.
    assert len(w) == len(layout.pairs)
    return w


def _single_additive(data):
    _, w = _pair_weights_from_tensors(data.x, data.edge_index, data.edge_attr)
    return w.sum().reshape(1)


def _single_interaction(data):
    pairs, w = _pair_weights_from_tensors(data.x, data.edge_index, data.edge_attr)
    total = torch.zeros(1)
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if len(set(pairs[i]) | set(pairs[j])) == 4:
                total = total + w[i] * w[j]
    return total.reshape(1)


def perfect_matching_products(data):
    """The three full-K4 perfect-matching w products (diagnostic target rule)."""
    pairs, w = _pair_weights_from_tensors(data.x, data.edge_index, data.edge_attr)
    idx = {p: i for i, p in enumerate(pairs)}
    matchings = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    out = []
    for m in matchings:
        if all(p in idx for p in m):
            out.append(float(w[idx[m[0]]] * w[idx[m[1]]]))
    return out


def analytic_target_for(data, family):
    if family == 'additive':
        return float(_single_additive_full(data))
    if family == 'interaction':
        prods = perfect_matching_products_full(data)
        return float(max(prods))
    raise ValueError(f'unknown family {family!r}')


def _single_additive_full(data):
    return _single_additive(data)


def perfect_matching_products_full(data):
    return perfect_matching_products(data)


class SyntheticAdditivePredictor(nn.Module):
    """Frozen diagnostic: sum of physical w over sliced graph."""

    def forward(self, batch, return_embeddings=False, **kwargs):
        datas = _single_graphs(batch)
        preds = [_single_additive(d).to(torch.float32) for d in datas]
        out = torch.stack(preds).reshape(-1)
        if out.numel() == 1:
            return out[0].reshape(1)
        return out.reshape(1)


class SyntheticInteractionPredictor(nn.Module):
    """Frozen diagnostic: disjoint-pair w products over sliced graph."""

    def forward(self, batch, return_embeddings=False, **kwargs):
        datas = _single_graphs(batch)
        preds = [_single_interaction(d).to(torch.float32) for d in datas]
        out = torch.stack(preds).reshape(-1)
        if out.numel() == 1:
            return out[0].reshape(1)
        return out.reshape(1)


class LiteralAdditivePredictor(nn.Module):
    """Fixture predictor using additive weights transported in edge_attr."""

    def forward(self, batch, return_embeddings=False, **kwargs):
        datas = _single_graphs(batch)
        vals = []
        for d in datas:
            _, attrs = _literal_pair_attributes(d)
            vals.append(attrs[:, 0].sum().reshape(1).to(torch.float32))
        out = torch.stack(vals).reshape(-1).to(torch.float32)
        return out.reshape(1)


class LiteralInteractionPredictor(nn.Module):
    """Fixture predictor using matching factors transported in edge_attr."""

    def forward(self, batch, return_embeddings=False, **kwargs):
        datas = _single_graphs(batch)
        vals = []
        for d in datas:
            pairs, attrs = _literal_pair_attributes(d)
            total = 0.0
            for i, (u, v) in enumerate(pairs):
                for j in range(i + 1, len(pairs)):
                    a, b = pairs[j]
                    if len({u, v, a, b}) == 4:
                        total += float(attrs[i, 1] * attrs[j, 1])
            vals.append(torch.tensor([total], dtype=torch.float32))
        out = torch.stack(vals).reshape(-1).to(torch.float32)
        return out.reshape(1)


def make_literal_k4():
    pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
             (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    ei = torch.tensor(pairs, dtype=torch.long).t()
    additive = dict(zip(_LITERAL_W_BY_ENDPOINT, LITERAL_ADDITIVE_W))
    interaction = {(0, 1): 8.0, (0, 2): 5.0, (0, 3): 2.0,
                   (1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    attrs = []
    for u, v in pairs:
        key = (min(u, v), max(u, v))
        attrs.append([additive[key], interaction[key]])
    data = Data(
        x=torch.zeros(4, 4, dtype=torch.float32),
        edge_index=ei,
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
    )
    return data, LITERAL_ADDITIVE_TARGET


def make_k4_graph(generator, family, graph_id):
    if family not in FAMILY_CODE:
        raise ValueError(f'unknown family {family!r}')
    code = FAMILY_CODE[family]
    for _ in range(1000):
        strengths = 0.5 + 1.5 * torch.rand(4, generator=generator)
        pos = torch.rand(4, 2, generator=generator)
        dists = [(math.dist(pos[i].tolist(), pos[j].tolist()))
                 for i in range(4) for j in range(i + 1, 4)]
        if min(dists) < 1e-6 or not all(s > 0 for s in strengths.tolist()):
            continue
        x = torch.stack([strengths, pos[:, 0], pos[:, 1],
                         torch.full((4,), code)], dim=1).to(torch.float32)
        cols = []
        for u in range(4):
            for v in range(4):
                if u != v:
                    cols.append((u, v))
        ei = torch.tensor(cols, dtype=torch.long).t()
        ea = torch.tensor([[math.dist(pos[u].tolist(), pos[v].tolist())]
                           for (u, v) in cols], dtype=torch.float32)
        data = Data(x=x, edge_index=ei, edge_attr=ea)
        data.cluster_id = graph_id
        target = analytic_target_for(data, family)
        ch = content_hash_for(data, family, graph_id, target)
        return {'graph_id': graph_id, 'family': family, 'target': float(target),
                'data': data, 'content_hash': ch}
    raise RuntimeError('failed to generate nondegenerate K4')


def content_hash_for(data, family, graph_id, target):
    tensors = {name: getattr(data, name) for name in
               ('x', 'edge_index', 'edge_attr')}
    tensor_hash = model_state_hash(tensors)
    return hash_payload({'graph_id': str(graph_id), 'family': str(family),
                         'target': float(target),
                         'tensor_state_sha256': tensor_hash})


def model_definition_identity(model):
    """Separate parameter-state identity from predictor formula identity."""
    cls = type(model)
    source_path = inspect.getsourcefile(cls)
    source = sha256_file(source_path)['sha256'] if source_path else None
    definition = {'class': f'{cls.__module__}.{cls.__qualname__}',
                  'source_sha256': source}
    return {'definition_sha256': hash_payload(definition),
            'class': definition['class'],
            'source_sha256': source,
            'parameter_state_sha256': model_state_hash(model.state_dict())}


def build_datasets(data_seed=1001, val_seed=2001,
                   n_train_add=8, n_train_int=8, n_val_add=4, n_val_int=4):
    gen_train = torch.Generator().manual_seed(int(data_seed))
    gen_val = torch.Generator().manual_seed(int(val_seed))
    train, val = [], []
    for i in range(n_train_add):
        train.append(make_k4_graph(gen_train, 'additive', f'train_add_{i:02d}'))
    for i in range(n_train_int):
        train.append(make_k4_graph(gen_train, 'interaction', f'train_int_{i:02d}'))
    for i in range(n_val_add):
        val.append(make_k4_graph(gen_val, 'additive', f'val_add_{i:02d}'))
    for i in range(n_val_int):
        val.append(make_k4_graph(gen_val, 'interaction', f'val_int_{i:02d}'))
    return train, val


def exact_oracle(data, target, metered, k, constraint='no_isolates'):
    """Exhaustive feasible-mask optimum with actual metered forwards.

    Returns dict with objective/se/prediction/optimal_masks/n_calls.
    Bounded to P<=8 and k<=4; full-graph references are separate.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError('k must be an integer budget')
    layout = _layout_of(data)
    if len(layout.pairs) > 8 or k > 4:
        raise ValueError('exact oracle bounded to P<=8 and k<=4')
    marks = torch.arange(len(layout.pairs))
    masks = enumerate_feasible_masks(layout, k, constraint, pair_marks=marks)
    if masks.shape[0] == 0:
        raise ValueError('no feasible masks')
    best_se, best_pred, best_masks = None, None, []
    with torch.no_grad():
        for m in range(masks.shape[0]):
            sel = masks[m]
            full = layout.expand(sel, keep_self_loops=True)
            res = metered.call(data, mask=full, phase='oracle')
            pred = float(res.prediction.detach().cpu().reshape(-1)[0])
            se = (pred - float(target)) ** 2
            if best_se is None or se < best_se - 1e-9:
                best_se, best_pred = se, pred
                best_masks = [sel.detach().cpu().clone()]
            elif abs(se - best_se) <= 1e-9:
                best_masks.append(sel.detach().cpu().clone())
    return {'objective': -best_se, 'se': best_se, 'prediction': best_pred,
            'optimal_masks': best_masks, 'n_calls': int(masks.shape[0]),
            'feasible_count': int(masks.shape[0])}
