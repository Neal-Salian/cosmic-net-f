"""Opt-in, deterministic direct decoding equivalent to the research reference.

Training/sampling and historical evaluation entry points are unchanged. Prepare
once per immutable topology on its target device. Rebuild preparation if graph
columns, node count or device change. Scores and marks remain per-call inputs.
"""
from dataclasses import dataclass
import numbers

import networkx as nx
import torch

from rls.constraints import (PhysicalPairLayout, pair_marks as make_pair_marks,
                             selection_diagnostics)
from rls.constrained_policy import PairSelection, select_pairs


@dataclass(frozen=True)
class PreparedTopology:
    """Owned layout snapshot and CPU feasibility graph; treat fields as read-only.

    Preparation includes one device-to-host topology copy and one maximum
    matching. No embeddings, scores, marks or selected masks are cached.
    """
    layout: PhysicalPairLayout
    endpoints: tuple
    graph: object
    minimum_no_isolates: object

    @classmethod
    def from_edge_index(cls, edge_index, num_nodes=None):
        # Own storage, so mutation of caller tensors cannot stale the cache.
        layout = PhysicalPairLayout.from_edge_index(edge_index.clone(), num_nodes)
        endpoints = tuple(map(tuple, layout.pairs.detach().cpu().tolist()))
        graph = nx.Graph()
        graph.add_nodes_from(range(layout.num_nodes))
        graph.add_edges_from(endpoints)
        if any(degree == 0 for _, degree in graph.degree):
            minimum = None
        else:
            minimum = layout.num_nodes - len(nx.max_weight_matching(graph, maxcardinality=True))
        return cls(layout, endpoints, nx.freeze(graph), minimum)


def _can_complete(topology, uncovered, remaining):
    """Exact edge-cover completion, with cheap sufficient/necessary bounds."""
    if len(uncovered) <= remaining:
        # Every uncovered node has an incident full-topology edge. At most one
        # additional pair per uncovered node suffices, even across components.
        return True
    if (len(uncovered) + 1) // 2 > remaining:
        return False
    matching = nx.max_weight_matching(topology.graph.subgraph(uncovered), maxcardinality=True)
    return len(uncovered) - len(matching) <= remaining


def select_pairs_fast(topology, scores, k, constraint='no_isolates', *,
                      pair_marks=None, generator=None, keep_self_loops=True):
    """Replay reference ``method='direct', sample=False`` with fewer checks.

    Stable score ranking after descending unique marks exactly reproduces
    reference argmax ties. Only higher-ranked candidates rejected by exact
    completion feasibility are skipped. Loops and every duplicate/orientation
    column are expanded using the original physical-pair inverse map.

    ``connected`` delegates to the reference; this optimization targets the
    benchmark's ``no_isolates`` constraint, not a new selection algorithm.
    """
    layout = topology.layout
    if constraint == 'connected':
        return select_pairs(layout, scores, k, constraint, method='direct', sample=False,
                            pair_marks=pair_marks, generator=generator,
                            keep_self_loops=keep_self_loops)
    if constraint not in ('none', 'no_isolates'):
        raise ValueError('constraint must be none, no_isolates, or connected')
    if (scores.ndim != 1 or scores.shape != (len(layout.pairs),)
            or scores.device != layout.pairs.device or not scores.is_floating_point()
            or not torch.isfinite(scores).all()):
        raise ValueError('scores must be finite floating pair logits on the layout device')
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError('k must be an integer physical-pair count')
    minimum = topology.minimum_no_isolates if constraint == 'no_isolates' else 0
    if minimum is None or not minimum <= k <= len(layout.pairs):
        raise ValueError(f'infeasible physical-pair budget: k={k}, minimum_pairs={minimum}')
    marks = make_pair_marks(layout, pair_marks, generator)
    priority = marks.argsort(descending=True)
    ranked = priority[scores.detach()[priority].argsort(descending=True, stable=True)]
    if constraint == 'none':
        order = ranked[:k]
    else:
        # One compact rank transfer, rather than scalar synchronization inside
        # the candidate loop. Endpoints and the full graph were copied once.
        ranking = ranked.detach().cpu().tolist()
        uncovered = set(range(layout.num_nodes))
        chosen = []
        for index in ranking:
            if len(chosen) == k:
                break
            u, v = topology.endpoints[index]
            proposed = uncovered.difference((u, v))
            if _can_complete(topology, proposed, k - len(chosen) - 1):
                chosen.append(index)
                uncovered = proposed
            # A rejected pair can never become feasible after adding further
            # pairs: such a completion would also complete the earlier prefix.
            # Therefore each ranked candidate needs to be visited at most once.
        if len(chosen) != k:
            raise RuntimeError('no feasible extension for a validated budget')
        order = torch.tensor(chosen, dtype=torch.long, device=scores.device)
    selected = torch.zeros(len(layout.pairs), dtype=torch.bool, device=scores.device)
    selected[order] = True
    scaffold = torch.zeros_like(selected)
    diagnostics = selection_diagnostics(layout, selected, k=k, sampled_count=len(order),
                                        scaffold_count=0, constraint=constraint)
    diagnostics.update(method='direct', selection_mode='rank', likelihood_kind=None,
                       tie_policy='exchangeable_pair_marks')
    return PairSelection(layout.expand(selected, keep_self_loops), selected, order,
                         scaffold, marks, None, diagnostics)
