"""Validated physical topology and exact cardinality feasibility.

Budgets count unordered non-self pairs, including one-way and duplicate input
columns. Loops are architecture edges and never cover a physical isolate.
NetworkX (a PyTorch dependency) supplies maximum-matching cardinalities only;
its arbitrary matching tie choices never determine a selected mask.
"""
from dataclasses import dataclass
import math
import numbers
import networkx as nx
import torch


@dataclass(frozen=True)
class PhysicalPairLayout:
    """Pair rows are canonical storage, NOT a preference or tie-breaking order.

    Pass num_nodes explicitly to include nodes absent from edge_index. Without
    it, nodes are inferred as 0..max(edge_index), including any interior gaps.
    ``inverse`` indexes pair rows for non-loop columns selected by ``real``.
    """
    pairs: torch.Tensor
    inverse: torch.Tensor
    real: torch.Tensor
    num_nodes: int

    @classmethod
    def from_edge_index(cls, edge_index, num_nodes=None):
        if (not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2
                or edge_index.shape[0] != 2 or edge_index.dtype not in (torch.int32, torch.int64)):
            raise ValueError('edge_index must be an integer tensor of shape [2, E]')
        if edge_index.numel() and bool((edge_index < 0).any()):
            raise ValueError('negative node indices are invalid')
        inferred = int(edge_index.max()) + 1 if edge_index.numel() else 0
        if num_nodes is None:
            num_nodes = inferred
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, numbers.Integral) or num_nodes < inferred:
            raise ValueError('num_nodes must be a nonnegative integer covering every endpoint')
        real = edge_index[0] != edge_index[1]
        pairs, inverse = torch.unique(edge_index[:, real].long().sort(dim=0).values.T,
                                      dim=0, sorted=True, return_inverse=True)
        return cls(pairs, inverse, real, int(num_nodes))

    def expand(self, selected, keep_self_loops=True):
        """Expand pair membership to every input copy; optionally keep all loops."""
        self.validate_selection(selected)
        mask = ~self.real if keep_self_loops else torch.zeros_like(self.real)
        mask = mask.clone()
        mask[self.real] = selected[self.inverse]
        return mask

    def validate_selection(self, selected):
        if (selected.dtype != torch.bool or selected.shape != (len(self.pairs),)
                or selected.device != self.pairs.device):
            raise ValueError('selected must be a bool vector on the pair-layout device')

    def collapse(self, mask):
        """OR over all copies, for legacy masks that may be asymmetric."""
        if mask.dtype != torch.bool or mask.shape != self.real.shape or mask.device != self.real.device:
            raise ValueError('mask must be a bool vector matching edge_index columns and device')
        counts = torch.zeros(len(self.pairs), dtype=torch.long, device=self.pairs.device)
        counts.index_add_(0, self.inverse, mask[self.real].long())
        return counts > 0


def pair_budget(layout, keep_fraction):
    """ceil(fraction * physical pairs), permitting zero; never repair or clamp."""
    if not isinstance(keep_fraction, numbers.Real) or not math.isfinite(keep_fraction) or not 0 <= keep_fraction <= 1:
        raise ValueError('keep_fraction must be finite and in [0, 1]')
    return math.ceil(float(keep_fraction) * len(layout.pairs))


def pair_marks(layout, marks=None, generator=None):
    """Return unique finite pair priorities (larger wins).

    Omitted marks use an exchangeable random permutation independent of logits.
    Supply/transport one mark per physical pair for reproducible coupled node
    relabeling. Duplicate explicit marks are rejected rather than resolved by
    node IDs. A generator must match the layout device.
    """
    n = len(layout.pairs)
    if marks is None:
        return torch.randperm(n, device=layout.pairs.device, generator=generator)
    marks = torch.as_tensor(marks, device=layout.pairs.device).detach()
    if marks.shape != (n,) or not torch.isfinite(marks).all() or marks.unique().numel() != n:
        raise ValueError('pair marks must be a unique finite value per physical pair')
    return marks


def _validate_constraint(constraint):
    if constraint not in ('none', 'no_isolates', 'connected'):
        raise ValueError('constraint must be none, no_isolates, or connected')


def _graph(layout, selected=None):
    graph = nx.Graph()
    graph.add_nodes_from(range(layout.num_nodes))
    pairs = layout.pairs if selected is None else layout.pairs[selected]
    graph.add_edges_from(pairs.detach().cpu().tolist())
    return graph


def minimum_additions(layout, selected, constraint):
    """Exact minimum number of extra pairs required from the input topology.

    For uncovered nodes U, an added edge covers at most two nodes of U. A
    maximum matching of the graph induced by U pairs as many as possible;
    every unmatched node needs one extra edge. Thus |U|-matching_size is exact.
    Returns infinity when the original topology cannot satisfy the constraint.
    """
    _validate_constraint(constraint)
    layout.validate_selection(selected)
    if constraint == 'none':
        return 0
    full = _graph(layout)
    chosen = _graph(layout, selected)
    if constraint == 'connected':
        if layout.num_nodes <= 1:
            return 0
        if not nx.is_connected(full):
            return math.inf
        return nx.number_connected_components(chosen) - 1
    uncovered = [node for node, degree in chosen.degree if degree == 0]
    if any(full.degree[node] == 0 for node in uncovered):
        return math.inf
    matching = nx.max_weight_matching(full.subgraph(uncovered), maxcardinality=True)
    return len(uncovered) - len(matching)


def feasible_budget(layout, k, constraint='none', selected=None):
    """Report exact feasibility of extending selected to k total physical pairs.

    Infeasible budgets are reported, never expanded. ``minimum_pairs`` is None
    if no size can meet the topology constraint; otherwise it includes selected.
    """
    _validate_constraint(constraint)
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise ValueError('k must be an integer physical-pair count')
    if selected is None:
        selected = torch.zeros(len(layout.pairs), dtype=torch.bool, device=layout.pairs.device)
    layout.validate_selection(selected)
    count = int(selected.sum())
    additional = minimum_additions(layout, selected, constraint)
    minimum = None if math.isinf(additional) else count + additional
    feasible = minimum is not None and count <= k <= len(layout.pairs) and k >= minimum
    return dict(feasible=bool(feasible), requested_pair_count=int(k), minimum_pairs=minimum,
                available_pair_count=len(layout.pairs), constraint=constraint)


def selection_diagnostics(layout, selected, *, k=None, sampled_count=None, scaffold_count=0, constraint='none'):
    """Physical counts/topology of a selection; metadata is explicit, not inferred."""
    _validate_constraint(constraint)
    layout.validate_selection(selected)
    graph = _graph(layout, selected)
    isolates = sum(degree == 0 for _, degree in graph.degree)
    components = nx.number_connected_components(graph)
    count = int(selected.sum())
    satisfied = constraint == 'none' or (isolates == 0 if constraint == 'no_isolates' else components <= 1)
    return dict(requested_pair_count=k, sampled_pair_count=sampled_count,
                scaffold_pair_count=int(scaffold_count), final_pair_count=count,
                available_pair_count=len(layout.pairs), physical_isolates=isolates,
                physical_components=components, constraint=constraint,
                constraint_satisfied=satisfied, budget_satisfied=k is None or count == k)
