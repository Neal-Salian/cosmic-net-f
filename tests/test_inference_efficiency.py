"""The unchanged research decoder is the oracle for this opt-in inference path."""
import itertools

import pytest
import torch
from torch_geometric.data import Batch, Data

from rls.constrained_policy import select_pairs
from rls.constraints import PhysicalPairLayout


def fast_api():
    import rls.inference_efficiency as efficiency
    return efficiency.PreparedTopology, efficiency.select_pairs_fast


def edges(pairs, nodes):
    # Include reciprocal and duplicate stored columns, plus architecture loops.
    columns = list(pairs) + [(v, u) for u, v in pairs] + list(pairs[:1])
    columns += [(u, u) for u in range(nodes)]
    return torch.tensor(columns, dtype=torch.long).reshape(-1, 2).T.contiguous()


def compare(ei, nodes, scores, k, marks, constraint="no_isolates", loops=True):
    PreparedTopology, fast = fast_api()
    prepared = PreparedTopology.from_edge_index(ei, nodes)
    layout = PhysicalPairLayout.from_edge_index(ei, nodes)
    try:
        original = select_pairs(layout, scores, k, constraint, pair_marks=marks,
                                keep_self_loops=loops)
    except ValueError:
        with pytest.raises(ValueError, match="infeasible"):
            fast(prepared, scores, k, constraint, pair_marks=marks)
        return
    result = fast(prepared, scores, k, constraint, pair_marks=marks,
                  keep_self_loops=loops)
    for field in ("mask", "selected", "order", "scaffold", "pair_marks"):
        assert torch.equal(getattr(result, field), getattr(original, field)), field
    assert result.diagnostics == original.diagnostics
    assert result.log_probability is None
    assert int(result.selected.sum()) == k
    assert bool(result.mask[ei[0] == ei[1]].all()) == loops or not (ei[0] == ei[1]).any()


def test_highest_score_is_rejected_when_it_prevents_exact_edge_cover():
    PreparedTopology, fast = fast_api()
    ei = edges([(0, 1), (1, 2), (2, 3)], 4)
    result = fast(PreparedTopology.from_edge_index(ei, 4), torch.tensor([1., 9., 2.]),
                  2, pair_marks=torch.tensor([0, 1, 2]))
    assert result.selected.tolist() == [True, False, True]
    assert result.order.tolist() == [2, 0]


def test_all_four_node_topologies_budgets_and_ties_match_reference():
    # Catches unsound feasibility bounds, tie ordering, and budget inflation.
    universe = list(itertools.combinations(range(4), 2))
    for bits in range(1 << len(universe)):
        pairs = [p for i, p in enumerate(universe) if bits & (1 << i)]
        ei = edges(pairs, 4)
        scores = torch.tensor([float(i % 2) for i in range(len(pairs))])
        marks = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(bits))
        for k in range(len(pairs) + 1):
            compare(ei, 4, scores, k, marks)


@pytest.mark.parametrize("constraint", ["none", "no_isolates", "connected"])
def test_random_graphs_match_actions_and_diagnostics(constraint):
    gen = torch.Generator().manual_seed(131)
    for nodes in range(3, 9):
        pairs = [p for p in itertools.combinations(range(nodes), 2)
                 if torch.rand((), generator=gen) > .35]
        ei = edges(pairs, nodes)
        scores = torch.randn(len(pairs), generator=gen)
        marks = torch.randperm(len(pairs), generator=gen)
        for k in {0, len(pairs) // 2, len(pairs)}:
            compare(ei, nodes, scores, k, marks, constraint)


def test_empty_one_way_loops_and_unrepresented_isolates():
    compare(torch.empty((2, 0), dtype=torch.long), 0, torch.empty(0), 0, torch.empty(0))
    ei = torch.tensor([[0, 1, 2, 2], [1, 2, 2, 2]])
    for nodes in (3, 4):
        compare(ei, nodes, torch.tensor([2., 1.]), 2, torch.tensor([1, 0]))
    compare(ei, 3, torch.tensor([2., 1.]), 1, torch.tensor([1, 0]), "none", False)


def test_owned_cache_survives_source_mutation_and_replays_without_rng_change():
    PreparedTopology, fast = fast_api()
    ei = edges([(0, 1), (0, 2), (1, 2)], 3)
    prepared = PreparedTopology.from_edge_index(ei, 3)
    scores, marks = torch.tensor([2., 1., 0.]), torch.tensor([2, 1, 0])
    before = fast(prepared, scores, 2, pair_marks=marks)
    ei.zero_()
    rng = torch.random.get_rng_state().clone()
    after = fast(prepared, scores, 2, pair_marks=marks)
    assert torch.equal(before.mask, after.mask)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert before.selected.tolist() == [True, True, False]


def test_transported_marks_preserve_relabeling_and_column_permutation():
    PreparedTopology, fast = fast_api()
    ei = edges(list(itertools.combinations(range(5), 2)), 5)
    original = PreparedTopology.from_edge_index(ei, 5)
    marks = torch.randperm(10, generator=torch.Generator().manual_seed(9))
    result = fast(original, torch.zeros(10), 4, pair_marks=marks)
    relabel = torch.tensor([2, 4, 0, 1, 3])
    permutation = torch.randperm(ei.shape[1], generator=torch.Generator().manual_seed(8))
    changed = relabel[ei[:, permutation]]
    layout = PhysicalPairLayout.from_edge_index(changed, 5)
    pair_map = {tuple(sorted(relabel[p].tolist())): i for i, p in enumerate(original.layout.pairs)}
    transport = torch.tensor([pair_map[tuple(p.tolist())] for p in layout.pairs])
    other = fast(PreparedTopology.from_edge_index(changed, 5), torch.zeros(10), 4,
                 pair_marks=marks[transport])
    assert torch.equal(other.mask, result.mask[permutation])


def test_identical_pruned_graph_produces_identical_real_gnn_prediction():
    from model.model import CosmicNetGNN
    PreparedTopology, fast = fast_api()
    torch.manual_seed(12)
    ei = edges(list(itertools.combinations(range(5), 2)), 5)
    graph = Data(x=torch.randn(5, 4), edge_index=ei, edge_attr=torch.randn(ei.shape[1], 5))
    layout = PhysicalPairLayout.from_edge_index(ei, 5)
    scores, marks = torch.randn(10), torch.randperm(10)
    a = select_pairs(layout, scores, 4, "no_isolates", pair_marks=marks)
    b = fast(PreparedTopology.from_edge_index(ei, 5), scores, 4, pair_marks=marks)
    model = CosmicNetGNN({"model": {"hidden_dim": 8, "output_dim": 8, "num_layers": 2}}).eval()
    def predict(mask):
        return model(Batch.from_data_list([Data(x=graph.x, edge_index=ei[:, mask],
                                               edge_attr=graph.edge_attr[mask])]))[0]
    with torch.no_grad():
        assert torch.equal(predict(a.mask), predict(b.mask))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scores_rejected(bad):
    PreparedTopology, fast = fast_api()
    topology = PreparedTopology.from_edge_index(edges([(0, 1)], 2), 2)
    with pytest.raises(ValueError, match="finite"):
        fast(topology, torch.tensor([bad]), 1)


def test_invalid_budget_and_marks_rejected():
    PreparedTopology, fast = fast_api()
    topology = PreparedTopology.from_edge_index(edges([(0, 1), (1, 2)], 3), 3)
    for k in (-1, 3, True, 1.5):
        with pytest.raises(ValueError):
            fast(topology, torch.ones(2), k)
    with pytest.raises(ValueError, match="unique"):
        fast(topology, torch.ones(2), 2, pair_marks=torch.ones(2))
