"""Tests for rls.structured_selection: bounded enumeration of feasible masks."""
import itertools, math, pytest, torch
from rls.constraints import PhysicalPairLayout, feasible_budget, pair_marks as make_pair_marks
from rls import structured_selection
from rls.constraints import pair_marks

def _api():
    import importlib.util
    assert importlib.util.find_spec('rls.structured_selection') is not None, 'structured_selection API missing'
    return structured_selection

def _independent_bfs_checker(layout, k, constraint):
    import networkx as nx
    g = nx.Graph(); g.add_nodes_from(range(layout.num_nodes))
    for p in layout.pairs.tolist(): g.add_edge(p[0], p[1])
    all_masks = []
    for combo in itertools.combinations(range(len(layout.pairs)), k):
        mask = [False] * len(layout.pairs)
        for i in combo: mask[i] = True
        selected_pairs = [layout.pairs[i].tolist() for i in range(len(layout.pairs)) if mask[i]]
        if not selected_pairs: continue
        if constraint == 'none': all_masks.append(mask)
        elif constraint == 'no_isolates':
            sg = nx.Graph(); sg.add_nodes_from(range(layout.num_nodes))
            for p in selected_pairs: sg.add_edge(p[0], p[1])
            if not any(deg == 0 for n, deg in sg.degree()): all_masks.append(mask)
        elif constraint == 'connected':
            sg = nx.Graph(); sg.add_nodes_from(range(layout.num_nodes))
            for p in selected_pairs: sg.add_edge(p[0], p[1])
            if sg.number_of_nodes() > 0 and nx.is_connected(sg): all_masks.append(mask)
    return all_masks

def _sets_equal(a, b):
    return {tuple(m.tolist()) for m in a} == {tuple(m.tolist()) for m in b}

def _compute_mapping(layout, other, rename):
    transported = torch.sort(rename[layout.pairs], dim=1).values
    mapping = []
    for t in transported.tolist():
        t_tensor = torch.tensor(t, dtype=torch.long, device=other.pairs.device)
        idx = (other.pairs == t_tensor).all(dim=1).nonzero(as_tuple=True)[0][0].item()
        mapping.append(idx)
    return torch.tensor(mapping)

def _K4_layout_with_scores(scores):
    ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
    return PhysicalPairLayout.from_edge_index(ei, 4), scores

class TestEnumerateFeasibleMasks:
    def test_K4_no_isolate_k2_count(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        assert masks.dtype == torch.bool and masks.shape[1] == len(layout.pairs) and masks.shape[0] == 3

    def test_K4_no_isolate_k3_count(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        assert s.enumerate_feasible_masks(layout, 3, constraint='no_isolates', pair_marks=marks).shape[0] == 16

    def test_K4_no_isolate_k4_count(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        assert s.enumerate_feasible_masks(layout, 4, constraint='no_isolates', pair_marks=marks).shape[0] == 15

    def test_connected_k2_infeasible(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='infeasible'):
            s.enumerate_feasible_masks(layout, 2, constraint='connected', pair_marks=marks)

    def test_connected_k3_count(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        assert s.enumerate_feasible_masks(layout, 3, constraint='connected', pair_marks=marks).shape[0] == 16

    def test_star_infeasible_tight_budget(self):
        s = _api()
        ei = torch.tensor([[0, 0, 0], [1, 2, 3]])
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='infeasible'):
            s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)

    def test_mask_cardinality_matches_k(self):
        s = _api()
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        assert s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks).sum(dim=1).eq(2).all()

    def test_unique_finite_marks_required(self):
        s = _api()
        ei = torch.tensor([[0, 1], [1, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        with pytest.raises(ValueError, match='unique'):
            s.enumerate_feasible_masks(layout, 2, pair_marks=torch.tensor([1.0, 1.0]))
        with pytest.raises(ValueError, match='finite'):
            s.enumerate_feasible_masks(layout, 2, pair_marks=torch.tensor([1.0, float('nan')]))

    def test_ordering_by_transported_pair_mark_ranks(self):
        s = _api()
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks1 = torch.tensor([0.5, 0.1])
        masks1 = s.enumerate_feasible_masks(layout, 2, pair_marks=marks1)
        marks2 = torch.tensor([0.1, 0.5])
        masks2 = s.enumerate_feasible_masks(layout, 2, pair_marks=marks2)
        assert {tuple(m.tolist()) for m in masks1} == {tuple(m.tolist()) for m in masks2}

    def test_unsupported_size_raises_valueerror(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(10), torch.arange(10)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 10)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='unsupported'):
            s.enumerate_feasible_masks(layout, 5, constraint='no_isolates', pair_marks=marks)

    def test_empty_layout_zero_pairs(self):
        s = _api()
        ei = torch.empty((2, 0), dtype=torch.long)
        layout = PhysicalPairLayout.from_edge_index(ei, 0)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        assert s.enumerate_feasible_masks(layout, 0, constraint='none', pair_marks=marks).shape[0] == 1

    def test_K4_no_isolate_k2_independent_checker(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        prod_masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        ind_masks = [torch.tensor(m, dtype=torch.bool) for m in _independent_bfs_checker(layout, 2, 'no_isolates')]
        assert _sets_equal(prod_masks, ind_masks)

    def test_K4_no_isolate_k3_independent_checker(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        prod_masks = s.enumerate_feasible_masks(layout, 3, constraint='no_isolates', pair_marks=marks)
        ind_masks = [torch.tensor(m, dtype=torch.bool) for m in _independent_bfs_checker(layout, 3, 'no_isolates')]
        assert _sets_equal(prod_masks, ind_masks)

    def test_K4_no_isolate_k4_independent_checker(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        prod_masks = s.enumerate_feasible_masks(layout, 4, constraint='no_isolates', pair_marks=marks)
        ind_masks = [torch.tensor(m, dtype=torch.bool) for m in _independent_bfs_checker(layout, 4, 'no_isolates')]
        assert _sets_equal(prod_masks, ind_masks)

    def test_K4_connected_k3_independent_checker(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        prod_masks = s.enumerate_feasible_masks(layout, 3, constraint='connected', pair_marks=marks)
        ind_masks = [torch.tensor(m, dtype=torch.bool) for m in _independent_bfs_checker(layout, 3, 'connected')]
        assert _sets_equal(prod_masks, ind_masks)

    def test_K4_no_isolate_k2_three_disconnected_matchings(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        assert masks.shape[0] == 3
        import networkx as nx
        for m in masks:
            sg = nx.Graph(); sg.add_nodes_from(range(4))
            for p in layout.pairs[m].tolist(): sg.add_edge(p[0], p[1])
            assert not nx.is_connected(sg)

    def test_incomplete_topology_and_isolate(self):
        s = _api()
        ei = torch.tensor([[0, 1], [1, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        assert masks.shape[0] >= 1
        ind_masks = [torch.tensor(m, dtype=torch.bool) for m in _independent_bfs_checker(layout, 2, 'no_isolates')]
        assert _sets_equal(masks, ind_masks)
        import networkx as nx
        for m in masks:
            sg = nx.Graph(); sg.add_nodes_from(range(3))
            for p in layout.pairs[m].tolist(): sg.add_edge(p[0], p[1])
            assert not any(deg == 0 for n, deg in sg.degree())

    def test_physical_isolate_no_isolates_infeasible(self):
        s = _api()
        ei = torch.tensor([[0, 1], [1, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='infeasible'):
            s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)

    def test_enumerate_rejects_omitted_marks(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        with pytest.raises(ValueError):
            s.enumerate_feasible_masks(layout, 2, constraint='no_isolates')

class TestStructuredGibbs:
    def test_equal_scores_uniform(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        scores = torch.zeros(len(layout.pairs))
        logps = s.structured_gibbs(scores, masks, temperature=1.0)
        assert logps.shape == (masks.shape[0],)
        torch.testing.assert_close(logps.exp(), logps.exp()[0].expand_as(logps.exp()))
        torch.testing.assert_close(logps.exp().sum(), torch.tensor(1.0))

    def test_positive_temperature_required(self):
        s = _api()
        masks = torch.ones(1, 2, dtype=torch.bool)
        scores = torch.tensor([0.0, 0.0])
        with pytest.raises(ValueError, match='temperature'):
            s.structured_gibbs(scores, masks, temperature=0.0)
        with pytest.raises(ValueError, match='temperature'):
            s.structured_gibbs(scores, masks, temperature=-1.0)

    def test_finite_scores_required(self):
        s = _api()
        masks = torch.ones(1, 2, dtype=torch.bool)
        scores = torch.tensor([0.0, float('nan')])
        with pytest.raises(ValueError, match='scores'):
            s.structured_gibbs(scores, masks)

    def test_inconsistent_cardinality_raises(self):
        s = _api()
        masks = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)
        scores = torch.tensor([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match='cardinality'): s.structured_gibbs(scores, masks)

    def test_empty_matrix_raises(self):
        s = _api()
        masks = torch.zeros(0, 2, dtype=torch.bool)
        scores = torch.tensor([0.0, 0.0])
        with pytest.raises(ValueError): s.structured_gibbs(scores, masks)

    def test_offset_invariance_float64(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        scores = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        logps1 = s.structured_gibbs(scores, masks)
        logps2 = s.structured_gibbs(scores + 1e8, masks)
        torch.testing.assert_close(logps1, logps2)

    def test_offset_invariance_representable(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        scores = torch.tensor([1.0, 2.0, 3.0])
        logps1 = s.structured_gibbs(scores, masks)
        logps2 = s.structured_gibbs(scores + 16, masks)
        torch.testing.assert_close(logps1, logps2)

    def test_equal_logits_at_0_plus_1e8(self):
        s = _api()
        masks = torch.eye(3, dtype=torch.bool)
        scores = torch.tensor([0., 0., 0.])
        logps = s.structured_gibbs(scores, masks)
        torch.testing.assert_close(logps.exp(), torch.ones(3) / 3)
        logps_large = s.structured_gibbs(torch.tensor([1e8, 1e8, 1e8]), masks)
        torch.testing.assert_close(logps, logps_large)

    def test_equal_logits_at_minus_1e8(self):
        s = _api()
        masks = torch.eye(3, dtype=torch.bool)
        scores = torch.tensor([0., 0., 0.])
        logps = s.structured_gibbs(scores, masks)
        logps_neg = s.structured_gibbs(torch.tensor([-1e8, -1e8, -1e8]), masks)
        torch.testing.assert_close(logps, logps_neg)

    def test_large_offset_float32(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        scores = torch.tensor([1.0, 2.0, 3.0])
        logps = s.structured_gibbs(scores + 16, masks)
        assert torch.isfinite(logps).all()
        torch.testing.assert_close(logps.exp().sum(), torch.tensor(1.0))

    def test_autograd(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        logps = s.structured_gibbs(scores, masks)
        logps.sum().backward()
        assert scores.grad is not None

    def test_float64_gradient_exact(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        assert masks.shape == (3, 6)
        scores = torch.tensor([0.5, -0.2, 0.3, 0.8, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
        logps = s.structured_gibbs(scores, masks, temperature=0.7)
        reward = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
        probs = logps.exp(); p = probs; A = masks.to(scores.dtype); r = reward
        term1 = (p * r) @ A
        term2 = (p * r).sum() * (p @ A)
        grad_analytic = (term1 - term2) / 0.7
        loss = (probs * reward).sum(); loss.backward()
        torch.testing.assert_close(scores.grad, grad_analytic, atol=1e-10, rtol=1e-10)

    def test_mismatched_device_raises(self):
        s = _api()
        masks = torch.ones(1, 2, dtype=torch.bool)
        scores = torch.tensor([1.0, 2.0], dtype=torch.float64)
        masks_gpu = masks.to('cuda') if torch.cuda.is_available() else masks
        if masks_gpu.device != scores.device:
            with pytest.raises(RuntimeError): s.structured_gibbs(scores, masks_gpu)

    def test_nonfinite_temperature_raises(self):
        s = _api()
        masks = torch.ones(1, 2, dtype=torch.bool)
        scores = torch.tensor([0.0, 0.0])
        with pytest.raises(ValueError, match='temperature'):
            s.structured_gibbs(scores, masks, temperature=float('inf'))
        with pytest.raises(ValueError, match='temperature'):
            s.structured_gibbs(scores, masks, temperature=float('nan'))

class TestSelectStructured:
    def test_MAP_offset_bug_regression(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        scores = torch.tensor([16., -32., 16., 32., 0., 80.], dtype=torch.float32)
        marks = torch.arange(6, dtype=torch.float64)
        k = 3
        masks = s.enumerate_feasible_masks(layout, k, constraint='no_isolates', pair_marks=marks)
        assert (masks.to(torch.float64) @ scores.to(torch.float64)).max().item() == 128
        result = s.select_structured(layout, scores, k, constraint='no_isolates', pair_marks=marks)
        assert float(result.selected.to(torch.float64) @ scores.to(torch.float64)) == 128
        for offset in [0.0, 1e8, -1e8]:
            shifted = (scores + offset).to(torch.float32)
            result_offset = s.select_structured(layout, shifted, k, constraint='no_isolates', pair_marks=marks)
            assert torch.equal(result.selected, result_offset.selected)
            assert float(result_offset.selected.to(torch.float64) @ scores.to(torch.float64)) == 128

    def test_MAP_deterministic(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.9, 0.1, 0.5])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks)
        assert result.selected.sum() == 2 and result.log_probability is None and result.diagnostics['method'] == 'enumerated_structured_map'
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        assert torch.equal(result.selected, masks[(masks.to(scores.dtype) @ scores).argmax()])

    def test_MAP_exact_tie_broken_by_marks(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.tensor([0.1, 0.2, 0.3])
        scores = torch.tensor([0.5, 0.5, 0.5])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        max_score = (masks.to(scores.dtype) @ scores).max()
        tied_indices = (masks.to(scores.dtype) @ scores == max_score).nonzero().flatten()
        assert result.selected.equal(masks[tied_indices[0]])

    def test_MAP_no_epsilon_scalarization(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([1e-8, 0., 0.])
        result = s.select_structured(layout, scores, 1, constraint='none', pair_marks=marks)
        assert result.selected[0], "pair01 with score 1e-8 must be the MAP selection"

    def test_MAP_lower_candidates_one_better(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.5, 0.5, 0.5001])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks)
        assert result.selected.sum() == 2
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        assert (masks.to(scores.dtype) @ scores).max() == (scores[2] + max(scores[0], scores[1]))

    def test_K4_literal_MAP_answers(self):
        s = _api()
        layout, scores = _K4_layout_with_scores(torch.tensor([8., 5., 1., 2., 4., 3.]))
        marks = torch.arange(6, dtype=torch.float64)
        assert torch.equal(s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks).selected, torch.tensor([True, False, False, False, False, True]))
        assert torch.equal(s.select_structured(layout, scores, 3, constraint='no_isolates', pair_marks=marks).selected, torch.tensor([True, True, False, False, True, False]))
        assert torch.equal(s.select_structured(layout, scores, 4, constraint='no_isolates', pair_marks=marks).selected, torch.tensor([True, True, False, False, True, True]))

    def test_K4_literal_MAP_answers_float64(self):
        s = _api()
        layout, scores = _K4_layout_with_scores(torch.tensor([8., 5., 1., 2., 4., 3.], dtype=torch.float64))
        marks = torch.arange(6, dtype=torch.float64)
        assert torch.equal(s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks).selected, torch.tensor([True, False, False, False, False, True]))
        assert torch.equal(s.select_structured(layout, scores, 3, constraint='no_isolates', pair_marks=marks).selected, torch.tensor([True, True, False, False, True, False]))
        assert torch.equal(s.select_structured(layout, scores, 4, constraint='no_isolates', pair_marks=marks).selected, torch.tensor([True, True, False, False, True, True]))

    def test_sampling_stochastic(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.9, 0.1, 0.5])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', sample=True, pair_marks=marks)
        assert result.selected.sum() == 2 and result.log_probability is not None and result.diagnostics['method'] == 'enumerated_structured_gibbs' and result.diagnostics['likelihood_kind'] == 'unordered_feasible_mask'

    def test_sampling_uses_marks(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.tensor([0.1, 0.2, 0.3])
        gen = torch.Generator().manual_seed(42)
        result = s.select_structured(layout, torch.tensor([0.5, 0.5, 0.5]), 2, constraint='no_isolates', sample=True, pair_marks=marks, generator=gen)
        assert torch.equal(result.pair_marks, marks)

    def test_deterministic_mode_log_probability_none(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.5, 0.5, 0.5])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', sample=False, pair_marks=marks)
        assert result.log_probability is None

    def test_expands_reciprocal_and_duplicate_copies(self):
        s = _api()
        ei = torch.tensor([[0, 0, 1], [0, 1, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.9, 0.5])
        assert s.select_structured(layout, scores, 1, constraint='none', pair_marks=marks).selected.sum() == 1

    def test_selfloops_stay_outside_budget(self):
        s = _api()
        ei = torch.tensor([[0, 0, 1], [0, 1, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.9, 0.5])
        assert s.select_structured(layout, scores, 1, constraint='none', pair_marks=marks).selected.sum() == 1

    def test_infeasible_budget_raises(self):
        s = _api()
        ei = torch.tensor([[0, 0, 0], [1, 2, 3]])
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='infeasible'):
            s.select_structured(layout, torch.zeros(3), 2, constraint='no_isolates', pair_marks=marks)

    def test_diagnostics_counts(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        scores = torch.tensor([0.9, 0.1, 0.5])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks)
        assert result.diagnostics['requested_pair_count'] == 2 and result.diagnostics['final_pair_count'] == 2 and 'method' in result.diagnostics

    def test_selected_log_likelihood_equals_mask_probability(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        masks = s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)
        scores = torch.tensor([0.9, 0.1, 0.5])
        result = s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks)
        log_probs = s.structured_gibbs(scores, masks, temperature=0.7)
        idx = (masks.to(scores.dtype) @ scores == (masks.to(scores.dtype) @ scores).max()).nonzero().flatten()[0]
        assert torch.isfinite(log_probs[idx].exp()) and log_probs[idx].exp() > 0

    def test_cap_enforced_before_enumeration(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        with pytest.raises(ValueError, match='unsupported'):
            s.select_structured(layout, torch.ones(6), 5, constraint='none')

    def test_empty_candidate_matrix_raises(self):
        s = _api()
        ei = torch.tensor([[0, 2], [1, 3]])
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='infeasible'):
            s.select_structured(layout, torch.tensor([0.5, 0.3]), 2, constraint='connected', pair_marks=marks)

    def test_nonfinite_scores_raises(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='finite'):
            s.select_structured(layout, torch.tensor([0.5, float('nan'), 0.7]), 2, constraint='none', pair_marks=marks)

    def test_invalid_marks_raises(self):
        s = _api()
        ei = torch.tensor([[0, 1], [1, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        with pytest.raises(ValueError, match='finite'):
            s.select_structured(layout, torch.tensor([0.5, float('nan')]), 2, pair_marks=torch.tensor([1.0, float('nan')]))
        with pytest.raises(ValueError, match='unique'):
            s.select_structured(layout, torch.tensor([0.5, 0.3]), 2, pair_marks=torch.tensor([1.0, 1.0]))

    def test_inconsistent_cardinality_raises(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        with pytest.raises(ValueError):
            s.select_structured(layout, torch.tensor([0.5, 0.3]), 2, constraint='none')

    def test_boolean_k_raises(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError):
            s.select_structured(layout, torch.tensor([0.5, 0.3, 0.7]), True, constraint='none', pair_marks=marks)

    def test_float_k_raises(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError):
            s.select_structured(layout, torch.tensor([0.5, 0.3, 0.7]), 2.0, constraint='none', pair_marks=marks)

    def test_finite_logits_nonfinite_marks_raises(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        marks = torch.tensor([1.0, float('inf'), 2.0])
        with pytest.raises(ValueError, match='finite'):
            s.select_structured(layout, torch.tensor([0.5, 0.3, 0.7]), 2, constraint='none', pair_marks=marks)

    def test_empty_candidate_matrix_select_raises(self):
        s = _api()
        ei = torch.empty((2, 0), dtype=torch.long)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError, match='infeasible'):
            s.select_structured(layout, torch.tensor([], dtype=torch.float64), 1, constraint='none', pair_marks=marks)

class TestRelabelingSymmetry:
    def test_node_relabeling_MAP_all_equal_scores(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        scores = torch.tensor([0.5, 0.5, 0.5])
        marks = torch.tensor([0.1, 0.2, 0.3])
        rename = torch.tensor([2, 0, 1])
        other_ei = rename[ei]
        other = PhysicalPairLayout.from_edge_index(other_ei, 3)
        mapping = _compute_mapping(layout, other, rename)
        inv_mapping = torch.zeros_like(mapping); inv_mapping[mapping] = torch.arange(len(mapping))
        other_scores = scores[inv_mapping]; other_marks = marks[inv_mapping]
        result1 = s.select_structured(layout, scores, 2, constraint='no_isolates', pair_marks=marks)
        result2 = s.select_structured(other, other_scores, 2, constraint='no_isolates', pair_marks=other_marks)
        assert torch.equal(result1.selected, result2.selected[mapping])

    def test_node_relabeling_MAP(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        scores = torch.tensor([0.5, 0.3, 0.7])
        rename = torch.tensor([2, 0, 1])
        other_ei = rename[ei]
        other = PhysicalPairLayout.from_edge_index(other_ei, 3)
        mapping = _compute_mapping(layout, other, rename)
        inv_mapping = torch.zeros_like(mapping); inv_mapping[mapping] = torch.arange(len(mapping))
        other_scores = scores[inv_mapping]
        result1 = s.select_structured(layout, scores, 2, constraint='no_isolates')
        result2 = s.select_structured(other, other_scores, 2, constraint='no_isolates')
        assert torch.equal(result1.selected, result2.selected[mapping])

    def test_node_relabeling_sampled(self):
        s = _api()
        ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        scores = torch.tensor([0.5, 0.3, 0.7])
        marks = torch.tensor([0.1, 0.2, 0.3])
        rename = torch.tensor([2, 0, 1])
        other_ei = rename[ei]
        other = PhysicalPairLayout.from_edge_index(other_ei, 3)
        mapping = _compute_mapping(layout, other, rename)
        inv_mapping = torch.zeros_like(mapping); inv_mapping[mapping] = torch.arange(len(mapping))
        other_scores = scores[inv_mapping]; other_marks = marks[inv_mapping]
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)
        result1 = s.select_structured(layout, scores, 2, constraint='no_isolates', sample=True, pair_marks=marks, generator=gen1)
        result2 = s.select_structured(other, other_scores, 2, constraint='no_isolates', sample=True, pair_marks=other_marks, generator=gen2)
        assert torch.equal(result1.selected, result2.selected[mapping])

    def test_duplicate_reverse_symmetry(self):
        s = _api()
        ei = torch.tensor([[0, 1], [1, 0]])
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        scores = torch.tensor([0.8])
        result = s.select_structured(layout, scores, 1, constraint='none')
        assert result.mask.sum() >= 1 and result.selected.sum() == 1

class TestLimits:
    def test_max_pairs_exceeded(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(10), torch.arange(10)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 10)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError):
            s.enumerate_feasible_masks(layout, 2, constraint='no_isolates', pair_marks=marks)

    def test_max_k_exceeded(self):
        s = _api()
        ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
        layout = PhysicalPairLayout.from_edge_index(ei, 4)
        marks = torch.arange(len(layout.pairs), dtype=torch.float64)
        with pytest.raises(ValueError):
            s.enumerate_feasible_masks(layout, 5, constraint='no_isolates', pair_marks=marks)
