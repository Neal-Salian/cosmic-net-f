"""Physical-pair budgets and feasible ordered actions, with hand-checked toys."""
import itertools
import pytest
import torch
from rls import pair_policy


def api():
    # Missing APIs are assertion failures before the implementation exists.
    import importlib.util
    assert importlib.util.find_spec('rls.constraints') is not None, 'physical constraint API missing'
    assert importlib.util.find_spec('rls.constrained_policy') is not None, 'constrained policy API missing'
    from rls import constraints, constrained_policy
    return constraints, constrained_policy


def test_layout_validates_indices_and_counts_every_physical_pair_once():
    c, _ = api()
    ei = torch.tensor([[1, 0, 0, 2, 2], [0, 1, 1, 1, 2]])
    layout = c.PhysicalPairLayout.from_edge_index(ei, num_nodes=4)
    assert layout.pairs.tolist() == [[0, 1], [1, 2]]
    assert layout.expand(torch.tensor([True, False])).tolist() == [True, True, True, False, True]
    assert c.pair_budget(layout, .5) == 1
    assert c.pair_budget(layout, 0.) == 0
    for bad in [torch.tensor([[0], [-1]]), torch.tensor([[0.], [1.]]), torch.tensor([0, 1])]:
        with pytest.raises(ValueError): c.PhysicalPairLayout.from_edge_index(bad)
    with pytest.raises(ValueError): c.PhysicalPairLayout.from_edge_index(ei, num_nodes=2)
    for keep in [-.1, 1.1, float('nan')]:
        with pytest.raises(ValueError): c.pair_budget(layout, keep)


def test_feasibility_uses_matching_not_half_the_node_count():
    c, p = api()
    star = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 0, 0], [1, 2, 3]]), 4)
    assert c.feasible_budget(star, 2, 'no_isolates')['minimum_pairs'] == 3
    assert not c.feasible_budget(star, 2, 'no_isolates')['feasible']
    for constraint in ['no_isolates', 'connected']:
        with pytest.raises(ValueError, match='infeasible'):
            p.select_pairs(star, torch.zeros(3), 2, constraint=constraint)
    disconnected = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 2], [1, 3]]), 4)
    assert c.feasible_budget(disconnected, 2, 'no_isolates')['feasible']
    assert not c.feasible_budget(disconnected, 2, 'connected')['feasible']
    isolated = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 2], [1, 2]]), 3)
    assert not c.feasible_budget(isolated, 1, 'no_isolates')['feasible']
    empty = c.PhysicalPairLayout.from_edge_index(torch.empty((2, 0), dtype=torch.long), 0)
    assert c.feasible_budget(empty, 0, 'none')['feasible']
    assert p.select_pairs(empty, torch.empty(0), 0).mask.numel() == 0


@pytest.mark.parametrize('constraint,k', [('none', 0), ('none', 2), ('no_isolates', 2), ('connected', 3)])
@pytest.mark.parametrize('method', ['direct', 'scaffold'])
def test_exact_budget_and_pair_membership(constraint, k, method):
    c, p = api()
    ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
    layout = c.PhysicalPairLayout.from_edge_index(ei, 4)
    for sample in [False, True]:
        out = p.select_pairs(layout, torch.arange(6.).requires_grad_(), k,
                             constraint=constraint, method=method, sample=sample)
        assert out.diagnostics['requested_pair_count'] == k
        assert out.diagnostics['final_pair_count'] == k
        assert out.diagnostics['sampled_pair_count'] + out.diagnostics['scaffold_pair_count'] == k
        assert int(out.selected.sum()) == k
        assert out.mask[ei[0] == ei[1]].all()
        assert torch.equal(out.mask.reshape(4, 4), out.mask.reshape(4, 4).T)
        assert out.diagnostics['constraint_satisfied']
        if not sample: assert out.log_probability is None


def test_constrained_order_probabilities_and_exact_reward_gradient():
    c, p = api()
    # 4-cycle has exactly two 2-edge covers; after any first edge the opposite is forced.
    layout = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]]), 4)
    scores = torch.tensor([.4, -.2, .7, .1], dtype=torch.double, requires_grad=True)
    orders = list(itertools.permutations(range(4), 2))
    logps = torch.stack([p.constrained_order_log_probability(layout, scores, torch.tensor(o), 2, 'no_isolates') for o in orders])
    finite = logps.isfinite()
    assert finite.sum() == 4
    probs = logps[finite].exp()
    torch.testing.assert_close(probs.sum(), torch.tensor(1., dtype=torch.double))
    torch.testing.assert_close(logps[orders.index((0, 3))], scores[0] - scores.logsumexp(0))
    rewards = torch.tensor([float(0 in o) for o in orders], dtype=torch.double)[finite]
    exact = torch.autograd.grad((probs * rewards).sum(), scores, retain_graph=True)[0]
    reinforce = torch.autograd.grad((probs.detach() * rewards * logps[finite]).sum(), scores, retain_graph=True)[0]
    torch.testing.assert_close(exact, reinforce)
    expected = torch.softmax(scores, 0)[[0, 3]].sum()
    torch.testing.assert_close(exact, torch.autograd.grad(expected, scores)[0])
    mask_logp = p.exact_mask_log_probability(layout, scores, torch.tensor([True, False, False, True]), 'no_isolates')
    torch.testing.assert_close(mask_logp.exp(), expected)
    assert mask_logp > logps[orders.index((0, 3))]


def test_scaffold_is_parameter_independent_and_likelihood_only_covers_fill():
    c, p = api()
    layout = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]]), 4)
    marks = torch.tensor([.6, .5, .4, .3, .2, .1])
    a = p.select_pairs(layout, torch.arange(6.), 3, constraint='no_isolates', method='scaffold', pair_marks=marks)
    b = p.select_pairs(layout, -torch.arange(6.), 3, constraint='no_isolates', method='scaffold', pair_marks=marks)
    assert torch.equal(a.scaffold, b.scaffold)
    assert a.scaffold.tolist() == [True, False, False, False, False, True]
    assert a.diagnostics['scaffold_pair_count'] == 2
    scores = torch.arange(6., dtype=torch.double).requires_grad_()
    possible = (~a.scaffold).nonzero().flatten()
    logps = torch.stack([p.constrained_order_log_probability(layout, scores, i[None], 3, 'no_isolates', method='scaffold', pair_marks=marks) for i in possible])
    torch.testing.assert_close(logps.exp().sum(), torch.tensor(1., dtype=torch.double))
    torch.testing.assert_close(logps, torch.log_softmax(scores[possible], 0))


@pytest.mark.parametrize('method', ['direct', 'scaffold'])
@pytest.mark.parametrize('sample', [False, True])
def test_selection_coupled_node_relabeling_and_reversal(method, sample):
    c, p = api()
    ei = torch.tensor([[0, 0, 0, 1, 1, 2, 0, 2], [1, 2, 3, 2, 3, 3, 0, 1]])
    layout = c.PhysicalPairLayout.from_edge_index(ei, 4)
    rename = torch.tensor([3, 0, 2, 1])
    other = c.PhysicalPairLayout.from_edge_index(rename[ei.flip(0)], 4)
    transported = {tuple(sorted(rename[pair].tolist())): i for i, pair in enumerate(layout.pairs)}
    mapping = torch.tensor([transported[tuple(pair)] for pair in other.pairs.tolist()])
    marks = torch.tensor([.12, .93, .42, .77, .21, .58])
    scores = torch.zeros(6)
    a = p.select_pairs(layout, scores, 3, constraint='connected', method=method, sample=sample, pair_marks=marks, generator=torch.Generator().manual_seed(17))
    b = p.select_pairs(other, scores[mapping], 3, constraint='connected', method=method, sample=sample, pair_marks=marks[mapping], generator=torch.Generator().manual_seed(17))
    assert torch.equal(a.mask, b.mask)
    if sample: torch.testing.assert_close(a.log_probability, b.log_probability)


def test_legacy_repair_coupled_node_relabeling_counterexample():
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    chosen = torch.tensor([False, False, False, False, False, True])
    rename = torch.tensor([0, 2, 1, 3])
    # Canonical storage order changes under a real node relabeling; transport
    # both membership and marks to the new physical rows.
    marks = torch.tensor([.9, .1, .2, .3, .4, .5])
    import inspect
    assert 'pair_marks' in inspect.signature(pair_policy.repair_pairs).parameters, 'repair tie policy missing'
    a = pair_policy.repair_pairs(pairs, chosen, 4, pair_marks=marks)
    renamed = rename[pairs].sort(dim=1).values
    mapping = torch.argsort(renamed[:, 0] * 4 + renamed[:, 1])
    b = pair_policy.repair_pairs(renamed[mapping], chosen[mapping], 4, pair_marks=marks[mapping])
    assert torch.equal(a[mapping], b)
    assert a.tolist() == [True, False, False, False, False, True]


def test_legacy_repaired_likelihood_is_original_order_and_inference_skips_it(monkeypatch):
    ei = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
    scores = torch.arange(16., requires_grad=True)
    out = pair_policy.pair_mask(ei, scores, .1, sample=True)
    expected = pair_policy.ordered_log_probability(pair_policy.pair_scores(ei, scores), out[3])[0]
    torch.testing.assert_close(out[1], expected)
    def forbidden(*args): raise AssertionError('inference evaluated ordered likelihood')
    monkeypatch.setattr(pair_policy, 'ordered_log_probability', forbidden)
    pair_policy.pair_mask(ei, scores, .5)


def test_legacy_sparsify_counts_oneway_pairs_and_mirrors_all_duplicates():
    from rls.sparsify import topk_scheduled_mask, repair_connectivity, symmetrize_probs
    # Three physical pairs: duplicate 0->1, one-way 1->2, reciprocal 2<->3, loop.
    ei = torch.tensor([[0, 0, 1, 2, 3, 3], [1, 1, 2, 3, 2, 3]])
    scores = torch.tensor([.9, .7, .1, .4, .4, .8])
    torch.testing.assert_close(symmetrize_probs(ei, scores), torch.tensor([.8, .8, .1, .4, .4, .8]))
    assert topk_scheduled_mask(ei, scores, 0.).tolist() == [False, False, False, False, False, True]
    assert topk_scheduled_mask(ei, scores, 1 / 3).tolist() == [True, True, False, False, False, True]
    repaired = repair_connectivity(ei, torch.tensor([True, False, False, True, False, False]))
    assert repaired.tolist() == [True, True, False, True, True, False]


def test_legacy_floors_use_tie_inclusive_cutoff_without_index_preference():
    from rls.sparsify import hard_mask, apply_min_keep_floor
    values = torch.tensor([.1, .1, .1, .1])
    for mask in [hard_mask(values, .25), apply_min_keep_floor(torch.zeros(4, dtype=torch.bool), values, .25)]:
        assert mask.all()


def test_ordered_pl_rejects_non_actions():
    for order in [torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([3])]:
        with pytest.raises(ValueError):
            pair_policy.ordered_log_probability(torch.zeros(3), order)
    with pytest.raises(ValueError):
        pair_policy.ordered_log_probability(torch.tensor([float('nan')]), torch.tensor([0]))


def test_policy_action_reuses_independent_marks_for_repeatable_validation():
    class ConstantPolicy(torch.nn.Module):
        def forward(self, edge_attr, emb, edge_index, ctx):
            return torch.zeros(edge_index.shape[1], 1)
    graph = dict(edge_index=torch.cartesian_prod(torch.arange(4), torch.arange(4)).T,
                 edge_attr=None, emb=None, ctx=None, x=torch.zeros(4, 1))
    a = pair_policy.policy_action(ConstantPolicy(), graph, dict(target_sparsity_end=.2))[0]
    assert 'pair_marks' in graph, 'validation must retain marks for replay and provenance'
    for _ in range(5):
        b = pair_policy.policy_action(ConstantPolicy(), graph, dict(target_sparsity_end=.2))[0]
        assert torch.equal(a, b)
    stats = pair_policy.pair_stats(graph['edge_index'], a, 4, requested_count=2, sampled_count=2)
    assert stats['requested_pair_count'] == stats['sampled_pair_count'] == 2
    assert stats['repair_added_pair_count'] == stats['final_pair_count'] - 2


def test_connected_orders_normalize_and_sampling_replays_exact_likelihood():
    c, p = api()
    # Triangle with a pendant node: pair 2-3 is mandatory in every spanning tree.
    layout = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]), 4)
    scores = torch.tensor([.4, -.3, .9, -.2], dtype=torch.double, requires_grad=True)
    orders = list(itertools.permutations(range(4), 3))
    logps = torch.stack([p.constrained_order_log_probability(layout, scores, torch.tensor(o), 3, 'connected') for o in orders])
    assert logps.isfinite().sum() == 18
    torch.testing.assert_close(logps.exp().sum(), torch.tensor(1., dtype=torch.double))
    for method in ['direct', 'scaffold']:
        out = p.select_pairs(layout, scores, 3, 'connected', method=method, sample=True)
        assert out.selected[3]
        replay = p.constrained_order_log_probability(layout, scores, out.order, 3, 'connected', method=method, pair_marks=out.pair_marks)
        torch.testing.assert_close(out.log_probability, replay)


@pytest.mark.parametrize('method', ['direct', 'scaffold'])
@pytest.mark.parametrize('offset', [0., 1e8, -1e8])
def test_order_replay_and_normalization_survive_common_logit_offset(method, offset):
    c, p = api()
    layout = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 1], [1, 2]]), 3)
    scores = torch.full((2,), offset, dtype=torch.float32)
    marks = torch.tensor([1., 0.])
    out = p.select_pairs(layout, scores, 1, method=method, sample=True,
                         pair_marks=marks, generator=torch.Generator().manual_seed(9))
    replay = p.constrained_order_log_probability(layout, scores, out.order, 1,
                                                method=method, pair_marks=marks)
    torch.testing.assert_close(replay, out.log_probability)
    logps = torch.stack([p.constrained_order_log_probability(
        layout, scores, torch.tensor([index]), 1, method=method, pair_marks=marks)
        for index in range(2)])
    # Both actions remain equiprobable under any shared shift of the logits.
    torch.testing.assert_close(logps.exp(), torch.tensor([.5, .5]))
    torch.testing.assert_close(logps.exp().sum(), torch.tensor(1.))


def test_invalid_marks_budgets_and_inputs_are_not_silently_repaired():
    c, p = api()
    layout = c.PhysicalPairLayout.from_edge_index(torch.tensor([[0, 0, 1], [1, 2, 2]]), 3)
    for k in [-1, 4, 1.5, True]:
        with pytest.raises(ValueError): p.select_pairs(layout, torch.zeros(3), k)
    for marks in [torch.ones(3), torch.tensor([.1, float('nan'), .3]), torch.ones(2)]:
        with pytest.raises(ValueError): p.select_pairs(layout, torch.zeros(3), 2, pair_marks=marks)
    with pytest.raises(ValueError): p.select_pairs(layout, torch.zeros(3), 2, constraint='components')
    with pytest.raises(ValueError): p.select_pairs(layout, torch.zeros(2), 2)
    with pytest.raises(ValueError): p.select_pairs(layout, torch.tensor([0., float('inf'), 0.]), 2)
    with pytest.raises(ValueError): p.constrained_order_log_probability(layout, torch.zeros(3), torch.tensor([0]), 2, method='scaffold')
    assert torch.isneginf(p.constrained_order_log_probability(layout, torch.zeros(3), torch.tensor([0, 0]), 2))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_selection_preserves_device_and_duplicate_copy_gradients(device):
    c, p = api()
    ei = torch.tensor([[0, 1, 0, 0, 1, 2], [1, 0, 1, 2, 2, 2]], device=device)
    edge_scores = torch.tensor([.4, .4, .4, -.2, .7, 0.], device=device, requires_grad=True)
    layout = c.PhysicalPairLayout.from_edge_index(ei, 3)
    scores = pair_policy.pair_scores(ei, edge_scores)
    out = p.select_pairs(layout, scores, 1, sample=True)
    assert out.mask.device == edge_scores.device
    assert out.order.device == edge_scores.device
    out.log_probability.backward()
    torch.testing.assert_close(edge_scores.grad[:3], edge_scores.grad[0].expand(3))
    assert edge_scores.grad[-1] == 0
    assert out.mask[0] == out.mask[1] == out.mask[2]
