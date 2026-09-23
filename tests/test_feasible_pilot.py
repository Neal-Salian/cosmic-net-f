"""Bounded feasible-selection pilot tests (Task 4, tests-only pass).

Source is read-only this pass: rls/feasible_pilot.py does not exist yet, so
pilot-driver tests below import it lazily and stay RED for the genuine reason
(ModuleNotFoundError / missing contract). No stubs, no weakened assertions.
"""
import importlib
import json
import math
import os

import pytest
import torch
from torch_geometric.data import Batch, Data

from rls.benchmark_predictor import MeteredPredictor
from rls.constrained_policy import select_pairs
from rls.constraints import (
    PhysicalPairLayout,
    feasible_budget,
    pair_marks,
)
from rls.pilot_synthetic import (
    EA_FEATURE_NAMES,
    LITERAL_ADDITIVE_TARGET,
    RAW_X_COLUMNS,
    SyntheticAdditivePredictor,
    analytic_target_for,
    build_datasets,
    exact_oracle,
    make_literal_k4,
)
from data.provenance import model_state_hash
from rls.structured_selection import (
    enumerate_feasible_masks,
    structured_gibbs,
)

# One literal reciprocal K4 edge_index; layouts always derived from it.
CLEAN_K4_EI = torch.tensor(
    [[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3,
      1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2][:12],
     [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2][:12]],
    dtype=torch.long,
)
# Pair index order: a=01, b=02, c=03, d=12, e=13, f=23.
K4_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PERM = [2, 0, 3, 1]  # nontrivial node relabeling used by invariance tests.


def k4_layout():
    return PhysicalPairLayout.from_edge_index(CLEAN_K4_EI, 4)


def _permute_data(data, perm):
    """Relabel nodes: new node i holds old node perm[i]."""
    pinv = [0] * len(perm)
    for new, old in enumerate(perm):
        pinv[old] = new
    mp = torch.tensor(pinv, dtype=torch.long)
    return Data(
        x=data.x[torch.tensor(perm)],
        edge_index=mp[data.edge_index],
        edge_attr=data.edge_attr.clone(),
    )


def _edge_slice(data, layout, selected):
    edge_mask = layout.expand(selected, keep_self_loops=True)
    return Data(x=data.x.clone(), edge_index=data.edge_index[:, edge_mask],
                edge_attr=data.edge_attr[edge_mask].clone())


def _pair_index_map(layout_old, layout_new, perm):
    """Map new-layout pair row -> old-layout pair row under node perm."""
    old_pos = {tuple(p): i for i, p in enumerate(layout_old.pairs.tolist())}
    mapping = []
    for (a, b) in layout_new.pairs.tolist():
        key = tuple(sorted((perm[a], perm[b])))
        mapping.append(old_pos[key])
    return torch.tensor(mapping, dtype=torch.long)


def test_k4_layout_canonical():
    layout = k4_layout()
    assert layout.pairs.tolist() == [list(p) for p in K4_PAIRS]
    assert int(layout.num_nodes) == 4
    assert int(layout.real.sum()) == 12


def test_literal_additive_optima_values_masks_and_counts():
    from rls.pilot_synthetic import LiteralAdditivePredictor

    data, target = make_literal_k4()
    assert target == LITERAL_ADDITIVE_TARGET == 23
    metered = MeteredPredictor(LiteralAdditivePredictor())
    # (budget, optimal SE, optimal prediction, optimal pair-index sets)
    cases = {2: (144, 11, [{0, 5}]),
             3: (36, 17, [{0, 1, 4}]),
             4: (9, 20, [{0, 1, 4, 5}])}
    total = 0
    for k, (se, pred, masks) in cases.items():
        before = metered.snapshot()['oracle']['forward_calls']
        out = exact_oracle(data, float(target), metered, k,
                           constraint='no_isolates')
        assert metered.snapshot()['oracle']['forward_calls'] - before == out['n_calls']
        assert out['se'] == pytest.approx(se)
        assert out['objective'] == pytest.approx(-se)
        assert out['prediction'] == pytest.approx(pred)
        got = [set(m.nonzero().flatten().tolist()) for m in out['optimal_masks']]
        assert len(got) == len(masks) == 1
        assert got[0] == masks[0]
        # Independent feasible-count check, not the production constant alone.
        indep = enumerate_feasible_masks(k4_layout(), k, 'no_isolates',
                                         pair_marks=torch.arange(6))
        assert out['n_calls'] == out['feasible_count'] == indep.shape[0]
        total += out['n_calls']
    assert total == 34
    # Full-graph reference is one separate query predicting 23.
    layout = k4_layout()
    full = torch.ones(12, dtype=torch.bool)
    res = metered.call(data, mask=full, phase='shared_full')
    assert float(res.prediction.reshape(-1)[0]) == pytest.approx(23)


def test_literal_interaction_k3_memberships_and_ties():
    from rls.pilot_synthetic import LiteralInteractionPredictor

    data, _ = make_literal_k4()
    metered = MeteredPredictor(LiteralInteractionPredictor())
    out2 = exact_oracle(data, 8.0, metered, 2, constraint='no_isolates')
    assert out2['se'] == pytest.approx(0.0)
    assert out2['prediction'] == pytest.approx(8.0)
    assert [set(m.nonzero().flatten().tolist())
            for m in out2['optimal_masks']] == [{0, 5}]
    out3 = exact_oracle(data, 8.0, metered, 3, constraint='no_isolates')
    assert out3['se'] == pytest.approx(0.0)
    assert out3['prediction'] == pytest.approx(8.0)
    got3 = [set(m.nonzero().flatten().tolist()) for m in out3['optimal_masks']]
    assert len(got3) == 4
    for m in got3:
        # Each optimal k=3 mask holds a and f plus exactly one of b/c/d/e.
        assert {0, 5} <= m
        assert len(m & {1, 2, 3, 4}) == 1
        assert len(m) == 3
    assert {frozenset(m) for m in got3} == {
        frozenset({0, 5, 1}), frozenset({0, 5, 2}),
        frozenset({0, 5, 3}), frozenset({0, 5, 4})}
    out4 = exact_oracle(data, 8.0, metered, 4, constraint='no_isolates')
    assert out4['se'] == pytest.approx(0.0)
    assert {frozenset(m.nonzero().flatten().tolist())
            for m in out4['optimal_masks']} == {
        frozenset({0, 5, 1, 2}), frozenset({0, 5, 1, 3}),
        frozenset({0, 5, 2, 4}), frozenset({0, 5, 3, 4})}
    assert out2['n_calls'] + out3['n_calls'] + out4['n_calls'] == 3 + 16 + 15


def test_literal_predictors_and_oracle_transport_under_node_permutation():
    from rls.pilot_synthetic import (
        LiteralAdditivePredictor, LiteralInteractionPredictor,
    )

    data, target = make_literal_k4()
    # A sliced graph must carry its fixture weights in edge_attr.
    old_layout = PhysicalPairLayout.from_edge_index(data.edge_index, 4)
    selected = torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.bool)
    sliced_old = _edge_slice(data, old_layout, selected)
    old_prediction = LiteralAdditivePredictor()(sliced_old).item()

    new_data = _permute_data(data, PERM)
    new_layout = PhysicalPairLayout.from_edge_index(new_data.edge_index, 4)
    idx_map = _pair_index_map(old_layout, new_layout, PERM)
    new_selected = selected[idx_map]
    new_prediction = LiteralAdditivePredictor()(
        _edge_slice(new_data, new_layout, new_selected)).item()
    assert new_prediction == pytest.approx(old_prediction)
    old_interaction = LiteralInteractionPredictor()(
        _edge_slice(data, old_layout, selected)).item()
    new_interaction = LiteralInteractionPredictor()(
        _edge_slice(new_data, new_layout, new_selected)).item()
    assert old_interaction == pytest.approx(8)
    assert new_interaction == pytest.approx(old_interaction)

    old_oracle = exact_oracle(data, target, MeteredPredictor(LiteralAdditivePredictor()),
                              3, constraint='no_isolates')
    new_oracle = exact_oracle(new_data, target, MeteredPredictor(LiteralAdditivePredictor()),
                              3, constraint='no_isolates')
    assert new_oracle['objective'] == pytest.approx(old_oracle['objective'])
    inverse = torch.empty_like(idx_map)
    inverse[idx_map] = torch.arange(6)
    mapped_old = {frozenset(inverse[m.nonzero().flatten()].tolist())
                  for m in old_oracle['optimal_masks']}
    actual_new = {frozenset(m.nonzero().flatten().tolist())
                  for m in new_oracle['optimal_masks']}
    assert actual_new == mapped_old


def test_synthetic_content_hash_and_model_definition_identity():
    from rls.pilot_synthetic import (
        LiteralAdditivePredictor, LiteralInteractionPredictor,
        content_hash_for, model_definition_identity,
    )

    data, target = make_literal_k4()
    h = content_hash_for(data, 'additive', 'fixture', target)
    assert len(h) == 64
    assert h != content_hash_for(data, 'additive', 'other', target)
    assert h != content_hash_for(data, 'interaction', 'fixture', target)
    assert h != content_hash_for(data, 'additive', 'fixture', target + 1)
    dtype_changed = data.clone()
    dtype_changed.x = dtype_changed.x.to(torch.float64)
    assert h != content_hash_for(dtype_changed, 'additive', 'fixture', target)
    assert model_state_hash(LiteralAdditivePredictor().state_dict()) == model_state_hash(
        LiteralInteractionPredictor().state_dict())
    additive_id = model_definition_identity(LiteralAdditivePredictor())
    interaction_id = model_definition_identity(LiteralInteractionPredictor())
    assert additive_id['parameter_state_sha256'] == model_state_hash(
        LiteralAdditivePredictor().state_dict())
    assert len(additive_id['definition_sha256']) == 64
    assert additive_id['definition_sha256'] != interaction_id['definition_sha256']


def test_predictors_reject_multi_graph_batches():
    from rls.pilot_synthetic import (
        LiteralAdditivePredictor, LiteralInteractionPredictor,
        SyntheticInteractionPredictor,
    )

    data, _ = make_literal_k4()
    batch = Batch.from_data_list([data, data.clone()])
    for predictor in (SyntheticAdditivePredictor(), SyntheticInteractionPredictor(),
                      LiteralAdditivePredictor(), LiteralInteractionPredictor()):
        with pytest.raises(ValueError, match='one graph'):
            predictor(batch)


def test_exact_oracle_exhaustive_and_bounded():
    from rls.pilot_synthetic import LiteralAdditivePredictor

    data, target = make_literal_k4()
    metered = MeteredPredictor(LiteralAdditivePredictor())
    out = exact_oracle(data, float(target), metered, 3,
                       constraint='no_isolates')
    assert out['n_calls'] == 16
    # Every returned optimum re-evaluates to the optimal prediction.
    layout = k4_layout()
    for mask in out['optimal_masks']:
        res = metered.call(data, mask=layout.expand(mask), phase='oracle')
        assert float(res.prediction.reshape(-1)[0]) == pytest.approx(out['prediction'])
    # No other feasible mask beats it: independent analytic brute force.
    w = [8.0, 5.0, 1.0, 2.0, 4.0, 3.0]
    indep = enumerate_feasible_masks(layout, 3, 'no_isolates',
                                     pair_marks=torch.arange(6))
    best = min((sum(w[i] for i in m.nonzero().flatten().tolist()) - 23) ** 2
               for m in indep)
    assert out['se'] == pytest.approx(best)
    with pytest.raises(ValueError):
        exact_oracle(data, float(target), metered, 5, constraint='no_isolates')
    k5_ei = torch.tensor([[u, v] for u in range(5) for v in range(5) if u != v],
                         dtype=torch.long).t()
    k5 = Data(x=torch.zeros(5, 4), edge_index=k5_ei,
              edge_attr=torch.zeros(k5_ei.shape[1], 1))
    with pytest.raises(ValueError):
        exact_oracle(k5, 0.0, metered, 3, constraint='no_isolates')


def test_structured_score_gradient_matches_audit_equation():
    torch.manual_seed(0)
    layout = k4_layout()
    masks = enumerate_feasible_masks(layout, 3, 'no_isolates',
                                     pair_marks=torch.randperm(6))
    scores = torch.randn(6, requires_grad=True)
    log_probs = structured_gibbs(scores, masks, temperature=1.0)
    log_probs[2].backward()
    with torch.no_grad():
        probs = log_probs.exp()
        expected = (masks[2].to(torch.float32)
                    - (probs[:, None] * masks.to(torch.float32)).sum(0))
    assert torch.allclose(scores.grad, expected, atol=1e-5)


def test_loo_gradient_exact_detached_values():
    fp = importlib.import_module('rls.feasible_pilot')
    logps = torch.zeros(4, requires_grad=True)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    loss = fp.reinforce_loss(logps, rewards)
    # LOO baselines [3, 8/3, 7/3, 2]; advantages [-2, -2/3, 2/3, 2].
    assert loss.item() == pytest.approx(0.0)
    g_logp, g_rew = torch.autograd.grad(loss, [logps, rewards],
                                        allow_unused=True)
    assert g_rew is None
    assert g_logp is not None
    assert torch.allclose(g_logp, torch.tensor([0.5, 1 / 6, -1 / 6, -0.5]),
                          atol=1e-6)
    # A constant reward shift must not change the logp gradient (fixed scale,
    # LOO baseline, no batch-std normalization).
    logps2 = torch.zeros(4, requires_grad=True)
    loss2 = fp.reinforce_loss(logps2, rewards.detach() + 5.0)
    (g2,) = torch.autograd.grad(loss2, [logps2], allow_unused=True)
    assert torch.allclose(g2, g_logp, atol=1e-6)


def test_dataset_hashes_unique_reproducible_and_varied():
    train, val = build_datasets()
    assert len(train) == 16 and len(val) == 8
    assert [t['family'] for t in train].count('additive') == 8
    assert [t['family'] for t in train].count('interaction') == 8
    assert [v['family'] for v in val].count('additive') == 4
    assert [v['family'] for v in val].count('interaction') == 4
    hashes = [e['content_hash'] for e in train + val]
    assert len(set(hashes)) == 24
    assert not ({t['graph_id'] for t in train} & {v['graph_id'] for v in val})
    # Same generator seeds reproduce tensor content exactly.
    train2, val2 = build_datasets()
    for a, b in zip(train + val, train2 + val2):
        assert torch.equal(a['data'].x, b['data'].x)
        assert torch.equal(a['data'].edge_attr, b['data'].edge_attr)
        assert a['target'] == b['target']
    # Changing the validation seed changes only validation, not train.
    train3, val3 = build_datasets(val_seed=2002)
    for a, b in zip(train, train3):
        assert torch.equal(a['data'].x, b['data'].x)
    assert any(not torch.equal(a['data'].x, b['data'].x)
               for a, b in zip(val, val3))
    # Examples vary beyond mere permutation: distance/strength multisets differ.
    def multiset(e):
        d = e['data'].edge_attr[:, 0].tolist()
        s = e['data'].x[:, 0].tolist()
        return (tuple(sorted(round(v, 6) for v in d)),
                tuple(sorted(round(v, 6) for v in s)))
    assert len({multiset(e) for e in train + val}) > 1


def test_raw_schema_and_distance_contract():
    assert RAW_X_COLUMNS == ['strength', 'pos_x', 'pos_y', 'family_code']
    assert EA_FEATURE_NAMES == ['distance']
    train, _ = build_datasets()
    g = train[0]['data']
    assert g.x.shape[1] == 4 and g.edge_attr.shape[1] == 1
    for col in range(g.edge_index.shape[1]):
        u, v = int(g.edge_index[0, col]), int(g.edge_index[1, col])
        d = math.dist(g.x[u, 1:3].tolist(), g.x[v, 1:3].tolist())
        assert g.edge_attr[col, 0].item() == pytest.approx(d)


def test_primitives_relabel_invariant_with_transported_marks():
    torch.manual_seed(7)
    layout_old = k4_layout()
    data_old = make_literal_k4()[0]
    data_old.x = torch.randn(4, 4)
    layout_new = PhysicalPairLayout.from_edge_index(
        torch.tensor([[3, 1, 1, 2, 2, 0, 0, 3, 3, 2, 1, 0],
                      [1, 3, 2, 1, 0, 2, 3, 0, 2, 3, 0, 1]], dtype=torch.long), 4)
    # Sanity: the hand-built edge_index must be the permuted K4.
    data_new = _permute_data(data_old, PERM)
    assert torch.equal(
        torch.sort(layout_new.pairs.flatten()).values,
        torch.sort(PhysicalPairLayout.from_edge_index(
            data_new.edge_index, 4).pairs.flatten()).values)
    idx_map = _pair_index_map(
        PhysicalPairLayout.from_edge_index(data_old.edge_index, 4),
        PhysicalPairLayout.from_edge_index(data_new.edge_index, 4), PERM)
    scores_old = torch.randn(6)
    scores_new = scores_old[idx_map]
    marks_old = pair_marks(layout_old,
                           generator=torch.Generator().manual_seed(11))
    marks_new = marks_old[idx_map]
    assert marks_new.unique().numel() == 6
    for method in ('direct', 'scaffold'):
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        s1 = select_pairs(layout_old, scores_old, 3, constraint='no_isolates',
                          method=method, sample=True, pair_marks=marks_old,
                          generator=g1)
        s2 = select_pairs(
            PhysicalPairLayout.from_edge_index(data_new.edge_index, 4),
            scores_new, 3, constraint='no_isolates', method=method,
            sample=True, pair_marks=marks_new, generator=g2)
        assert int(s2.selected.sum()) == 3
        # Membership maps through the endpoint correspondence.
        assert torch.equal(s2.selected, s1.selected[idx_map])
        assert int(s2.selected.sum()) == int(s1.selected.sum())
    # Structured Gibbs sample with transported marks and cloned generator.
    masks_old = enumerate_feasible_masks(layout_old, 3, 'no_isolates',
                                         pair_marks=marks_old)
    layout_n = PhysicalPairLayout.from_edge_index(data_new.edge_index, 4)
    masks_new = enumerate_feasible_masks(layout_n, 3, 'no_isolates',
                                         pair_marks=marks_new)
    lp_old = structured_gibbs(scores_old, masks_old, temperature=1.0)
    lp_new = structured_gibbs(scores_new, masks_new, temperature=1.0)
    i_old = torch.multinomial(lp_old.exp(),
                              1, generator=torch.Generator().manual_seed(5)).item()
    i_new = torch.multinomial(lp_new.exp(),
                              1, generator=torch.Generator().manual_seed(5)).item()
    assert torch.equal(masks_new[i_new], masks_old[i_old][idx_map])
    # Scorer values and synthetic predictions transport / stay invariant.
    from rls.raw_selector import RawBudgetPairScorer
    scorer = RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'],
                                 hidden_dim=16)
    synth, _ = build_datasets(n_train_add=1, n_train_int=0,
                              n_val_add=0, n_val_int=0)
    d = synth[0]['data']
    lay = PhysicalPairLayout.from_edge_index(d.edge_index, 4)
    gd = {'edge_index': d.edge_index, 'edge_attr': d.edge_attr, 'x': d.x}
    sc_old = scorer(gd, lay, 3).detach()
    dn = _permute_data(d, PERM)
    lay_n = PhysicalPairLayout.from_edge_index(dn.edge_index, 4)
    mp = _pair_index_map(lay, lay_n, PERM)
    sc_new = scorer({'edge_index': dn.edge_index, 'edge_attr': dn.edge_attr,
                     'x': dn.x}, lay_n, 3).detach()
    assert torch.allclose(sc_new, sc_old[mp], atol=1e-5)
    m_add = SyntheticAdditivePredictor()
    assert m_add(Batch.from_data_list([d])).item() == pytest.approx(
        m_add(Batch.from_data_list([dn])).item())
    # Oracle minimum objective is invariant; optimal masks map equivariantly.
    # Literal fixture weights travel in edge_attr; this learning-family check
    # uses node strengths/positions and confirms their relabeling transport.
    t_old = analytic_target_for(d, 'additive')
    t_new = analytic_target_for(dn, 'additive')
    assert t_new == pytest.approx(t_old)
    mo = MeteredPredictor(SyntheticAdditivePredictor())
    mn = MeteredPredictor(SyntheticAdditivePredictor())
    o_old = exact_oracle(d, t_old, mo, 3, constraint='no_isolates')
    o_new = exact_oracle(dn, t_new, mn, 3, constraint='no_isolates')
    assert o_new['objective'] == pytest.approx(o_old['objective'])
    inv = torch.empty_like(mp)
    inv[mp] = torch.arange(6)
    # mp maps new rows -> old rows; inv maps old rows -> new rows.
    want = {frozenset((inv[m.nonzero().flatten()]).tolist())
            for m in o_old['optimal_masks']}
    got = {frozenset(m.nonzero().flatten().tolist())
           for m in o_new['optimal_masks']}
    assert got == want


def test_mask_expansion_copies_loops_and_collapse():
    ei = torch.tensor([[0, 1, 1, 2, 2, 2],
                       [1, 2, 2, 1, 2, 0]], dtype=torch.long)
    layout = PhysicalPairLayout.from_edge_index(ei, 3)
    # Pairs: (0,1) one-way single copy, (1,2) reciprocal + self-loop col ignored.
    sel = torch.zeros(len(layout.pairs), dtype=torch.bool)
    sel[0] = True
    full = layout.expand(sel, keep_self_loops=True)
    assert full.shape[0] == ei.shape[1]
    assert torch.equal(layout.collapse(full), sel)
    # The one-way/duplicate pair-0 columns follow the selection; other real
    # columns are dropped; the self-loop column is kept.
    assert bool(full[0])
    assert not bool(full[1:4].any())
    assert bool(full[4])
    assert not bool(full[5])
    # All copies of a selected pair share membership; loops are kept.
    sel2 = torch.ones(len(layout.pairs), dtype=torch.bool)
    full2 = layout.expand(sel2, keep_self_loops=True)
    assert bool(full2[~layout.real].all())
    for p in range(len(layout.pairs)):
        cols = (layout.inverse == p).nonzero().flatten()
        assert bool((full2[layout.real][cols] == full2[layout.real][cols[0]]).all())


def test_pilot_four_decoder_interfaces_relabel_invariant():
    fp = importlib.import_module('rls.feasible_pilot')
    assert set(fp.METHODS) == {'repaired_pl', 'scaffold', 'direct', 'structured'}
    entry = next((getattr(fp, n, None) for n in ('decode', 'rollout', 'select')),
                 None)
    assert callable(entry), 'no documented decoder entry point'
    layout = k4_layout()
    torch.manual_seed(0)
    scores = torch.randn(6)
    marks = pair_marks(layout, generator=torch.Generator().manual_seed(9))
    data_old = make_literal_k4()[0]
    data_new = _permute_data(data_old, PERM)
    layout_new = PhysicalPairLayout.from_edge_index(data_new.edge_index, 4)
    idx_map = _pair_index_map(layout, layout_new, PERM)
    scores_new = scores[idx_map]
    marks_new = marks[idx_map]
    for method in fp.METHODS:
        out = entry(layout, scores, 3, method=method, pair_marks=marks,
                    generator=torch.Generator().manual_seed(21))
        out_new = entry(layout_new, scores_new, 3, method=method,
                        pair_marks=marks_new,
                        generator=torch.Generator().manual_seed(21))
        assert int(out['selected'].sum()) == 3
        rep = feasible_budget(layout, 3, 'no_isolates', out['selected'])
        assert rep['feasible']
        assert torch.equal(out_new['selected'], out['selected'][idx_map]), method


def test_pilot_decoders_have_deterministic_map_path_with_no_likelihood():
    fp = importlib.import_module('rls.feasible_pilot')
    layout = k4_layout()
    scores = torch.tensor([0.2, 1.3, -0.7, 0.9, -1.2, 0.4])
    marks = pair_marks(layout, generator=torch.Generator().manual_seed(17))
    data = make_literal_k4()[0]
    permuted = _permute_data(data, PERM)
    new_layout = PhysicalPairLayout.from_edge_index(permuted.edge_index, 4)
    idx_map = _pair_index_map(layout, new_layout, PERM)
    for method in fp.METHODS:
        first = fp.decode(layout, scores, 3, method=method, pair_marks=marks,
                          sample=False)
        second = fp.decode(new_layout, scores[idx_map], 3, method=method,
                           pair_marks=marks[idx_map], sample=False)
        assert first['sampled_log_probability'] is None
        assert first['log_probability'] is None
        assert int(first['selected'].sum()) == first['final_count'] == 3
        assert torch.equal(second['selected'], first['selected'][idx_map]), method


def test_repaired_decoder_reports_each_membership_transition():
    fp = importlib.import_module('rls.feasible_pilot')
    layout = k4_layout()
    out = fp.decode(layout, torch.tensor([5., 4., 3., 2., 1., 0.]), 2,
                    method='repaired_pl',
                    pair_marks=torch.tensor([0., 1., 2., 3., 4., 5.]),
                    generator=torch.Generator().manual_seed(1))
    sampled = out['sampled_selection']
    repaired = out['repaired_selection']
    projected = out['projected_selection']
    assert torch.equal(out['repaired_added'], repaired & ~sampled)
    assert torch.equal(out['repaired_removed'], sampled & ~repaired)
    assert torch.equal(out['repaired_xor'], sampled ^ repaired)
    assert torch.equal(out['projected_added'], projected & ~repaired)
    assert torch.equal(out['projected_removed'], repaired & ~projected)
    assert torch.equal(out['projected_xor'], repaired ^ projected)
    assert int(repaired.sum()) == out['repaired_count']
    assert int(projected.sum()) == out['projected_count'] == 2


def test_caps_reject_oversized_smoke_and_invalid_family_counts():
    fp = importlib.import_module('rls.feasible_pilot')
    smoke = fp.smoke_config()
    cases = [
        dict(smoke, epochs=1000, n_train_add=100000),
        dict(smoke, n_train_add=-1, n_train_int=3),
        dict(smoke, n_val_add=0),
        dict(smoke, policy_seeds=[-1]),
        dict(smoke, rollouts_B=5),
        dict(smoke, n_train_add=33, n_train_int=32),
        dict(smoke, n_val_add=9, n_val_int=8),
    ]
    for bad in cases:
        with pytest.raises(ValueError):
            fp.validate_caps(bad, allow_smoke=True)


def test_oversized_smoke_rejected_before_output_side_effect(tmp_path):
    fp = importlib.import_module('rls.feasible_pilot')
    bad = dict(fp.smoke_config(), n_train_add=100000, epochs=1000)
    target = tmp_path / 'oversized_smoke'
    with pytest.raises(ValueError):
        fp.run_pilot(bad, str(target), smoke=True)
    assert not target.exists()


def test_caps_reject_noninteger_or_negative_capped_counts_and_seeds():
    fp = importlib.import_module('rls.feasible_pilot')
    base = fp.default_config()
    for bad in (
        dict(base, n_train_add=-1, n_train_int=17),
        dict(base, n_train_add=8.5),
        dict(base, policy_seeds=[0, 1, -2]),
        dict(base, data_seed=-1),
        dict(base, validation_seed=1001),
    ):
        with pytest.raises(ValueError):
            fp.validate_caps(bad)


def test_caps_reject_invalid_configs():
    fp = importlib.import_module('rls.feasible_pilot')
    base = fp.default_config()
    assert base['mode'] == 'capped_pilot'
    bad = dict(base, policy_seeds=[0, 1, 2, 3])
    with pytest.raises(ValueError):
        fp.validate_caps(bad)
    bad = dict(base)
    bad['n_train_add'], bad['n_train_int'] = 33, 32  # 65 aggregate > 64 cap
    with pytest.raises(ValueError):
        fp.validate_caps(bad)
    bad = dict(base, epochs=11)
    with pytest.raises(ValueError):
        fp.validate_caps(bad)
    for budgets in ([2, 3], [2, 3, 4, 5], [2, 3, 4.0], [2, 3, 5]):
        bad = dict(base, budgets=list(budgets))
        with pytest.raises(ValueError):
            fp.validate_caps(bad)
    bad = dict(base, budgets=[2, 3, 4], max_pairs=9)
    with pytest.raises(ValueError):
        fp.validate_caps(bad)
    bad = dict(base, budgets=[2, 3, 4], max_k=5)
    with pytest.raises(ValueError):
        fp.validate_caps(bad)
    # Smoke may lower seeds/epochs but capped mode still requires 3/10/3.
    smoke = fp.smoke_config()
    assert smoke['mode'] == 'smoke'
    with pytest.raises(ValueError):
        fp.validate_caps(smoke)
    with pytest.raises(ValueError):
        fp.validate_caps(dict(smoke, mode='capped_pilot'))


def test_invalid_config_creates_no_output(tmp_path):
    fp = importlib.import_module('rls.feasible_pilot')
    bad = fp.default_config()
    bad['budgets'] = [2, 3, 5]
    target = str(tmp_path / 'must_not_exist')
    with pytest.raises(ValueError):
        fp.run_pilot(bad, target)
    assert not os.path.exists(target)
    occupied = tmp_path / 'occupied'
    occupied.mkdir()
    (occupied / 'sentinel.txt').write_text('x')
    with pytest.raises(ValueError):
        fp.run_pilot(fp.default_config(), str(occupied))


def test_smoke_artifact_counts_and_identities(tmp_path):
    fp = importlib.import_module('rls.feasible_pilot')
    cfg = fp.smoke_config()
    out = str(tmp_path / 'smoke_out')
    fp.run_pilot(cfg, out, smoke=True)
    with open(os.path.join(out, 'summary.json')) as f:
        summary = json.load(f)
    assert summary['mode'] == 'smoke'
    n_seeds = len(cfg['policy_seeds'])
    n_budgets = len(cfg['budgets'])
    n_methods = 4
    n_train = cfg['n_train_add'] + cfg['n_train_int']
    n_val = cfg['n_val_add'] + cfg['n_val_int']
    B, E = cfg['rollouts_B'], cfg['epochs']
    per_method_train = n_seeds * n_budgets * E * n_train * B
    assert summary['queries']['training'] == n_methods * per_method_train
    assert summary['queries']['validation'] == n_methods * n_seeds * n_budgets * E * n_val
    assert summary['queries']['oracle'] == (n_train + n_val) * sum(
        fp.ORACLE_MASKS_PER_BUDGET[k] for k in cfg['budgets'])
    assert summary['queries']['full'] == n_train + n_val
    with open(os.path.join(out, 'curves.json')) as f:
        curves = json.load(f)
    assert {c['method'] for c in curves} == set(fp.METHODS)
    assert {c['budget'] for c in curves} == set(cfg['budgets'])
    with open(os.path.join(out, 'val_per_example.json')) as f:
        per_ex = json.load(f)
    assert per_ex
    for row in per_ex:
        assert int(row['final_pair_count']) == int(row['budget'])
        assert row['constraint_satisfied'] is True
        assert row['regret'] >= -1e-9
        assert row['method'] in fp.METHODS
    with open(os.path.join(out, 'provenance.json')) as f:
        prov = json.load(f)
    for key in ('source_hash', 'dataset_hash', 'split_hash', 'config_hash',
                'seed_roles', 'train_ids', 'val_ids', 'reward_scale'):
        assert key in prov
    assert len(prov['train_ids']) == n_train
    assert len(prov['val_ids']) == n_val
    assert prov['reward_scale'] >= 1.0


def test_smoke_matched_init_and_scaffold_support(tmp_path):
    fp = importlib.import_module('rls.feasible_pilot')
    cfg = fp.smoke_config()
    out = str(tmp_path / 'smoke_out2')
    fp.run_pilot(cfg, out, smoke=True)
    with open(os.path.join(out, 'summary.json')) as f:
        summary = json.load(f)
    # Same train-only normalization and reward scale across methods; initial
    # scorer hash matched per seed/budget across the four methods.
    assert summary['normalization']['fitted_on'] == 'train_only'
    by_seed_budget = {}
    for h in summary['initial_scorer_hashes']:
        by_seed_budget.setdefault((h['seed'], h['budget']), set()).add(h['hash'])
    assert by_seed_budget
    for key, hashes in by_seed_budget.items():
        assert len(hashes) == 1, key
    # Scaffold at kmin has no learnable residual and unchanged parameters.
    scaf = [c for c in summary['support_limits'] if c['method'] == 'scaffold']
    assert scaf and all(s['no_residual_zero_gradient'] for s in scaf)
    assert summary['param_update_norm']['scaffold'] == pytest.approx(0.0)
    # Another method shows an actual nonzero update on the fixture.
    assert summary['param_update_norm']['direct'] > 0.0
