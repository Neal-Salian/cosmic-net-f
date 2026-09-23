"""Tests for rls.raw_selector: raw pair scorer and supervised training.

Physical-pair semantics: P is the number of unique unordered
non-self-loop pairs, not E (total edge columns).  All fixtures are
validated against this contract.
"""
import copy
import math
import pytest
import torch
from rls.constraints import PhysicalPairLayout
from rls import raw_selector


def _api():
    import importlib.util
    assert importlib.util.find_spec('rls.raw_selector') is not None, 'raw_selector API missing'
    return raw_selector


def _make_graph(node_dim, edge_feature_names, n_nodes, edge_pairs,
                cluster_id='g0', x_init=None):
    """Create a valid graph dict with consistent edge_attr dimensions."""
    ei = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    num_edges = ei.shape[1] if ei.numel() > 0 else 0
    edge_attr = torch.ones(num_edges, len(edge_feature_names), dtype=torch.float32)
    if x_init is not None:
        x = x_init.clone()
    else:
        x = torch.randn(n_nodes, node_dim, dtype=torch.float32)
    return {
        'edge_index': ei,
        'edge_attr': edge_attr,
        'x': x,
        'cluster_id': cluster_id,
    }


# Shared helper for supervised examples (module-level, not class-scoped)
def _make_supervised_example(graph, k, labels, label_def, label_calls, label_prov):
    return {
        'graph': graph,
        'k': k,
        'labels': labels,
        'label_definition': label_def,
        'label_predictor_calls': label_calls,
        'label_provenance': label_prov,
    }


def _expected_schema(scorer):
    """Declared schema contract: ordered feature names + node dimension."""
    return f"node_dim={scorer.node_dim};features={','.join(scorer.edge_feature_names)}"


def _p2_graph(cluster_id='g0'):
    """Valid P=2 graph: 3 nodes, 4 stored edges, 2 physical pairs."""
    return _make_graph(4, ['distance'], 3, [[0, 1], [1, 0], [1, 2], [2, 1]], cluster_id=cluster_id)


# ===================== RawBudgetPairScorer =====================

class TestRawBudgetPairScorer:

    def test_forward_output_shape(self):
        """P is unique unordered non-self pairs, not E."""
        r = _api()
        scorer = r.RawBudgetPairScorer(
            node_dim=4,
            edge_feature_names=['distance', 'delta_v', 'cos_theta',
                                'mass_ratio', 'proj_sep'],
            hidden_dim=16,
        )
        # 4 edges but P=2 physical pairs
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        edge_attr = torch.randn(4, 5)
        x = torch.randn(3, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        out = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=2)
        # P=2, not E=4
        assert out.shape == (2,)
        assert torch.isfinite(out).all()

    def test_forward_empty_pairs(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        ei = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, 1)
        x = torch.empty(0, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 0)
        out = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=0)
        assert out.shape == (0,)

    def test_forward_preserves_device(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=2, edge_feature_names=['distance'], hidden_dim=8)
        # 2 reciprocal edges = 1 physical pair, edge_attr needs 2 rows
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[1.0], [1.0]])  # 2 rows, one per edge column
        x = torch.randn(2, 2)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        out = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=1)
        assert out.device == x.device
        assert out.shape == (1,)

    def test_edge_reversal_preserves_scores(self):
        """Same node features and same physical pair under reversal."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance', 'mass_ratio'], hidden_dim=8)
        # Same physical pair (0,1) under different edge orderings
        x = torch.randn(2, 4)
        ei1 = torch.tensor([[0, 1], [1, 0]])
        edge_attr1 = torch.tensor([[1.0, 0.5], [1.0, -0.5]])
        layout1 = PhysicalPairLayout.from_edge_index(ei1, 2)
        out1 = scorer.forward({'edge_index': ei1, 'edge_attr': edge_attr1, 'x': x}, layout1, k=1)

        ei2 = torch.tensor([[1, 0], [0, 1]])
        edge_attr2 = torch.tensor([[1.0, -0.5], [1.0, 0.5]])
        layout2 = PhysicalPairLayout.from_edge_index(ei2, 2)
        out2 = scorer.forward({'edge_index': ei2, 'edge_attr': edge_attr2, 'x': x}, layout2, k=1)

        assert out1.shape == out2.shape == (1,)
        # Same physical pair, same features, same scores
        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_mass_ratio_abs_before_collapse(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['mass_ratio'], hidden_dim=8)
        # Two copies of same physical pair with opposite mass_ratio signs
        ei = torch.tensor([[0, 0, 1], [1, 1, 2]])
        # edge_attr has 3 rows (one per edge column)
        edge_attr = torch.tensor([[0.5], [-0.5], [0.3]])
        x = torch.randn(3, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        out = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=1)
        # P=2 physical pairs
        assert out.shape == (2,)
        assert torch.isfinite(out).all()
        # After abs, both copies of (0,1) have mass_ratio=0.5
        # The mean for physical pair 0 should be 0.5
        # Verify edge_attr was not mutated
        assert torch.allclose(edge_attr[0], torch.tensor([0.5]))
        assert torch.allclose(edge_attr[1], torch.tensor([-0.5]))

    def test_different_budgets_reach_scorer(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        edge_attr = torch.randn(4, 1)
        x = torch.randn(3, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        out1 = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=1)
        out2 = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=2)
        # P=2 physical pairs
        assert out1.shape == (2,)
        assert out2.shape == (2,)

    def test_invalid_k_negative(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[1.0], [1.0]])
        x = torch.randn(2, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        with pytest.raises(ValueError, match='k'):
            scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=-1)

    def test_invalid_k_exceeds_pairs(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[1.0], [1.0]])
        x = torch.randn(2, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        with pytest.raises(ValueError, match='k'):
            scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=3)

    def test_unknown_feature_name_raises(self):
        r = _api()
        with pytest.raises(ValueError, match='unknown'):
            r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['bad_feature'], hidden_dim=8)

    def test_no_target_predictor_access(self):
        """Scorer should not call any external predictor."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[1.0], [1.0]])
        x = torch.randn(2, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        out = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=1)
        # P=1 physical pair
        assert out.shape == (1,)

    # ---- Exact raw feature dimension and k/P column ----

    def test_feature_dimensions_and_kp_column(self):
        """Verify exact feature width 4*D+F+1 and k/P column values."""
        r = _api()
        D = 4
        F = 3
        scorer = r.RawBudgetPairScorer(node_dim=D, edge_feature_names=['distance', 'delta_v', 'cos_theta'], hidden_dim=16)
        expected_width = 4 * D + F + 1  # 4*4+3+1 = 20
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.randn(2, F)
        x = torch.randn(2, D)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        features = scorer._construct_pair_features(x, edge_attr, layout, k=1)
        assert features.shape[1] == expected_width
        # k/P column should be 1.0 (k=1, P=1)
        assert torch.isclose(features[0, -1], torch.tensor(1.0))

    def test_feature_dimensions_multiple_pairs(self):
        """Verify feature width with multiple physical pairs."""
        r = _api()
        D = 3
        F = 2
        scorer = r.RawBudgetPairScorer(node_dim=D, edge_feature_names=['distance', 'proj_sep'], hidden_dim=16)
        expected_width = 4 * D + F + 1  # 4*3+2+1 = 15
        # 4 edges = 2 physical pairs
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        edge_attr = torch.randn(4, F)
        x = torch.randn(3, D)
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        features = scorer._construct_pair_features(x, edge_attr, layout, k=1)
        assert features.shape[1] == expected_width
        # k/P should be 0.5 (k=1, P=2)
        assert torch.isclose(features[0, -1], torch.tensor(0.5))
        assert torch.isclose(features[1, -1], torch.tensor(0.5))

    # ---- Signed reversal + duplicates without input mutation ----

    def test_no_input_mutation(self):
        """Forward should not mutate edge_attr or x."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['mass_ratio'], hidden_dim=8)
        ei = torch.tensor([[0, 0, 1], [1, 1, 2]])
        edge_attr = torch.tensor([[0.5], [-0.5], [0.3]])
        x = torch.randn(3, 4)
        edge_attr_before = edge_attr.clone()
        x_before = x.clone()
        layout = PhysicalPairLayout.from_edge_index(ei, 3)
        scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=1)
        torch.testing.assert_close(edge_attr, edge_attr_before)
        torch.testing.assert_close(x, x_before)

    # ---- Coupled node permutation ----

    def test_coupled_node_permutation(self):
        """Scores should be invariant under coupled node relabeling."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        x = torch.randn(3, 4)
        # Original: nodes 0,1,2 with edge (0,1)
        ei1 = torch.tensor([[0, 1], [1, 0]])
        edge_attr1 = torch.tensor([[1.0], [1.0]])
        layout1 = PhysicalPairLayout.from_edge_index(ei1, 3)
        out1 = scorer.forward({'edge_index': ei1, 'edge_attr': edge_attr1, 'x': x}, layout1, k=1)

        # Permuted: relabel 0->1, 1->0, same physical pair
        ei2 = torch.tensor([[1, 0], [0, 1]])
        edge_attr2 = torch.tensor([[1.0], [1.0]])
        layout2 = PhysicalPairLayout.from_edge_index(ei2, 3)
        out2 = scorer.forward({'edge_index': ei2, 'edge_attr': edge_attr2, 'x': x}, layout2, k=1)

        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    # ---- Float64 path ----

    def test_float64_path(self):
        """scorer.double() with double graphs must work."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer = scorer.double()
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
        x = torch.randn(2, 4, dtype=torch.float64)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        out = scorer.forward({'edge_index': ei, 'edge_attr': edge_attr, 'x': x}, layout, k=1)
        assert out.dtype == torch.float64
        assert out.shape == (1,)
        # Verify buffers are also double
        assert scorer.norm_mean.dtype == torch.float64
        assert scorer.norm_std.dtype == torch.float64


# ===================== fit_normalization =====================

class TestFitNormalization:

    def test_fit_and_apply(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc123'
        )
        assert scorer._norm_fitted
        assert scorer._norm_split_hash == 'abc123'
        # Check named_buffers includes norm_mean and norm_std
        buffer_names = [n for n, _ in scorer.named_buffers()]
        assert 'norm_mean' in buffer_names
        assert 'norm_std' in buffer_names

    def test_reject_empty_examples(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        with pytest.raises(ValueError, match='empty'):
            scorer.fit_normalization(examples=[], train_ids=[], heldout_ids=[], split_hash='abc')

    def test_reject_overlapping_ids(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        with pytest.raises(ValueError, match='overlap'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=['g0'],
                split_hash='abc'
            )

    def test_reject_unknown_graph_ids(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # Graph has cluster_id='g0' which is in train_ids, but heldout_ids=['g1']
        # is unknown. The validation checks that graph cluster_id is in train_ids OR heldout_ids.
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g1')
        with pytest.raises(ValueError, match='unknown'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=['g1'],
                split_hash='abc'
            )

    def test_reject_empty_split_hash(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        with pytest.raises(ValueError, match='hash'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash=''
            )

    def test_reject_zero_pair_fit(self):
        """Zero-pair normalization input should be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # Graph has nodes but no physical pairs (empty edges)
        ei = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, 1)
        x = torch.randn(2, 4)  # non-empty x so not zero-node
        graph = {'edge_index': ei, 'edge_attr': edge_attr, 'x': x, 'cluster_id': 'g0'}
        with pytest.raises(ValueError, match='zero-pair'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )

    def test_reject_zero_node_fit(self):
        """Zero-node normalization input should be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        ei = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, 1)
        x = torch.empty(0, 4)
        graph = {'edge_index': ei, 'edge_attr': edge_attr, 'x': x, 'cluster_id': 'g0'}
        with pytest.raises(ValueError, match='zero-node'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )

    def test_heldout_not_affect_fit(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        mean_before = scorer.norm_mean.clone()
        std_before = scorer.norm_std.clone()
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=['g1'],
            split_hash='abc'
        )
        torch.testing.assert_close(scorer.norm_mean, mean_before)
        torch.testing.assert_close(scorer.norm_std, std_before)

    def test_validation_before_state_change(self):
        """Validation errors should not leave _norm_fitted=True."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # Graph with unknown cluster_id
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='unknown_id')
        assert not scorer._norm_fitted
        with pytest.raises(ValueError, match='unknown'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=['g1'],
                split_hash='abc'
            )
        assert not scorer._norm_fitted, 'state should not change on validation error'

    def test_duplicate_id_k_rejected(self):
        """Same ID with different budgets is allowed; duplicate (ID,k) should be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        # Same ID, same k - should be rejected via duplicate (ID,k) check
        with pytest.raises(ValueError, match='duplicate'):
            scorer.fit_normalization(
                examples=[(graph, 1), (graph, 1)],
                train_ids=['g0', 'g1'],
                heldout_ids=[],
                split_hash='abc'
            )

    def test_multiple_budgets_same_id_allowed(self):
        """Same ID across several budgets should be allowed."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # Graph with 3 nodes and 4 edges = 2 physical pairs (P=2)
        graph = _make_graph(4, ['distance'], 3, [[0, 1], [1, 0], [1, 2], [2, 1]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1), (graph, 2)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        assert scorer._norm_fitted
        assert scorer._norm_budgets == [1, 2]


# ===================== train_supervised_selector =====================

class TestTrainSupervisedSelector:

    def _make_supervised_example(self, graph, k, labels, label_def, label_calls, label_prov):
        return {
            'graph': graph,
            'k': k,
            'labels': labels,
            'label_definition': label_def,
            'label_predictor_calls': label_calls,
            'label_provenance': label_prov,
        }

    def test_training_reduces_error(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])  # P=1
        ex = self._make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=5,
            lr=1e-2
        )
        assert 'model_state' in artifact
        assert 'training_error_before' in artifact
        assert 'training_error_after' in artifact
        assert artifact['training_error_after'] < 1.0

    def test_wrong_split_rejects(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = self._make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        with pytest.raises(ValueError, match='split'):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='xyz',
                seed=42,
                epochs=1,
                lr=1e-3
            )

    def test_records_label_provenance(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        label_calls = 5
        ex = self._make_supervised_example(graph, 1, labels, 'good', label_calls, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=1,
            lr=1e-3
        )
        assert artifact['label_provenance']['total_calls'] == label_calls
        assert artifact['label_provenance']['label_predictor_calls'] == [label_calls]

    def test_artifact_contains_model_state_hash(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = self._make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=1,
            lr=1e-3
        )
        # Verify model state hash matches scorer's current state
        from data.provenance import model_state_hash
        current_hash = model_state_hash(scorer.state_dict())
        assert artifact['model_state_hash'] == current_hash

    def test_tiny_label_training_changes_weights_lowers_mse(self):
        """A tiny supplied-label training case changes weights and lowers MSE."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([1.0])  # P=1
        ex = _make_supervised_example(graph, 1, labels, 'positive', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Compute loss before training
        scorer.eval()
        with torch.no_grad():
            feat = scorer._construct_pair_features(graph['x'], graph['edge_attr'],
                                                     __import__('rls.constraints', fromlist=['PhysicalPairLayout']).PhysicalPairLayout.from_edge_index(graph['edge_index'], 2), 1)
            feat = scorer.apply_normalization(feat)
            loss_before = torch.nn.functional.mse_loss(scorer.mlp(feat).squeeze(-1), labels.float()).item()
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=10,
            lr=1e-2
        )
        # Compute loss after training
        scorer.eval()
        with torch.no_grad():
            loss_after = torch.nn.functional.mse_loss(scorer.mlp(feat).squeeze(-1), labels.float()).item()
        assert loss_after < loss_before, 'MSE should decrease after training'

    def test_artifact_wrong_split_rejected(self):
        """validate_supervised_artifact should reject wrong split."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = self._make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=1,
            lr=1e-3
        )
        with pytest.raises(ValueError, match='split'):
            r.validate_supervised_artifact(
                scorer, artifact, expected_split='wrong', expected_state_hash=artifact['model_state_hash'], expected_schema=_expected_schema(scorer)
            )

    def test_artifact_wrong_state_hash_rejected(self):
        """validate_supervised_artifact should reject wrong state hash."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = self._make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=1,
            lr=1e-3
        )
        # Create a modified state
        modified_scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        modified_graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        modified_labels = torch.tensor([0.0])
        modified_ex = self._make_supervised_example(modified_graph, 1, modified_labels, 'good', 0, {'source': 'test'})
        modified_scorer.fit_normalization(
            examples=[(modified_graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                modified_scorer, artifact, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=_expected_schema(scorer)
            )

    def test_caller_rng_preserved(self):
        """Training should not alter the caller's RNG state."""
        r = _api()
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Save caller RNG state immediately before train_supervised_selector
        rng_before = torch.get_rng_state().clone()
        r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42,
            epochs=1,
            lr=1e-3
        )
        rng_after = torch.get_rng_state()
        torch.testing.assert_close(rng_before, rng_after, atol=1e-6, rtol=1e-6)

    def test_reproducibility_same_seed(self):
        """Same initial model + seed should produce reproducible results."""
        r = _api()
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})

        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Save initial weights
        initial_weights = {k: v.clone() for k, v in scorer.mlp.named_parameters()}
        artifact1 = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=5, lr=1e-3
        )

        # Reset weights and retrain with same seed
        scorer2 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer2.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Restore initial weights
        for name, param in scorer2.mlp.named_parameters():
            param.data.copy_(initial_weights[name])
        # Also copy normalization buffers
        scorer2.norm_mean = scorer.norm_mean.clone()
        scorer2.norm_std = scorer.norm_std.clone()
        scorer2._norm_fitted = True
        scorer2._norm_split_hash = scorer._norm_split_hash
        scorer2._norm_ids = list(scorer._norm_ids)
        scorer2._norm_budgets = list(scorer._norm_budgets)
        scorer2._norm_schema = scorer._norm_schema
        artifact2 = r.train_supervised_selector(
            scorer2, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=5, lr=1e-3
        )

        torch.testing.assert_close(scorer.mlp[0].weight, scorer2.mlp[0].weight, atol=1e-6, rtol=1e-6)
        assert artifact1['training_error_after'] == artifact2['training_error_after']

    def test_exact_p_shape_labels(self):
        """Labels must have exact P shape; truncation is not permitted."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # 4 edges = 2 physical pairs, so labels must have shape [2]
        graph = _make_graph(4, ['distance'], 3, [[0, 1], [1, 0], [1, 2], [2, 1]], cluster_id='g0')
        labels = torch.tensor([0.0, 0.0])  # P=2
        ex = self._make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Should work fine
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        assert 'model_state_hash' in artifact

        # Wrong label shape should raise
        labels_wrong = torch.tensor([0.0])  # P=2 but only 1 label
        ex_wrong = self._make_supervised_example(graph, 1, labels_wrong, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError, match='shape'):
            r.train_supervised_selector(
                scorer, [ex_wrong],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )

        # True 2-D shape [P,1] must be rejected (code must check exact 1-D [P])
        labels_2d = torch.zeros((2, 1))
        ex_2d = self._make_supervised_example(graph, 1, labels_2d, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError, match='shape'):
            r.train_supervised_selector(
                scorer, [ex_2d],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )

    def test_finite_float64_labels_train(self):
        """Finite float64 labels for a float64 scorer must train (not be rejected)."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer = scorer.double()
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['x'] = graph['x'].double()
        graph['edge_attr'] = graph['edge_attr'].double()
        labels = torch.tensor([1.0], dtype=torch.float64)
        assert torch.isfinite(labels).all()
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(examples=[(graph, 1)], train_ids=['g0'], heldout_ids=[], split_hash='abc')
        artifact = r.train_supervised_selector(
            scorer, [ex], train_ids=['g0'], heldout_ids=[], split_hash='abc',
            seed=42, epochs=2, lr=1e-3)
        assert 'model_state_hash' in artifact


# ===================== validate_supervised_artifact =====================

class TestValidateSupervisedArtifact:

    def test_validate_success(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        result = r.validate_supervised_artifact(
            scorer, artifact, expected_split='abc',
            expected_state_hash=artifact['model_state_hash'],
            expected_schema=_expected_schema(scorer)
        )
        assert result is True

    def test_validate_wrong_schema_rejected(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, artifact, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
            expected_schema=_expected_schema(scorer) + ':wrong'
        )


# ===================== Gap 1: Unique declared train_ids =====================

class TestUniqueDeclaredIds:

    def test_unique_train_ids_for_multiple_budgets(self):
        """One graph at k=1 and k=2 should use train_ids=['g0'] (unique cohort)."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 3, [[0, 1], [1, 0], [1, 2], [2, 1]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1), (graph, 2)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        assert scorer._norm_fitted
        assert scorer._norm_budgets == [1, 2]
        assert len(set(scorer._norm_ids)) == 1

    def test_reject_duplicate_declared_train_ids(self):
        """Duplicate declared train IDs should be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        with pytest.raises(ValueError, match='duplicate'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0', 'g0'],
                heldout_ids=[],
                split_hash='abc'
            )

    def test_reject_duplicate_heldout_ids(self):
        """Duplicate declared heldout IDs should be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        with pytest.raises(ValueError, match='duplicate'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=['h0', 'h0'],
                split_hash='abc'
            )

    def test_reject_missing_declared_train_id(self):
        """A declared train ID missing from examples should be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        with pytest.raises(ValueError, match='unknown|missing|declared|cover'):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0', 'g1'],
                heldout_ids=[],
                split_hash='abc'
            )

    def test_heldout_actual_graph_rejected_and_state_unchanged(self):
        """Pass an actual heldout graph into normalization; reject and verify state unchanged."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph_train = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph_heldout = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='h0')
        with pytest.raises(ValueError, match='unknown'):
            scorer.fit_normalization(
                examples=[(graph_heldout, 1)],
                train_ids=['g0'],
                heldout_ids=['h0'],
                split_hash='abc'
            )
        assert not scorer._norm_fitted


# ===================== Gap 2: Validation before state change =====================

class TestValidationBeforeStateChange:

    def test_reject_bool_k_in_normalization(self):
        """Normalization should reject bool k before modifying state."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, True)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_reject_nan_features(self):
        """Normalization should reject NaN features before modifying state."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['x'][:, 0] = float('nan')
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_reject_inf_features(self):
        """Normalization should reject Inf features before modifying state."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['x'][:, 0] = float('inf')
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_reject_wrong_x_width(self):
        """Normalization should reject wrong x width before modifying state."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['x'] = torch.randn(2, 3)
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_reject_wrong_edge_attr_dimensions(self):
        """Normalization should reject wrong edge_attr row count before modifying state."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['edge_attr'] = torch.randn(1, 1)
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_reject_dtype_mismatch(self):
        """Normalization should reject dtype mismatch before modifying state."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['x'] = graph['x'].double()
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_refit_preserves_state_on_invalid_input(self):
        """Refitting an already-fitted scorer with invalid input should leave buffers unchanged."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        mean_before = scorer.norm_mean.clone()
        std_before = scorer.norm_std.clone()
        split_before = scorer._norm_split_hash
        bad_graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        bad_graph['x'][:, 0] = float('nan')
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(bad_graph, 1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        torch.testing.assert_close(scorer.norm_mean, mean_before)
        torch.testing.assert_close(scorer.norm_std, std_before)
        assert scorer._norm_split_hash == split_before

    def test_reject_foreign_layout_same_num_nodes(self):
        """Forward must reject a foreign layout with SAME N AND P (same E too) but changed topology."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # Same N=3, same E=4, same P=2, but different topology:
        # pairs (0,1),(1,2) vs pairs (0,1),(0,2)
        ei1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        ei2 = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
        x = torch.randn(3, 4)
        layout1 = PhysicalPairLayout.from_edge_index(ei1, 3)
        layout2 = PhysicalPairLayout.from_edge_index(ei2, 3)
        assert len(layout1.pairs) == len(layout2.pairs) == 2
        assert layout1.num_nodes == layout2.num_nodes == 3
        assert ei1.shape[1] == ei2.shape[1] == 4
        assert not torch.equal(layout1.pairs, layout2.pairs)
        graph = {'edge_index': ei1, 'edge_attr': torch.randn(4, 1), 'x': x}
        with pytest.raises(ValueError):
            scorer.forward(graph, layout2, k=1)

    def test_reject_negative_and_float_budgets(self):
        """Normalization must reject negative / float budgets, isolating the budget check."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _p2_graph(cluster_id='g0')
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, -1)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1.0)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted
        with pytest.raises(ValueError):
            scorer.fit_normalization(
                examples=[(graph, 1.5)],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc'
            )
        assert not scorer._norm_fitted

    def test_unknown_feature_name_rejected(self):
        """Schema names must be unique and supported."""
        r = _api()
        with pytest.raises(ValueError, match='duplicate'):
            r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance', 'distance'], hidden_dim=8)
        with pytest.raises(ValueError, match='unknown'):
            r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance', 'bad_name'], hidden_dim=8)


# ===================== Gap 3: Fixed-width buffers and load_state_dict =====================

class TestFixedWidthBuffers:

    def test_buffers_have_fixed_width_after_construction(self):
        """norm_mean/norm_std must have fixed feature width immediately after construction."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        buffer_names = [n for n, _ in scorer.named_buffers()]
        assert 'norm_mean' in buffer_names
        assert 'norm_std' in buffer_names
        expected_width = 4 * 4 + 1 + 1  # 4*D+F+1 = 18
        assert tuple(scorer.norm_mean.shape) == (expected_width,)
        assert tuple(scorer.norm_std.shape) == (expected_width,)
        torch.testing.assert_close(scorer.norm_mean, torch.zeros(expected_width))
        torch.testing.assert_close(scorer.norm_std, torch.ones(expected_width))

    def test_load_state_dict_restores_predictions(self):
        """Fit, save state_dict, create fresh scorer, load_state_dict, verify predictions."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        state = scorer.state_dict()
        fresh = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        fresh.load_state_dict(state)
        torch.testing.assert_close(fresh.norm_mean, scorer.norm_mean)
        torch.testing.assert_close(fresh.norm_std, scorer.norm_std)
        ei = torch.tensor([[0, 1], [1, 0]])
        x = torch.randn(2, 4)
        layout = PhysicalPairLayout.from_edge_index(ei, 2)
        out1 = scorer.forward({'edge_index': ei, 'edge_attr': torch.ones(2, 1), 'x': x}, layout, k=1)
        out2 = fresh.forward({'edge_index': ei, 'edge_attr': torch.ones(2, 1), 'x': x}, layout, k=1)
        torch.testing.assert_close(out1, out2)

    def test_double_after_fit_and_restore(self):
        """.double() must work after fit and after restore."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        scorer_double = scorer.double()
        assert scorer_double.norm_mean.dtype == torch.float64
        assert scorer_double.norm_std.dtype == torch.float64
        fresh = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        fresh.load_state_dict(scorer.state_dict())
        fresh_double = fresh.double()
        assert fresh_double.norm_mean.dtype == torch.float64

    def test_model_state_hash_includes_buffers(self):
        """Model state hash must include normalization buffers."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        from data.provenance import model_state_hash as _msh
        state = scorer.state_dict()
        assert 'norm_mean' in state
        assert 'norm_std' in state
        _msh(state)

    def test_load_state_dict_from_fresh_scorer(self):
        """load_state_dict into a fresh scorer should not fail."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        fresh = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        fresh.load_state_dict(scorer.state_dict())
        torch.testing.assert_close(fresh.norm_mean, scorer.norm_mean)
        torch.testing.assert_close(fresh.norm_std, scorer.norm_std)


# ===================== Gap 4: Content identity =====================

class TestContentIdentity:

    def test_content_hash_changes_with_different_data(self):
        """Content hash must hash actual per-ID training data, not just means/std."""
        r = _api()
        x0 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        x1 = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
        ei = torch.tensor([[0, 1], [1, 0]])
        ea0 = torch.tensor([[2.0], [2.0]])
        ea1 = torch.tensor([[8.0], [8.0]])
        graph0 = {'edge_index': ei.clone(), 'edge_attr': ea0.clone(), 'x': x0.clone(), 'cluster_id': 'g0'}
        graph1 = {'edge_index': ei.clone(), 'edge_attr': ea1.clone(), 'x': x1.clone(), 'cluster_id': 'g1'}
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer.fit_normalization(
            examples=[(graph0, 1), (graph1, 1)],
            train_ids=['g0', 'g1'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Clone both, swap complete x/edge_attr/edge_index data while retaining IDs
        graph0_swapped = {
            'edge_index': graph1['edge_index'].clone(),
            'edge_attr': graph1['edge_attr'].clone(),
            'x': graph1['x'].clone(),
            'cluster_id': 'g0',
        }
        graph1_swapped = {
            'edge_index': graph0['edge_index'].clone(),
            'edge_attr': graph0['edge_attr'].clone(),
            'x': graph0['x'].clone(),
            'cluster_id': 'g1',
        }
        scorer2 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer2.fit_normalization(
            examples=[(graph0_swapped, 1), (graph1_swapped, 1)],
            train_ids=['g0', 'g1'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Aggregate multiset is identical, so population stats must match
        torch.testing.assert_close(scorer.norm_mean, scorer2.norm_mean, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(scorer.norm_std, scorer2.norm_std, atol=1e-6, rtol=1e-6)
        hash1 = scorer._norm_training_content_hash
        hash2 = scorer2._norm_training_content_hash
        assert hash1 != hash2, 'Content hash must change when per-ID data is swapped'

    def test_changed_feature_schema_changes_schema_identity(self):
        """Changing feature schema must change schema identity even when width is unchanged."""
        r = _api()
        # TWO valid feature columns with equal values; names differ, width equal
        ei = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[3.0, 5.0], [3.0, 5.0]])
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        graph = {'edge_index': ei, 'edge_attr': edge_attr, 'x': x, 'cluster_id': 'g0'}
        scorer1 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance', 'delta_v'], hidden_dim=8)
        scorer1.fit_normalization(
            examples=[({'edge_index': ei.clone(), 'edge_attr': edge_attr.clone(), 'x': x.clone(), 'cluster_id': 'g0'}, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        scorer2 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance', 'cos_theta'], hidden_dim=8)
        scorer2.fit_normalization(
            examples=[({'edge_index': ei.clone(), 'edge_attr': edge_attr.clone(), 'x': x.clone(), 'cluster_id': 'g0'}, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        assert scorer1._norm_schema != scorer2._norm_schema
        assert scorer1._norm_schema == _expected_schema(scorer1)
        assert scorer2._norm_schema == _expected_schema(scorer2)

    def test_schema_encodes_ordered_names_and_node_dim(self):
        """Schema must describe ordered feature names + node dimension, not a fixed string."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['delta_v', 'distance'], hidden_dim=8)
        ei = torch.tensor([[0, 1], [1, 0]])
        graph = {
            'edge_index': ei,
            'edge_attr': torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            'x': torch.randn(2, 4),
            'cluster_id': 'g0',
        }
        scorer.fit_normalization(examples=[(graph, 1)], train_ids=['g0'], heldout_ids=[], split_hash='abc')
        assert scorer._norm_schema == _expected_schema(scorer)
        assert 'delta_v' in scorer._norm_schema and 'distance' in scorer._norm_schema
        # Order matters: swapped names must give a different schema
        scorer_rev = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance', 'delta_v'], hidden_dim=8)
        assert _expected_schema(scorer) != _expected_schema(scorer_rev)

    def test_reordered_examples_must_not_silently_pair_ids_with_different_budgets(self):
        """Reordered equivalent examples must canonicalize: same canonical (ID,k) content."""
        r = _api()
        # Valid P=2 graph, one unique ID with budgets 1,2
        base = _p2_graph(cluster_id='g0')
        graph_a = {'edge_index': base['edge_index'].clone(), 'edge_attr': base['edge_attr'].clone(),
                   'x': base['x'].clone(), 'cluster_id': 'g0'}
        graph_b = {'edge_index': base['edge_index'].clone(), 'edge_attr': base['edge_attr'].clone(),
                   'x': base['x'].clone(), 'cluster_id': 'g0'}
        scorer1 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer1.fit_normalization(
            examples=[(graph_a, 1), (graph_b, 2)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        scorer2 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer2.fit_normalization(
            examples=[(graph_b, 2), (graph_a, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Corresponding (ID,k) record sets must match, not input order
        assert sorted(scorer1._norm_budgets) == sorted(scorer2._norm_budgets) == [1, 2]
        assert set(scorer1._norm_budgets) == {1, 2}
        torch.testing.assert_close(scorer1.norm_mean, scorer2.norm_mean, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(scorer1.norm_std, scorer2.norm_std, atol=1e-6, rtol=1e-6)
        # Canonical content identity must be order-invariant
        assert scorer1._norm_training_content_hash == scorer2._norm_training_content_hash


# ===================== Gap 5: Supervised training requires fitted normalization =====================

class TestSupervisedRequiresFit:

    def test_training_without_fit_rejects(self):
        """Supervised training requires fitted normalization (no implicit fit)."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )

    def test_heldout_graph_rejected_in_training(self):
        """Fit on g0, then attempt training on heldout g1 while passing train_ids=['g0']."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph_g0 = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph_g1 = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g1')
        scorer.fit_normalization(
            examples=[(graph_g0, 1)],
            train_ids=['g0'],
            heldout_ids=['g1'],
            split_hash='abc'
        )
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph_g1, 1, labels, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=['g1'],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )

    def test_changed_k_rejected(self):
        """Training with changed k under same ID should be rejected (both budgets individually valid)."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _p2_graph(cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # P=2 so k=2 is individually valid, but differs from fitted k=1
        labels_p2 = torch.tensor([0.0, 0.0])
        ex = _make_supervised_example(graph, 2, labels_p2, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )
        # Changed x under the same ID and valid budget must also be rejected
        changed = {'edge_index': graph['edge_index'].clone(), 'edge_attr': graph['edge_attr'].clone(),
                   'x': graph['x'].clone() + 5.0, 'cluster_id': 'g0'}
        ex_changed = _make_supervised_example(changed, 1, labels_p2, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex_changed],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )

    def test_no_model_changes_on_rejection(self):
        """No tensor state / metadata / RNG changes on rejection."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _p2_graph(cluster_id='g0')
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        import copy as _copy2
        state_before = {k: (v.clone() if isinstance(v, torch.Tensor) else _copy2.deepcopy(v))
                        for k, v in scorer.state_dict().items()}
        meta_before = (scorer._norm_fitted, scorer._norm_split_hash, list(scorer._norm_ids),
                       list(scorer._norm_budgets), scorer._norm_schema, scorer._norm_training_content_hash)
        rng_before = torch.get_rng_state().clone()
        labels_p2 = torch.tensor([0.0, 0.0])
        ex = _make_supervised_example(graph, 2, labels_p2, 'good', 0, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )
        for k, v in scorer.state_dict().items():
            b = state_before[k]
            if isinstance(v, torch.Tensor) and isinstance(b, torch.Tensor):
                torch.testing.assert_close(v, b)
            else:
                assert v == b
        assert (scorer._norm_fitted, scorer._norm_split_hash, list(scorer._norm_ids),
                list(scorer._norm_budgets), scorer._norm_schema,
                scorer._norm_training_content_hash) == meta_before
        torch.testing.assert_close(torch.get_rng_state(), rng_before)

    def test_label_predictor_calls_must_be_integer_not_bool(self):
        """label_predictor_calls must be integer >=0, not bool/float."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        ex_bool = _make_supervised_example(graph, 1, labels, 'good', True, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex_bool],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )
        ex_float = _make_supervised_example(graph, 1, labels, 'good', 1.5, {'source': 'test'})
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex_float],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )

    def test_label_provenance_nonempty(self):
        """label_provenance must be a nonempty mapping; truthy non-mappings rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        for bad_prov in ({}, 'provenance-string', ['source'], 42):
            ex_bad = _make_supervised_example(graph, 1, labels, 'good', 0, bad_prov)
            with pytest.raises(ValueError):
                r.train_supervised_selector(
                    scorer, [ex_bad],
                    train_ids=['g0'],
                    heldout_ids=[],
                    split_hash='abc',
                    seed=42, epochs=1, lr=1e-3
                )

    def test_label_definition_nonempty(self):
        """Empty label_definition must be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        for bad_def in ('', None, 0):
            ex_bad = _make_supervised_example(graph, 1, labels, bad_def, 0, {'source': 'test'})
            with pytest.raises(ValueError):
                r.train_supervised_selector(
                    scorer, [ex_bad],
                    train_ids=['g0'],
                    heldout_ids=[],
                    split_hash='abc',
                    seed=42, epochs=1, lr=1e-3
                )

    def test_epochs_and_lr_validation(self):
        """epochs must be positive integer and lr finite positive."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=0, lr=1e-3
            )
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=0.0
            )


# ===================== Gap 6: Outer no_grad and RNG =====================

class TestOuterNoGrad:

    def test_training_under_outer_no_grad(self):
        """Training under an outer torch.no_grad context must locally enable gradients."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        with torch.no_grad():
            artifact = r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=1, lr=1e-3
            )
        assert artifact['training_error_after'] < artifact['training_error_before']

    def test_float64_training(self):
        """Verify float64 training works."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer = scorer.double()
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        graph['x'] = graph['x'].double()
        graph['edge_attr'] = graph['edge_attr'].double()
        labels = torch.tensor([0.0], dtype=torch.float64)
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        assert artifact['model_state_hash'] is not None

    def test_reproducibility_copy_same_starting_scorer(self):
        """Seeded reproducibility test must copy the SAME starting scorer."""
        import copy as _copy
        r = _api()
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        initial_state = _copy.deepcopy(scorer.state_dict())
        initial_meta = _copy.deepcopy((scorer._norm_ids, scorer._norm_budgets, scorer._norm_split_hash,
                                       scorer._norm_schema, scorer._norm_training_content_hash,
                                       scorer._norm_fit_records if hasattr(scorer, '_norm_fit_records') else None))
        artifact1 = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=5, lr=1e-3
        )
        scorer2 = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer2.load_state_dict(_copy.deepcopy(initial_state))
        artifact2 = r.train_supervised_selector(
            scorer2, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=5, lr=1e-3
        )
        for (n1, p1), (n2, p2) in zip(scorer.mlp.named_parameters(), scorer2.mlp.named_parameters()):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)
        assert artifact1['training_error_after'] == artifact2['training_error_after']
        assert artifact1['training_error_before'] == artifact2['training_error_before']

    def test_preserve_module_training_flags(self):
        """Training should preserve every module's training flag (mixed modes)."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        # Mixed modes: MLP Linear/ReLU alternating train/eval
        scorer.train()
        scorer.mlp[0].eval()
        scorer.mlp[1].train()
        scorer.mlp[2].eval()
        flags_before = [(name, m.training) for name, m in scorer.named_modules()]
        assert {f for _, f in flags_before} == {True, False}, 'fixture must mix train/eval modes'
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Re-apply mixed modes after fit (fit must not change them either)
        scorer.mlp[0].eval()
        scorer.mlp[1].train()
        scorer.mlp[2].eval()
        flags_before = [(name, m.training) for name, m in scorer.named_modules()]
        r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        flags_after = [(name, m.training) for name, m in scorer.named_modules()]
        assert flags_before == flags_after


# ===================== Gap 7: Artifact validation =====================

class TestArtifactValidation:

    def test_reject_false_norm_fitted(self):
        """Artifact validation must reject when the SAME scorer's _norm_fitted is flipped false."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        # Isolate missing-metadata validation: flip only the flag on the SAME scorer
        scorer._norm_fitted = False
        # Use artifact's own schema to isolate the flag check from the schema contract
        # (schema contract itself is covered by test_schema_encodes_ordered_names_and_node_dim)
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, artifact, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=artifact['normalization_metadata']['schema']
            )

    def test_reject_changed_metadata(self):
        """Reject artifact if metadata was changed after training."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        scorer._norm_split_hash = 'changed'
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, artifact, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=artifact['normalization_metadata']['schema']
            )

    def test_reject_epochs_zero(self):
        """Producing an epochs=0 artifact is prohibited; validator must also reject epochs=0."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        # Training with epochs=0 must raise (artifact production prohibited)
        with pytest.raises(ValueError):
            r.train_supervised_selector(
                scorer, [ex],
                train_ids=['g0'],
                heldout_ids=[],
                split_hash='abc',
                seed=42, epochs=0, lr=1e-3
            )
        # Validator rejection: copy a valid artifact, set epochs to 0, assert rejection
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        import copy as _copy
        bad = _copy.deepcopy(artifact)
        bad['training_settings']['epochs'] = 0
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, bad, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=artifact['normalization_metadata']['schema']
            )

    def test_reject_changed_current_metadata(self):
        """Reject if normalization buffer/weights changed."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(
            examples=[(graph, 1)],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc'
        )
        artifact = r.train_supervised_selector(
            scorer, [ex],
            train_ids=['g0'],
            heldout_ids=[],
            split_hash='abc',
            seed=42, epochs=1, lr=1e-3
        )
        scorer.norm_mean += 1.0
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, artifact, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=_expected_schema(scorer)
            )

    def test_artifact_records_label_identities(self):
        """Artifact must record a per-example label digest that changes with labels/definition/provenance/cost."""
        r = _api()
        # Fixed graph data across all variants to isolate label/definition/provenance/cost
        _base_x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        _base_ei = torch.tensor([[0, 1], [1, 0]])
        _base_ea = torch.tensor([[2.0], [2.0]])

        def _fixed_graph():
            return {'edge_index': _base_ei.clone(), 'edge_attr': _base_ea.clone(),
                    'x': _base_x.clone(), 'cluster_id': 'g0'}

        def _train_once(labels, definition, calls, prov):
            scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
            graph = _fixed_graph()
            scorer.fit_normalization(examples=[(graph, 1)], train_ids=['g0'], heldout_ids=[], split_hash='abc')
            ex = _make_supervised_example(_fixed_graph(), 1, labels, definition, calls, prov)
            art = r.train_supervised_selector(
                scorer, [ex], train_ids=['g0'], heldout_ids=[], split_hash='abc',
                seed=42, epochs=1, lr=1e-3)
            return art

        base_labels = torch.tensor([0.0])
        base = _train_once(base_labels, 'positive_label', 5, {'source': 'test', 'version': 1})
        # Explicit per-example digest contract
        assert 'label_hashes' in base, 'artifact must expose per-example label_hashes'
        base_hashes = base['label_hashes']
        assert isinstance(base_hashes, list) and len(base_hashes) == 1
        # Existing settings keys still recorded
        lp = base['label_provenance']
        assert lp['total_calls'] == 5
        assert lp['label_definitions'] == ['positive_label']
        assert lp['label_predictor_calls'] == [5]
        ts = base['training_settings']
        assert ts['epochs'] == 1
        assert ts['seed'] == 42
        assert ts['objective'] == 'mse'
        assert 'model_state_hash' in base
        assert 'normalization_metadata' in base
        # Changed labels must change the digest
        alt = _train_once(torch.tensor([1.0]), 'positive_label', 5, {'source': 'test', 'version': 1})
        assert alt['label_hashes'] != base_hashes
        # Changed definition must change the digest
        alt = _train_once(base_labels, 'other_label', 5, {'source': 'test', 'version': 1})
        assert alt['label_hashes'] != base_hashes
        # Changed provenance must change the digest
        alt = _train_once(base_labels, 'positive_label', 5, {'source': 'other'})
        assert alt['label_hashes'] != base_hashes
        # Changed cost must change the digest
        alt = _train_once(base_labels, 'positive_label', 6, {'source': 'test', 'version': 1})
        assert alt['label_hashes'] != base_hashes


# ===================== Final review regressions (TDD RED first) =====================

class TestFinalReviewRegressions:

    def _valid_artifact(self):
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        labels = torch.tensor([0.0])
        ex = _make_supervised_example(graph, 1, labels, 'good', 0, {'source': 'test'})
        scorer.fit_normalization(examples=[(graph, 1)], train_ids=['g0'], heldout_ids=[], split_hash='abc')
        artifact = r.train_supervised_selector(
            scorer, [ex], train_ids=['g0'], heldout_ids=[], split_hash='abc',
            seed=42, epochs=1, lr=1e-3)
        return scorer, artifact

    def test_tampered_saved_model_state_rejected(self):
        """Validator must hash artifact[model_state], not only declared hash vs scorer."""
        import copy as _copy
        r = _api()
        scorer, artifact = self._valid_artifact()
        bad = _copy.deepcopy(artifact)
        wkey = 'mlp.0.weight'
        assert wkey in bad['model_state']
        bad['model_state'][wkey] = bad['model_state'][wkey] + 1.0
        # Scorer unchanged; saved weights tampered -> must reject
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, bad, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=_expected_schema(scorer))

    def test_artifact_cost_and_hash_consistency(self):
        """total_calls must be consistent nonnegative; label arrays aligned/nonempty."""
        import copy as _copy
        r = _api()
        scorer, artifact = self._valid_artifact()
        # Negative total calls rejected
        bad = _copy.deepcopy(artifact)
        bad['label_provenance']['total_calls'] = -1
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, bad, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=_expected_schema(scorer))
        # Disagreeing sum rejected
        bad = _copy.deepcopy(artifact)
        bad['label_provenance']['total_calls'] = 999
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, bad, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=_expected_schema(scorer))
        # Empty label_hashes rejected (must be nonempty, aligned to fit records)
        bad = _copy.deepcopy(artifact)
        bad['label_hashes'] = []
        with pytest.raises(ValueError):
            r.validate_supervised_artifact(
                scorer, bad, expected_split='abc',
                expected_state_hash=artifact['model_state_hash'],
                expected_schema=_expected_schema(scorer))

    def test_unfitted_apply_normalization_preserves_double_dtype(self):
        """Unfitted apply_normalization must preserve dtype/device (identity)."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        scorer = scorer.double()
        assert not scorer._norm_fitted
        feats = torch.randn(2, scorer.total_input_width, dtype=torch.float64)
        out = scorer.apply_normalization(feats)
        assert out.dtype == torch.float64
        assert out.device == feats.device
        torch.testing.assert_close(out, feats)

    def test_edge_index_device_must_match_graph_and_scorer(self):
        """edge_index device must cohere with x/edge_attr/scorer; meta-device mismatch rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        bad = dict(graph)
        bad['edge_index'] = graph['edge_index'].to('meta')
        with pytest.raises(ValueError, match='device'):
            scorer.fit_normalization(
                examples=[(bad, 1)], train_ids=['g0'], heldout_ids=[], split_hash='abc')

    def test_layout_device_must_match_graph(self):
        """Foreign-device layout must be rejected without relying on .cpu() fallback."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        layout = PhysicalPairLayout.from_edge_index(graph['edge_index'], 2)
        object.__setattr__(layout, 'pairs', layout.pairs.to('meta'))
        with pytest.raises(ValueError, match='device'):
            scorer.forward(graph, layout, k=1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available locally')
    def test_cuda_mixed_edge_index_device_rejected(self):
        """CUDA: edge_index on cuda vs cpu graph must be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        bad = dict(graph)
        bad['edge_index'] = graph['edge_index'].to('cuda')
        with pytest.raises(ValueError, match='device'):
            scorer.fit_normalization(
                examples=[(bad, 1)], train_ids=['g0'], heldout_ids=[], split_hash='abc')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available locally')
    def test_cuda_mixed_layout_device_rejected(self):
        """CUDA: layout on cuda vs cpu graph must be rejected."""
        r = _api()
        scorer = r.RawBudgetPairScorer(node_dim=4, edge_feature_names=['distance'], hidden_dim=8)
        graph = _make_graph(4, ['distance'], 2, [[0, 1], [1, 0]], cluster_id='g0')
        layout = PhysicalPairLayout.from_edge_index(graph['edge_index'], 2)
        object.__setattr__(layout, 'pairs', layout.pairs.to('cuda'))
        with pytest.raises(ValueError, match='device'):
            scorer.forward(graph, layout, k=1)
