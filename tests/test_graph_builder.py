"""
test_graph_builder.py
Purpose: Unit tests for graph/graph_builder.py
Tests: Edge construction methods, self-loops, determinism, symmetry, isolated nodes
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph_builder import GraphBuilder, GraphDataset, compute_graph_statistics
from data.loaders.base_loader import HaloData, SubhaloData


@pytest.fixture
def base_config():
    """Base configuration for graph builder tests."""
    return {
        'seed': 42,
        'graph': {
            'method': 'radius',
            'radius_mpc': 2.0,
            'k_neighbors': 5,
            'self_loops': True,
            'edge_features': ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep'],
            'hierarchical': False
        }
    }


@pytest.fixture
def graph_builder(base_config):
    """Create GraphBuilder instance."""
    return GraphBuilder(base_config)


def create_test_halo(
    num_subhalos: int = 10,
    cluster_id: str = 'test_halo',
    spread: float = 1.0,
    halo_mass: float = 1e13
) -> HaloData:
    """Create a test halo with specified number of subhalos."""
    np.random.seed(42)

    subhalos = []
    for i in range(num_subhalos):
        position = np.random.randn(3) * spread
        velocity = np.random.randn(3) * 200.0

        subhalo = SubhaloData(
            subhalo_id=i,
            position=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            stellar_mass=1e10 * (1 + 0.1 * np.random.rand()),
            velocity_dispersion=150.0 + 50 * np.random.rand(),
            half_mass_radius=0.005 + 0.01 * np.random.rand(),
            metallicity=0.01 + 0.02 * np.random.rand()
        )
        subhalos.append(subhalo)

    return HaloData(
        cluster_id=cluster_id,
        subhalos=subhalos,
        halo_mass=halo_mass,
        redshift=0.0
    )


class TestRadiusMethod:
    """Tests for radius-based edge construction."""

    def test_no_isolated_nodes_with_fallback(self, base_config):
        """Test that no nodes are isolated (fallback to nearest neighbor)."""
        # Create halo with one very isolated node
        halo = create_test_halo(num_subhalos=10, spread=0.5)

        # Add an isolated node far away
        isolated_subhalo = SubhaloData(
            subhalo_id=100,
            position=np.array([100.0, 100.0, 100.0], dtype=np.float32),  # Very far
            velocity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            stellar_mass=1e10,
            velocity_dispersion=200.0,
            half_mass_radius=0.01,
            metallicity=0.02
        )
        halo.subhalos.append(isolated_subhalo)

        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        # Check no isolated nodes (every node should have at least one edge)
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes

        # Get unique nodes that appear in edges (excluding self-loops for this check)
        non_self_mask = edge_index[0] != edge_index[1]
        connected_nodes = torch.unique(edge_index[:, non_self_mask].flatten())

        # With fallback, all nodes should be connected
        assert len(connected_nodes) == num_nodes, \
            f"Expected {num_nodes} connected nodes, got {len(connected_nodes)}"

    def test_radius_connects_nearby_nodes(self, base_config):
        """Test that nodes within radius are connected."""
        # Create compact halo where all nodes are within radius
        base_config['graph']['radius_mpc'] = 5.0  # Large radius
        halo = create_test_halo(num_subhalos=5, spread=0.5)  # Small spread

        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        # Remove self-loops for edge count
        non_self_mask = graph.edge_index[0] != graph.edge_index[1]
        num_non_self_edges = non_self_mask.sum().item()

        # For 5 nodes all within radius, should have many edges
        # (5 nodes, each connected to others = up to 5*4 = 20 directed edges)
        assert num_non_self_edges > 0, "Should have edges between nearby nodes"


class TestKNNMethod:
    """Tests for KNN-based edge construction."""

    def test_knn_every_node_has_k_edges(self, base_config):
        """Test that every node has exactly k outgoing edges in KNN."""
        base_config['graph']['method'] = 'knn'
        base_config['graph']['k_neighbors'] = 3

        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        edge_index = graph.edge_index

        # Count outgoing edges per node (excluding self-loops added separately)
        # Note: KNN graph is made undirected, so edge count may vary
        # But each node should have at least k connections
        for node in range(graph.num_nodes):
            # Count edges where this node is source or target (undirected)
            node_edges = ((edge_index[0] == node) | (edge_index[1] == node)).sum().item()

            # Should have at least k edges (may have more due to bidirectional + self-loop)
            assert node_edges >= base_config['graph']['k_neighbors'], \
                f"Node {node} has only {node_edges} edges, expected >= {base_config['graph']['k_neighbors']}"


class TestSelfLoops:
    """Tests for self-loop handling."""

    def test_self_loops_present_when_enabled(self, base_config):
        """Test that self-loops are added when config flag is True."""
        base_config['graph']['self_loops'] = True

        halo = create_test_halo(num_subhalos=5)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        edge_index = graph.edge_index

        # Count self-loops
        self_loop_mask = edge_index[0] == edge_index[1]
        num_self_loops = self_loop_mask.sum().item()

        assert num_self_loops == graph.num_nodes, \
            f"Expected {graph.num_nodes} self-loops, got {num_self_loops}"

    def test_self_loops_absent_when_disabled(self, base_config):
        """Test that self-loops are NOT added when config flag is False."""
        base_config['graph']['self_loops'] = False

        halo = create_test_halo(num_subhalos=5)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        edge_index = graph.edge_index

        # Count self-loops
        self_loop_mask = edge_index[0] == edge_index[1]
        num_self_loops = self_loop_mask.sum().item()

        assert num_self_loops == 0, f"Expected 0 self-loops, got {num_self_loops}"


class TestEdgeFeatures:
    """Tests for edge feature computation."""

    def test_edge_features_shape(self, base_config):
        """Test that edge features have shape [E, 5]."""
        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        edge_attr = graph.edge_attr
        num_edges = graph.edge_index.shape[1]

        assert edge_attr.shape == (num_edges, 5), \
            f"Expected edge_attr shape ({num_edges}, 5), got {edge_attr.shape}"

    def test_edge_features_no_nan(self, base_config):
        """Test that edge features contain no NaN values."""
        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        assert not torch.any(torch.isnan(graph.edge_attr)), "Edge features contain NaN"
        assert not torch.any(torch.isinf(graph.edge_attr)), "Edge features contain Inf"

    def test_subset_of_edge_features(self, base_config):
        """Test that subset of edge features works."""
        base_config['graph']['edge_features'] = ['distance', 'delta_v']

        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        # Should have 2 edge features
        assert graph.edge_attr.shape[1] == 2, \
            f"Expected 2 edge features, got {graph.edge_attr.shape[1]}"

    def test_distance_feature_positive(self, base_config):
        """Test that distance edge feature is non-negative."""
        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        # First feature is distance
        distances = graph.edge_attr[:, 0]
        assert torch.all(distances >= 0), "Distances should be non-negative"


class TestGraphSymmetry:
    """Tests for graph undirectedness."""

    def test_graph_is_undirected(self, base_config):
        """Test that graph is undirected (symmetric edge_index)."""
        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        edge_index = graph.edge_index

        # Create set of edges as tuples
        edges = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edges.add((src, dst))

        # For each edge (a, b), reverse (b, a) should also exist
        for src, dst in edges:
            if src != dst:  # Skip self-loops
                assert (dst, src) in edges, f"Edge ({dst}, {src}) missing for ({src}, {dst})"


class TestDeterminism:
    """Tests for reproducibility."""

    def test_same_seed_same_graph(self, base_config):
        """Test that same input + same seed = identical graph."""
        halo1 = create_test_halo(num_subhalos=15, cluster_id='test')
        halo2 = create_test_halo(num_subhalos=15, cluster_id='test')

        builder1 = GraphBuilder(base_config)
        builder2 = GraphBuilder(base_config)

        graph1 = builder1.build_graph(halo1)
        graph2 = builder2.build_graph(halo2)

        # Edge indices should be identical
        assert torch.equal(graph1.edge_index, graph2.edge_index), \
            "Edge indices should be identical for same seed"

        # Node features should be identical
        assert torch.allclose(graph1.x, graph2.x), \
            "Node features should be identical"

        # Edge features should be identical
        assert torch.allclose(graph1.edge_attr, graph2.edge_attr), \
            "Edge features should be identical"

    def test_different_halos_different_graphs(self, base_config):
        """Test that different halos produce different graphs."""
        halo1 = create_test_halo(num_subhalos=10, cluster_id='halo1', spread=0.5)

        # Create different halo
        np.random.seed(123)  # Different seed
        halo2 = HaloData(
            cluster_id='halo2',
            subhalos=[
                SubhaloData(
                    subhalo_id=i,
                    position=np.random.randn(3).astype(np.float32) * 2.0,
                    velocity=np.random.randn(3).astype(np.float32) * 300.0,
                    stellar_mass=1e11,
                    velocity_dispersion=250.0,
                    half_mass_radius=0.02,
                    metallicity=0.03
                ) for i in range(10)
            ],
            halo_mass=1e14
        )

        builder = GraphBuilder(base_config)
        graph1 = builder.build_graph(halo1)
        graph2 = builder.build_graph(halo2)

        # Should have different features
        assert not torch.allclose(graph1.x, graph2.x), \
            "Different halos should have different node features"


class TestBuildGraphs:
    """Tests for batch graph building."""

    def test_build_graphs_multiple(self, base_config):
        """Test building graphs for multiple halos."""
        halos = [create_test_halo(num_subhalos=5 + i, cluster_id=f'halo_{i}')
                 for i in range(5)]

        builder = GraphBuilder(base_config)
        graphs = builder.build_graphs(halos)

        assert len(graphs) == len(halos)

        for i, graph in enumerate(graphs):
            assert graph.num_nodes == 5 + i
            assert graph.cluster_id == f'halo_{i}'


class TestGraphDataset:
    """Tests for GraphDataset wrapper."""

    def test_dataset_length(self, base_config):
        """Test dataset length."""
        halos = [create_test_halo(num_subhalos=5, cluster_id=f'h{i}') for i in range(10)]
        builder = GraphBuilder(base_config)
        graphs = builder.build_graphs(halos)

        dataset = GraphDataset(graphs)
        assert len(dataset) == 10

    def test_dataset_getitem(self, base_config):
        """Test dataset indexing."""
        halos = [create_test_halo(num_subhalos=5, cluster_id=f'h{i}') for i in range(5)]
        builder = GraphBuilder(base_config)
        graphs = builder.build_graphs(halos)

        dataset = GraphDataset(graphs)

        for i in range(5):
            assert dataset[i].cluster_id == f'h{i}'


class TestGraphStatistics:
    """Tests for graph statistics computation."""

    def test_compute_statistics(self, base_config):
        """Test statistics computation."""
        halos = [create_test_halo(num_subhalos=10 + i) for i in range(5)]
        builder = GraphBuilder(base_config)
        graphs = builder.build_graphs(halos)

        stats = compute_graph_statistics(graphs)

        assert 'num_graphs' in stats
        assert stats['num_graphs'] == 5
        assert 'nodes_mean' in stats
        assert 'edges_mean' in stats
        assert stats['nodes_mean'] > 0

    def test_empty_graph_collection_has_explicit_error(self):
        """Removing the explicit empty guard must expose an incidental NumPy error."""
        with pytest.raises(ValueError, match="at least one graph"):
            compute_graph_statistics([])


class TestHierarchicalGraphs:
    """Tests for hierarchical graph scaffolding."""

    def test_hierarchical_flag(self, base_config):
        """Test that hierarchical flag adds hierarchy_level attribute."""
        base_config['graph']['hierarchical'] = True

        halo = create_test_halo(num_subhalos=10)
        builder = GraphBuilder(base_config)
        graph = builder.build_graph(halo)

        # Should have hierarchy_level attribute
        assert hasattr(graph, 'hierarchy_level')
        assert graph.hierarchy_level.shape[0] == graph.num_nodes

    def test_hierarchical_batch_uses_same_public_behavior(self, base_config):
        base_config['graph']['hierarchical'] = True
        graphs = GraphBuilder(base_config).build_graphs([
            create_test_halo(num_subhalos=4, cluster_id='h0'),
            create_test_halo(num_subhalos=5, cluster_id='h1'),
        ])
        assert [graph.hierarchy_level.shape[0] for graph in graphs] == [4, 5]


def _geometry_halo(positions, cluster_id='geometry'):
    return HaloData(
        cluster_id=cluster_id,
        subhalos=[
            SubhaloData(
                subhalo_id=index,
                position=np.asarray(position, dtype=np.float32),
                velocity=np.asarray(position, dtype=np.float32) * 10,
                stellar_mass=1e10 + index,
                velocity_dispersion=100 + index,
                half_mass_radius=0.01,
                metallicity=0.02,
            )
            for index, position in enumerate(positions)
        ],
        halo_mass=1e13,
    )


def _edge_set(edge_index):
    return {tuple(edge) for edge in edge_index.t().tolist()}


class TestPortableTieInclusiveConstruction:
    def test_builder_initialization_does_not_mutate_global_rng(self, base_config):
        np.random.seed(1701)
        torch.manual_seed(1701)
        expected_np = np.random.random(3)
        expected_torch = torch.rand(3)

        np.random.seed(1701)
        torch.manual_seed(1701)
        GraphBuilder(base_config)

        assert np.array_equal(np.random.random(3), expected_np)
        assert torch.equal(torch.rand(3), expected_torch)

    @pytest.mark.parametrize('method', ['radius', 'knn'])
    def test_exact_ties_are_permutation_equivariant(self, base_config, method):
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        base_config['graph'].update({
            'method': method,
            'radius_mpc': 1.0,
            'k_neighbors': 1,
            'self_loops': False,
        })
        original = GraphBuilder(base_config).build_graph(_geometry_halo(positions))
        permutation = np.array([2, 0, 3, 1])
        permuted = GraphBuilder(base_config).build_graph(
            _geometry_halo(positions[permutation], cluster_id='permuted'))
        transported = permutation[permuted.edge_index.cpu().numpy()]

        assert _edge_set(original.edge_index) == {
            tuple(edge) for edge in transported.T.tolist()
        }

    def test_edge_helpers_preserve_input_device(self, base_config):
        positions = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        builder = GraphBuilder(base_config)
        assert builder._build_radius_edges(positions, 2).device == positions.device
        builder.method = 'knn'
        assert builder._build_knn_edges(positions, 2).device == positions.device


class TestBatchFailurePolicy:
    def test_build_graphs_fails_closed_with_halo_context(self, base_config, monkeypatch):
        halo = _geometry_halo([[0, 0, 0]], cluster_id='broken-halo')
        builder = GraphBuilder(base_config)
        monkeypatch.setattr(builder, 'build_graph', lambda _: (_ for _ in ()).throw(RuntimeError('boom')))

        with pytest.raises(RuntimeError, match='broken-halo'):
            builder.build_graphs([halo])

    def test_build_graphs_skip_mode_is_explicit(self, base_config, monkeypatch):
        halo = _geometry_halo([[0, 0, 0]], cluster_id='broken-halo')
        builder = GraphBuilder(base_config)
        monkeypatch.setattr(builder, 'build_graph', lambda _: (_ for _ in ()).throw(RuntimeError('boom')))

        assert builder.build_graphs([halo], on_error='skip') == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
