"""
graph_builder.py
Purpose: Converts HaloData objects into PyTorch Geometric Data objects for GNN training.
         Supports radius-based and KNN edge construction with configurable edge features.
         Implements hierarchical graph scaffolding for multi-scale physics.
Inputs: List[HaloData] - Halo data from any loader
        config (dict) - Configuration dictionary from config.yaml
Outputs: List[torch_geometric.data.Data] - PyG Data objects ready for batching
         node_features: Tensor[N, 4] - [log_stellar_mass, log_vel_disp, log_half_mass_r, log_metallicity]
         edge_features: Tensor[E, D] - D depends on config (up to 5 features)
         edge_index: Tensor[2, E] - COO format edge indices
         y: Tensor[1] - log10(M_halo / M_sun)
Config keys: graph.method, graph.radius_mpc, graph.k_neighbors, graph.self_loops,
             graph.edge_features, graph.hierarchical, seed
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from data.loaders.base_loader import HaloData

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds PyTorch Geometric graphs from HaloData objects.

    Supports multiple edge construction methods:
    - radius: Connect nodes within a radius (physics-motivated)
    - knn: Connect to k-nearest neighbors

    Edge features encode physical relationships:
    - distance: Euclidean distance (gravitational PE proxy)
    - delta_v: Relative velocity magnitude
    - cos_theta: Velocity approach/recession angle
    - mass_ratio: Log stellar mass ratio
    - proj_sep: Projected separation (mock observational)

    All construction parameters are configurable via config.yaml
    for ablation studies.
    """

    # Edge feature names and their indices
    EDGE_FEATURE_MAP = {
        'distance': 0,
        'delta_v': 1,
        'cos_theta': 2,
        'mass_ratio': 3,
        'proj_sep': 4
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the graph builder.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.graph_config = config.get('graph', {})
        self.seed = config.get('seed', 42)

        # Edge construction method
        self.method = self.graph_config.get('method', 'radius')
        self.radius_mpc = self.graph_config.get('radius_mpc', 2.0)
        self.k_neighbors = self.graph_config.get('k_neighbors', 10)

        # Self-loops
        self.self_loops = self.graph_config.get('self_loops', True)

        # Edge features to include
        self.edge_feature_names = self.graph_config.get('edge_features', [
            'distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep'
        ])
        self.num_edge_features = len(self.edge_feature_names)

        # Hierarchical graph settings
        self.hierarchical = self.graph_config.get('hierarchical', False)
        self.hierarchical_settings = self.graph_config.get('hierarchical_settings', {})

        self.distance_chunk_size = int(self.graph_config.get('distance_chunk_size', 1024))
        self.tie_rtol = float(self.graph_config.get('tie_rtol', 1e-6))
        self.tie_atol = float(self.graph_config.get('tie_atol', 1e-8))

        logger.info(f"GraphBuilder initialized: method={self.method}, "
                   f"edge_features={self.edge_feature_names}, "
                   f"hierarchical={self.hierarchical}")

    def build_graph(self, halo: HaloData) -> Data:
        """
        Build a PyG Data object from a single HaloData.

        Args:
            halo: HaloData object with subhalo information

        Returns:
            torch_geometric.data.Data object
        """
        if (isinstance(getattr(halo, "metadata", None), dict)
                and isinstance(halo.metadata.get("observation"), dict)
                and halo.metadata["observation"].get("schema") == "projected_observation_v1"):
            raise ValueError(
                "projected observations require data.projected_graph.build_projected_graph"
            )
        if self.hierarchical:
            return self._build_hierarchical_graph(halo)
        return self._build_standard_graph(halo)

    def _build_standard_graph(self, halo: HaloData) -> Data:
        """Build the single-level physical graph used by every public path."""
        # Get node features [N, 4]
        node_features = torch.tensor(halo.get_node_features(), dtype=torch.float32)
        num_nodes = node_features.shape[0]

        if num_nodes == 0:
            logger.warning(f"Halo {halo.cluster_id} has no subhalos")
            return self._create_empty_graph(halo)

        # Get positions and velocities for edge construction
        positions = torch.tensor(halo.get_positions(), dtype=torch.float32)
        velocities = torch.tensor(halo.get_velocities(), dtype=torch.float32)

        # Build edge index based on method
        if self.method == 'radius':
            edge_index = self._build_radius_edges(positions, num_nodes)
        elif self.method == 'knn':
            edge_index = self._build_knn_edges(positions, num_nodes)
        else:
            raise ValueError(f"Unknown edge method: {self.method}")

        # Add self-loops if configured
        if self.self_loops:
            edge_index = self._add_self_loops(edge_index, num_nodes)

        # Ensure all isolated nodes have at least one edge (nearest neighbor fallback)
        edge_index = self._connect_isolated_nodes(edge_index, positions, num_nodes)

        # Compute edge features
        edge_attr = self._compute_edge_features(
            edge_index, positions, velocities, node_features
        )

        # Target: log10(halo mass)
        y = torch.tensor([halo.log_halo_mass], dtype=torch.float32)

        # Physical quantities for virial loss computation
        stellar_masses = torch.tensor(
            [s.stellar_mass for s in halo.subhalos], dtype=torch.float32
        )
        vel_dispersions = torch.tensor(
            [s.velocity_dispersion for s in halo.subhalos], dtype=torch.float32
        )
        half_mass_radii = torch.tensor(
            [s.half_mass_radius for s in halo.subhalos], dtype=torch.float32
        )

        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=positions,
            vel=velocities,
            stellar_mass=stellar_masses,
            vel_disp=vel_dispersions,
            half_mass_r=half_mass_radii,
            cluster_id=halo.cluster_id,
            metadata=dict(halo.metadata),
            num_nodes=num_nodes
        )

        return data

    def _build_radius_edges(
        self,
        positions: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Build edges connecting nodes within a radius.

        Args:
            positions: Node positions [N, 3]
            num_nodes: Number of nodes

        Returns:
            edge_index: Tensor [2, E] in COO format
        """
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

        edges = []
        for start in range(0, num_nodes, self.distance_chunk_size):
            stop = min(start + self.distance_chunk_size, num_nodes)
            distances = torch.cdist(positions[start:stop], positions)
            rows = torch.arange(start, stop, device=positions.device)[:, None]
            cols = torch.arange(num_nodes, device=positions.device)[None, :]
            mask = (rows != cols) & (distances <= self.radius_mpc)
            local_row, dst = torch.where(mask)
            if local_row.numel():
                edges.append(torch.stack((local_row + start, dst)))
        edge_index = (torch.cat(edges, dim=1) if edges else
                      torch.zeros((2, 0), dtype=torch.long, device=positions.device))
        return self._symmetrize_and_coalesce(edge_index, num_nodes)

    def _build_knn_edges(
        self,
        positions: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Build edges connecting to k-nearest neighbors.

        Args:
            positions: Node positions [N, 3]
            num_nodes: Number of nodes

        Returns:
            edge_index: Tensor [2, E] in COO format
        """
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

        k = min(self.k_neighbors, num_nodes - 1)

        if k <= 0:
            return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

        edges = []
        for start in range(0, num_nodes, self.distance_chunk_size):
            stop = min(start + self.distance_chunk_size, num_nodes)
            distances = torch.cdist(positions[start:stop], positions)
            rows = torch.arange(start, stop, device=positions.device)[:, None]
            cols = torch.arange(num_nodes, device=positions.device)[None, :]
            distances = distances.masked_fill(rows == cols, torch.inf)
            threshold = torch.kthvalue(distances, k, dim=1).values[:, None]
            # Include every exact/tolerance-level tie at the kth boundary.
            mask = distances <= (
                threshold + self.tie_atol + self.tie_rtol * threshold.abs()
            )
            local_row, dst = torch.where(mask)
            if local_row.numel():
                edges.append(torch.stack((local_row + start, dst)))
        edge_index = (torch.cat(edges, dim=1) if edges else
                      torch.zeros((2, 0), dtype=torch.long, device=positions.device))
        return self._symmetrize_and_coalesce(edge_index, num_nodes)

    @staticmethod
    def _symmetrize_and_coalesce(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Return sorted, unique directed edges for an undirected physical graph."""
        if edge_index.numel() == 0:
            return edge_index.reshape(2, 0)
        symmetric = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        linear = symmetric[0] * num_nodes + symmetric[1]
        linear = torch.unique(linear, sorted=True)
        return torch.stack((linear // num_nodes, linear % num_nodes))

    def _add_self_loops(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Add self-loops to the graph.

        Args:
            edge_index: Current edges [2, E]
            num_nodes: Number of nodes

        Returns:
            edge_index with self-loops added
        """
        self_loops = torch.arange(
            num_nodes, dtype=torch.long, device=edge_index.device
        ).unsqueeze(0).repeat(2, 1)
        return self._symmetrize_and_coalesce(
            torch.cat([edge_index, self_loops], dim=1), num_nodes
        )

    def _connect_isolated_nodes(
        self,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Connect isolated nodes to their nearest neighbor.

        Args:
            edge_index: Current edges [2, E]
            positions: Node positions [N, 3]
            num_nodes: Number of nodes

        Returns:
            edge_index with isolated nodes connected
        """
        if num_nodes <= 1:
            return edge_index

        # Find nodes with no edges (excluding self-loops)
        mask = edge_index[0] != edge_index[1]
        connected_nodes = torch.unique(edge_index[:, mask].flatten())

        all_nodes = torch.arange(num_nodes, device=positions.device)
        isolated_mask = ~torch.isin(all_nodes, connected_nodes)
        isolated_nodes = all_nodes[isolated_mask]

        if len(isolated_nodes) == 0:
            return edge_index

        distances = torch.cdist(positions[isolated_nodes], positions)
        distances.scatter_(1, isolated_nodes[:, None], torch.inf)
        minimum = distances.min(dim=1).values[:, None]
        tied = torch.isclose(
            distances, minimum, rtol=self.tie_rtol, atol=self.tie_atol
        )
        local_row, nearest = torch.where(tied)
        if local_row.numel():
            new_edge_tensor = torch.stack((isolated_nodes[local_row], nearest))
            edge_index = self._symmetrize_and_coalesce(
                torch.cat([edge_index, new_edge_tensor], dim=1), num_nodes
            )
            logger.debug(f"Connected {len(isolated_nodes)} isolated nodes")

        return edge_index

    def _compute_edge_features(
        self,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge features based on configuration.

        Args:
            edge_index: Edge indices [2, E]
            positions: Node positions [N, 3]
            velocities: Node velocities [N, 3]
            node_features: Node features [N, 4]

        Returns:
            edge_attr: Tensor [E, D] where D is number of selected edge features
        """
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return torch.zeros(
                (0, self.num_edge_features), dtype=torch.float32,
                device=positions.device
            )

        src, dst = edge_index

        # Compute all potential edge features
        all_features = {}

        # 1. Euclidean distance (Mpc) - gravitational PE proxy
        diff_pos = positions[dst] - positions[src]
        distances = torch.norm(diff_pos, dim=1, keepdim=True)
        all_features['distance'] = distances

        # 2. Relative velocity magnitude (km/s)
        diff_vel = velocities[dst] - velocities[src]
        delta_v = torch.norm(diff_vel, dim=1, keepdim=True)
        all_features['delta_v'] = delta_v

        # 3. Cosine of velocity angle (approach vs recession)
        # cos(theta) = (r_ij · v_ij) / (|r_ij| * |v_ij|)
        eps = 1e-8
        dot_product = (diff_pos * diff_vel).sum(dim=1, keepdim=True)
        cos_theta = dot_product / (distances * delta_v + eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        all_features['cos_theta'] = cos_theta

        # 4. Log stellar mass ratio
        # node_features[:, 0] is log_stellar_mass
        log_mass_src = node_features[src, 0:1]
        log_mass_dst = node_features[dst, 0:1]
        mass_ratio = log_mass_src - log_mass_dst  # log(M_i / M_j)
        all_features['mass_ratio'] = mass_ratio

        # 5. Projected separation (2D distance in xy-plane) - mock observational
        proj_sep = torch.norm(diff_pos[:, :2], dim=1, keepdim=True)
        all_features['proj_sep'] = proj_sep

        # Select only configured features
        selected_features = []
        for name in self.edge_feature_names:
            if name in all_features:
                selected_features.append(all_features[name])
            else:
                logger.warning(f"Unknown edge feature: {name}")

        if not selected_features:
            return torch.zeros(
                (num_edges, 1), dtype=torch.float32, device=positions.device
            )

        edge_attr = torch.cat(selected_features, dim=1)

        # Handle NaN/Inf values
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)

        return edge_attr

    def _create_empty_graph(self, halo: HaloData) -> Data:
        """
        Create an empty graph for halos with no subhalos.

        Args:
            halo: HaloData object

        Returns:
            Empty Data object
        """
        return Data(
            x=torch.zeros((0, 4), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, self.num_edge_features), dtype=torch.float32),
            y=torch.tensor([halo.log_halo_mass], dtype=torch.float32),
            pos=torch.zeros((0, 3), dtype=torch.float32),
            vel=torch.zeros((0, 3), dtype=torch.float32),
            stellar_mass=torch.zeros((0,), dtype=torch.float32),
            vel_disp=torch.zeros((0,), dtype=torch.float32),
            half_mass_r=torch.zeros((0,), dtype=torch.float32),
            cluster_id=halo.cluster_id,
            metadata=dict(halo.metadata),
            num_nodes=0
        )

    def build_graphs(self, halos: List[HaloData], on_error: str = 'raise') -> List[Data]:
        """
        Build graphs for a list of halos.

        Args:
            halos: List of HaloData objects

        Returns:
            List of PyG Data objects
        """
        if on_error not in {'raise', 'skip'}:
            raise ValueError("on_error must be 'raise' or 'skip'")
        graphs = []

        for halo in halos:
            try:
                graphs.append(self.build_graph(halo))
            except Exception as e:
                if on_error == 'raise':
                    raise RuntimeError(
                        f"Failed to build graph for halo {halo.cluster_id}: {e}"
                    ) from e
                logger.warning(f"Failed to build graph for {halo.cluster_id}: {e}")

        logger.info(f"Built {len(graphs)} graphs from {len(halos)} halos")

        return graphs

    def _build_hierarchical_graph(self, halo: HaloData) -> Data:
        """
        Build a hierarchical multi-scale graph (scaffold for future implementation).

        When hierarchical mode is enabled, this creates a 2-level graph:
        - Level 0: Subhalo nodes (fine scale)
        - Level 1: Halo-level aggregation nodes (coarse scale)

        Currently implements the standard single-level graph as a placeholder.
        The hierarchical logic can be filled in later.

        Args:
            halo: HaloData object

        Returns:
            Data object (currently single-level, ready for hierarchical extension)
        """
        # Build standard graph first
        data = self._build_standard_graph(halo)

        # Add hierarchical placeholders
        data.hierarchy_level = torch.zeros(data.num_nodes, dtype=torch.long)

        # Scaffold for hierarchical extension:
        # When implementing full hierarchical graphs:
        # 1. Cluster subhalos into sub-groups (spatial clustering)
        # 2. Create level-1 nodes representing these clusters
        # 3. Add inter-level edges connecting subhalos to cluster nodes
        # 4. Add level-1 edges between cluster nodes
        # 5. Store hierarchical structure in data object

        if self.hierarchical:
            logger.debug(f"Hierarchical graph scaffold created for {halo.cluster_id}")
            # Future implementation:
            # data = self._add_hierarchy_levels(data, halo)

        return data

    def get_edge_feature_dim(self) -> int:
        """Get the dimension of edge features based on config."""
        return self.num_edge_features

    def get_node_feature_dim(self) -> int:
        """Get the dimension of node features (fixed at 4)."""
        return 4
    def build(self, halos: List[HaloData]) -> List[Data]:
        """Alias for build_graphs — maintains API consistency."""
        return self.build_graphs(halos)


class GraphDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for pre-built graphs.

    Used with PyTorch Geometric DataLoader for batching.
    """

    def __init__(self, graphs: List[Data]):
        """
        Initialize dataset.

        Args:
            graphs: List of PyG Data objects
        """
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def build_dataloaders(
    config: Dict[str, Any],
    train_halos: List[HaloData],
    val_halos: List[HaloData],
    test_halos: List[HaloData],
    seed: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build PyTorch Geometric DataLoaders for train/val/test sets.

    Args:
        config: Configuration dictionary
        train_halos: Training halo data
        val_halos: Validation halo data
        test_halos: Test halo data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader

    augmentation_manifest = None
    augmentation_config = config.get('augmentation', {})
    if augmentation_config.get('enabled', False):
        from data.augmentation import augment_training_split
        train_halos, augmentation_manifest = augment_training_split(
            train_halos,
            copies=int(augmentation_config.get('copies', 1)),
            seed=int(augmentation_config.get('seed', config.get('seed', 42))),
            jitter_std=float(augmentation_config.get('jitter_std', 0.0)),
            rotate=bool(augmentation_config.get('rotate', True)),
            line_of_sight=augmentation_config.get('line_of_sight', 'z'),
        )

    # Build graph builder
    graph_builder = GraphBuilder(config)

    # Build graphs
    train_graphs = graph_builder.build_graphs(train_halos)
    val_graphs = graph_builder.build_graphs(val_halos)
    test_graphs = graph_builder.build_graphs(test_halos)

    # Get batch size
    batch_size = config.get('data', {}).get('batch_size', 32)
    num_workers = config.get('data', {}).get('num_workers', 4)

    # Create dataloaders
    train_gen = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    train_loader.augmentation_manifest = augmentation_manifest

    logger.info(f"Created dataloaders: train={len(train_graphs)}, "
               f"val={len(val_graphs)}, test={len(test_graphs)}")

    return train_loader, val_loader, test_loader


def compute_graph_statistics(graphs: List[Data]) -> Dict[str, Any]:
    """
    Compute statistics for a list of graphs.

    Args:
        graphs: List of PyG Data objects

    Returns:
        Dictionary of statistics
    """
    if not graphs:
        raise ValueError("compute_graph_statistics requires at least one graph")

    num_nodes_list = []
    num_edges_list = []
    edge_lengths = []

    for g in graphs:
        num_nodes_list.append(g.num_nodes)
        num_edges_list.append(g.edge_index.shape[1])

        if g.edge_attr is not None and g.edge_attr.shape[0] > 0:
            # First feature is distance
            edge_lengths.extend(g.edge_attr[:, 0].tolist())

    stats = {
        'num_graphs': len(graphs),
        'nodes_mean': np.mean(num_nodes_list),
        'nodes_std': np.std(num_nodes_list),
        'nodes_min': np.min(num_nodes_list),
        'nodes_max': np.max(num_nodes_list),
        'edges_mean': np.mean(num_edges_list),
        'edges_std': np.std(num_edges_list),
        'edges_min': np.min(num_edges_list),
        'edges_max': np.max(num_edges_list),
        'avg_degree': np.mean(num_edges_list) / max(np.mean(num_nodes_list), 1),
    }

    if edge_lengths:
        stats['edge_length_mean'] = np.mean(edge_lengths)
        stats['edge_length_std'] = np.std(edge_lengths)
        stats['edge_length_min'] = np.min(edge_lengths)
        stats['edge_length_max'] = np.max(edge_lengths)

    return stats
