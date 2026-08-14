"""
model.py
Purpose: Physics-informed Graph Neural Network for dark matter halo mass prediction.
         Implements NNConv (Edge-Conditioned Convolution) with configurable pooling,
         residual connections, and MC-Dropout for uncertainty quantification.
Inputs: batch_data: PyG Batch - Batched graph data
        node_features: Tensor[N, 4] - [log_stellar_mass, log_vel_disp, log_half_mass_r, log_metallicity]
        edge_features: Tensor[E, D] - Edge features from graph builder
        edge_index: Tensor[2, E] - Edge connectivity
Outputs: predictions: Tensor[B] - Predicted log10(M_halo) per cluster
         embeddings: Tensor[N, hidden_dim] - Node embeddings (for symbolic regression)
Config keys: model.node_features, model.edge_features, model.hidden_dim, model.output_dim,
             model.num_layers, model.dropout, model.pooling, model.residual,
             model.mc_dropout, model.mc_samples, model.activation
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, Set2Set
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU()
    }
    return activations.get(name, nn.LeakyReLU(0.1))


class EdgeMLP(nn.Module):
    """
    MLP that transforms edge features for NNConv.

    For NNConv, this MLP takes edge features and outputs a weight matrix
    that transforms the source node features during message passing.
    Output shape: (edge_features_dim,) -> (in_channels * out_channels,)
    """

    def __init__(
        self,
        edge_dim: int,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        activation: str = 'leaky_relu'
    ):
        """
        Initialize EdgeMLP.

        Args:
            edge_dim: Number of edge features
            in_channels: Input node feature dimension
            out_channels: Output node feature dimension
            hidden_dim: Hidden dimension of MLP
            activation: Activation function name
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, in_channels * out_channels)
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Weight matrices [E, in_channels * out_channels]
        """
        return self.mlp(edge_attr)


class NNConvBlock(nn.Module):
    """
    Single NNConv block with optional residual connection and dropout.

    Structure:
    - NNConv (edge-conditioned message passing)
    - LayerNorm (post-norm style)
    - Activation
    - Dropout
    - Optional residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        residual: bool = True,
        activation: str = 'leaky_relu'
    ):
        """
        Initialize NNConv block.

        Args:
            in_channels: Input node feature dimension
            out_channels: Output node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for edge MLP
            dropout: Dropout probability
            residual: Whether to use residual connection
            activation: Activation function name
        """
        super().__init__()

        self.residual = residual and (in_channels == out_channels)

        # Edge MLP for this layer (separate weights per layer)
        self.edge_mlp = EdgeMLP(
            edge_dim=edge_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            activation=activation
        )

        # NNConv layer
        self.conv = NNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            nn=self.edge_mlp,
            aggr='mean'  # Mean aggregation
        )

        # Normalization and activation
        self.norm = nn.LayerNorm(out_channels)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        if residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
            self.residual = True
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Updated node features [N, out_channels]
        """
        # Store input for residual
        identity = x

        # NNConv message passing
        out = self.conv(x, edge_index, edge_attr)

        # Post-norm style: conv -> norm -> activation -> dropout
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity

        return out


class CosmicNetGNN(nn.Module):
    """
    Physics-informed GNN for dark matter halo mass prediction.

    Architecture:
    1. Input projection: node_features (4) -> hidden_dim
    2. Stack of NNConv blocks with residual connections
    3. Global pooling (mean or Set2Set)
    4. MLP prediction head

    Features:
    - Edge-conditioned message passing (NNConv)
    - Separate edge MLPs per layer (no weight sharing)
    - Configurable pooling (mean or Set2Set)
    - MC-Dropout for uncertainty quantification
    - Embedding extraction for symbolic regression
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GNN model.

        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__()

        self.config = config
        model_config = config.get('model', {})
        graph_config = config.get('graph', {})

        # Model dimensions
        self.node_input_dim = model_config.get('node_features', 4)
        self.edge_input_dim = len(graph_config.get('edge_features', ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep']))
        self.hidden_dim = model_config.get('hidden_dim', 128)
        self.output_dim = model_config.get('output_dim', 64)
        self.num_layers = model_config.get('num_layers', 3)

        # Dropout and residual
        self.dropout = model_config.get('dropout', 0.2)
        self.residual = model_config.get('residual', True)
        self.activation = model_config.get('activation', 'leaky_relu')

        # Pooling method
        self.pooling_method = model_config.get('pooling', 'mean')
        set2set_config = model_config.get('set2set', {})
        self.set2set_steps = set2set_config.get('processing_steps', 4)
        self.set2set_layers = set2set_config.get('num_layers', 1)

        # MC-Dropout settings
        self.mc_dropout = model_config.get('mc_dropout', True)
        self.mc_samples = model_config.get('mc_samples', 50)

        # Build network
        self._build_network()

        logger.info(f"CosmicNetGNN initialized: layers={self.num_layers}, "
                   f"hidden={self.hidden_dim}, pooling={self.pooling_method}, "
                   f"mc_dropout={self.mc_dropout}")

    def _build_network(self):
        """Build the network layers."""
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.node_input_dim, self.hidden_dim),
            get_activation(self.activation),
            nn.Dropout(self.dropout)
        )

        # NNConv blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.hidden_dim
            out_dim = self.hidden_dim if i < self.num_layers - 1 else self.output_dim

            block = NNConvBlock(
                in_channels=in_dim,
                out_channels=out_dim,
                edge_dim=self.edge_input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                residual=self.residual,
                activation=self.activation
            )
            self.conv_blocks.append(block)

        # Pooling layer
        if self.pooling_method == 'set2set':
            self.pooling = Set2Set(
                self.output_dim,
                processing_steps=self.set2set_steps,
                num_layers=self.set2set_layers
            )
            pool_output_dim = 2 * self.output_dim  # Set2Set doubles dimension
        else:
            self.pooling = None  # Will use global_mean_pool
            pool_output_dim = self.output_dim

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(pool_output_dim, self.hidden_dim),
            get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(
        self,
        batch_data: Batch,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            batch_data: PyG Batch object
            return_embeddings: Whether to return node embeddings

        Returns:
            predictions: Tensor [B] - Predicted log10(M_halo)
            embeddings: Tensor [N, output_dim] (if return_embeddings=True)
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch

        # Input projection
        x = self.input_proj(x)

        # NNConv blocks
        for block in self.conv_blocks:
            x = block(x, edge_index, edge_attr)

        # Store node embeddings (pre-pooling)
        node_embeddings = x

        # Global pooling
        if self.pooling_method == 'set2set':
            x = self.pooling(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # Prediction head
        predictions = self.pred_head(x).squeeze(-1)

        if return_embeddings:
            return predictions, node_embeddings
        return predictions, None

    def predict_with_uncertainty(
        self,
        batch_data: Batch,
        n_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with MC-Dropout uncertainty estimation.

        Runs multiple forward passes with dropout enabled to estimate
        prediction uncertainty.

        Args:
            batch_data: PyG Batch object
            n_samples: Number of MC samples (uses config default if None)

        Returns:
            Dictionary with:
                - mean: Mean prediction [B]
                - std: Standard deviation [B]
                - samples: All samples [n_samples, B]
                - confidence_95_low: 2.5th percentile [B]
                - confidence_95_high: 97.5th percentile [B]
        """
        n_samples = n_samples or self.mc_samples

        # Enable dropout during inference
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.forward(batch_data)
                samples.append(pred)

        samples = torch.stack(samples, dim=0)  # [n_samples, B]

        # Compute statistics
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        # 95% confidence interval (2.5th and 97.5th percentiles)
        sorted_samples, _ = samples.sort(dim=0)
        low_idx = int(0.025 * n_samples)
        high_idx = int(0.975 * n_samples)
        conf_low = sorted_samples[low_idx]
        conf_high = sorted_samples[high_idx]

        # Reset to eval mode
        self.eval()

        return {
            'mean': mean,
            'std': std,
            'samples': samples,
            'confidence_95_low': conf_low,
            'confidence_95_high': conf_high
        }

    def get_embeddings(
        self,
        batch_data: Batch,
        embedding_point: str = 'pre_pooling'
    ) -> torch.Tensor:
        """
        Extract embeddings for symbolic regression.

        Args:
            batch_data: PyG Batch object
            embedding_point: 'pre_pooling' (per-node) or 'post_pooling' (per-graph)

        Returns:
            embeddings: Tensor of embeddings
        """
        self.eval()
        with torch.no_grad():
            x = batch_data.x
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            batch = batch_data.batch

            # Input projection
            x = self.input_proj(x)

            # NNConv blocks
            for block in self.conv_blocks:
                x = block(x, edge_index, edge_attr)

            if embedding_point == 'pre_pooling':
                return x
            else:
                # Post-pooling embeddings
                if self.pooling_method == 'set2set':
                    return self.pooling(x, batch)
                else:
                    return global_mean_pool(x, batch)

    def get_weight_norms(self) -> Dict[str, float]:
        """
        Get L2 norms of model weights for logging.

        Returns:
            Dictionary mapping layer names to weight norms
        """
        norms = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                norms[name] = param.data.norm(2).item()
        return norms

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleGNN(nn.Module):
    """
    Ensemble of GNN models for improved predictions and uncertainty.

    Trains multiple models with different random seeds and averages
    their predictions. Provides epistemic uncertainty estimation.
    """

    def __init__(self, config: Dict[str, Any], n_models: int = 5):
        """
        Initialize ensemble.

        Args:
            config: Configuration dictionary
            n_models: Number of models in ensemble
        """
        super().__init__()

        self.n_models = n_models
        self.models = nn.ModuleList()

        for i in range(n_models):
            # Create config with different seed for each model
            model_config = config.copy()
            model_config['seed'] = config.get('seed', 42) + i
            self.models.append(CosmicNetGNN(model_config))

        logger.info(f"EnsembleGNN initialized with {n_models} models")

    def forward(
        self,
        batch_data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all models.

        Args:
            batch_data: PyG Batch object

        Returns:
            mean_pred: Mean prediction across models [B]
            std_pred: Standard deviation across models [B]
        """
        predictions = []
        for model in self.models:
            pred, _ = model(batch_data)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [n_models, B]
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred


def build_model(config: Dict[str, Any]) -> CosmicNetGNN:
    """
    Factory function to build the GNN model.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized CosmicNetGNN model
    """
    model = CosmicNetGNN(config)
    logger.info(f"Built model with {model.count_parameters():,} parameters")
    return model


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> CosmicNetGNN:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model to

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Prefer the checkpoint's own embedded config for the model architecture.
    # This makes loading robust to config.yaml drift: the checkpoint knows the
    # exact hidden_dim/output_dim/num_layers it was trained with, and a stale
    # config.yaml must not silently build a mismatched architecture.
    ckpt_config = checkpoint.get('config') if isinstance(checkpoint, dict) else None
    if ckpt_config and 'model' in ckpt_config:
        model = build_model(ckpt_config)
    else:
        model = build_model(config)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path} "
                f"(hidden_dim={model.hidden_dim}, output_dim={model.output_dim})")

    return model
