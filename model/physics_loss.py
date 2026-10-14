"""
physics_loss.py
Purpose: Physics-informed loss functions implementing the Virial Theorem constraint.
         Computes virial violation as a soft penalty to encourage physically consistent predictions.
         Fully standalone and independently unit-testable.
Inputs: predictions: Tensor[B] - Predicted log10(M_halo) per cluster
        batch_data: PyG Batch - Batched graph data with physical quantities
        config (dict) - Configuration dictionary from config.yaml
Outputs: total_loss: Tensor[1] - Combined MSE + λ * virial_loss
         loss_dict: Dict[str, Tensor] - Individual loss components for logging
Config keys: physics.G, physics.virial_coefficient, physics.use_virial_loss,
             training.lambda_schedule, training.lambda_start, training.lambda_end
"""

import logging
import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


class VirialLoss(nn.Module):
    """
    Physics-informed loss based on the Virial Theorem.

    The Virial Theorem for a gravitationally bound system states:
        2 * KE + PE = 0  (for virialized systems)

    Where:
        KE = (1/2) * M * σ²  (kinetic energy, σ = velocity dispersion)
        PE = -G * M² / R     (gravitational potential energy)

    We compute:
        virial_ratio = 2 * KE / |PE|

    The penalty is the existing (virial_ratio - 1)^2 proxy. It combines
    stellar kinetic energy with predicted halo potential energy and mean
    subhalo radius; it is not a calibrated halo equilibrium measurement.
    Forward evaluates energies in log space and clips ln(virial_ratio) to
    +/- physics.virial_log_ratio_bound (default 20, maximum 30). This
    deliberately saturates the penalty and its gradient in numerical tails.
    Logged KE/PE are also clipped to exp(+/- 70) for finite diagnostics.

    This is computed per-cluster (after pooling node features),
    not per-node, as the virial theorem applies to the whole system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Virial loss module.

        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__()

        self.physics_config = config.get('physics', {})

        # Gravitational constant in appropriate units
        # G = 4.302e-9 Mpc (km/s)² / M_sun
        self.G = self.physics_config.get('G', 4.302e-9)

        # Virial coefficient (2 for virial equilibrium)
        self.virial_coeff = self.physics_config.get('virial_coefficient', 2.0)

        # Whether to use virial loss
        self.use_virial = self.physics_config.get('use_virial_loss', True)

        self.log_ratio_bound = self.physics_config.get('virial_log_ratio_bound', 20.0)
        for name, value in [('G', self.G), ('virial_coefficient', self.virial_coeff),
                            ('virial_log_ratio_bound', self.log_ratio_bound)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"physics.{name} must be finite and positive")
        if self.log_ratio_bound > 30:
            raise ValueError("physics.virial_log_ratio_bound must be <= 30 for finite float32 squared penalties")

        logger.info(f"VirialLoss initialized: G={self.G}, use_virial={self.use_virial}")

    def compute_kinetic_energy(
        self,
        stellar_masses: torch.Tensor,
        vel_dispersions: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cluster kinetic energy from subhalo properties.

        KE_cluster = (1/2) * Σ_i (M_stellar_i * σ_i²)

        Args:
            stellar_masses: Tensor [N] - Stellar mass per subhalo (M_sun)
            vel_dispersions: Tensor [N] - Velocity dispersion per subhalo (km/s)
            batch: Tensor [N] - Batch assignment per node

        Returns:
            ke: Tensor [B] - Kinetic energy per cluster
        """
        # KE per subhalo: (1/2) * M * σ²
        ke_per_subhalo = 0.5 * stellar_masses * vel_dispersions ** 2

        # Sum over subhalos in each cluster
        num_clusters = batch.max().item() + 1 if len(batch) > 0 else 0
        ke_per_cluster = torch.zeros(num_clusters, device=stellar_masses.device)

        for i in range(num_clusters):
            mask = batch == i
            if mask.any():
                ke_per_cluster[i] = ke_per_subhalo[mask].sum()

        return ke_per_cluster

    def compute_potential_energy(
        self,
        halo_masses_pred: torch.Tensor,
        half_mass_radii: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cluster potential energy from predicted halo mass.

        PE_cluster ≈ -G * M_halo² / R_half

        Uses the mean half-mass radius of subhalos as characteristic scale.

        Args:
            halo_masses_pred: Tensor [B] - Predicted halo mass per cluster (M_sun)
            half_mass_radii: Tensor [N] - Half-mass radius per subhalo (Mpc)
            batch: Tensor [N] - Batch assignment per node

        Returns:
            pe: Tensor [B] - Potential energy magnitude per cluster (positive)
        """
        num_clusters = len(halo_masses_pred)

        # Compute mean R_half per cluster
        r_half_per_cluster = torch.zeros(num_clusters, device=halo_masses_pred.device)

        for i in range(num_clusters):
            mask = batch == i
            if mask.any():
                r_half_per_cluster[i] = half_mass_radii[mask].mean()

        # Avoid division by zero
        r_half_per_cluster = torch.clamp(r_half_per_cluster, min=1e-6)

        # |PE| = G * M² / R (magnitude, positive)
        pe_magnitude = self.G * halo_masses_pred ** 2 / r_half_per_cluster

        return pe_magnitude

    def compute_virial_ratio(
        self,
        ke: torch.Tensor,
        pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the virial ratio: 2 * KE / |PE|

        For virialized systems, this should be ≈ 1.

        Args:
            ke: Tensor [B] - Kinetic energy per cluster
            pe: Tensor [B] - Potential energy magnitude per cluster

        Returns:
            virial_ratio: Tensor [B] - Virial ratio per cluster
        """
        # Avoid division by zero
        pe_safe = torch.clamp(pe, min=1e-10)

        # virial_ratio = 2 * KE / |PE|
        virial_ratio = self.virial_coeff * ke / pe_safe

        return virial_ratio

    def forward(
        self,
        predictions: torch.Tensor,
        batch_data: Batch,
        lambda_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the virial loss.

        Args:
            predictions: Tensor [B] - Predicted log10(M_halo) per cluster
            batch_data: PyG Batch object with physical quantities
            lambda_weight: Weight for virial loss term

        Returns:
            virial_loss: Tensor [1] - Mean virial violation
            loss_dict: Dictionary with detailed loss components
        """
        if isinstance(lambda_weight, bool) or not isinstance(lambda_weight, (int, float)) or not math.isfinite(lambda_weight) or lambda_weight < 0:
            raise ValueError("lambda_weight must be finite and nonnegative")

        if not self.use_virial or lambda_weight == 0:
            # No physics observations are required by a disabled term. Keep a
            # differentiable zero so standalone loss.backward() remains valid.
            zero = (predictions * 0.0).sum()
            return zero, {'virial_loss': zero, 'virial_ratio_mean': zero + 1.0,
                          'virial_ratio_std': zero, 'ke_mean': zero, 'pe_mean': zero,
                          'virial_ratio_clipped_fraction': zero}

        if predictions.ndim != 1 or predictions.numel() == 0 or not torch.isfinite(predictions).all():
            raise ValueError("physics predictions must be a nonempty finite vector of log10 halo masses")

        # Get physical quantities from batch
        stellar_masses = batch_data.stellar_mass
        vel_dispersions = batch_data.vel_disp
        half_mass_radii = batch_data.half_mass_r
        batch = batch_data.batch

        for name, values in [('stellar_mass', stellar_masses), ('vel_disp', vel_dispersions),
                             ('half_mass_r', half_mass_radii)]:
            if values.ndim != 1 or values.shape != batch.shape or not torch.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"physics {name} must be a positive finite vector matching batch assignments")
        if batch.dtype not in (torch.int32, torch.int64) or (batch < 0).any() or (batch >= len(predictions)).any():
            raise ValueError("physics batch assignments must index the predicted clusters")

        # Double precision in log space avoids forming 10**prediction or M**2.
        # The only exponentials used for the objective are explicitly bounded.
        log_ke_nodes = math.log(.5) + stellar_masses.double().log() + 2 * vel_dispersions.double().log()
        log_ke, log_radius = [], []
        for i in range(len(predictions)):
            mask = batch == i
            if not mask.any():
                raise ValueError(f"physics cluster {i} has no observations")
            log_ke.append(torch.logsumexp(log_ke_nodes[mask], dim=0))
            log_radius.append(half_mass_radii[mask].double().mean().clamp_min(1e-6).log())
        log_ke = torch.stack(log_ke)
        log_pe = math.log(self.G) + 2 * math.log(10) * predictions.double() - torch.stack(log_radius)
        log_ratio = math.log(self.virial_coeff) + log_ke - log_pe
        virial_ratio = log_ratio.clamp(-self.log_ratio_bound, self.log_ratio_bound).exp()
        virial_loss = (virial_ratio - 1.0).square().mean()

        # Components retain float64 precision and finite, explicitly bounded
        # diagnostic values even when the prediction lies far outside training.
        loss_dict = {'virial_loss': virial_loss,
                     'virial_ratio_mean': virial_ratio.mean(),
                     'virial_ratio_std': virial_ratio.std() if len(virial_ratio) > 1 else predictions.new_zeros(()),
                     'ke_mean': log_ke.clamp(-70, 70).exp().mean(),
                     'pe_mean': log_pe.clamp(-70, 70).exp().mean(),
                     'virial_ratio_clipped_fraction': (log_ratio.abs() > self.log_ratio_bound).double().mean()}

        return lambda_weight * virial_loss, loss_dict


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function: MSE + λ * Virial Loss

    The total loss balances data fitting (MSE on halo mass prediction)
    with physics constraints (virial equilibrium).

    Lambda (λ) can be annealed during training to gradually introduce
    the physics constraint after initial fitting.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the physics-informed loss.

        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__()

        self.config = config
        self.virial_loss_fn = VirialLoss(config)

        # Training config for gradient clipping
        self.training_config = config.get('training', {})
        self.virial_grad_clip = self.training_config.get('virial_grad_clip', 1.0)

        logger.info("PhysicsInformedLoss initialized")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch_data: Batch,
        lambda_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total physics-informed loss.

        Args:
            predictions: Tensor [B] - Predicted log10(M_halo)
            targets: Tensor [B] - True log10(M_halo)
            batch_data: PyG Batch object
            lambda_weight: Current weight for virial loss

        Returns:
            total_loss: Tensor [1] - Combined loss
            loss_dict: Dictionary with all loss components
        """
        loss_dict = {}

        # MSE loss on log halo mass
        mse_loss = F.mse_loss(predictions, targets)
        loss_dict['mse_loss'] = mse_loss

        # Virial loss
        virial_loss, virial_dict = self.virial_loss_fn(
            predictions, batch_data, lambda_weight
        )
        loss_dict.update(virial_dict)

        # Total loss
        total_loss = mse_loss + virial_loss
        loss_dict['total_loss'] = total_loss
        loss_dict['lambda'] = torch.tensor(lambda_weight, device=predictions.device)

        return total_loss, loss_dict

    def compute_mse_only(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only (for validation/evaluation).

        Args:
            predictions: Tensor [B] - Predicted log10(M_halo)
            targets: Tensor [B] - True log10(M_halo)

        Returns:
            mse_loss: Tensor [1]
        """
        return F.mse_loss(predictions, targets)


class GradientScaler:
    """
    Utility class for separate gradient scaling of physics vs data loss.

    Allows clipping virial loss gradients independently to prevent
    physics constraints from overwhelming the data fitting.
    """

    def __init__(self, virial_grad_clip: float = 1.0):
        """
        Initialize gradient scaler.

        Args:
            virial_grad_clip: Maximum gradient norm for virial loss
        """
        self.virial_grad_clip = virial_grad_clip

    def scale_gradients(
        self,
        model: nn.Module,
        mse_loss: torch.Tensor,
        virial_loss: torch.Tensor
    ) -> None:
        """
        Apply separate gradient scaling for MSE and virial losses.

        This is called during the backward pass to ensure virial loss
        gradients don't dominate.

        Note: This requires manual backward calls instead of optimizer.step()
        on total loss. Use when fine control over gradient balance is needed.

        Args:
            model: The GNN model
            mse_loss: MSE loss tensor
            virial_loss: Virial loss tensor
        """
        # Compute gradients separately
        mse_loss.backward(retain_graph=True)

        # Store MSE gradients
        mse_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                mse_grads[name] = param.grad.clone()

        # Zero gradients for virial loss computation
        model.zero_grad()

        # Compute virial gradients
        virial_loss.backward()

        # Clip virial gradients
        virial_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.virial_grad_clip
        )

        # Add MSE gradients back
        for name, param in model.named_parameters():
            if param.grad is not None and name in mse_grads:
                param.grad += mse_grads[name]


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Tensor [B]
        targets: Tensor [B]

    Returns:
        rmse: Tensor [1]
    """
    return torch.sqrt(F.mse_loss(predictions, targets))


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Tensor [B]
        targets: Tensor [B]

    Returns:
        mae: Tensor [1]
    """
    return F.l1_loss(predictions, targets)


def compute_r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute R² coefficient of determination.

    Args:
        predictions: Tensor [B]
        targets: Tensor [B]

    Returns:
        r2: Tensor [1]
    """
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2


def compute_scatter(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute scatter (standard deviation of residuals).

    Common metric in astrophysics papers.

    Args:
        predictions: Tensor [B]
        targets: Tensor [B]

    Returns:
        scatter: Tensor [1] - Standard deviation of (pred - target)
    """
    residuals = predictions - targets
    return residuals.std()


class MetricsComputer:
    """
    Computes all evaluation metrics for model performance tracking.
    """

    @staticmethod
    def compute_all(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predictions: Tensor [B]
            targets: Tensor [B]

        Returns:
            Dictionary of metrics
        """
        return {
            'mse': F.mse_loss(predictions, targets).item(),
            'rmse': compute_rmse(predictions, targets).item(),
            'mae': compute_mae(predictions, targets).item(),
            'r2': compute_r2_score(predictions, targets).item(),
            'scatter': compute_scatter(predictions, targets).item()
        }


# Unit tests for standalone testing
def _test_virial_loss():
    """Test VirialLoss computation."""
    config = {
        'physics': {
            'G': 4.302e-9,
            'virial_coefficient': 2.0,
            'use_virial_loss': True
        }
    }

    loss_fn = VirialLoss(config)

    # Create mock data
    batch_size = 4
    nodes_per_cluster = 10
    total_nodes = batch_size * nodes_per_cluster

    # Mock batch data
    class MockBatch:
        stellar_mass = torch.ones(total_nodes) * 1e10  # M_sun
        vel_disp = torch.ones(total_nodes) * 200  # km/s
        half_mass_r = torch.ones(total_nodes) * 0.01  # Mpc
        batch = torch.repeat_interleave(
            torch.arange(batch_size),
            nodes_per_cluster
        )

    batch_data = MockBatch()

    # Predictions: log10(M_halo)
    predictions = torch.ones(batch_size) * 12.0  # 10^12 M_sun

    # Compute loss
    loss, loss_dict = loss_fn(predictions, batch_data, lambda_weight=1.0)

    print(f"Virial loss: {loss.item():.6f}")
    print(f"Virial ratio mean: {loss_dict['virial_ratio_mean'].item():.4f}")
    print("Test passed!")


if __name__ == "__main__":
    _test_virial_loss()
