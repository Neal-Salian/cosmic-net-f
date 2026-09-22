"""
base_loader.py
Purpose: Abstract base class defining the interface for all data loaders.
         Ensures consistent schema across synthetic, TNG, and CAMELS sources.
Inputs: config (dict) - Configuration dictionary from config.yaml
Outputs: List[HaloData] - List of normalized halo data objects
Config keys: data.source, data.grouping, data.train_ratio, data.val_ratio, data.test_ratio
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubhaloData:
    """
    Represents a single subhalo (galaxy) within a halo.

    Attributes:
        subhalo_id: Unique identifier for the subhalo
        position: 3D position [x, y, z] in Mpc
        velocity: 3D velocity [vx, vy, vz] in km/s
        stellar_mass: Stellar mass in M_sun (linear scale)
        velocity_dispersion: Velocity dispersion in km/s
        half_mass_radius: Stellar half-mass radius in Mpc
        metallicity: Stellar metallicity (dimensionless)
    """
    subhalo_id: int
    position: np.ndarray  # Shape: (3,)
    velocity: np.ndarray  # Shape: (3,)
    stellar_mass: float
    velocity_dispersion: float
    half_mass_radius: float
    metallicity: float

    def get_node_features(self) -> np.ndarray:
        """
        Returns normalized node features for GNN.

        Returns:
            np.ndarray: Shape (4,) - [log_stellar_mass, log_vel_disp,
                                      log_half_mass_r, log_metallicity]
        """
        eps = 1e-10  # Avoid log(0)
        return np.array([
            np.log10(max(self.stellar_mass, eps)),
            np.log10(max(self.velocity_dispersion, eps)),
            np.log10(max(self.half_mass_radius, eps)),
            np.log10(max(self.metallicity, eps))
        ], dtype=np.float32)


@dataclass
class HaloData:
    """
    Represents a complete halo with all its subhalos.

    This is the standardized internal schema that all data loaders
    must produce, regardless of the source (synthetic, TNG, CAMELS).

    Attributes:
        cluster_id: Unique identifier for the halo/cluster
        subhalos: List of SubhaloData objects
        halo_mass: Total halo mass in M_sun (target for prediction)
        redshift: Cosmological redshift (z)
        metadata: Additional source-specific metadata
    """
    cluster_id: str
    subhalos: List[SubhaloData]
    halo_mass: float  # M_sun, target for prediction
    redshift: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_subhalos(self) -> int:
        """Number of subhalos in this halo."""
        return len(self.subhalos)

    @property
    def log_halo_mass(self) -> float:
        """Log10 of halo mass (the prediction target)."""
        return np.log10(max(self.halo_mass, 1e-10))

    def get_positions(self) -> np.ndarray:
        """Get all subhalo positions as array. Shape: (N, 3)"""
        return np.array([s.position for s in self.subhalos], dtype=np.float32)

    def get_velocities(self) -> np.ndarray:
        """Get all subhalo velocities as array. Shape: (N, 3)"""
        return np.array([s.velocity for s in self.subhalos], dtype=np.float32)

    def get_node_features(self) -> np.ndarray:
        """Get all node features as array. Shape: (N, 4)"""
        return np.array([s.get_node_features() for s in self.subhalos], dtype=np.float32)

    def get_total_stellar_mass(self) -> float:
        """Sum of all subhalo stellar masses."""
        return sum(s.stellar_mass for s in self.subhalos)

    def get_mean_velocity_dispersion(self) -> float:
        """Mean velocity dispersion across subhalos."""
        if not self.subhalos:
            return 0.0
        return np.mean([s.velocity_dispersion for s in self.subhalos])

    def get_mean_half_mass_radius(self) -> float:
        """Mean half-mass radius across subhalos."""
        if not self.subhalos:
            return 0.0
        return np.mean([s.half_mass_radius for s in self.subhalos])


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.

    Subclasses must implement:
        - load_raw(): Load raw data from source
        - _parse_subhalo(): Parse a single subhalo record
        - _group_into_halos(): Group subhalos into halos

    Provides:
        - Consistent interface across all data sources
        - Train/val/test splitting with deterministic seeds
        - Data validation and logging
    """

    supports_strict_research = False

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.seed = config.get('seed', 42)
        self.source_name = self.__class__.__name__
        self.strict_research = bool(self.data_config.get('strict_research', False))
        self.declared_catalog_contract = self.data_config.get('catalog_contract')

        # Split ratios
        self.train_ratio = self.data_config.get('train_ratio', 0.7)
        self.val_ratio = self.data_config.get('val_ratio', 0.15)
        self.test_ratio = self.data_config.get('test_ratio', 0.15)

        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Storage for loaded data
        self._halos: Optional[List[HaloData]] = None

        logger.info(f"Initialized {self.source_name} with seed={self.seed}")

    @abstractmethod
    def load_raw(self) -> Any:
        """
        Load raw data from the source.

        Returns:
            Raw data in source-specific format.
        """
        pass

    @abstractmethod
    def _parse_subhalo(self, raw_record: Any) -> SubhaloData:
        """
        Parse a single raw record into a SubhaloData object.

        Args:
            raw_record: Source-specific raw data record

        Returns:
            SubhaloData object with normalized fields
        """
        pass

    @abstractmethod
    def _group_into_halos(self, subhalos: List[SubhaloData], raw_data: Any) -> List[HaloData]:
        """
        Group subhalos into their parent halos.

        Args:
            subhalos: List of parsed SubhaloData objects
            raw_data: Original raw data (may contain grouping info)

        Returns:
            List of HaloData objects
        """
        pass

    def load(self) -> List[HaloData]:
        """
        Main entry point: load and process all data.

        Returns:
            List of HaloData objects
        """
        if self._halos is not None:
            logger.info(f"Returning cached data ({len(self._halos)} halos)")
            return self._halos

        # Strict runs accept only a positive, explicit source declaration.
        # Source adapters normalize fields and unit conversions before this
        # boundary; Task 5 supplies the astronomy-specific adapter.
        if self.strict_research:
            if self.declared_catalog_contract is None:
                raise ValueError(
                    "strict_research requires an explicit corrected_research "
                    "catalog_contract before parsing source records"
                )
            from data.catalog import CatalogContract
            CatalogContract.from_mapping(self.declared_catalog_contract).validate(
                strict_research=True
            )
            if not self.supports_strict_research:
                raise ValueError(
                    f"{self.source_name} is not a strict normalized catalog "
                    "adapter; use a source adapter that validates fields before parsing"
                )

        logger.info(f"Loading data from {self.source_name}...")

        # Load raw data
        raw_data = self.load_raw()
        logger.info(f"Raw data loaded from {self.source_name}")

        # Parse individual subhalos
        subhalos = self._parse_all_subhalos(raw_data)
        logger.info(f"Parsed {len(subhalos)} subhalos")

        # Group into halos
        self._halos = self._group_into_halos(subhalos, raw_data)
        logger.info(f"Grouped into {len(self._halos)} halos")

        contract = self.get_catalog_contract()
        if contract is not None:
            from data.catalog import attach_catalog_contract
            attach_catalog_contract(
                self._halos, contract, strict_research=self.strict_research
            )

        # Validate
        self._validate_data()

        return self._halos

    def get_catalog_contract(self):
        """Return a source contract; subclasses provide legacy defaults."""
        return self.declared_catalog_contract

    def _parse_all_subhalos(self, raw_data: Any) -> List[SubhaloData]:
        """
        Parse all subhalos from raw data.
        Override in subclass if needed.

        Args:
            raw_data: Raw data from source

        Returns:
            List of SubhaloData objects
        """
        # Default implementation assumes iterable raw_data
        subhalos = []
        for record in raw_data:
            try:
                subhalo = self._parse_subhalo(record)
                subhalos.append(subhalo)
            except Exception as e:
                logger.warning(f"Failed to parse subhalo: {e}")
        return subhalos

    def _validate_data(self) -> None:
        """Validate loaded data for consistency."""
        if not self._halos:
            raise ValueError("No halos loaded")

        total_subhalos = sum(h.num_subhalos for h in self._halos)
        logger.info(f"Validation: {len(self._halos)} halos, {total_subhalos} total subhalos")

        # Check for empty halos
        empty_halos = sum(1 for h in self._halos if h.num_subhalos == 0)
        if empty_halos > 0:
            logger.warning(f"Found {empty_halos} empty halos (no subhalos)")

        # Check for NaN values
        nan_count = 0
        for halo in self._halos:
            features = halo.get_node_features()
            if np.any(np.isnan(features)):
                nan_count += 1
        if nan_count > 0:
            logger.warning(f"Found {nan_count} halos with NaN node features")

    def split_data(
        self,
        halos: Optional[List[HaloData]] = None
    ) -> Tuple[List[HaloData], List[HaloData], List[HaloData]]:
        """
        Split data into train/val/test sets.

        Args:
            halos: Optional list of halos (uses loaded data if None)

        Returns:
            Tuple of (train_halos, val_halos, test_halos)
        """
        if halos is None:
            if self._halos is None:
                self.load()
            halos = self._halos

        # Preserve historical MT19937 membership without mutating global RNG.
        n = len(halos)
        indices = np.random.RandomState(self.seed).permutation(n)

        # Calculate split points
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_halos = [halos[i] for i in train_indices]
        val_halos = [halos[i] for i in val_indices]
        test_halos = [halos[i] for i in test_indices]

        logger.info(f"Split: train={len(train_halos)}, val={len(val_halos)}, test={len(test_halos)}")

        return train_halos, val_halos, test_halos

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics on loaded data for normalization and logging.

        Returns:
            Dictionary of statistics
        """
        if self._halos is None:
            self.load()

        # Collect all node features
        all_features = []
        all_masses = []
        subhalo_counts = []

        for halo in self._halos:
            all_features.append(halo.get_node_features())
            all_masses.append(halo.log_halo_mass)
            subhalo_counts.append(halo.num_subhalos)

        all_features = np.concatenate(all_features, axis=0)
        all_masses = np.array(all_masses)
        subhalo_counts = np.array(subhalo_counts)

        stats = {
            'num_halos': len(self._halos),
            'total_subhalos': len(all_features),
            'feature_mean': all_features.mean(axis=0).tolist(),
            'feature_std': all_features.std(axis=0).tolist(),
            'feature_min': all_features.min(axis=0).tolist(),
            'feature_max': all_features.max(axis=0).tolist(),
            'log_mass_mean': float(all_masses.mean()),
            'log_mass_std': float(all_masses.std()),
            'log_mass_min': float(all_masses.min()),
            'log_mass_max': float(all_masses.max()),
            'subhalos_per_halo_mean': float(subhalo_counts.mean()),
            'subhalos_per_halo_std': float(subhalo_counts.std()),
            'subhalos_per_halo_min': int(subhalo_counts.min()),
            'subhalos_per_halo_max': int(subhalo_counts.max()),
        }

        return stats


class HaloDataset(Dataset):
    """
    PyTorch Dataset wrapper for HaloData objects.

    Used with PyTorch Geometric DataLoader after graph construction.
    """

    def __init__(self, halos: List[HaloData]):
        """
        Initialize dataset.

        Args:
            halos: List of HaloData objects
        """
        self.halos = halos

    def __len__(self) -> int:
        return len(self.halos)

    def __getitem__(self, idx: int) -> HaloData:
        return self.halos[idx]


def get_loader(config: Dict[str, Any]) -> BaseDataLoader:
    """
    Factory function to get the appropriate data loader based on config.

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        Instance of appropriate BaseDataLoader subclass
    """
    source = config.get('data', {}).get('source', 'synthetic')

    # Import here to avoid circular imports
    if source == 'synthetic':
        from data.loaders.synthetic_loader import SyntheticLoader
        return SyntheticLoader(config)
    elif source == 'tng':
        from data.loaders.tng_loader import TNGLoader
        return TNGLoader(config)
    elif source == 'camels':
        from data.loaders.camels_loader import CAMELSLoader
        return CAMELSLoader(config)
    elif source == 'camels_hf':
        from data.loaders.camels_loader import CAMELSHuggingFaceLoader
        return CAMELSHuggingFaceLoader(config)
    else:
        raise ValueError(f"Unknown data source: {source}")


def get_train_test_loaders(config: Dict[str, Any]) -> Tuple[BaseDataLoader, BaseDataLoader]:
    """
    Get separate loaders for training and testing (cross-simulation experiments).

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_source = config.get('training', {}).get('train_source', 'synthetic')
    test_source = config.get('training', {}).get('test_source', 'synthetic')

    # Create modified configs for each source
    train_config = config.copy()
    train_config['data'] = config['data'].copy()
    train_config['data']['source'] = train_source

    test_config = config.copy()
    test_config['data'] = config['data'].copy()
    test_config['data']['source'] = test_source

    train_loader = get_loader(train_config)
    test_loader = get_loader(test_config)

    if train_source != test_source:
        logger.info(f"Cross-simulation experiment: train on {train_source}, test on {test_source}")

    return train_loader, test_loader
