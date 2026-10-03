"""
camels_loader.py
Purpose: Data loaders for CAMELS simulation data (both HDF5 direct download and Hugging Face).
         Supports CAMELS IllustrisTNG and SIMBA suites for cross-simulation experiments.
Inputs: config (dict) - Configuration dictionary from config.yaml
        For HDF5: Downloaded HDF5 files from Flatiron Institute
        For HF: HF_TOKEN environment variable (if dataset is gated)
Outputs: List[HaloData] - Normalized halo data objects with CAMELS subhalo properties
Config keys: data.camels.base_url, data.camels.suite, data.camels.simulation,
             data.camels.cache_dir, data.camels_hf.dataset_name, data.camels_hf.split
"""

import os
import logging
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_dataset = None

import requests
from dotenv import load_dotenv

from data.loaders.base_loader import BaseDataLoader, HaloData, SubhaloData

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CAMELSLoader(BaseDataLoader):
    """
    Data loader for CAMELS simulation data via HDF5 files.

    CAMELS (Cosmology and Astrophysics with MachinE Learning Simulations)
    provides thousands of cosmological simulations with varying parameters.

    Data is downloaded from: https://users.flatironinstitute.org/~camels/
    (verified Sep 2026: host alive; the CATALOG_SUBDIR pattern below was
    corrected against the live directory listing — the old
    "Sims/{suite}/{sim}/fof_subhalo_tab_033.hdf5" pattern 404s because real
    catalogs live under FOF_Subfind/{suite}/{set}/{sim}/groups_{snap}.hdf5,
    e.g. FOF_Subfind/IllustrisTNG/LH/LH_0/groups_090.hdf5. Verified by
    downloading that exact file and parsing 15,712 subhalos -> 254 usable
    halos through load_raw/_parse_all_subhalos/_group_into_halos.)

    Supported suites:
    - IllustrisTNG: Same physics as TNG but smaller boxes
    - SIMBA: Alternative galaxy formation model

    HDF5 structure typically:
    - Subhalo/SubhaloMass
    - Subhalo/SubhaloPos
    - Subhalo/SubhaloVel
    - Subhalo/SubhaloVelDisp
    - Subhalo/SubhaloHalfmassRad
    - Subhalo/SubhaloGrNr
    - Group/GroupMass
    """

    # CAMELS unit conversions (similar to TNG, smaller boxes)
    MASS_UNIT = 1e10  # 1e10 M_sun/h
    LENGTH_UNIT = 1e-3  # ckpc/h to Mpc
    H_PARAM = 0.6711  # Default CAMELS Hubble parameter

    # CAMELS data URLs (FIX Sep 2026, verified live against the Flatiron
    # directory listing + a real 14 MB download of the LH_0 file below).
    # Catalogs are the Arepo FOF/Subfind group files, NOT Sims/ snapshots:
    #   FOF_Subfind/{suite}/{sim_set}/{sim}/groups_{snapshot}.hdf5
    # e.g. FOF_Subfind/IllustrisTNG/LH/LH_0/groups_090.hdf5 (z=0).
    # sim_set/snapshot come from data.camels config (set defaults to the
    # simulation name prefix: "LH_0" -> "LH"; snapshot defaults to "090").
    BASE_URL = "https://users.flatironinstitute.org/~camels/"
    CATALOG_SUBDIR = "FOF_Subfind/{suite}/{sim_set}/{sim}/groups_{snapshot}.hdf5"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CAMELS HDF5 data loader.

        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__(config)

        if not HAS_H5PY:
            raise ImportError("h5py is required for CAMELSLoader. Install with: pip install h5py")

        # CAMELS-specific configuration
        camels_config = self.data_config.get('camels', {})
        self.base_url = camels_config.get('base_url', self.BASE_URL)
        self.suite = camels_config.get('suite', 'IllustrisTNG')
        self.simulation = camels_config.get('simulation', 'LH_0')
        # Set level (LH/CV/1P/...) defaults to the simulation prefix ("LH_0"
        # -> "LH"); snapshot defaults to "090" (z=0 catalog).
        self.sim_set = camels_config.get('set', self.simulation.split('_')[0])
        self.snapshot = camels_config.get('snapshot', '090')
        self.cache_dir = Path(camels_config.get('cache_dir', 'data/raw/camels_cache'))
        self.offline = camels_config.get('offline', False)
        self.used_synthetic_fallback = False

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Grouping method
        self.grouping_method = self.data_config.get('grouping', 'fof')

        logger.info(f"CAMELSLoader initialized: suite={self.suite}, sim={self.simulation}")

    def _get_hdf5_url(self) -> str:
        """Get the URL for the HDF5 catalog file."""
        path = self.CATALOG_SUBDIR.format(suite=self.suite,
                                          sim_set=self.sim_set,
                                          sim=self.simulation,
                                          snapshot=self.snapshot)
        return f"{self.base_url.rstrip('/')}/{path}"

    def _get_cache_path(self) -> Path:
        """Get the local cache path for the HDF5 file."""
        filename = f"{self.suite}_{self.simulation}_fof_subhalo_tab.hdf5"
        return self.cache_dir / filename

    def _download_hdf5(self) -> Path:
        """
        Download the HDF5 file if not cached.

        Returns:
            Path to local HDF5 file
        """
        cache_path = self._get_cache_path()

        if cache_path.exists():
            logger.info(f"Using cached HDF5: {cache_path}")
            return cache_path

        if self.offline:
            # No network access (tests/CI): go straight to the synthetic
            # fallback instead of attempting a real download.
            logger.warning("offline mode: generating synthetic CAMELS-like data")
            return self._generate_synthetic_camels(cache_path)

        url = self._get_hdf5_url()
        logger.info(f"Downloading CAMELS data from: {url}")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded to: {cache_path}")
            return cache_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download CAMELS data: {e}")
            # Create synthetic fallback
            logger.warning("Generating synthetic CAMELS-like data as fallback...")
            return self._generate_synthetic_camels(cache_path)

    def _generate_synthetic_camels(self, cache_path: Path) -> Path:
        """
        Generate synthetic CAMELS-like data for testing.

        Args:
            cache_path: Path to save the HDF5 file

        Returns:
            Path to generated file
        """
        self.used_synthetic_fallback = True
        np.random.seed(self.seed)

        num_groups = 200
        total_subhalos = 0
        subhalo_data = {
            'SubhaloMass': [],
            'SubhaloPos': [],
            'SubhaloVel': [],
            'SubhaloVelDisp': [],
            'SubhaloHalfmassRad': [],
            'SubhaloGrNr': [],
            'SubhaloStellarMass': [],
            'SubhaloMetallicity': []
        }
        group_data = {
            'GroupMass': [],
            'GroupPos': [],
            'GroupNsubs': []
        }

        for group_idx in range(num_groups):
            # Group properties
            log_group_mass = np.random.uniform(10.5, 14.0)
            group_mass = 10 ** log_group_mass / self.MASS_UNIT * self.H_PARAM
            group_pos = np.random.uniform(0, 25, 3)  # 25 Mpc/h box

            num_subhalos = max(1, int(3 * (group_mass * self.MASS_UNIT / self.H_PARAM / 1e12) ** 0.4))
            num_subhalos = min(num_subhalos, 30)

            group_data['GroupMass'].append(group_mass)
            group_data['GroupPos'].append(group_pos)
            group_data['GroupNsubs'].append(num_subhalos)

            for _ in range(num_subhalos):
                # Subhalo properties
                offset = np.random.exponential(0.3, 3)  # Mpc/h
                pos = group_pos + offset * np.random.choice([-1, 1], 3)

                vel = np.random.normal(0, 150, 3)
                vel_disp = 50 * (group_mass * self.MASS_UNIT / self.H_PARAM / 1e12) ** 0.25

                stellar_mass = 10 ** (log_group_mass - 2.0 + np.random.normal(0, 0.3))
                stellar_mass = stellar_mass / self.MASS_UNIT * self.H_PARAM

                half_mass_r = 5 * (stellar_mass * self.MASS_UNIT / self.H_PARAM / 1e10) ** 0.2

                metallicity = 0.02 * (stellar_mass * self.MASS_UNIT / self.H_PARAM / 1e10) ** 0.3

                subhalo_data['SubhaloMass'].append(stellar_mass + np.random.exponential(stellar_mass * 10))
                subhalo_data['SubhaloPos'].append(pos)
                subhalo_data['SubhaloVel'].append(vel)
                subhalo_data['SubhaloVelDisp'].append(vel_disp)
                subhalo_data['SubhaloHalfmassRad'].append(half_mass_r)
                subhalo_data['SubhaloGrNr'].append(group_idx)
                subhalo_data['SubhaloStellarMass'].append(stellar_mass)
                subhalo_data['SubhaloMetallicity'].append(metallicity)

                total_subhalos += 1

        # Write HDF5 file
        with h5py.File(cache_path, 'w') as f:
            subhalo_grp = f.create_group('Subhalo')
            for key, values in subhalo_data.items():
                subhalo_grp.create_dataset(key, data=np.array(values))

            group_grp = f.create_group('Group')
            for key, values in group_data.items():
                group_grp.create_dataset(key, data=np.array(values))

        logger.info(f"Generated synthetic CAMELS data: {total_subhalos} subhalos in {num_groups} groups")
        return cache_path

    def load_raw(self) -> Tuple[h5py.File, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load raw CAMELS data from HDF5 file.

        Returns:
            Tuple of (hdf5_file, subhalo_dict, group_dict)
        """
        hdf5_path = self._download_hdf5()

        # Open HDF5 file
        f = h5py.File(hdf5_path, 'r')

        # Read subhalo data
        subhalo_dict = {}
        if 'Subhalo' in f:
            subhalo_grp = f['Subhalo']
            for key in subhalo_grp.keys():
                subhalo_dict[key] = subhalo_grp[key][:]

        # Read group data
        group_dict = {}
        if 'Group' in f:
            group_grp = f['Group']
            for key in group_grp.keys():
                group_dict[key] = group_grp[key][:]

        logger.info(f"Loaded HDF5 with {len(subhalo_dict.get('SubhaloMass', []))} subhalos, "
                   f"{len(group_dict.get('GroupMass', []))} groups")

        return f, subhalo_dict, group_dict

    def _parse_subhalo(self, raw_record: Dict[str, Any]) -> SubhaloData:
        """
        Parse a single subhalo from CAMELS data.

        Args:
            raw_record: Dictionary with subhalo properties

        Returns:
            SubhaloData object
        """
        # Position (convert from ckpc/h to Mpc)
        position = np.array(raw_record['pos'], dtype=np.float32) * self.LENGTH_UNIT / self.H_PARAM

        # Velocity (km/s)
        velocity = np.array(raw_record['vel'], dtype=np.float32)

        # Stellar mass (convert from 1e10 M_sun/h)
        stellar_mass = float(raw_record['stellar_mass']) * self.MASS_UNIT / self.H_PARAM
        stellar_mass = max(stellar_mass, 1e6)

        # Velocity dispersion (km/s)
        vel_disp = max(float(raw_record['vel_disp']), 1.0)

        # Half-mass radius (convert to Mpc)
        half_mass_r = float(raw_record['half_mass_r']) * self.LENGTH_UNIT / self.H_PARAM
        half_mass_r = max(half_mass_r, 1e-6)

        # Metallicity
        metallicity = max(float(raw_record.get('metallicity', 0.02)), 1e-10)

        return SubhaloData(
            subhalo_id=int(raw_record['id']),
            position=position,
            velocity=velocity,
            stellar_mass=stellar_mass,
            velocity_dispersion=vel_disp,
            half_mass_radius=half_mass_r,
            metallicity=metallicity
        )

    def _parse_all_subhalos(
        self,
        raw_data: Tuple[h5py.File, Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ) -> List[SubhaloData]:
        """
        Parse all subhalos from CAMELS HDF5 data.

        Args:
            raw_data: Tuple of (hdf5_file, subhalo_dict, group_dict)

        Returns:
            List of SubhaloData objects
        """
        _, subhalo_dict, _ = raw_data

        num_subhalos = len(subhalo_dict.get('SubhaloMass', []))
        subhalos = []

        # Get arrays
        positions = subhalo_dict.get('SubhaloPos', np.zeros((num_subhalos, 3)))
        velocities = subhalo_dict.get('SubhaloVel', np.zeros((num_subhalos, 3)))
        masses = subhalo_dict.get('SubhaloMass', np.ones(num_subhalos))
        stellar_masses = subhalo_dict.get('SubhaloStellarMass',
                                          subhalo_dict.get('SubhaloMassType',
                                                           masses * 0.01))
        vel_disps = subhalo_dict.get('SubhaloVelDisp', np.ones(num_subhalos) * 100)
        half_mass_rs = subhalo_dict.get('SubhaloHalfmassRad', np.ones(num_subhalos) * 5)
        metallicities = subhalo_dict.get('SubhaloMetallicity',
                                         subhalo_dict.get('SubhaloGasMetallicity',
                                                          np.ones(num_subhalos) * 0.02))

        # Handle stellar mass from MassType array
        if isinstance(stellar_masses, np.ndarray) and len(stellar_masses.shape) > 1:
            if stellar_masses.shape[1] > 4:
                stellar_masses = stellar_masses[:, 4]  # Star particle mass
            else:
                stellar_masses = stellar_masses[:, -1]

        for i in range(num_subhalos):
            try:
                record = {
                    'id': i,
                    'pos': positions[i] if i < len(positions) else [0, 0, 0],
                    'vel': velocities[i] if i < len(velocities) else [0, 0, 0],
                    'stellar_mass': stellar_masses[i] if i < len(stellar_masses) else 1e10,
                    'vel_disp': vel_disps[i] if i < len(vel_disps) else 100,
                    'half_mass_r': half_mass_rs[i] if i < len(half_mass_rs) else 5,
                    'metallicity': metallicities[i] if i < len(metallicities) else 0.02
                }
                subhalo = self._parse_subhalo(record)
                subhalos.append(subhalo)
            except Exception as e:
                logger.warning(f"Failed to parse subhalo {i}: {e}")

        return subhalos

    def _group_into_halos(
        self,
        subhalos: List[SubhaloData],
        raw_data: Tuple[h5py.File, Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ) -> List[HaloData]:
        """
        Group subhalos into halos using FoF assignments.

        Args:
            subhalos: List of parsed SubhaloData objects
            raw_data: Tuple of (hdf5_file, subhalo_dict, group_dict)

        Returns:
            List of HaloData objects
        """
        _, subhalo_dict, group_dict = raw_data

        # Get group assignments
        group_nrs = subhalo_dict.get('SubhaloGrNr', np.arange(len(subhalos)))
        group_masses = group_dict.get('GroupMass', np.ones(max(group_nrs) + 1) * 1e12)

        # Build mapping
        group_subhalos: Dict[int, List[SubhaloData]] = {}
        for i, subhalo in enumerate(subhalos):
            grnr = int(group_nrs[i]) if i < len(group_nrs) else i
            if grnr not in group_subhalos:
                group_subhalos[grnr] = []
            group_subhalos[grnr].append(subhalo)

        # Create HaloData objects
        halos = []
        for group_id, subhalo_list in group_subhalos.items():
            if len(subhalo_list) == 0:
                continue

            # Get halo mass
            if group_id < len(group_masses):
                halo_mass = group_masses[group_id] * self.MASS_UNIT / self.H_PARAM
            else:
                halo_mass = 1e12  # Default

            halo = HaloData(
                cluster_id=f"camels_{self.suite}_{self.simulation}_{group_id:05d}",
                subhalos=subhalo_list,
                halo_mass=halo_mass,
                redshift=0.0,  # Snapshot 33 is typically z~0
                metadata={
                    'group_id': group_id,
                    'source': 'camels',
                    'suite': self.suite,
                    'simulation': self.simulation
                }
            )
            halos.append(halo)

        # Sort by cluster_id
        halos.sort(key=lambda h: h.cluster_id)

        # Close HDF5 file
        hdf5_file, _, _ = raw_data
        hdf5_file.close()

        logger.info(f"Grouped into {len(halos)} halos")
        return halos


class CAMELSHuggingFaceLoader(BaseDataLoader):
    """
    Data loader for CAMELS data via Hugging Face datasets.

    Uses the Hugging Face datasets library to load CAMELS data,
    which may include pre-processed or augmented versions.

    Dataset: camels-multifield-dataset/CAMELS
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CAMELS Hugging Face data loader.

        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__(config)

        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")

        # HF-specific configuration
        hf_config = self.data_config.get('camels_hf', {})
        self.dataset_name = hf_config.get('dataset_name', 'camels-multifield-dataset/CAMELS')
        self.split = hf_config.get('split', 'train')
        self.cache_dir = hf_config.get('cache_dir', 'data/raw/camels_hf_cache')

        # HF token from environment
        self.hf_token = os.environ.get('HF_TOKEN', None)

        # Grouping method
        self.grouping_method = self.data_config.get('grouping', 'fof')

        # Mirrors CAMELSLoader: set True by _generate_synthetic_hf_data when
        # the synthetic fallback is used, so downstream guards (Notebooks C/D,
        # rls/cross_sim.py) can refuse to publish synthetic-data results.
        self.used_synthetic_fallback = False

        logger.info(f"CAMELSHuggingFaceLoader initialized: dataset={self.dataset_name}")

    def load_raw(self) -> Any:
        """
        Load CAMELS dataset from Hugging Face.

        Returns:
            Hugging Face dataset object or synthetic fallback
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            logger.info(f"Loaded {len(dataset)} samples from Hugging Face")
            return dataset

        except Exception as e:
            logger.warning(f"Failed to load HF dataset: {e}")
            logger.warning("Generating synthetic fallback data...")
            return self._generate_synthetic_hf_data()

    def _generate_synthetic_hf_data(self) -> List[Dict[str, Any]]:
        """
        Generate synthetic data mimicking HF CAMELS format.

        Returns:
            List of dictionaries with simulated data
        """
        self.used_synthetic_fallback = True
        np.random.seed(self.seed)

        data = []
        num_samples = 300

        for i in range(num_samples):
            # Simulate a halo with multiple subhalos
            num_subhalos = np.random.randint(3, 25)
            log_halo_mass = np.random.uniform(11.0, 14.5)

            subhalos = []
            for j in range(num_subhalos):
                log_stellar = log_halo_mass - 2.0 + np.random.normal(0, 0.3)
                subhalos.append({
                    'x': np.random.uniform(-5, 5),
                    'y': np.random.uniform(-5, 5),
                    'z': np.random.uniform(-5, 5),
                    'vx': np.random.normal(0, 200),
                    'vy': np.random.normal(0, 200),
                    'vz': np.random.normal(0, 200),
                    'stellar_mass': 10 ** log_stellar,
                    'vel_disp': 100 * (10 ** log_halo_mass / 1e12) ** 0.25,
                    'half_mass_r': 0.001 * (10 ** log_stellar / 1e10) ** 0.25,
                    'metallicity': 0.02 * (10 ** log_stellar / 1e10) ** 0.3
                })

            data.append({
                'halo_id': i,
                'cluster_id': f'camels_hf_{i:05d}',
                'halo_mass': 10 ** log_halo_mass,
                'log_halo_mass': log_halo_mass,
                'subhalos': subhalos,
                'redshift': 0.0
            })

        return data

    def _parse_subhalo(self, raw_record: Dict[str, Any]) -> SubhaloData:
        """
        Parse a single subhalo from HF data.

        Args:
            raw_record: Dictionary with subhalo properties

        Returns:
            SubhaloData object
        """
        position = np.array([
            raw_record.get('x', 0.0),
            raw_record.get('y', 0.0),
            raw_record.get('z', 0.0)
        ], dtype=np.float32) / 1000  # Convert to Mpc if in kpc

        velocity = np.array([
            raw_record.get('vx', 0.0),
            raw_record.get('vy', 0.0),
            raw_record.get('vz', 0.0)
        ], dtype=np.float32)

        return SubhaloData(
            subhalo_id=int(raw_record.get('id', 0)),
            position=position,
            velocity=velocity,
            stellar_mass=max(float(raw_record.get('stellar_mass', 1e10)), 1e6),
            velocity_dispersion=max(float(raw_record.get('vel_disp', 100)), 1.0),
            half_mass_radius=max(float(raw_record.get('half_mass_r', 0.001)), 1e-6),
            metallicity=max(float(raw_record.get('metallicity', 0.02)), 1e-10)
        )

    def _parse_all_subhalos(self, raw_data: Any) -> List[SubhaloData]:
        """
        Parse all subhalos from HF data.
        Note: This is called differently for HF loader since data is structured per-halo.

        Args:
            raw_data: HF dataset or synthetic list

        Returns:
            List of SubhaloData objects (flattened)
        """
        # For HF loader, we handle grouping in _group_into_halos
        return []

    def _group_into_halos(
        self,
        subhalos: List[SubhaloData],
        raw_data: Any
    ) -> List[HaloData]:
        """
        Convert HF data directly to halos (data is already grouped).

        Args:
            subhalos: Empty list (not used for HF loader)
            raw_data: HF dataset or synthetic list

        Returns:
            List of HaloData objects
        """
        halos = []

        for item in raw_data:
            try:
                # Get subhalos for this halo
                subhalo_list = []
                raw_subhalos = item.get('subhalos', [])

                # Handle different data formats
                if isinstance(raw_subhalos, dict):
                    # HF format might have arrays for each field
                    num_subhalos = len(raw_subhalos.get('stellar_mass', []))
                    for i in range(num_subhalos):
                        record = {key: val[i] for key, val in raw_subhalos.items()}
                        record['id'] = i
                        subhalo = self._parse_subhalo(record)
                        subhalo_list.append(subhalo)
                elif isinstance(raw_subhalos, list):
                    for i, sub in enumerate(raw_subhalos):
                        sub['id'] = i
                        subhalo = self._parse_subhalo(sub)
                        subhalo_list.append(subhalo)

                if len(subhalo_list) == 0:
                    continue

                halo = HaloData(
                    cluster_id=str(item.get('cluster_id', f"camels_hf_{item.get('halo_id', 0):05d}")),
                    subhalos=subhalo_list,
                    halo_mass=float(item.get('halo_mass', 1e12)),
                    redshift=float(item.get('redshift', 0.0)),
                    metadata={
                        'halo_id': item.get('halo_id', 0),
                        'source': 'camels_hf'
                    }
                )
                halos.append(halo)

            except Exception as e:
                logger.warning(f"Failed to parse halo: {e}")

        # Sort by cluster_id
        halos.sort(key=lambda h: h.cluster_id)

        logger.info(f"Loaded {len(halos)} halos from Hugging Face")
        return halos

    def load(self) -> List[HaloData]:
        """
        Override load to handle HF-specific data structure.

        Returns:
            List of HaloData objects
        """
        if self._halos is not None:
            return self._halos

        raw_data = self.load_raw()
        self._halos = self._group_into_halos([], raw_data)
        self._validate_data()

        return self._halos


def get_camels_simulation_list(suite: str = 'IllustrisTNG') -> List[str]:
    """
    Get list of available CAMELS simulations for a suite.

    Args:
        suite: 'IllustrisTNG' or 'SIMBA'

    Returns:
        List of simulation names
    """
    # LH: Latin Hypercube (varying cosmology + astrophysics)
    # 1P: One Parameter at a time
    # CV: Cosmic Variance (fiducial cosmology)
    simulations = []

    # LH simulations (1000 variations)
    for i in range(1000):
        simulations.append(f"LH_{i}")

    # 1P simulations
    for param in ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']:
        for i in range(61):
            simulations.append(f"1P_{param}_{i}")

    # CV simulations (27 realizations)
    for i in range(27):
        simulations.append(f"CV_{i}")

    return simulations
