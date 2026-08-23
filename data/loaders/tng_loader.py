"""
tng_loader.py
Purpose: Data loader for IllustrisTNG-100 simulation data via REST API or local CSV.
         Fetches subhalo data from TNG-100 snapshot 99 (z=0) with ALL subhalos per halo
         (central + satellites) grouped by FoF halo ID for multi-node graphs.
Inputs: config (dict) - Configuration dictionary from config.yaml
        TNG_API_KEY - Environment variable for API authentication
        OR local CSV files: data.tng.raw_file, data.tng.clustered_file
Outputs: List[HaloData] - Normalized halo data objects with TNG subhalo properties
Config keys: data.tng.base_url, data.tng.snapshot, data.tng.n_halos,
             data.tng.min_subhalos_per_halo, data.tng.cache_dir,
             data.tng.raw_file, data.tng.clustered_file, data.grouping, seed
"""

import os
import json
import logging
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from data.loaders.base_loader import BaseDataLoader, HaloData, SubhaloData

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TNGLoader(BaseDataLoader):
    """
    Data loader for IllustrisTNG-100 simulation data.

    Supports two modes:
    1. Local CSV: Load pre-downloaded data from CSV files
    2. API: Fetch from TNG API (https://www.tng-project.org/api/)

    For proper graph construction, this loader fetches ALL subhalos per halo
    (central + satellites) grouped by FoF halo ID. Halos with fewer than
    min_subhalos_per_halo members are dropped.

    TNG-100 specific fields mapped to internal schema:
    - SubhaloMassType[:, 4] -> stellar_mass (Type 4 = stars)
    - SubhaloVelDisp -> velocity_dispersion
    - SubhaloHalfmassRadType[:, 4] -> half_mass_radius
    - SubhaloGasMetallicity -> metallicity
    - SubhaloPos -> position
    - SubhaloVel -> velocity
    - SubhaloGrNr -> halo group assignment (FoF)
    - Group_M_Crit200 -> halo mass (target)

    NOTE (schema discrepancy, verified Aug 2026 against raw_hdf5/groupcat_99_*.hdf5):
    the mapping above describes the API path. The clustered CSV's
    `half_mass_radius` column actually contains the TOTAL SubhaloHalfmassRad
    (ckpc/h), NOT the stellar type-4 radius (e.g. 682.5 total vs 88.3 stellar
    for the matched reference subhalo). Consumers that assume stellar radii
    (e.g. model/physics_loss.py's R_half scale) get the total radius when fed
    from the CSV path.
    """

    # TNG unit conversions
    MASS_UNIT = 1e10  # TNG mass unit to M_sun (1e10 M_sun / h, assume h=0.6774)
    LENGTH_UNIT = 1e-3  # TNG length unit to Mpc (ckpc/h to Mpc)
    H_PARAM = 0.6774  # Hubble parameter for TNG100

    # Required subhalo fields from API
    SUBHALO_FIELDS = [
        'SubhaloMassType',
        'SubhaloVelDisp',
        'SubhaloHalfmassRadType',
        'SubhaloGasMetallicity',
        'SubhaloStarMetallicity',
        'SubhaloPos',
        'SubhaloVel',
        'SubhaloGrNr',
        'SubhaloFlag'
    ]

    # Required group/halo fields
    GROUP_FIELDS = [
        'Group_M_Crit200',
        'GroupPos',
        'GroupVel',
        'GroupNsubs'
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TNG data loader.

        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__(config)

        # TNG-specific configuration
        tng_config = self.data_config.get('tng', {})
        self.base_url = tng_config.get('base_url', 'https://www.tng-project.org/api/TNG100-1/')
        self.snapshot = tng_config.get('snapshot', 99)
        self.n_halos = tng_config.get('n_halos', 500)
        self.min_subhalos = tng_config.get('min_subhalos_per_halo', 3)
        self.cache_dir = Path(tng_config.get('cache_dir', 'data/raw/tng_cache'))

        # Local CSV file paths (new)
        self.raw_file = tng_config.get('raw_file', None)
        self.clustered_file = tng_config.get('clustered_file', None)

        # API key from environment
        self.api_key = os.environ.get('TNG_API_KEY', '')
        if not self.api_key:
            logger.warning("TNG_API_KEY not set. Will try local files first.")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Grouping method
        self.grouping_method = self.data_config.get('grouping', 'fof')

        logger.info(f"TNGLoader initialized: snapshot={self.snapshot}, "
                   f"n_halos={self.n_halos}, min_subhalos={self.min_subhalos}")

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            'api-key': self.api_key,
            'Content-Type': 'application/json'
        }

    def _cache_key(self, request_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key for a request."""
        param_str = json.dumps(params, sort_keys=True)
        key = f"{request_type}_{param_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if available."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                logger.debug(f"Loading from cache: {cache_key}")
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Saved to cache: {cache_key}")

    def _api_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the TNG API.

        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        params = params or {}

        # Check cache first
        cache_key = self._cache_key(url, params)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Make API request
        try:
            response = requests.get(url, headers=self._get_headers(), params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Cache successful response
            self._save_to_cache(cache_key, data)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"TNG API request failed: {e}")
            raise

    def load_raw(self) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
        """
        Load raw TNG data from local CSV or API.

        Priority:
        1. Clustered CSV (already processed multi-subhalo data)
        2. Raw CSV (needs clustering)
        3. API fetch (slowest, requires API key)

        Returns:
            Tuple of (subhalos_df, groups_dict)
        """
        # Try clustered file first
        if self.clustered_file and Path(self.clustered_file).exists():
            logger.info(f"Loading clustered data from {self.clustered_file}")
            return self._load_from_clustered_csv(self.clustered_file)

        # Try raw file and cluster it
        if self.raw_file and Path(self.raw_file).exists():
            logger.info(f"Loading raw data from {self.raw_file}")
            df, groups = self._load_from_raw_csv(self.raw_file)

            # Save clustered version for next time
            if self.clustered_file:
                self._save_clustered_csv(df, self.clustered_file)

            return df, groups

        # Fall back to API
        logger.info("No local files found, fetching from TNG API...")
        return self._fetch_from_api()

    def _load_from_clustered_csv(self, filepath: str) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
        """
        Load pre-clustered data from CSV.
        """
        H = 0.6774
        CKPC_TO_MPC = 1e-3 / H

        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} subhalos from {filepath}")

        # Fix column name variant
        if 'vel_dispersion' in df.columns and 'velocity_dispersion' not in df.columns:
            df = df.rename(columns={'vel_dispersion': 'velocity_dispersion'})

        # Fix positions: ckpc/h → Mpc
        if df['pos_x'].abs().max() > 100:
            df['pos_x'] = df['pos_x'] * CKPC_TO_MPC
            df['pos_y'] = df['pos_y'] * CKPC_TO_MPC
            df['pos_z'] = df['pos_z'] * CKPC_TO_MPC
            logger.info("Converted positions from ckpc/h to Mpc")

        # Fix stellar_mass: log10 → linear
        if df['stellar_mass'].between(6, 16).all():
            df['stellar_mass'] = 10 ** df['stellar_mass']
            logger.info("Converted stellar_mass from log10 to linear")

        # Fix half_mass_radius: ckpc/h → Mpc
        if df['half_mass_radius'].abs().max() > 1.0:
            df['half_mass_radius'] = df['half_mass_radius'] * CKPC_TO_MPC
            logger.info("Converted half_mass_radius from ckpc/h to Mpc")

        # Add linear halo_mass column if missing
        if 'halo_mass' not in df.columns:
            df['halo_mass'] = 10 ** df['halo_mass_log']
            logger.info("Created halo_mass from halo_mass_log")
        elif df['halo_mass'].between(10, 20).all():
            df['halo_mass'] = 10 ** df['halo_mass']
            logger.info("Fixed halo_mass from log to linear scale")

        # Build groups dict
        groups = {}
        for group_id, group_df in df.groupby('group_id'):
            groups[int(group_id)] = {
                'Group_M_Crit200': float(group_df['halo_mass'].iloc[0]),
                'GroupNsubs': len(group_df)
            }

        return df, groups

    def _load_from_raw_csv(self, filepath: str) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
        """
        Load raw CSV (single subhalo per halo) and prepare for API expansion.

        The raw CSV has primary_flag=1 only (central subhalos).
        This method identifies the group_ids and will fetch all satellites.

        Args:
            filepath: Path to raw CSV file

        Returns:
            Tuple of (subhalos_df, groups_dict)
        """
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} central subhalos from {filepath}")

        # Get unique group IDs and their halo masses
        # Need to fetch ALL subhalos per group from API
        group_ids = df['group_id'].unique().tolist()

        # Limit to n_halos most massive
        if 'halo_mass_log' in df.columns:
            mass_by_group = df.groupby('group_id')['halo_mass_log'].first()
            top_groups = mass_by_group.nlargest(self.n_halos).index.tolist()
            group_ids = top_groups

        logger.info(f"Will fetch all subhalos for {len(group_ids)} halos...")

        # Fetch all subhalos for each group
        all_subhalos = []
        groups = {}

        for i, group_id in enumerate(group_ids):
            if i % 50 == 0:
                logger.info(f"Processing group {i+1}/{len(group_ids)}...")

            try:
                # Fetch group info
                group_data = self._api_get(f"snapshots/{self.snapshot}/groups/{group_id}/")
                groups[group_id] = group_data

                # Fetch all subhalos in this group (no primary_flag filter!)
                subhalos = self._fetch_group_subhalos(group_id)

                for sub in subhalos:
                    sub['group_id'] = group_id
                    sub['halo_mass'] = group_data.get('Group_M_Crit200', 1e12) * self.MASS_UNIT / self.H_PARAM
                all_subhalos.extend(subhalos)

            except Exception as e:
                logger.warning(f"Failed to fetch group {group_id}: {e}")

        # Create DataFrame
        result_df = self._subhalos_to_dataframe(all_subhalos)
        logger.info(f"Fetched {len(result_df)} total subhalos in {len(groups)} groups")

        return result_df, groups

    def _fetch_group_subhalos(self, group_id: int) -> List[Dict[str, Any]]:
        """
        Fetch ALL subhalos belonging to a specific FoF group.

        Args:
            group_id: FoF group ID

        Returns:
            List of subhalo dictionaries
        """
        cache_key = f"group_subhalos_{self.snapshot}_{group_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Query subhalos with this group_id
        endpoint = f"snapshots/{self.snapshot}/subhalos/"
        params = {'grnr': group_id}  # Filter by group number

        all_subhalos = []
        url = endpoint

        while url:
            try:
                response = self._api_get(url, params)
                subhalos = response.get('results', [])
                all_subhalos.extend(subhalos)

                url = response.get('next')
                if url:
                    url = url.replace(self.base_url, '')
                params = {}  # Pagination in URL

            except Exception as e:
                logger.warning(f"Failed to fetch subhalos for group {group_id}: {e}")
                break

        self._save_to_cache(cache_key, all_subhalos)
        return all_subhalos

    def _fetch_from_api(self) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
        """
        Fetch data directly from TNG API.

        Fetches the top n_halos most massive halos and ALL their subhalos.

        Returns:
            Tuple of (subhalos_df, groups_dict)
        """
        # Get most massive groups
        endpoint = f"snapshots/{self.snapshot}/groups/"
        params = {'limit': self.n_halos, 'order_by': '-Group_M_Crit200'}

        logger.info(f"Fetching top {self.n_halos} massive groups...")
        response = self._api_get(endpoint, params)
        top_groups = response.get('results', [])

        all_subhalos = []
        groups = {}

        for i, group in enumerate(top_groups):
            group_id = group.get('id', i)
            groups[group_id] = group

            if i % 50 == 0:
                logger.info(f"Processing group {i+1}/{len(top_groups)}...")

            # Fetch all subhalos for this group
            subhalos = self._fetch_group_subhalos(group_id)
            for sub in subhalos:
                sub['group_id'] = group_id
                sub['halo_mass'] = group.get('Group_M_Crit200', 1e12) * self.MASS_UNIT / self.H_PARAM
            all_subhalos.extend(subhalos)

        result_df = self._subhalos_to_dataframe(all_subhalos)
        logger.info(f"Fetched {len(result_df)} subhalos in {len(groups)} groups")

        return result_df, groups

    def _subhalos_to_dataframe(self, subhalos: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert list of subhalo dicts to DataFrame with standard columns.

        Args:
            subhalos: List of subhalo dictionaries from API

        Returns:
            DataFrame with standardized columns
        """
        records = []

        for sub in subhalos:
            # Extract position
            pos = sub.get('SubhaloPos', sub.get('pos', [0, 0, 0]))
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                pos_x, pos_y, pos_z = pos[0], pos[1], pos[2]
            else:
                pos_x, pos_y, pos_z = 0, 0, 0

            # Extract velocity
            vel = sub.get('SubhaloVel', sub.get('vel', [0, 0, 0]))
            if isinstance(vel, (list, np.ndarray)) and len(vel) >= 3:
                vel_x, vel_y, vel_z = vel[0], vel[1], vel[2]
            else:
                vel_x, vel_y, vel_z = 0, 0, 0

            # Extract stellar mass
            mass_type = sub.get('SubhaloMassType', sub.get('mass_type', [0] * 6))
            if isinstance(mass_type, (list, np.ndarray)) and len(mass_type) > 4:
                stellar_mass = mass_type[4] * self.MASS_UNIT / self.H_PARAM
            else:
                stellar_mass = sub.get('mass_stars', 1e10) * self.MASS_UNIT / self.H_PARAM

            # Velocity dispersion
            vel_disp = sub.get('SubhaloVelDisp', sub.get('vel_disp', 100.0))

            # Half mass radius
            hmr_type = sub.get('SubhaloHalfmassRadType', sub.get('halfmassrad_type', [0] * 6))
            if isinstance(hmr_type, (list, np.ndarray)) and len(hmr_type) > 4:
                half_mass_r = hmr_type[4] * self.LENGTH_UNIT / self.H_PARAM
            else:
                half_mass_r = sub.get('halfmassrad', 1.0) * self.LENGTH_UNIT / self.H_PARAM

            # Metallicity
            metallicity = sub.get('SubhaloStarMetallicity',
                                  sub.get('SubhaloGasMetallicity',
                                          sub.get('metallicity', 0.02)))

            records.append({
                'subhalo_id': sub.get('id', sub.get('subhalo_id', 0)),
                'group_id': sub.get('group_id', sub.get('SubhaloGrNr', sub.get('grnr', 0))),
                'stellar_mass': max(stellar_mass, 1e6),
                'velocity_dispersion': max(float(vel_disp), 1.0),
                'half_mass_radius': max(float(half_mass_r), 1e-6),
                'metallicity': max(float(metallicity), 1e-10),
                'pos_x': float(pos_x) * self.LENGTH_UNIT / self.H_PARAM,
                'pos_y': float(pos_y) * self.LENGTH_UNIT / self.H_PARAM,
                'pos_z': float(pos_z) * self.LENGTH_UNIT / self.H_PARAM,
                'vel_x': float(vel_x),
                'vel_y': float(vel_y),
                'vel_z': float(vel_z),
                'halo_mass': sub.get('halo_mass', 1e12),
                'halo_mass_log': np.log10(max(sub.get('halo_mass', 1e12), 1e6)),
                'log_stellar_mass': np.log10(max(stellar_mass, 1e6)),
                'log_vel_dispersion': np.log10(max(float(vel_disp), 1.0)),
                'log_half_mass_radius': np.log10(max(float(half_mass_r), 1e-6)),
                'log_metallicity': np.log10(max(float(metallicity), 1e-10))
            })

        return pd.DataFrame(records)

    def _save_clustered_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save clustered DataFrame to CSV."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved clustered data to {filepath}")

    def _parse_subhalo(self, raw_record: pd.Series) -> SubhaloData:
        """
        Parse a single row from DataFrame into SubhaloData.

        Args:
            raw_record: pandas Series representing one subhalo

        Returns:
            SubhaloData object
        """
        H = 0.6774
        CKPC_TO_MPC = 1e-3 / H

        position = np.array([
            raw_record.get('pos_x', 0.0),
            raw_record.get('pos_y', 0.0),
            raw_record.get('pos_z', 0.0)
        ], dtype=np.float32)

        # Convert positions from ckpc/h to Mpc if still in raw units
        if np.abs(position).max() > 100:
            position = position * CKPC_TO_MPC

        velocity = np.array([
            raw_record.get('vel_x', 0.0),
            raw_record.get('vel_y', 0.0),
            raw_record.get('vel_z', 0.0)
        ], dtype=np.float32)

        # Stellar mass — CSV stores log10 scale (8-16), need linear
        raw_sm = float(raw_record.get('stellar_mass', 10.0))
        if 6 <= raw_sm <= 16:
            stellar_mass = 10 ** raw_sm
        else:
            stellar_mass = max(raw_sm, 1e6)

        # Velocity dispersion — try both column name variants
        vel_disp = float(raw_record.get(
            'vel_dispersion',
            raw_record.get('velocity_dispersion', 100.0)
        ))

        # Half mass radius — convert ckpc/h to Mpc if needed
        raw_hmr = float(raw_record.get('half_mass_radius', 0.001))
        if raw_hmr > 1.0:
            half_mass_radius = raw_hmr * CKPC_TO_MPC
        else:
            half_mass_radius = max(raw_hmr, 1e-6)

        # Metallicity
        metallicity = float(raw_record.get('metallicity', 0.02))
        metallicity = max(metallicity, 1e-10)

        return SubhaloData(
            subhalo_id=int(raw_record.get('subhalo_id', 0)),
            position=position,
            velocity=velocity,
            stellar_mass=stellar_mass,
            velocity_dispersion=vel_disp,
            half_mass_radius=half_mass_radius,
            metallicity=metallicity
        )

    def _parse_all_subhalos(
        self,
        raw_data: Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]
    ) -> List[SubhaloData]:
        """
        Parse all subhalos from DataFrame.

        Args:
            raw_data: Tuple of (subhalos_df, groups_dict)

        Returns:
            List of SubhaloData objects
        """
        df, _ = raw_data

        subhalos = []
        for _, row in df.iterrows():
            try:
                subhalo = self._parse_subhalo(row)
                subhalos.append(subhalo)
            except Exception as e:
                logger.warning(f"Failed to parse subhalo: {e}")

        return subhalos

    def _group_into_halos(
        self,
        subhalos: List[SubhaloData],
        raw_data: Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]
    ) -> List[HaloData]:
        """
        Group subhalos into halos using FoF group assignments.

        Only includes halos with >= min_subhalos_per_halo members.

        Args:
            subhalos: List of parsed SubhaloData objects
            raw_data: Tuple of (subhalos_df, groups_dict)

        Returns:
            List of HaloData objects
        """
        df, groups = raw_data

        # Build mapping from group_id to subhalos
        group_subhalos: Dict[int, List[SubhaloData]] = {}

        for subhalo, (_, row) in zip(subhalos, df.iterrows()):
            grnr = int(row.get('group_id', -1))
            if grnr not in group_subhalos:
                group_subhalos[grnr] = []
            group_subhalos[grnr].append(subhalo)

        # Create HaloData objects (filter by min subhalos)
        halos = []
        dropped_count = 0

        for group_id, subhalo_list in group_subhalos.items():
            # Filter: minimum subhalos per halo
            if len(subhalo_list) < self.min_subhalos:
                dropped_count += 1
                continue

            # Get halo mass
            if group_id in groups:
                group = groups[group_id]
                halo_mass = group.get('Group_M_Crit200', group.get('mass', 1e12))

            else:
                # Get from first subhalo's halo_mass
                first_row = df[df['group_id'] == group_id].iloc[0]
                halo_mass_log = first_row.get('halo_mass_log', 12.0)
                halo_mass = first_row.get('halo_mass', 10 ** float(halo_mass_log))

            halo = HaloData(
                cluster_id=f"tng100_{group_id:07d}",
                subhalos=subhalo_list,
                halo_mass=halo_mass,
                redshift=0.0,
                metadata={
                    'group_id': group_id,
                    'source': 'tng100',
                    'snapshot': self.snapshot,
                    'num_subhalos': len(subhalo_list)
                }
            )
            halos.append(halo)

        # Sort by cluster_id for reproducibility
        halos.sort(key=lambda h: h.cluster_id)

        logger.info(f"Grouped into {len(halos)} halos "
                   f"(dropped {dropped_count} with < {self.min_subhalos} subhalos)")

        return halos

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob('*.pkl'):
                f.unlink()
            logger.info(f"Cleared cache: {self.cache_dir}")

    def get_simulation_info(self) -> Dict[str, Any]:
        """
        Get information about the TNG simulation.

        Returns:
            Dictionary with simulation metadata
        """
        try:
            info = self._api_get('')
            return info
        except Exception as e:
            logger.error(f"Failed to fetch simulation info: {e}")
            return {}


def download_tng_sample(
    config: Dict[str, Any],
    output_dir: str = 'data/raw/tng_cache',
    max_halos: int = 100
) -> str:
    """
    Download a sample of TNG data for testing.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save data
        max_halos: Maximum number of halos to download

    Returns:
        Path to cached data directory
    """
    # Override max_halos for sample
    sample_config = config.copy()
    sample_config['data'] = config.get('data', {}).copy()
    sample_config['data']['tng'] = config.get('data', {}).get('tng', {}).copy()
    sample_config['data']['tng']['max_halos'] = max_halos
    sample_config['data']['tng']['cache_dir'] = output_dir

    loader = TNGLoader(sample_config)
    halos = loader.load()

    logger.info(f"Downloaded {len(halos)} halos to {output_dir}")
    return output_dir
