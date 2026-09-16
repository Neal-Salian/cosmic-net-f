"""Regression test: CAMELS HDF5 catalog URL must match the real Flatiron
layout (FIX Sep 2026).

The old pattern "Sims/{suite}/{sim}/fof_subhalo_tab_033.hdf5" 404s. Verified
live against https://users.flatironinstitute.org/~camels/ that catalogs are
the Arepo FOF/Subfind group files at
"FOF_Subfind/{suite}/{set}/{sim}/groups_{snapshot}.hdf5"
(e.g. .../FOF_Subfind/IllustrisTNG/LH/LH_0/groups_090.hdf5 — HEAD 200,
14,840,600 bytes; downloaded and parsed 15,712 subhalos locally).

Network-free: only URL construction is asserted here (no download).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

pytest.importorskip("h5py")

from data.loaders.camels_loader import CAMELSLoader


def _loader(tmp_path, **camels):
    cfg = {"data": {"camels": {"cache_dir": str(tmp_path), **camels}},
           "seed": 42}
    return CAMELSLoader(cfg)


def test_default_url_matches_verified_live_layout(tmp_path):
    loader = _loader(tmp_path)
    assert loader._get_hdf5_url() == (
        "https://users.flatironinstitute.org/~camels/"
        "FOF_Subfind/IllustrisTNG/LH/LH_0/groups_090.hdf5")


def test_set_derived_from_simulation_prefix(tmp_path):
    assert _loader(tmp_path, simulation="CV_5").sim_set == "CV"
    assert _loader(tmp_path, simulation="LH_0").sim_set == "LH"


def test_set_and_snapshot_overridable(tmp_path):
    loader = _loader(tmp_path, simulation="CV_5", set="CV", snapshot="088")
    assert loader._get_hdf5_url() == (
        "https://users.flatironinstitute.org/~camels/"
        "FOF_Subfind/IllustrisTNG/CV/CV_5/groups_088.hdf5")


def test_old_broken_pattern_is_gone():
    assert "fof_subhalo_tab_033" not in CAMELSLoader.CATALOG_SUBDIR
    assert CAMELSLoader.CATALOG_SUBDIR.startswith("FOF_Subfind/")
