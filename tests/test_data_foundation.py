import copy

import numpy as np
import pandas as pd
import pytest

from data.catalog import CatalogContract, attach_catalog_contract, source_file_record
from data.loaders.base_loader import HaloData
from data.loaders.synthetic_loader import SyntheticLoader
from data.loaders.tng_loader import TNGLoader
from data.loaders.camels_loader import CAMELSLoader
from data.splits import grouped_split, validate_split_manifest
from data.provenance import sha256_file


def halo(cluster_id, **metadata):
    return HaloData(cluster_id=cluster_id, subhalos=[], halo_mass=1e12,
                    metadata=metadata)


def corrected_contract(**overrides):
    values = {
        "research_label": "corrected_research",
        "source": "illustris_tng",
        "suite": "TNG100-1",
        "simulation": "TNG100-1",
        "snapshot": 99,
        "volume_or_ic_group": "TNG100",
        "target_field": "Group_M_Crit200",
        "target_definition": "log10(M200c/M_sun)",
        "mass_units": "M_sun",
        "position_units": "Mpc",
        "radius_units": "Mpc",
        "velocity_units": "km/s",
        "coordinate_frame": "physical",
        "velocity_convention": "peculiar_physical",
        "hubble_param": 0.6774,
        "scale_factor": 1.0,
        "radius_source_field": "SubhaloHalfmassRadType[:,4]",
        "radius_semantic": "stellar_half_mass_radius_type4",
        "synthetic": False,
        "fallback": False,
        "field_mapping": {"halo_mass": "Group_M_Crit200"},
        "conversion_record": {"position": "ckpc/h -> physical Mpc"},
        "source_files": [{"path": "catalog.hdf5", "size_bytes": 12,
                          "sha256": "a" * 64}],
    }
    values.update(overrides)
    return CatalogContract.from_mapping(values)


def test_strict_catalog_contract_requires_physical_semantics_and_hashes():
    corrected_contract().validate(strict_research=True)
    with pytest.raises(ValueError, match="Group_M_Crit200"):
        corrected_contract(target_field="GroupMass").validate(strict_research=True)
    with pytest.raises(ValueError, match="stellar_half_mass_radius_type4"):
        corrected_contract(radius_semantic="total_subhalo_half_mass_radius").validate(
            strict_research=True)
    with pytest.raises(ValueError, match="coordinate_frame"):
        corrected_contract(coordinate_frame=None).validate(strict_research=True)
    with pytest.raises(ValueError, match="source_files"):
        corrected_contract(source_files=[]).validate(strict_research=True)


def test_legacy_total_radius_keeps_unknown_target_semantics():
    contract = corrected_contract(
        research_label="legacy_total_radius",
        target_field="unknown",
        target_definition="unknown",
        radius_source_field="SubhaloHalfmassRad",
        radius_semantic="total_subhalo_half_mass_radius",
    )
    contract.validate(strict_research=False)
    with pytest.raises(ValueError, match="corrected_research"):
        contract.validate(strict_research=True)


def test_catalog_contract_is_copied_into_each_halo():
    halos = [halo("a"), halo("b")]
    attached = attach_catalog_contract(halos, corrected_contract())
    assert attached is halos
    assert all(item.metadata["catalog_contract"]["target_field"] ==
               "Group_M_Crit200" for item in halos)
    halos[0].metadata["catalog_contract"]["source"] = "changed"
    assert halos[1].metadata["catalog_contract"]["source"] == "illustris_tng"


def test_source_file_record_hashes_exact_bytes(tmp_path):
    path = tmp_path / "catalog.bin"
    path.write_bytes(b"cosmic-net")
    assert source_file_record(path) == {
        "path": str(path),
        "size_bytes": 10,
        "sha256": "46f36118da308451a14b1104ade801f9fde5567b68debf93677bbf76fdf517f9",
    }
    assert sha256_file(path) == source_file_record(path)


def test_grouped_split_keeps_lineage_together_and_manifest_detects_tampering():
    halos = [
        halo("a0", ic_group="ic-a"),
        halo("a1", ic_group="ic-a", augmentation_parent_id="a0"),
        halo("b0", ic_group="ic-b"),
        halo("c0", ic_group="ic-c"),
        halo("d0", ic_group="ic-d"),
        halo("e0", ic_group="ic-e"),
    ]
    train, val, test, manifest = grouped_split(
        halos, group_key="ic_group", ratios=(0.6, 0.2, 0.2), seed=7)
    locations = {
        item.cluster_id: name
        for name, split in (("train", train), ("val", val), ("test", test))
        for item in split
    }
    assert locations["a0"] == locations["a1"]
    validate_split_manifest(manifest, halos)

    tampered = copy.deepcopy(manifest)
    tampered["splits"]["test"]["cluster_ids"].append("a0")
    with pytest.raises(ValueError, match="hash|disjoint|reuse"):
        validate_split_manifest(tampered, halos)


def test_grouped_split_requires_group_key_and_preserves_global_rng():
    np.random.seed(99)
    expected = np.random.random(4)
    np.random.seed(99)
    with pytest.raises(ValueError, match="ic_group"):
        grouped_split([halo("missing")], group_key="ic_group", seed=3)
    assert np.array_equal(np.random.random(4), expected)


def test_legacy_split_uses_local_mt19937_without_changing_membership(tmp_path):
    cfg = {"seed": 42, "data": {"train_ratio": .5, "val_ratio": .25,
           "test_ratio": .25, "synthetic": {
               "features_path": str(tmp_path / "features.pt"),
               "csv_path": str(tmp_path / "data.csv")}}}
    loader = SyntheticLoader(cfg)
    halos = [halo(str(index)) for index in range(8)]
    expected_order = np.random.RandomState(42).permutation(8).tolist()
    np.random.seed(1234)
    expected_next = np.random.random(3)
    np.random.seed(1234)

    train, val, test = loader.split_data(halos)

    assert [int(item.cluster_id) for item in train + val + test] == expected_order
    assert np.array_equal(np.random.random(3), expected_next)


def _write_synthetic_csv(path):
    rows = []
    for index in range(3):
        rows.append({
            "subhalo_id": index, "halo_id": 0, "cluster_id": "synthetic-0",
            "x": index, "y": 0, "z": 0, "vx": 0, "vy": index, "vz": 0,
            "stellar_mass": 1e10, "velocity_dispersion": 100,
            "half_mass_radius": .01, "metallicity": .02,
            "halo_mass": 1e13, "redshift": 0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loader_attaches_synthetic_contract_and_strict_mode_rejects_it(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)
    cfg = {"data": {"source": "synthetic", "synthetic": {
        "features_path": str(tmp_path / "missing.pt"), "csv_path": str(csv_path)}}}
    halo_data = SyntheticLoader(cfg).load()
    contract = halo_data[0].metadata["catalog_contract"]
    assert contract["research_label"] == "synthetic"
    assert contract["synthetic"] is True
    assert contract["hubble_param"] is None

    cfg["data"]["strict_research"] = True
    with pytest.raises(ValueError, match="synthetic|corrected_research"):
        SyntheticLoader(cfg).load()

    cfg["data"]["catalog_contract"] = corrected_contract().to_dict()
    with pytest.raises(ValueError, match="adapter|strict"):
        SyntheticLoader(cfg).load()


def test_committed_tng_csv_is_labeled_legacy_with_unknown_target():
    cfg = {"data": {"source": "tng", "tng": {
        "clustered_file": "data/raw/tng100_clustered.csv",
        "min_subhalos_per_halo": 3}}}
    contract = TNGLoader(cfg).load()[0].metadata["catalog_contract"]
    assert contract["research_label"] == "legacy_total_radius"
    assert contract["target_field"] == "unknown"
    assert contract["target_definition"] == "unknown"
    assert contract["radius_semantic"] == "total_subhalo_half_mass_radius"

    cfg["data"]["strict_research"] = True
    with pytest.raises(ValueError, match="corrected_research"):
        TNGLoader(cfg).load()


def test_camels_synthetic_generation_uses_local_rng(tmp_path):
    cfg = {"seed": 8, "data": {"camels": {"cache_dir": str(tmp_path)}}}
    loader = CAMELSLoader(cfg)
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    loader._generate_synthetic_camels(tmp_path / "fixture.hdf5")
    assert np.array_equal(np.random.random(3), expected)
