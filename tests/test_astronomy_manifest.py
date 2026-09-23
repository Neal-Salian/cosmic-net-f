"""Offline manifest and HDF5 audit contract tests using synthetic fixtures only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from data.astronomy_manifest import load_astronomy_manifest
from data.astronomy_audit import audit_astronomy_manifest


def _catalog(path, *, role="development", ic="ic-a", source_kind="synthetic_fixture"):
    return {
        "id": f"cat-{role}-{ic}", "role": role, "suite": "IllustrisTNG",
        "simulation": "L25n256", "simulation_set": "LH", "initial_condition_id": ic,
        "snapshot": "099", "source": {"kind": source_kind, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes() if path.exists() else b"expected local file").hexdigest()},
        "epoch": {"redshift": 0.0, "scale_factor": 1.0, "h": 0.6774},
        "conventions": {"coordinates": "comoving", "velocities": "peculiar_km_per_s",
            "periodic_box": False},
        "fields": {
            "group_mass": {"path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "shape": [2]},
            "subhalo_mass_anchor": {"path": "Subhalo/SubhaloMass", "unit": "1e10_Msun_per_h", "shape": [3]},
            "position": {"path": "Subhalo/SubhaloPos", "unit": "ckpc_per_h", "shape": [3, 3]},
            "velocity": {"path": "Subhalo/SubhaloVel", "unit": "km_per_s", "shape": [3, 3]},
            "stellar_mass": {"path": "Subhalo/SubhaloMassType", "component": 4, "unit": "1e10_Msun_per_h", "shape": [3, 6]},
            "half_mass_radius": {"path": "Subhalo/SubhaloHalfmassRadType", "component": 4, "unit": "ckpc_per_h", "shape": [3, 6]},
            "group_index": {"path": "Subhalo/SubhaloGrNr", "unit": "index", "shape": [3]},
        },
        "target": {"name": "halo_mass", "path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h",
            "definition": "log10(M200c/M_sun)", "shape": [2]},
        "selection": {"predicate": "stellar_mass_type4_gt_zero"},
        "release": "synthetic fixture only",
    }


def _fixture(path):
    with h5py.File(path, "w") as f:
        f.attrs["h"] = 0.6774
        f.attrs["Time"] = 1.0
        f.create_dataset("Group/Group_M_Crit200", data=[10., 20.])
        f.create_dataset("Subhalo/SubhaloMass", data=[1., 2., 3.])
        f.create_dataset("Subhalo/SubhaloPos", data=np.zeros((3, 3)))
        f.create_dataset("Subhalo/SubhaloVel", data=np.zeros((3, 3)))
        f.create_dataset("Subhalo/SubhaloMassType", data=np.ones((3, 6)))
        f.create_dataset("Subhalo/SubhaloHalfmassRadType", data=np.ones((3, 6)))
        f.create_dataset("Subhalo/SubhaloGrNr", data=[0, 1, 1])


def _write_manifest(tmp_path, entries):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": entries}))
    return path


def test_load_manifest_and_audit_synthetic_local_hdf5(tmp_path):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    manifest_path = _write_manifest(tmp_path, [_catalog(file_path)])
    manifest = load_astronomy_manifest(manifest_path)
    report = audit_astronomy_manifest(manifest)
    assert report.ok
    assert report.to_dict()["verification_level"] == "content_verified"
    report.require_content_verified()
    assert report.catalogs[0]["source_kind"] == "synthetic_fixture"
    assert report.catalogs[0]["datasets"]["Subhalo/SubhaloGrNr"]["shape"] == [3]


def test_audit_reads_standard_header_group_epoch_attributes(tmp_path):
    file_path = tmp_path / "header_fixture.h5"
    _fixture(file_path)
    with h5py.File(file_path, "a") as f:
        del f.attrs["h"]
        del f.attrs["Time"]
        header = f.create_group("Header")
        header.attrs["HubbleParam"] = 0.6774
        header.attrs["Time"] = 1.0
    c = _catalog(file_path)
    c["source"]["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    report = audit_astronomy_manifest(load_astronomy_manifest(_write_manifest(tmp_path, [c])))
    assert report.ok, report.errors


@pytest.mark.parametrize("mutation, match", [
    (lambda c: c["fields"].pop("velocity"), "velocity"),
    (lambda c: c["fields"]["position"].update(unit="parsec"), "unit"),
    (lambda c: c["epoch"].update(h=0.0), "h"),
    (lambda c: c["epoch"].update(scale_factor=1.1), "scale_factor"),
    (lambda c: c["target"].update(path="Group/GroupMass"), "target"),
    (lambda c: c["fields"]["half_mass_radius"].update(path="Subhalo/SubhaloHalfmassRad"), "radius"),
])
def test_manifest_rejects_incomplete_or_wrong_science_contract(tmp_path, mutation, match):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    catalog = _catalog(file_path)
    mutation(catalog)
    with pytest.raises(ValueError, match=match):
        load_astronomy_manifest(_write_manifest(tmp_path, [catalog]))


def test_manifest_rejects_missing_file_and_hash_mismatch(tmp_path):
    file_path = tmp_path / "gone.h5"
    c = _catalog(file_path)
    missing = audit_astronomy_manifest(load_astronomy_manifest(_write_manifest(tmp_path, [c])))
    assert not missing.ok and "does not exist" in missing.errors[0]["error"]
    file_path.write_bytes(b"changed")
    mismatch = audit_astronomy_manifest(load_astronomy_manifest(_write_manifest(tmp_path, [c])))
    assert not mismatch.ok and "hash mismatch" in mismatch.errors[0]["error"]


def test_cross_role_initial_condition_overlap_is_rejected_across_suites(tmp_path):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    dev = _catalog(file_path, role="external_development", ic="shared")
    conf = _catalog(file_path, role="confirmation", ic="shared")
    conf["suite"] = "SIMBA"
    with pytest.raises(ValueError, match="initial.condition|lineage|overlap"):
        load_astronomy_manifest(_write_manifest(tmp_path, [dev, conf]))


def test_audit_rejects_wrong_shapes_and_group_membership_bounds(tmp_path):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    with h5py.File(file_path, "a") as f:
        del f["Subhalo/SubhaloPos"]
        f.create_dataset("Subhalo/SubhaloPos", data=np.zeros((3, 2)))
        f["Subhalo/SubhaloGrNr"][2] = 9
    c = _catalog(file_path)
    c["source"]["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    report = audit_astronomy_manifest(load_astronomy_manifest(_write_manifest(tmp_path, [c])))
    assert not report.ok
    assert "shape" in report.errors[0]["error"]


def test_audit_rejects_out_of_range_membership_when_shapes_match(tmp_path):
    file_path = tmp_path / "bad_membership.h5"
    _fixture(file_path)
    with h5py.File(file_path, "a") as f:
        f["Subhalo/SubhaloGrNr"][2] = 2
    c = _catalog(file_path)
    c["source"]["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    report = audit_astronomy_manifest(load_astronomy_manifest(_write_manifest(tmp_path, [c])))
    assert not report.ok
    assert "outside" in report.errors[0]["error"]


def test_cli_json_success_and_failure_have_no_filesystem_side_effects(tmp_path):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    manifest_path = _write_manifest(tmp_path, [_catalog(file_path)])
    before = set(tmp_path.iterdir())
    ok = subprocess.run([sys.executable, "-m", "data.astronomy_audit", "--manifest", str(manifest_path), "--json"],
                        capture_output=True, text=True, check=False)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["ok"] is True
    assert json.loads(ok.stdout)["verification_level"] == "content_verified"
    assert set(tmp_path.iterdir()) == before
    bad_path = _write_manifest(tmp_path, [_catalog(tmp_path / "missing.h5")])
    before = set(tmp_path.iterdir())
    bad = subprocess.run([sys.executable, "-m", "data.astronomy_audit", "--manifest", str(bad_path), "--json"],
                         capture_output=True, text=True, check=False)
    assert bad.returncode != 0
    assert json.loads(bad.stdout)["ok"] is False
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("where,key", [
    ("root", "surprise"), ("catalog", "surprise"), ("source", "remote_url"),
    ("epoch", "cosmology"), ("conventions", "frame"), ("field", "comment"),
    ("selection", "minimum_richness"), ("target", "alias"),
])
def test_manifest_rejects_unknown_keys_at_every_level(tmp_path, where, key):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    catalog = _catalog(file_path)
    document = {"schema_version": 1, "catalogs": [catalog]}
    if where == "root":
        document[key] = True
    elif where == "catalog":
        catalog[key] = True
    elif where == "source":
        catalog["source"][key] = "https://example.invalid/data"
    elif where == "epoch":
        catalog["epoch"][key] = 1
    elif where == "conventions":
        catalog["conventions"][key] = "unknown"
    elif where == "field":
        catalog["fields"]["position"][key] = "metadata"
    elif where == "selection":
        catalog["selection"][key] = 5
    else:
        catalog["target"][key] = "GroupMass"
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="unknown"):
        load_astronomy_manifest(manifest_path)


@pytest.mark.parametrize("field,value", [
    ("id", []), ("role", {}), ("suite", 8), ("simulation", None),
    ("simulation_set", []), ("initial_condition_id", {}), ("snapshot", 99),
])
def test_manifest_rejects_non_string_catalog_identity_values(tmp_path, field, value):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    catalog = _catalog(file_path)
    catalog[field] = value
    with pytest.raises(ValueError, match="string"):
        load_astronomy_manifest(_write_manifest(tmp_path, [catalog]))


@pytest.mark.parametrize("field,value", [("kind", []), ("path", {}), ("sha256", [])])
def test_manifest_rejects_non_string_source_values(tmp_path, field, value):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    catalog = _catalog(file_path)
    catalog["source"][field] = value
    with pytest.raises(ValueError, match="string"):
        load_astronomy_manifest(_write_manifest(tmp_path, [catalog]))


@pytest.mark.parametrize("logical,shape", [
    ("position", [3, 2]), ("velocity", [3, 4]),
    ("stellar_mass", [3, 5]), ("half_mass_radius", [3, 7]),
    ("group_index", [3, 1]),
])
def test_manifest_requires_physical_component_shapes(tmp_path, logical, shape):
    file_path = tmp_path / "fixture.h5"
    _fixture(file_path)
    catalog = _catalog(file_path)
    catalog["fields"][logical]["shape"] = shape
    if logical in {"position", "velocity", "stellar_mass", "half_mass_radius"}:
        with h5py.File(file_path, "a") as f:
            name = catalog["fields"][logical]["path"]
            del f[name]
            f.create_dataset(name, shape=shape, dtype="f8")
        catalog["source"]["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="shape"):
        load_astronomy_manifest(_write_manifest(tmp_path, [catalog]))


def test_schema_only_audit_is_not_content_verified(tmp_path):
    file_path = tmp_path / "missing.h5"
    manifest = load_astronomy_manifest(_write_manifest(tmp_path, [_catalog(file_path)]))
    report = audit_astronomy_manifest(manifest, inspect_hdf5=False)
    assert report.ok
    assert report.to_dict()["verification_level"] == "schema_only"
    assert not report.content_verified
    with pytest.raises(ValueError, match="content-verified"):
        report.require_content_verified()
