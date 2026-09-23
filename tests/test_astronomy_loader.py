"""Strict audited astronomy row conversion using tiny synthetic HDF5 fixtures."""
import hashlib

import h5py
import numpy as np
import pytest
import yaml

from data.astronomy_manifest import load_astronomy_manifest
from data.astronomy_audit import AuditReport, audit_astronomy_manifest
from data.astronomy_loader import load_audited_catalog


def _fixture(path):
    with h5py.File(path, "w") as f:
        f.attrs["h"] = 0.5
        f.attrs["Time"] = 0.5
        f.create_dataset("Group/Group_M_Crit200", data=[5., 10.])
        f.create_dataset("Subhalo/SubhaloMass", data=[1., 2., 3.])
        f.create_dataset("Subhalo/SubhaloPos", data=[[100., 0., -100.], [200., 0., 0.], [300., 0., 0.]])
        f.create_dataset("Subhalo/SubhaloVel", data=[[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]])
        f.create_dataset("Subhalo/SubhaloMassType", data=np.array([[0, 0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0.0, 0], [0, 0, 0, 0, 0.4, 0]]))
        f.create_dataset("Subhalo/SubhaloHalfmassRadType", data=np.array([[0, 0, 0, 0, 50., 0], [0, 0, 0, 0, 0., 0], [0, 0, 0, 0, 100., 0]]))
        f.create_dataset("Subhalo/SubhaloGrNr", data=np.array([0, 0, 1], dtype="i8"))


def _entry(path, *, kind="local_hdf5"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "id": "tng-test", "role": "external_development", "suite": "IllustrisTNG",
        "simulation": "L25n256", "simulation_set": "LH", "initial_condition_id": "ic-001",
        "snapshot": "099", "source": {"kind": kind, "path": str(path), "sha256": digest},
        "epoch": {"redshift": 1., "scale_factor": .5, "h": .5},
        "conventions": {"coordinates": "comoving", "velocities": "peculiar_km_per_s", "periodic_box": False},
        "fields": {
            "group_mass": {"path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "shape": [2]},
            "subhalo_mass_anchor": {"path": "Subhalo/SubhaloMass", "unit": "1e10_Msun_per_h", "shape": [3]},
            "position": {"path": "Subhalo/SubhaloPos", "unit": "ckpc_per_h", "shape": [3, 3]},
            "velocity": {"path": "Subhalo/SubhaloVel", "unit": "km_per_s", "shape": [3, 3]},
            "stellar_mass": {"path": "Subhalo/SubhaloMassType", "component": 4, "unit": "1e10_Msun_per_h", "shape": [3, 6]},
            "half_mass_radius": {"path": "Subhalo/SubhaloHalfmassRadType", "component": 4, "unit": "ckpc_per_h", "shape": [3, 6]},
            "group_index": {"path": "Subhalo/SubhaloGrNr", "unit": "index", "shape": [3]},
        },
        "target": {"name": "halo_mass", "path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "definition": "log10(M200c/M_sun)", "shape": [2]},
        "selection": {"predicate": "stellar_mass_type4_gt_zero"}, "release": "test-only fixture",
    }


def _audited(tmp_path, *, kind="local_hdf5"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "tiny.h5"
    _fixture(source)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": [_entry(source, kind=kind)]}))
    manifest = load_astronomy_manifest(manifest_path)
    report = audit_astronomy_manifest(manifest)
    assert report.ok, report.errors
    return manifest, report, source


def test_converts_units_and_preserves_identity_and_partial_schema(tmp_path):
    manifest, report, _ = _audited(tmp_path)
    halos = load_audited_catalog(manifest, report, "tng-test")
    assert len(halos) == 2
    assert halos[0].halo_mass_msun == pytest.approx(1e11)
    assert halos[0].target_log10_msun == pytest.approx(11.)
    assert halos[0].members[0].stellar_mass_msun == pytest.approx(4e9)
    assert halos[0].members[0].position_mpc.tolist() == pytest.approx([0.1, 0., -0.1])
    assert halos[0].members[0].half_mass_radius_mpc == pytest.approx(0.05)
    assert halos[0].members[0].velocity_km_s.tolist() == [10., 20., 30.]
    assert halos[0].members[0].member_id == "tng-test:group:0:row:0"
    assert halos[0].metadata["role"] == "external_development"
    assert halos[0].metadata["initial_condition_id"] == "ic-001"
    contract = halos[0].metadata["catalog_contract"]
    assert contract["source_files"][0]["sha256"] == manifest.catalogs[0]["source"]["sha256"]
    assert contract["radius_source_field"] == "SubhaloHalfmassRadType[4]"
    assert contract["velocity_convention"] == "peculiar_km_per_s"
    assert halos[0].metadata["selection_accounting"] == {"input_rows": 2, "selected_rows": 1, "excluded_rows": 1, "excluded_reasons": {"zero_stellar_mass_type4": 1}}
    assert halos[0].schema == "partial_observation_inputs_v1"
    assert "velocity_dispersion" not in halos[0].members[0].__dataclass_fields__
    assert "metallicity" not in halos[0].members[0].__dataclass_fields__


def test_velocity_remains_peculiar_km_per_second_without_epoch_scaling(tmp_path):
    manifest, report, _ = _audited(tmp_path)
    halo = load_audited_catalog(manifest, report, "tng-test")[0]
    assert halo.members[0].velocity_km_s[0] == 10.


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f["Subhalo/SubhaloGrNr"].__setitem__(0, -1), "membership"),
    (lambda f: f["Subhalo/SubhaloPos"].__setitem__((0, 0), np.nan), "finite"),
    (lambda f: f["Subhalo/SubhaloMassType"].__setitem__((0, 4), -1), "negative"),
    (lambda f: f["Subhalo/SubhaloHalfmassRadType"].__setitem__((0, 4), -1), "negative"),
    (lambda f: f["Group/Group_M_Crit200"].__setitem__(0, 0), "positive"),
])
def test_rejects_invalid_raw_rows_and_targets(tmp_path, mutation, match):
    manifest, _, source = _audited(tmp_path)
    with h5py.File(source, "a") as f:
        mutation(f)
    if match == "membership":
        catalog = manifest.catalogs[0]
        catalog["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest, report = _rewrite_and_audit(tmp_path, catalog)
        assert not report.ok and "outside" in report.errors[0]["error"]
        return
    _refresh_manifest_source_hash(manifest, source)
    manifest, report = _rewrite_and_audit(tmp_path, manifest.catalogs[0])
    assert report.ok, report.errors
    with pytest.raises(ValueError, match=match):
        load_audited_catalog(manifest, report, "tng-test")


def test_rejects_file_mutation_after_content_audit(tmp_path):
    manifest, report, source = _audited(tmp_path)
    with h5py.File(source, "a") as f:
        f["Subhalo/SubhaloVel"][0, 0] = 999.
    with pytest.raises(ValueError, match="hash mismatch|changed after audit"):
        load_audited_catalog(manifest, report, "tng-test")


def test_requires_content_verified_audit_and_rejects_synthetic_source(tmp_path):
    manifest, report, _ = _audited(tmp_path)
    with pytest.raises(ValueError, match="content-verified"):
        load_audited_catalog(manifest, audit_astronomy_manifest(manifest, inspect_hdf5=False), "tng-test")
    synthetic_manifest, synthetic_report, _ = _audited(tmp_path / "synthetic", kind="synthetic_fixture")
    with pytest.raises(ValueError, match="synthetic_fixture"):
        load_audited_catalog(synthetic_manifest, synthetic_report, "tng-test")


def _refresh_manifest_source_hash(manifest, source):
    manifest.catalogs[0]["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()


def _rewrite_and_audit(tmp_path, catalog):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": [catalog]}))
    manifest = load_astronomy_manifest(manifest_path)
    return manifest, audit_astronomy_manifest(manifest)


def test_rejects_manifest_file_changed_after_content_audit(tmp_path):
    manifest, report, _ = _audited(tmp_path)
    raw = (tmp_path / "manifest.yaml").read_text().replace("test-only fixture", "altered release")
    (tmp_path / "manifest.yaml").write_text(raw)
    with pytest.raises(ValueError, match="manifest.*changed|manifest.*match"):
        load_audited_catalog(manifest, report, "tng-test")


def test_rejects_mutated_in_memory_catalog_declaration(tmp_path):
    manifest, report, _ = _audited(tmp_path)
    manifest.catalogs[0]["role"] = "confirmation"
    with pytest.raises(ValueError, match="manifest.*declaration|manifest.*match"):
        load_audited_catalog(manifest, report, "tng-test")


@pytest.mark.parametrize("epoch_field,header_value,error_match", [
    ("h", 0.0, "positive and finite"),
    ("scale_factor", 0.0, "positive and finite"),
    ("h", 1e-12, "mismatch"),
    ("scale_factor", 1e-12, "mismatch"),
])
def test_loader_rejects_invalid_or_mismatched_tiny_header_epoch(tmp_path, epoch_field, header_value, error_match):
    source = tmp_path / "tiny_header.h5"
    _fixture(source)
    catalog = _entry(source)
    if epoch_field == "h":
        catalog["epoch"]["h"] = 1e-8
        with h5py.File(source, "a") as f:
            f.attrs["h"] = header_value
    else:
        catalog["epoch"]["scale_factor"] = 1e-8
        catalog["epoch"]["redshift"] = 1e8 - 1
        with h5py.File(source, "a") as f:
            f.attrs["Time"] = header_value
    catalog["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": [catalog]}))
    manifest = load_astronomy_manifest(manifest_path)
    # Construct fixture audit evidence to exercise the loader's independent
    # header validation after hash/declaration binding.
    evidence = {"catalog_id": "tng-test", "role": catalog["role"], "suite": catalog["suite"],
                "simulation": catalog["simulation"], "initial_condition_id": catalog["initial_condition_id"],
                "snapshot": catalog["snapshot"], "source": {"sha256": catalog["source"]["sha256"]}}
    report = AuditReport(True, [evidence], [], manifest.sha256, content_verified=True)
    with pytest.raises(ValueError, match=error_match):
        load_audited_catalog(manifest, report, "tng-test")


@pytest.mark.parametrize("field,value,match", [
    ("Group/Group_M_Crit200", np.finfo(np.float64).max, "converted.*mass|finite"),
    ("Subhalo/SubhaloPos", np.finfo(np.float64).max, "converted.*position|finite"),
    ("Subhalo/SubhaloHalfmassRadType", np.finfo(np.float64).max, "converted.*radius|finite"),
])
def test_rejects_nonfinite_physical_values_after_conversion(tmp_path, field, value, match):
    manifest, _, source = _audited(tmp_path)
    with h5py.File(source, "a") as f:
        if field == "Group/Group_M_Crit200":
            f[field][0] = value
        elif field == "Subhalo/SubhaloPos":
            f[field][0, 0] = value
        else:
            f[field][0, 4] = value
    catalog = manifest.catalogs[0]
    if field != "Group/Group_M_Crit200":
        # A tiny positive h makes otherwise finite coordinate/radius inputs
        # overflow the proper-unit conversion while preserving valid declarations.
        catalog["epoch"]["h"] = 1e-320
        with h5py.File(source, "a") as f:
            tiny = np.nextafter(0.0, 1.0)
            f.attrs["h"] = 1e-320
            f["Group/Group_M_Crit200"][...] = [tiny, tiny]
            f["Subhalo/SubhaloMass"][...] = [tiny, tiny, tiny]
            f["Subhalo/SubhaloMassType"][...] = np.array(
                [[0, 0, 0, 0, tiny, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, tiny, 0]])
            if field == "Subhalo/SubhaloPos":
                f["Subhalo/SubhaloPos"][0, 0] = value
            else:
                f["Subhalo/SubhaloHalfmassRadType"][0, 4] = value
        catalog["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        catalog["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest, report = _rewrite_and_audit(tmp_path, catalog)
    assert report.ok, report.errors
    with pytest.raises(ValueError, match=match):
        load_audited_catalog(manifest, report, "tng-test")


@pytest.mark.parametrize("component,match", [
    ("stellar_mass", "converted selected type-4 stellar mass must be positive"),
    ("radius", "converted selected type-4 radius must be positive"),
])
def test_rejects_positive_raw_selected_values_that_underflow_after_conversion(tmp_path, component, match):
    manifest, _, source = _audited(tmp_path)
    catalog = manifest.catalogs[0]
    catalog["epoch"]["h"] = 1e308
    with h5py.File(source, "a") as f:
        f.attrs["h"] = 1e308
        if component == "stellar_mass":
            f["Subhalo/SubhaloMassType"][0, 4] = np.nextafter(0.0, 1.0)
        else:
            f["Subhalo/SubhaloHalfmassRadType"][0, 4] = np.nextafter(0.0, 1.0)
    catalog["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest, report = _rewrite_and_audit(tmp_path, catalog)
    assert report.ok, report.errors
    with pytest.raises(ValueError, match=match):
        load_audited_catalog(manifest, report, "tng-test")
