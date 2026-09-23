"""Strict conversion of content-audited astronomy HDF5 catalogs.

The returned records intentionally describe only fields present in the audited
manifest. They do not satisfy the legacy four-feature 3D ``HaloData`` schema.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import h5py

from data.catalog import CatalogContract
from data.provenance import sha256_file
from data.astronomy_audit import AuditReport
from data.astronomy_manifest import _validate_catalog, load_astronomy_manifest


@dataclass(frozen=True)
class AstronomyMember:
    member_id: str
    source_row: int
    stellar_mass_msun: float
    position_mpc: np.ndarray
    velocity_km_s: np.ndarray
    half_mass_radius_mpc: float


@dataclass(frozen=True)
class AstronomyHalo:
    catalog_id: str
    group_index: int
    halo_mass_msun: float
    target_log10_msun: float
    members: tuple[AstronomyMember, ...]
    metadata: dict = field(default_factory=dict)
    schema: str = "partial_observation_inputs_v1"


def _finite_array(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite numeric values")
    return array


def _validate_dataset(h5, spec, logical):
    path = spec["path"]
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise ValueError(f"missing declared dataset {path} ({logical})")
    data = h5[path][...]
    if list(data.shape) != spec["shape"]:
        raise ValueError(f"shape mismatch for {path}: got {list(data.shape)}, expected {spec['shape']}")
    return _finite_array(data, path)


def load_audited_catalog(manifest, audit_report, catalog_id):
    """Load one explicitly selected catalog after audit and a fresh SHA-256 check."""
    if not isinstance(audit_report, AuditReport):
        raise TypeError("audit_report must be an AuditReport from audit_astronomy_manifest")
    audit_report.require_content_verified()
    current_manifest = load_astronomy_manifest(manifest.path)
    if (current_manifest.sha256 != audit_report.manifest_sha256
            or current_manifest.sha256 != manifest.sha256):
        raise ValueError("manifest changed after content audit or does not match the audited manifest")
    if current_manifest.catalogs != manifest.catalogs:
        raise ValueError("in-memory manifest declaration differs from the audited manifest declaration")
    manifest = current_manifest
    matches = [c for c in manifest.catalogs if c["id"] == catalog_id]
    if len(matches) != 1:
        raise ValueError(f"catalog_id {catalog_id!r} must identify exactly one manifest entry")
    catalog = matches[0]
    # The parser's entry mappings are mutable; revalidate the exact declaration
    # at this boundary instead of trusting that callers left it untouched.
    _validate_catalog(catalog, Path(manifest.path).parent)
    evidence = [item for item in audit_report.catalogs if item.get("catalog_id") == catalog_id]
    if len(evidence) != 1 or not isinstance(evidence[0].get("source"), dict):
        raise ValueError(f"content audit has no source evidence for catalog {catalog_id!r}")
    for key in ("role", "suite", "simulation", "initial_condition_id", "snapshot"):
        if evidence[0].get(key) != catalog[key]:
            raise ValueError(f"content audit {key} evidence disagrees with catalog declaration")
    source = catalog["source"]
    if source["kind"] == "synthetic_fixture":
        raise ValueError("strict scientific loading rejects source.kind=synthetic_fixture")
    if source["kind"] != "local_hdf5":
        raise ValueError(f"unsupported strict source kind {source['kind']!r}")
    path = Path(source["path"])
    digest = sha256_file(path)
    if digest["sha256"] != source["sha256"] or evidence[0]["source"].get("sha256") != digest["sha256"]:
        raise ValueError(f"catalog content hash mismatch or changed after audit for {path}")

    fields = catalog["fields"]
    with h5py.File(path, "r") as h5:
        header = h5["Header"].attrs if "Header" in h5 and isinstance(h5["Header"], h5py.Group) else h5.attrs
        actual_h = header.get("h", header.get("HubbleParam"))
        actual_a = header.get("Time")
        try:
            actual_h, actual_a = float(actual_h), float(actual_a)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("HDF5 header h/Time must be positive and finite") from None
        if (not np.isfinite(actual_h) or actual_h <= 0
                or not np.isfinite(actual_a) or actual_a <= 0):
            raise ValueError("HDF5 header h/Time must be positive and finite")
        if (not np.isclose(actual_h, catalog["epoch"]["h"], rtol=1e-5, atol=0.0)
                or not np.isclose(actual_a, catalog["epoch"]["scale_factor"], rtol=1e-5, atol=0.0)):
            raise ValueError("HDF5 header h/Time mismatch with catalog epoch declaration")
        arrays = {name: _validate_dataset(h5, spec, name) for name, spec in fields.items()}
        groups = arrays["group_index"]
        if groups.dtype.kind not in "iu" or groups.ndim != 1:
            raise ValueError("group membership indices must be a complete integer vector")
        target_raw = _finite_array(_validate_dataset(h5, catalog["target"], "target"), "target")
        if target_raw.ndim != 1 or target_raw.size == 0:
            raise ValueError("target group masses must be a nonempty vector")
        if np.any(target_raw <= 0):
            raise ValueError("Group_M_Crit200 target values must be positive")
        nrows = arrays["subhalo_mass_anchor"].shape[0]
        if any(arrays[name].shape[0] != nrows for name in fields if name != "group_mass"):
            raise ValueError("all subhalo field lengths must match before row selection")
        if groups.size and (int(groups.min()) < 0 or int(groups.max()) >= len(target_raw)):
            raise ValueError(f"group membership index outside [0, {len(target_raw)})")
        for name in ("group_mass", "subhalo_mass_anchor", "stellar_mass", "half_mass_radius"):
            if np.any(arrays[name] < 0):
                raise ValueError(f"negative values are invalid in {name}")
        stellar = arrays["stellar_mass"][:, 4]
        radii = arrays["half_mass_radius"][:, 4]
        if np.any((stellar > 0) & (radii <= 0)):
            raise ValueError("selected members require positive type-4 stellar radius")
        h = float(catalog["epoch"]["h"])
        a = float(catalog["epoch"]["scale_factor"])
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            group_mass_msun = target_raw * 1e10 / h
            target_log10_msun = np.log10(group_mass_msun)
            positions_mpc = arrays["position"] * a / (1000.0 * h)
            radii_mpc = radii * a / (1000.0 * h)
            stellar_msun = stellar * 1e10 / h
        for label, values in (("converted group mass", group_mass_msun),
                              ("converted log10 target", target_log10_msun),
                              ("converted position", positions_mpc),
                              ("converted type-4 radius", radii_mpc),
                              ("converted stellar mass", stellar_msun)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{label} must remain finite after physical conversion")
        selected_rows = stellar > 0
        if np.any(stellar_msun[selected_rows] <= 0):
            raise ValueError("converted selected type-4 stellar mass must be positive")
        if np.any(radii_mpc[selected_rows] <= 0):
            raise ValueError("converted selected type-4 radius must be positive")
        velocities = arrays["velocity"].astype(np.float64, copy=True)

    file_record = {**digest}
    contract = CatalogContract.from_mapping({
        "research_label": "corrected_research", "source": f"{catalog['suite']}/{catalog['simulation']}/{catalog['snapshot']}",
        "suite": catalog["suite"], "simulation": catalog["simulation"], "snapshot": catalog["snapshot"],
        "volume_or_ic_group": catalog["initial_condition_id"], "target_field": "Group_M_Crit200",
        "target_definition": "log10(M200c/M_sun)", "mass_units": "M_sun",
        "position_units": "physical_Mpc", "radius_units": "physical_Mpc",
        "velocity_units": "peculiar_km_per_s", "coordinate_frame": "physical_proper",
        "velocity_convention": "peculiar_km_per_s", "hubble_param": h, "scale_factor": a,
        "radius_source_field": "SubhaloHalfmassRadType[4]",
        "radius_semantic": "stellar_half_mass_radius_type4", "synthetic": False, "fallback": False,
        "source_files": [file_record], "field_mapping": {name: spec["path"] for name, spec in fields.items()},
        "conversion_record": {"mass": "raw*1e10/h", "position_radius": "raw*a/(1000*h)",
                              "velocity": "unchanged peculiar km/s"},
    }).validate(strict_research=True).to_dict()

    output = []
    for group_index, (mass, log_mass) in enumerate(zip(group_mass_msun, target_log10_msun)):
        row_indices = np.flatnonzero(groups == group_index)
        selected = [int(i) for i in row_indices if stellar[i] > 0]
        excluded = int(len(row_indices) - len(selected))
        members = tuple(AstronomyMember(
            member_id=f"{catalog_id}:group:{group_index}:row:{i}", source_row=i,
            stellar_mass_msun=float(stellar_msun[i]), position_mpc=positions_mpc[i].copy(),
            velocity_km_s=velocities[i].copy(), half_mass_radius_mpc=float(radii_mpc[i]),
        ) for i in selected)
        metadata = {
            "catalog_id": catalog_id, "source_kind": source["kind"], "source": file_record,
            "role": catalog["role"], "suite": catalog["suite"], "simulation": catalog["simulation"],
            "simulation_set": catalog["simulation_set"], "initial_condition_id": catalog["initial_condition_id"],
            "snapshot": catalog["snapshot"], "release": catalog["release"],
            "selection": dict(catalog["selection"]),
            "selection_accounting": {"input_rows": len(row_indices), "selected_rows": len(selected),
                                     "excluded_rows": excluded,
                                     "excluded_reasons": {"zero_stellar_mass_type4": excluded}},
            "catalog_contract": contract,
        }
        output.append(AstronomyHalo(catalog_id, group_index, float(mass), float(log_mass), members, metadata))
    return output
