"""Strict declarations for local astronomy catalogs used by research workflows."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml


ROLES = {"development", "external_development", "confirmation"}
REQUIRED_FIELDS = {
    "group_mass": ("Group/Group_M_Crit200", "1e10_Msun_per_h", 1),
    "subhalo_mass_anchor": ("Subhalo/SubhaloMass", "1e10_Msun_per_h", 1),
    "position": ("Subhalo/SubhaloPos", "ckpc_per_h", 2),
    "velocity": ("Subhalo/SubhaloVel", "km_per_s", 2),
    "stellar_mass": ("Subhalo/SubhaloMassType", "1e10_Msun_per_h", 2),
    "half_mass_radius": ("Subhalo/SubhaloHalfmassRadType", "ckpc_per_h", 2),
    "group_index": ("Subhalo/SubhaloGrNr", "index", 1),
}
VALID_SOURCE_KINDS = {"local_hdf5", "synthetic_fixture"}
ROOT_KEYS = {"schema_version", "catalogs"}
CATALOG_KEYS = {"id", "role", "suite", "simulation", "simulation_set", "initial_condition_id",
                "snapshot", "source", "epoch", "conventions", "fields", "target", "selection", "release"}
MAPPING_KEYS = {
    "source": {"kind", "path", "sha256"},
    "epoch": {"redshift", "scale_factor", "h"},
    "conventions": {"coordinates", "velocities", "periodic_box", "box_size"},
    "target": {"name", "path", "unit", "definition", "shape"},
    "selection": {"predicate"},
    "field": {"path", "unit", "shape", "component"},
    "box_size": {"value", "unit"},
}


@dataclass(frozen=True)
class AstronomyManifest:
    path: str
    schema_version: int
    catalogs: tuple[dict[str, Any], ...]
    sha256: str


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _reject_unknown_keys(value, allowed, label):
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown keys: {sorted(unknown, key=str)}")


def _required_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def _validate_catalog(c, manifest_dir):
    _require_mapping(c, "catalog entry")
    _reject_unknown_keys(c, CATALOG_KEYS, "catalog entry")
    for key in ("id", "role", "suite", "simulation", "simulation_set", "initial_condition_id", "snapshot"):
        _required_string(c.get(key), f"catalog.{key}")
    if c["role"] not in ROLES:
        raise ValueError(f"unknown catalog role {c['role']!r}")
    source = _require_mapping(c.get("source"), "source")
    _reject_unknown_keys(source, MAPPING_KEYS["source"], "source")
    _required_string(source.get("kind"), "source.kind")
    if source.get("kind") not in VALID_SOURCE_KINDS:
        raise ValueError("source.kind must be local_hdf5 or synthetic_fixture")
    _required_string(source.get("path"), "source.path")
    path = Path(source["path"])
    if not path.is_absolute():
        path = manifest_dir / path
    source["path"] = str(path.resolve())
    digest = source.get("sha256")
    if not isinstance(digest, str):
        raise ValueError("source.sha256 must be a string")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("source.sha256 must be a lowercase SHA-256 digest")
    epoch = _require_mapping(c.get("epoch"), "epoch")
    _reject_unknown_keys(epoch, MAPPING_KEYS["epoch"], "epoch")
    for name in ("h", "scale_factor"):
        _positive_number(epoch.get(name), f"epoch.{name}")
    redshift = epoch.get("redshift")
    if isinstance(redshift, bool) or not isinstance(redshift, (int, float)) or not math.isfinite(redshift) or redshift < 0:
        raise ValueError("epoch.redshift must be finite and nonnegative")
    if not math.isclose(epoch["scale_factor"], 1 / (1 + redshift), rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError("epoch.scale_factor contradicts redshift")
    conventions = _require_mapping(c.get("conventions"), "conventions")
    _reject_unknown_keys(conventions, MAPPING_KEYS["conventions"], "conventions")
    if conventions.get("coordinates") != "comoving":
        raise ValueError("coordinates must be declared as comoving")
    if conventions.get("velocities") != "peculiar_km_per_s":
        raise ValueError("velocity convention must be peculiar_km_per_s")
    if not isinstance(conventions.get("periodic_box"), bool):
        raise ValueError("conventions.periodic_box must be explicitly boolean")
    if conventions["periodic_box"]:
        box = _require_mapping(conventions.get("box_size"), "conventions.box_size")
        _reject_unknown_keys(box, MAPPING_KEYS["box_size"], "conventions.box_size")
        _positive_number(box.get("value"), "box_size.value")
        if box.get("unit") not in {"ckpc_per_h", "cMpc_per_h", "Mpc"}:
            raise ValueError("unknown box size unit")
    fields = _require_mapping(c.get("fields"), "fields")
    unknown = set(fields) - set(REQUIRED_FIELDS)
    missing = set(REQUIRED_FIELDS) - set(fields)
    if missing or unknown:
        raise ValueError(f"field map missing {sorted(missing)} or has unknown {sorted(unknown)} fields")
    for logical, (expected_path, expected_unit, rank) in REQUIRED_FIELDS.items():
        spec = _require_mapping(fields[logical], f"fields.{logical}")
        _reject_unknown_keys(spec, MAPPING_KEYS["field"], f"fields.{logical}")
        if spec.get("path") != expected_path:
            raise ValueError(f"{logical} field path has wrong {('radius ' if logical == 'half_mass_radius' else '')}semantics; expected {expected_path}")
        if spec.get("unit") != expected_unit:
            raise ValueError(f"unknown or mismatched unit for fields.{logical}: {spec.get('unit')!r}")
        shape = spec.get("shape")
        if not isinstance(shape, list) or len(shape) != rank or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f"fields.{logical}.shape must declare a positive rank-{rank} shape")
        if logical in {"position", "velocity"} and shape[-1] != 3:
            raise ValueError(f"fields.{logical}.shape must have a final coordinate dimension of 3")
        if logical in {"stellar_mass", "half_mass_radius"} and shape[-1] != 6:
            raise ValueError(f"fields.{logical}.shape must have a final particle-type dimension of 6")
        if logical in {"stellar_mass", "half_mass_radius"} and spec.get("component") != 4:
            raise ValueError(f"fields.{logical} must explicitly select type-4 component 4")
    target = _require_mapping(c.get("target"), "target")
    _reject_unknown_keys(target, MAPPING_KEYS["target"], "target")
    expected_target = {"name": "halo_mass", "path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h",
                       "definition": "log10(M200c/M_sun)"}
    for key, expected in expected_target.items():
        if target.get(key) != expected:
            raise ValueError(f"strict target requires {key}={expected!r}")
    if target.get("shape") != fields["group_mass"]["shape"]:
        raise ValueError("target shape must match group_mass field shape")
    selection = _require_mapping(c.get("selection"), "selection")
    _reject_unknown_keys(selection, MAPPING_KEYS["selection"], "selection")
    if selection.get("predicate") != "stellar_mass_type4_gt_zero":
        raise ValueError("selection must explicitly declare stellar_mass_type4_gt_zero")
    if not isinstance(c.get("release"), str) or not c["release"].strip():
        raise ValueError("release/schema declaration is required (use an explicit unknown marker if unavailable)")
    return c


def load_astronomy_manifest(path) -> AstronomyManifest:
    """Parse and validate a manifest without opening any catalog files."""
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML manifest: {exc}") from exc
    _require_mapping(data, "manifest")
    _reject_unknown_keys(data, ROOT_KEYS, "manifest")
    if data.get("schema_version") != 1:
        raise ValueError("unsupported astronomy manifest schema_version")
    entries = data.get("catalogs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest catalogs must be a nonempty list")
    catalogs = tuple(_validate_catalog(entry, path.parent) for entry in entries)
    ids = [c["id"] for c in catalogs]
    if len(set(ids)) != len(ids):
        raise ValueError("catalog IDs must be unique")
    by_role = {}
    for c in catalogs:
        # The explicit lineage identifier is global across suites; adding suite here
        # would miss the very cross-suite overlap this guard is intended to catch.
        by_role.setdefault(c["role"], set()).add(c["initial_condition_id"])
    if by_role.get("external_development", set()) & by_role.get("confirmation", set()):
        raise ValueError("initial-condition lineage overlap between external_development and confirmation")
    if by_role.get("development", set()) & by_role.get("confirmation", set()):
        raise ValueError("initial-condition lineage overlap between development and confirmation")
    return AstronomyManifest(str(path), 1, catalogs, hashlib.sha256(raw_bytes).hexdigest())
