"""Explicit scientific compatibility gates for the frozen intrinsic-feature path."""
import hashlib
import json
import math
from pathlib import Path

FEATURE_CONTRACT = {
    "target": "log10(M200c/M_sun)", "target_field": "Group_M_Crit200",
    "radius_field": "SubhaloHalfmassRad", "radius_semantic": "total_subhalo_half_mass_radius",
    "node_features": ["log10_stellar_mass_Msun", "log10_velocity_dispersion_kms",
                      "log10_total_half_mass_radius_Mpc", "log10_stellar_metallicity"],
    "edge_features": ["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"],
    "position_units": "physical_Mpc", "velocity_units": "peculiar_km/s",
    "mass_units": "M_sun", "metallicity": "SubhaloStarMetallicity_mass_fraction",
    "normalization": "log10_nodes_no_refit_raw_physical_edges",
    "geometry": "euclidean_per_halo_unwrapped",
}


def file_record(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty required file: {path}")
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:  # macOS UF_DATALESS
        raise ValueError(f"cloud placeholder is not locally usable: {path}")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer is not artifact content: {path}")
        stream.seek(0)
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def checked_file(spec, base):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("required file record must contain path and sha256")
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(base) / path
    record = file_record(path)
    if record["sha256"] != spec["sha256"]:
        raise ValueError(f"content hash mismatch: {path}")
    return record


def validate_pair(training, evaluation, *, fixture_smoke=False):
    expected_status = "fixture_smoke" if fixture_smoke else "audited"
    if training.get("schema_version") != 1 or training.get("status") != expected_status:
        raise ValueError("training receipt must be schema 1 and audited; unknown provenance blocks evaluation")
    required = {"target", "features", "training_transforms", "lineage", "backbone_policy_binding"}
    checks = training.get("verification", {})
    if set(checks) != required or any(checks[k] is not True for k in required):
        raise ValueError("training target, features, transforms, lineage and backbone/policy binding must be audited")
    source_kind = "synthetic_fixture" if fixture_smoke else "local_hdf5"
    if evaluation.get("source_kind") != source_kind:
        raise ValueError("scientific evaluation requires local_hdf5, never synthetic/fallback")
    if not evaluation.get("release") or any(marker in evaluation["release"].lower() for marker in ("pending", "unknown", "unverified", "tbd")):
        raise ValueError("catalog release and schema review required")
    for contract in (training.get("feature_contract"), evaluation.get("feature_contract")):
        if contract != FEATURE_CONTRACT:
            raise ValueError("incompatible target or feature schema; no coercion is permitted")
    population = training.get("population_contract")
    if (not isinstance(population, dict) or population != evaluation.get("population_contract")
            or set(population) != {"member_selection", "min_members"}
            or population.get("member_selection") not in {"all_stellar_positive", "native_first10_then_stellar_positive"}
            or population.get("min_members") != 3):
        raise ValueError("population/member selection must be audited and identical; no silent richness expansion")
    mass_range = training.get("target_range_log10_msun")
    if (not isinstance(mass_range, list) or len(mass_range) != 2
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in mass_range)
            or mass_range[0] > mass_range[1]):
        raise ValueError("audited training mass support requires a finite [min, max] log10 range")
    catalogs = training.get("training_catalogs")
    if not isinstance(catalogs, list) or len(catalogs) != 1:
        raise ValueError("this package supports exactly one training catalog; all development/model-selection populations must be in that audited source")
    fields = ("suite", "simulation", "initial_condition_id", "source_sha256")
    for catalog in [*catalogs, evaluation]:
        if any(not isinstance(catalog.get(k), str) or not catalog[k].strip() or any(marker in catalog[k].lower() for marker in ("pending", "unknown", "unverified", "tbd")) for k in fields):
            raise ValueError("catalog identity, lineage and source content must be explicit")
        digest = catalog["source_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("catalog source content must have a SHA-256 digest")
    if evaluation["initial_condition_id"] in {c["initial_condition_id"] for c in catalogs}:
        raise ValueError("independent validation rejects shared initial-condition lineage, including other snapshots")
    if evaluation["source_sha256"] in {c["source_sha256"] for c in catalogs}:
        raise ValueError("evaluation content is already in the development population")
    same_suite = evaluation["suite"] in {c["suite"] for c in catalogs}
    expected = "fixture_smoke" if fixture_smoke else ("independent_volume" if same_suite else "cross_simulation_transfer")
    if evaluation.get("claim") != expected:
        raise ValueError(f"this catalog comparison must be labeled {expected}")
    return {"claim": expected, "independent_initial_conditions": True,
            "domain_generalization_established": False,
            "scope": "single_prespecified_external_catalog"}


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)
