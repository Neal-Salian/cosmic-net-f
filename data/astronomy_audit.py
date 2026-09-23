"""Read-only offline audit for strict, explicitly declared astronomy catalogs."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from data.astronomy_manifest import AstronomyManifest, load_astronomy_manifest
from data.provenance import sha256_file


class AuditReport:
    def __init__(self, ok, catalogs, errors, manifest_sha256, content_verified=False):
        self.ok = bool(ok)
        self.catalogs = catalogs
        self.errors = errors
        self.manifest_sha256 = manifest_sha256
        self.content_verified = bool(content_verified and self.ok)

    def require_content_verified(self):
        """Raise unless this successful report inspected and hashed catalog content."""
        if not self.content_verified:
            raise ValueError("scientific use requires a successful content-verified audit")
        return self

    def to_dict(self):
        return {"ok": self.ok, "manifest_sha256": self.manifest_sha256,
                "verification_level": "content_verified" if self.content_verified else "schema_only",
                "content_verified": self.content_verified,
                "catalogs": self.catalogs, "errors": self.errors}


def _inspect(catalog):
    src = catalog["source"]
    path = Path(src["path"])
    if not path.is_file():
        raise FileNotFoundError(f"catalog file does not exist: {path}")
    digest = sha256_file(path)
    if digest["sha256"] != src["sha256"]:
        raise ValueError(f"catalog content hash mismatch for {path}")
    datasets = {}
    with h5py.File(path, "r") as h5:
        for logical, spec in catalog["fields"].items():
            name = spec["path"]
            if name not in h5:
                raise ValueError(f"missing declared field dataset {name} ({logical})")
            ds = h5[name]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"declared field {name} is not a dataset")
            shape = list(ds.shape)
            declared = spec["shape"]
            if len(shape) != len(declared) or any(actual != expected for actual, expected in zip(shape, declared)):
                raise ValueError(f"shape mismatch for {name}: actual {shape}, declared {declared}")
            if ds.dtype.kind not in "iuf":
                raise ValueError(f"field {name} must have numeric dtype")
            datasets[name] = {"shape": shape, "dtype": str(ds.dtype), "rank": ds.ndim,
                              "length": shape[0]}
        header_attrs = h5["Header"].attrs if "Header" in h5 and isinstance(h5["Header"], h5py.Group) else h5.attrs
        for attr, aliases, expected, label in (("h", ("h", "HubbleParam"), catalog["epoch"]["h"], "h"),
                                               ("Time", ("Time",), catalog["epoch"]["scale_factor"], "scale_factor")):
            actual_name = next((candidate for candidate in aliases if candidate in header_attrs), None)
            if actual_name is None:
                raise ValueError(f"HDF5 header is missing {attr} required for {label} check")
            actual = float(header_attrs[actual_name])
            if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-5, atol=1e-7):
                raise ValueError(f"HDF5 header {actual_name}={actual} mismatches declared {label}={expected}")
        target_len = h5[catalog["target"]["path"]].shape[0]
        groups = h5[catalog["fields"]["group_index"]["path"]][...]
        if groups.dtype.kind not in "iu":
            raise ValueError("group membership indices must use an integer dtype")
        if groups.size and (int(groups.min()) < 0 or int(groups.max()) >= target_len):
            raise ValueError(f"group membership index outside [0, {target_len})")
        anchor_len = h5[catalog["fields"]["subhalo_mass_anchor"]["path"]].shape[0]
        for name in (catalog["fields"][k]["path"] for k in
                     ("position", "velocity", "stellar_mass", "half_mass_radius", "group_index")):
            if h5[name].shape[0] != anchor_len:
                raise ValueError(f"subhalo field {name} length disagrees with membership anchor")
    return {"catalog_id": catalog["id"], "role": catalog["role"], "suite": catalog["suite"],
            "simulation": catalog["simulation"], "initial_condition_id": catalog["initial_condition_id"],
            "snapshot": catalog["snapshot"], "source_kind": src["kind"], "source": digest,
            "datasets": datasets, "group_count": target_len, "subhalo_count": anchor_len}


def audit_astronomy_manifest(manifest, roles=None, inspect_hdf5=True):
    if not isinstance(manifest, AstronomyManifest):
        raise TypeError("manifest must be an AstronomyManifest; parse it with load_astronomy_manifest")
    selected_roles = set(roles or ())
    if selected_roles - {"development", "external_development", "confirmation"}:
        raise ValueError(f"unknown requested roles: {sorted(selected_roles)}")
    selected = [c for c in manifest.catalogs if not selected_roles or c["role"] in selected_roles]
    if selected_roles and not selected:
        raise ValueError(f"manifest contains no catalogs for roles {sorted(selected_roles)}")
    evidence, errors = [], []
    for catalog in selected:
        try:
            if inspect_hdf5:
                evidence.append(_inspect(catalog))
            else:
                evidence.append({"catalog_id": catalog["id"], "role": catalog["role"],
                                 "source_kind": catalog["source"]["kind"]})
        except Exception as exc:
            errors.append({"catalog_id": catalog["id"], "error": f"{type(exc).__name__}: {exc}"})
    return AuditReport(not errors, evidence, errors, manifest.sha256,
                       content_verified=inspect_hdf5 and not errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="local YAML manifest")
    parser.add_argument("--role", choices=("development", "external_development", "confirmation"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        manifest = load_astronomy_manifest(args.manifest)
        report = audit_astronomy_manifest(manifest, roles=[args.role] if args.role else None)
    except Exception as exc:
        report = AuditReport(False, [], [{"error": f"{type(exc).__name__}: {exc}"}], None)
    payload = report.to_dict()
    if args.as_json:
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
    elif report.ok:
        print(f"Astronomy manifest audit passed: {len(report.catalogs)} catalog(s)")
        for item in report.catalogs:
            print(f"  {item['catalog_id']} [{item['role']}] {item['source_kind']} {item['source']['sha256']}")
    else:
        for error in report.errors:
            print(error.get("error"), file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
