# Local astronomy manifest audit

`data.astronomy_manifest.load_astronomy_manifest(path)` parses a version-1 YAML manifest and validates each catalog's explicit role, simulation and initial-condition lineage, epoch, conventions, strict target, type-4 stellar mass/radius field map, units, shape declarations, selection, source kind, local path, digest syntax, and release declaration. Unknown keys and non-string identity values fail closed. Component shapes must use three position/velocity coordinates, six particle-type values for type-4 mass/radius arrays, and rank-one group membership. The parser does not open catalog data. Relative paths resolve from the manifest directory.

Run the read-only content audit with:

```sh
python -m data.astronomy_audit --manifest config/astronomy_manifest.yaml --json
python -m data.astronomy_audit --manifest config/astronomy_manifest.yaml --role development
```

The audit checks each file's SHA-256 before opening it, confirms declared HDF5 datasets, exact shapes and numeric dtypes, checks header `h`/`HubbleParam` and `Time` against the manifest, and validates integer group membership bounds and aligned subhalo lengths. Successful reports and JSON output identify `verification_level: content_verified`; failures use a nonzero exit code and structured error entries. The optional Python API `inspect_hdf5=False` yields `schema_only`, which must not be used for scientific evaluation. Call `report.require_content_verified()` at a scientific-use boundary. The audit does not create output/cache directories or substitute data.

The shipped `config/astronomy_manifest.example.yaml` contains placeholders and is intentionally not evidence that any catalog is available. Test fixtures are synthetic-only.

## Strict audited row adapter

`data.astronomy_loader.load_audited_catalog(manifest, audit_report, catalog_id)`
requires a successful content-verified `AuditReport` containing evidence for the
selected catalog. It reparses the manifest on disk and matches its hash and
catalog declarations against both the passed manifest and audit report. It
recomputes the file SHA-256 immediately before opening the HDF5 file and
rejects `synthetic_fixture` in this strict path. It checks every
declared dataset shape and numeric value, aligned subhalo field lengths, full
integer membership bounds, nonnegative mass/radius fields, positive finite
group targets, and positive type-4 stellar radius for selected rows.
Header h and scale factor must be positive and finite and match the manifest
with a relative tolerance only, so tiny values are not accepted through an
absolute tolerance. Selected converted stellar masses and radii must remain
strictly positive after conversion.
Converted masses, positions, radii and logarithmic targets must also remain
finite; finite raw values that overflow during physical conversion are
rejected. Positive raw values that underflow to zero are rejected for selected
members.

The returned `AstronomyHalo` and `AstronomyMember` records use the declared
`Group_M_Crit200` target and type-4 stellar mass/radius. Conversion is
`1e10 M_sun/h * 1/h` for mass and `ckpc/h * a/(1000h)` for physical Mpc;
peculiar `SubhaloVel` remains in km/s unchanged. A zero type-4 stellar mass is
excluded by the explicit manifest selection and counted per halo. Stable IDs
include catalog, group and source row; each halo carries role, IC lineage,
source hash, field map, conversion record and selection accounting.

The adapter deliberately returns the `partial_observation_inputs_v1` schema.
The manifest does not declare intrinsic stellar velocity dispersion or
metallicity, so these records cannot be passed as complete legacy 3D
`HaloData` or used to claim the corrected 3D baseline is available. A separate
audited schema extension is needed for those fields.
