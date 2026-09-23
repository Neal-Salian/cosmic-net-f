# Local astronomy manifest audit

`data.astronomy_manifest.load_astronomy_manifest(path)` parses a version-1 YAML manifest and validates each catalog's explicit role, simulation and initial-condition lineage, epoch, conventions, strict target, type-4 stellar mass/radius field map, units, shape declarations, selection, source kind, local path, digest syntax, and release declaration. Unknown keys and non-string identity values fail closed. Component shapes must use three position/velocity coordinates, six particle-type values for type-4 mass/radius arrays, and rank-one group membership. The parser does not open catalog data. Relative paths resolve from the manifest directory.

Run the read-only content audit with:

```sh
python -m data.astronomy_audit --manifest config/astronomy_manifest.yaml --json
python -m data.astronomy_audit --manifest config/astronomy_manifest.yaml --role development
```

The audit checks each file's SHA-256 before opening it, confirms declared HDF5 datasets, exact shapes and numeric dtypes, checks header `h`/`HubbleParam` and `Time` against the manifest, and validates integer group membership bounds and aligned subhalo lengths. Successful reports and JSON output identify `verification_level: content_verified`; failures use a nonzero exit code and structured error entries. The optional Python API `inspect_hdf5=False` yields `schema_only`, which must not be used for scientific evaluation. Call `report.require_content_verified()` at a scientific-use boundary. The audit does not create output/cache directories or substitute data.

The shipped `config/astronomy_manifest.example.yaml` contains placeholders and is intentionally not evidence that any catalog is available. Test fixtures are synthetic-only. The current implementation audits metadata and membership; strict row conversion and physical unit conversion are reserved for Task 5a2.
