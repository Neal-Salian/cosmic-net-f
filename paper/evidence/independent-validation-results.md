# Independent validation results ledger — 16 October 2026

**Real independent datasets evaluated: 0. Real external halos evaluated: 0. No generalization, independent-volume performance or cross-simulation transfer result is claimed.**

| Work item | Executed? | Evidence / result |
|---|---|---|
| Local catalog and checkpoint inventory | Yes | `experiments/paper_generalization/local-inventory.json`: 32 inspected file records; actual hashes/header/tensor details |
| Desktop/Downloads shallow discovery | Yes | `downloads-discovery.json`: same CSV hash; no independently identified external catalog |
| Native-vs-CSV semantic spot-check | Yes | `semantic-spot-check.json`: one target/radius match and strong member-count mismatch; not full-catalog certification |
| Existing notebook provenance inspection | Yes | B/C share primary CSV/checkpoint; 378/81/82 split, legacy target declaration; no new prediction evaluation |
| Real backbone/policy strict state load | Yes, no predictions | `artifact-load-check.json`: original backbone and recovered B policy load strictly at 64-d embedding; semantic certification remains blocked |
| Frozen external evaluation | **No** | Missing independent catalog, parameters/release review and audited original training receipt |
| Shipped experiment template rejection | Yes | CLI exits 2, prints `BLOCKED: manifest is pending_inputs...`, creates no output directory |
| Frozen software fixture integration | Yes, tests only | Tiny generated native HDF5 and randomly initialized test models; verified masks, predictions, exports and frozen hashes. Permanently `fixture_smoke` / `scientific_result=false`. No astronomy output retained |

The current 82-halo main result remains a **within-volume** evaluation. None of this work turns those halos into 82 independent simulation realizations. The independent-volume, cross-simulation-transfer and domain-generalization rows all remain **not evaluated/not established**.

## Executable handoff

`experiments/paper_generalization/README.md` contains the prespecified protocol, exact commands and interpretation rules. `required-inputs.json` lists file-level availability and requirements. The example manifest targets CAMELS IllustrisTNG CV_0, snapshot 090; its receipt and population contract deliberately remain pending. No astronomy data was downloaded. The native single-file adapter is ready for software execution, conditional on the actual acquired catalog satisfying its declared header/field contract; no claim that a specific unavailable release has passed it is made.

A real run exports `results.json` and `per_halo_results.csv` with every eligible halo/method, full/pruned targets/predictions, feasibility/failure records, exact budgets, frozen hashes, population and mass-support records. The committed `per_halo_results.csv` is **header-only (zero rows)**, paired with `run-status.json`. It is an export contract, not an empty successful experiment.

## Validation

The final test counts and commands are recorded in `experiments/paper_generalization/validation-report.md`. Coverage includes target/feature order/unit mismatches, lineage/content overlap, unknown provenance, selection mismatch, unknown mass support, incomplete native catalogs, fractional memberships, h disagreement, unhashed HDF5 dependencies, explicit fixture labels and an executable frozen fixture/export path.

An independent code review identified external HDF5 storage dependencies and incomplete multi-catalog evidence as material provenance gaps. Regression tests reproduced both; the adapter rejects external/soft links, virtual datasets and external raw storage, and receipts are restricted to one completely evidenced training catalog. Inventory header declarations no longer masquerade as scientific completeness. Code and training-evidence hashes are rechecked after evaluation.

The initial full-suite attempt lacked python-dotenv in a newly isolated runtime; installing that dependency resolved collection. A subsequent run stalled in a historical Git-object lookup because original Git metadata had cloud-only files. Final validation used a separate hydrated object store and Git metadata view, without editing original checkout contents. The exact Python package environment is archived in `validation-environment.txt`.

## Remaining scientific dependencies and limits

- Audit original backbone training transforms, full training/augmentation and model-selection population, split leakage, target/radius semantics and exact galaxy extraction order. A single central match and current policy split are insufficient.
- Acquire and checksum a genuinely disjoint compatible native catalog, actual parameter/IC lineage record and current release schema review. Do not infer existence from a loader or example manifest.
- Resolve the observed 7–10-member selection before comparing it with native groups containing thousands of galaxies. If neither supported selection rule matches training, extend the adapter with the verified rule; do not relabel the current input.
- Use TNG300 for better cluster support when feasible; acquiring and consolidating complete native chunks remains separate work. TNG100's local chunks 0 and 7 are neither separate volumes nor interchangeable local group tables.
- Report common-support coverage, all failures and paired uncertainty conditional on the external volume. One small volume does not measure volume-to-volume variance. Domain transfer can fail despite schema compatibility.
- Keep simulation-family transfer, cosmology/resolution shifts, temporal shifts and observational domain transfer separate. This package does not create an observationally validated mass estimator, a corrected stellar-radius model or broad domain-generalization evidence.

Draft-ready statement: **“External validation is prepared but has not been executed because an independently identified compatible catalog and a complete original-training provenance audit are not yet available.”**
