# Paper generalization implementation plan

Goal: determine the fastest defensible independent validation before the 17 October 2026 draft, implementing the user's supplied mission in an isolated branch from c825bf5.

Architecture: keep production training and loaders unchanged. Inventory local files; add a local-only, fail-closed native Subfind experiment for one independent CAMELS IllustrisTNG CV volume, with frozen intrinsic-feature backbone/policy and exact matched budgets. Require an audited training receipt; never infer compatibility from tensor widths. TNG300 remains the preferred mass-support comparison but needs a complete multi-file ingestion product.

Global constraints: no original checkout edits; no merge, push, or PR; no synthetic astronomy results; no target/schema coercion; snapshots and catalog chunks are not independent volumes.

- [x] Audit loaders, local catalogs, native headers, checkpoints, notebook artifacts and primary data specifications.
- [x] Write failing tests for semantic mismatch, unknown provenance, shared ICs, incomplete catalogs, integer membership and file changes.
- [x] Implement contracts, native single-file adapter, inventory and frozen matched evaluator; export all per-halo outcomes and failures.
- [x] Execute the inventory and pending-input rejection. Execute real evaluation only if every gate passes.
- [x] Write compatibility matrix, independent protocol, results ledger and limitations.
- [x] Run targeted and available repository tests; independent code review; commit only this branch.

Review focus: same-width projected/intrinsic features; legacy raw-unit training ambiguity; same IC under another suite/snapshot alias; zero successful matched halos; partial or stale HDF5 inputs. Test fixtures verify software only.

Review: independent reviewer reproduced unhashed HDF5 dependency and incomplete multi-catalog evidence gaps. Fixed with RED→GREEN regression tests, self-contained native fields and a single fully evidenced training catalog. Unknown identity placeholders, source changes and the observed population selection mismatch also fail closed. No scientific evaluation was possible.
