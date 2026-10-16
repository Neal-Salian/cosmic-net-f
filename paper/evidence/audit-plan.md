# Paper provenance audit plan

Goal: distinguish campaign identity from scientific provenance at base commit
`c825bf57652005c80faff79cf4dfd58e3a9baa14`, preserving all completed artifacts.

Execution is inline and autonomous as requested. The design is bounded:
validate contradictory catalog identities/targets before the legacy loader
selects a group's first mass; add read-only measurements and explicit evidence
classifications. Do not change features, labels, graphs or historical weights.

- [x] Isolate the exact base in a sibling worktree; establish test baseline.
- [x] Trace A–D manifests, CSV/HDF5 content, graph construction, original
  notebook/checkpoint, policy features and reward code.
- [x] Add regression tests first for duplicate IDs, conflicting group targets,
  inconsistent redundant mass columns, periodic geometry diagnostics and raw
  reference coverage; run them red, then implement and run green.
- [x] Produce exclusive-output audit CLI with hashes, reconstruction of legacy
  splits, direct raw field matching, geometry diagnostics, checkpoint metadata
  and optional campaign/per-example verification. Unknown coverage stays unknown.
- [x] Document claim classifications, manuscript limits and an executable
  separate baseline experiment using the existing audited training boundary.
- [x] Run focused and relevant regression tests; review diff, verify preserved
  artifact hashes, commit only this branch. No push, PR or integration.

Review focus: a file name does not establish completeness; empty reference chunks
must not imply global offsets; repeated targets within a group are expected but
conflicting ones are invalid; log masses need explicit units; deterministic halo
membership does not establish historical backbone independence.
