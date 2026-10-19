# Paper statistics and figures implementation plan

**Goal:** Turn the completed legacy A–D campaign into reproducible manuscript artifacts.

**Spec:** The user mission in this chat: six supported figures, five LaTeX tables,
independent numerical verification, the existing paired design and correction,
no fabricated data, and no changes to the original checkout or notebook outputs.

**Architecture:** Lossless gzip snapshots preserve numerical source bytes and a
SHA-256 manifest. A small NumPy/SciPy analysis module validates evidence and
recomputes statistics. Matplotlib and plain LaTeX exporters consume that analysis.
Only `paper/` and focused statistical tests change. Execute inline.

**Constraints:** Branch `codex/paper-statistics-figures` from
`c825bf57652005c80faff79cf4dfd58e3a9baa14` in an independent worktree.
Commit locally; do not merge, push or open a PR. Preserve 50,000 paired BCa
bootstrap draws, 99,999 within-halo permutations, seed/order and both Holm families.
Treat all post hoc tests as exploratory; D aggregates cannot support paired tests.

## Tasks

- [x] Inspect prior scientific report, original analysis code, A–D provenance,
  CSVs, per-halo exports, and D archive contents; establish clean isolated branch.
- [x] Snapshot necessary evidence with lossless gzip and original-byte hashes.
- [x] Add focused tests for strict pairing, exact small-sample permutation,
  Holm adjustment, and mask/physical-budget integrity; observe failures first.
- [x] Implement `paper/scripts/analyze_results.py`: verify hashes, splits,
  all A/C metrics and masks, B aggregate consistency, D counts/timing/selection,
  then reproduce all 21 original contrasts and calculate interpretable effects.
- [x] Implement `generate_figures.py` and `generate_tables.py`, export six
  PDF/PNG figures and five numbered LaTeX tables (Table 3 includes all 21 tests).
- [x] Write statistical analysis, ready-to-use results text, figure captions,
  provenance, missing inputs, environment and regeneration instructions.
- [x] Run numerical checks, focused tests, repository suite, regeneration/hash
  comparison and visual QA; review the branch and commit all deliverables.

## Review focus

1. Pair by exact halo IDs, reject missing/duplicate IDs and target mismatches.
2. Validate selected-pair indices, reciprocal columns, loops and exact ceil(qP).
3. Full reference always retains 100%; never plot its nominal sweep tag as retention.
4. C q=.4 and D q=.5 are separate experiments; no cross-budget significance.
5. Floating-point differences, unadjusted CIs, repeated budgets, unavailable
   training seeds, and hardware/timing scope must remain explicit.

## Execution ledger

Ruling: Proceed directly under the user's detailed authorized mission, without
an additional design approval loop; keep the implementation small and inline.
Ruling: Commit 4.4 MB of lossless evidence snapshots for portable reproduction,
including the legacy CSV; do not copy model weights or entire notebook archives.
The CSV's presence verifies identity and counts, not acquisition semantics.

Validation completed: all 21 prior contrasts match exactly, all 21 generated
products regenerate byte for byte, 808 tests passed and 13 skipped, all five
LaTeX tables compile without overfull boxes, all figures/tables visually checked,
and an independent review found no actionable issues. Source preservation is
recorded in source-audit.json; detailed results are in validation.json.
