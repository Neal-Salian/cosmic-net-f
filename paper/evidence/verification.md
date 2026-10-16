# Verification record — 16 October 2026

Base commit: `c825bf57652005c80faff79cf4dfd58e3a9baa14`. All work executed from the isolated
`codex/paper-provenance-audit` worktree.

## Automated checks

- Initial focused baseline: **72 passed** (physics/data, data foundation,
  provenance foundation and notebook pipeline).
- New scientific validation regressions: **25 passed**. Red/green evidence
  included missing validation implementation, duplicate/alias IDs, conflicting
  labels, redundant target mismatch, periodic-neighbor changes, raw membership/
  coverage scope, absent/mismatched raw groups and changed aggregate source
  identity despite an agreeing commit.
- Final entire configured repository suite: **829 passed, 13 skipped,
  695 warnings in 34.96 seconds**. Command: `python3 -m pytest -q
  --basetemp=.audit-tmp/pytest-final`. Skips belong to optional/environment
  paths; warnings are existing PyTorch/PyG/Python deprecations. This suite
  follows pytest.ini (backend tests are excluded). No claim that skipped or
  excluded tests passed.
- Standalone prospective baseline command with the new example config:
  fails on the absent real manifest before creating its output directory, as
  intended. The independent reviewer also checked its configuration validator.
  No scientific training is represented by this check.

## Artifact and numerical checks

- A/B/C/D: all completion, commit, CSV/checkpoint/state identity, split IDs,
  aggregate source identity, declared Python source and artifact checks pass.
  Recomputed full base source identity:
  `8a05077184564309a97d339ff4773727cad7f2a400e50b4fb79c053af0f66545`.
- C q=0.4 targets, graph content hashes and exact unique full/RL populations
  match reconstruction. Saved residual arithmetic gives full RMSE
  `0.11479669852430192` dex and
  RL RMSE `0.13616044859036275` dex.
- Current CPU frozen replay: `0.11479662526768547` dex.
  Archived-loader/current-graph diagnostic test:
  `0.11671666960606404` dex;
  validation: `0.11749383528813533` dex
  versus checkpoint metadata 0.11749375611543655 dex. Numerical agreement
  supports the recipe trace and does not certify historical train exclusion.
- Raw content: 20 matching rows/two groups; incomplete and invalid files remain
  explicitly labelled. Final CSV/HDF5 match all 18 columns.
- Consumed campaign manifests/artifacts and primary CSV/checkpoint were
  rehashed unchanged. No existing source data, weights, notebooks or outputs
  were written. The audit's newly generated trial directories were removed;
  only the selected new measurement report is retained.

Measurement file SHA-256: `111566789baf637548d2d11ac0c3dae0ac7a393bbbd67260fdfb3d16ff882660`.

## Review and local Git filesystem limitation

Independent read-only review identified two important reporting gaps (raw group
coverage and aggregate source identity), both fixed with regression tests.
Final review found no actionable issues in code, documents, status JSON or the
prospective baseline config.

Ordinary Git reads later encountered local shared-store `mmap failed: Operation
 timed out` errors, including in otherwise passing source-identity tests. These
were environment access failures, not successful scientific checks. Final
verification used a worktree-local bare read cache with the copied loose Git
objects verified against their object SHA-1 identities and copied pack bytes;
no shared object was repaired, overwritten or removed. The cache's detached
HEAD was explicitly the base commit. Final source/tests/audit subprocesses used
`GIT_DIR="$PWD/.audit-tmp/source.git"` and
`GIT_OBJECT_DIRECTORY="$PWD/.audit-tmp/git-objects"`. This changes the location of
readable immutable Git objects, not their contents or the source revision.
The cache is temporary and is not part of the committed research evidence.

## Scientific limit

Passing code tests, agreeing hashes and reproducing predictions establish the
scoped measurements. They do not certify the remaining raw acquisition,
training separation, projected-policy validity or external generalization.

## Exact changed files

- `config/paper-provenance-baseline.example.yaml`
- `data/loaders/tng_loader.py`
- `data/scientific_validation.py`
- `scripts/audit_paper_provenance.py`
- `tests/test_paper_provenance.py`
- `paper/evidence/audit-plan.md`
- `paper/evidence/backbone-audit.md`
- `paper/evidence/data-and-target-definition.md`
- `paper/evidence/limitations-for-manuscript.md`
- `paper/evidence/provenance-audit.md`
- `paper/evidence/provenance-measurements.json`
- `paper/evidence/provenance-status.json`
- `paper/evidence/verification.md`

The same filesystem problem affected staging through the shared
`.git/info/exclude`. Staging/diff/commit creation therefore used the temporary
worktree-local metadata and an isolated copy of this audit worktree's index.
The resulting content-addressed objects are to be added without overwriting
existing shared objects, and the audit branch advanced from the exact base
with an atomic compare-and-swap ref update. Only this audit worktree's index
is restored to the committed tree; the original checkout index is untouched.
No source artifact or scientific dataset is changed by this Git workaround.
