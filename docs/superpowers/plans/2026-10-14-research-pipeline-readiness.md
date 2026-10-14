# Research Pipeline Readiness Implementation Plan

> **For agentic workers:** Use incremental test-first implementation and independent review. The user's implementation request authorizes execution, commits, and independent subagents; no push, merge, PR, expensive training, or full notebook execution.

**Goal:** Make the existing pipeline reproducible and scientifically honest enough for a controlled experiment campaign.

**Architecture:** Reuse physical-pair layouts and the matched evaluator as the common evaluation boundary. Keep historical repaired-policy training explicitly labelled; freeze its validation budget. Support corrected training only through audited, explicitly selected schemas. Reject unsupported periodic geometry rather than inventing conventions. Notebook handoffs must bind code, data, split, checkpoint and evaluation configuration.

**Tech stack:** Python, PyTorch/PyG, pytest, JSON notebooks, existing installed dependencies.

**Spec:** The user's 2026-10-14 implementation request and the preceding evidence-based repository audit.

## Global Constraints

- Start at f66bb570496e6062e755521512432d8476e7160e on fix/research-pipeline-readiness.
- Preserve historical commits, checkpoint bytes, saved notebook outputs, experimental evidence, paper/ and output/.
- No dependency installation, expensive training, full notebook execution, push, merge or PR.
- Smoke/fixture evidence must never be labelled a scientific result.
- Missing or incompatible scientific inputs fail explicitly; no random-backbone or synthetic-data fallback.
- Use exact final physical-pair budgets, reciprocal selection and common graph populations for primary comparisons.

## Review Focus

- Unknown RL historical cost must stay unknown, not zero; inference includes embedding/policy/decoding overhead.
- Low or infeasible pair budgets fail visibly, with common-success populations reported.
- Legacy weights/data may reproduce historical results only with explicit legacy labels.
- Strict partial astronomy schemas must never acquire fabricated dispersion or metallicity.
- Notebook outputs are historical evidence and are not evidence of execution of edited source.

## Tasks

### Task 1: Physics and ingestion correctness

- [x] Reproduce overflow, configuration drift, API target conversion and periodic acceptance in regression tests.
- [x] Implement finite log-domain physics calculations, zero-weight bypass and configured scheduling.
- [x] Remove repeated magnitude-based unit conversion from already converted API/CSV records; preserve explicit legacy semantics.
- [x] Reject periodic strict catalogs until supported minimum-image geometry exists.
- [x] Run targeted tests; commit independently of other groups.

### Task 2: Supported audited full-graph baseline

- [x] Add failing tests for audited manifest training, grouped splits, fixture labels and provenance-bound projected artifacts.
- [x] Wire audited partial observations into the full-graph baseline without fabricating missing intrinsic features.
- [x] Enforce strict catalog/target identity, supported schemas and fail-before-output behavior.
- [x] Run tiny CPU smoke/integration tests; document real-data prerequisites; commit.

### Task 3: Canonical research execution and matched evaluation

- [x] Add regression tests for the multiseed executable, missing checkpoints, mode-aware decoding and best-validation restoration.
- [x] Repair entry points using shared modules and explicit reproducible seeds.
- [x] Fix cost accounting and add supported physical/saliency/trained controls without false trained labels.
- [x] Preserve raw per-graph matched metrics/common populations and report inference overhead.
- [x] Verify and commit this execution/evaluation group.

### Task 4: Notebooks and artifact handoffs

- [x] Preserve existing output blocks while marking them historical; update setup branch and source identities.
- [x] Replace raw-column comparisons with the common physical-pair evaluator; explicitly bind requested training and inference budgets.
- [x] Correct reciprocal physics statistics, metric reporting and TTA timing/validation-first gates.
- [x] Validate A -> B -> C/D dependencies and config/checksum/source identity; reject ambiguous inputs and stale artifacts.
- [x] Add cell-level integration and static notebook tests without executing full notebooks; commit.

### Task 5: Documentation, verification and review

- [x] Document exact Kaggle inputs, order, outputs, limitations and remaining scientific blockers.
- [x] Recheck authenticated GitHub metadata/compare without changing remote settings.
- [x] Run available lightweight suite, syntax checks and git diff --check; compare preservation hashes.
- [x] Obtain independent whole-branch review; fix important findings and verify again.
- [x] Commit verified final changes and report implementation separately from scientific validation.

## Progress

- Initial branch and HEAD verified; new fix branch created in the existing checkout as requested. Untracked paper/ and output/ are left in place. No worktree or dependency setup is needed for this explicit in-place branch workflow.

- Completed test-first physics/units, strict projected baseline, canonical execution/evaluation and notebook groups in separate commits. Historical outputs remain archival; no new scientific results are claimed.
- Independent review found three population/device issues: raw rejected halo membership, common-success CSV metrics, and notebook CPU-to-GPU transfer. Regression tests reproduce each original failure and verify the fixes. Non-smoke notebooks also require an immutable checkout pin.
- Future pilot validation now uses fixed marks across epochs; scaffold sampled counts exclude deterministic actions. The legacy cross-source CLI fails before loading incompatible targets.
- Final lightweight suite: 757 passed, 13 skipped in 36.75 seconds. Python AST parsing (147 files), compilation (58 A-D code cells), entry-point help and branch-wide whitespace checks passed. CUDA and optional integration tests remain skipped locally; no full Kaggle notebooks or expensive training were run.
- Preservation checks: all 87 protected non-notebook files and all A-D execution counts/output blocks match the initial snapshot. Untracked paper/ and output/ remain untouched and unstaged.
- Remaining scientific blocker: the corrected projected observations do not supply the intrinsic quantities required by the old RL reward. A provenance-complete projected policy/reward design and real-data experiments remain necessary. No surrogate intrinsic fields or random scientific checkpoints were introduced.
- Authenticated GitHub metadata showed private upstream, a public personal repository outside its fork network, and a 404 cross-repository compare. No permissions, visibility or remote settings were changed. No push, merge or PR was performed.
