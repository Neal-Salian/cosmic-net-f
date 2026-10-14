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

- [ ] Reproduce overflow, configuration drift, API target conversion and periodic acceptance in regression tests.
- [ ] Implement finite log-domain physics calculations, zero-weight bypass and configured scheduling.
- [ ] Remove repeated magnitude-based unit conversion from already converted API/CSV records; preserve explicit legacy semantics.
- [ ] Reject periodic strict catalogs until supported minimum-image geometry exists.
- [ ] Run targeted tests; commit independently of other groups.

### Task 2: Supported audited full-graph baseline

- [ ] Add failing tests for audited manifest training, grouped splits, fixture labels and provenance-bound projected artifacts.
- [ ] Wire audited partial observations into the full-graph baseline without fabricating missing intrinsic features.
- [ ] Enforce strict catalog/target identity, supported schemas and fail-before-output behavior.
- [ ] Run tiny CPU smoke/integration tests; document real-data prerequisites; commit.

### Task 3: Canonical research execution and matched evaluation

- [ ] Add regression tests for the multiseed executable, missing checkpoints, mode-aware decoding and best-validation restoration.
- [ ] Repair entry points using shared modules and explicit reproducible seeds.
- [ ] Fix cost accounting and add supported physical/saliency/trained controls without false trained labels.
- [ ] Preserve raw per-graph matched metrics/common populations and report inference overhead.
- [ ] Verify and commit this execution/evaluation group.

### Task 4: Notebooks and artifact handoffs

- [ ] Preserve existing output blocks while marking them historical; update setup branch and source identities.
- [ ] Replace raw-column comparisons with the common physical-pair evaluator; explicitly bind requested training and inference budgets.
- [ ] Correct reciprocal physics statistics, metric reporting and TTA timing/validation-first gates.
- [ ] Validate A -> B -> C/D dependencies and config/checksum/source identity; reject ambiguous inputs and stale artifacts.
- [ ] Add cell-level integration and static notebook tests without executing full notebooks; commit.

### Task 5: Documentation, verification and review

- [ ] Document exact Kaggle inputs, order, outputs, limitations and remaining scientific blockers.
- [ ] Recheck authenticated GitHub metadata/compare without changing remote settings.
- [ ] Run available lightweight suite, syntax checks and git diff --check; compare preservation hashes.
- [ ] Obtain independent whole-branch review; fix important findings and verify again.
- [ ] Commit verified final changes and report implementation separately from scientific validation.

## Progress

- Initial branch and HEAD verified; new fix branch created in the existing checkout as requested. Untracked paper/ and output/ are left in place. No worktree or dependency setup is needed for this explicit in-place branch workflow.
