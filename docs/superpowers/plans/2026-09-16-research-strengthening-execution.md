# Research strengthening: scoped execution addendum

This addendum executes the approved research plan/design in this directory and `../specs/2026-09-16-research-strengthening-design.md`. The user's implementation request supersedes their earlier planning-only/no-commit constraints and authorizes the first four implementation stages, small commits, and a bounded pilot. It does not authorize an expensive research campaign, push, merge, PR, history rewrite, or publication/novelty claims.

Starting state: clean `fix/rl-pruning-symmetry` at `ac22c86`. Work proceeds in the existing checkout on the requested `feature/research-strengthening` branch. This avoids adding a separate worktree/consent workflow; the source branch remains unchanged.

## Global constraints

- Preserve committed data/notebook evidence; reusable Python modules own new core behavior.
- TDD for behavioral changes; reproduce and trace existing failures before fixes.
- All scientific budgets count unordered physical non-self pairs. Reciprocal copies share membership. Report requested/sampled/final counts and constraints; never hide infeasibility or silently expand a matched budget.
- Node relabeling must preserve behavior; symmetric ties require an explicit exchangeable-random or tie-inclusive convention, not numeric node IDs. Transport random marks when testing coupled equivariance.
- No fabricated external catalog data, units or metadata. Legacy total-radius checkpoints remain explicitly legacy; corrected stellar-radius models require suitable data and retraining.
- Small pilot only: three seeds, at most 64 training halos, ten epochs, three budgets. Synthetic optima are diagnostic evidence only. No full campaign, symbolic regression or major TTA sweep.
- Every output records model/data/split/config provenance and whether it is synthetic, legacy development or corrected research data.

### Task 1: Pair constraints and invariant selection foundation

Inspect `rls/pair_policy.py`, `rls/sparsify.py`, `rls/train_policy.py`, existing mask/reward tests and the current node-number repair counterexample. Add failing tests before changing behavior. Implement reusable physical-pair layout validation, no-isolate/connected feasibility and exact-budget selection (including direct feasible sequential sampling and a parameter-independent scaffold control). Preserve exact ordered-action likelihood and explicitly distinguish final mask probability. Avoid expensive likelihood calculations in deterministic inference. Preserve existing public callers where practical and remove numeric-node/edge-order repair preferences in legacy paths with a declared tie policy. Tests must cover node relabeling, reversal/duplicates, loops, infeasible stars/isolates, exact budgets, probability normalization and exact-gradient checks on small enumerated examples. Fix the two existing connectivity-contract failures by asserting the revised physical semantics rather than removing coverage.

Deliver a documented API for budget calculation, constrained sampling, pair marks and diagnostics. Planned files: `rls/constraints.py`, `rls/constrained_policy.py`, `rls/pair_policy.py`, `rls/sparsify.py`, relevant tests. Do not modify graph/data modules or benchmarks in this task.

### Task 2: Graph/data/augmentation/provenance foundation

Inspect graph construction, TNG/CAMELS loaders, original augmentation, checkpoint metadata and training. Test then fix missing portable graph construction, node-order-dependent nearest-neighbor ties and global RNG side effects where reproduced. Extract joint position/velocity rotation and jitter/rebuild augmentation into a reusable module. Preserve explicit radius semantics: do not substitute unavailable stellar radii in the legacy CSV. Add a strict research catalog contract with explicit target/units metadata and train-only/grouped split manifests/checkpoint hashes. Establish a reusable full-graph baseline entry point with physics proxy disabled by default for corrected research work; do not relabel or overwrite old weights. A lightweight synthetic baseline smoke is allowed; real corrected training remains conditional on suitable data. Planned files: graph builder, `data/augmentation.py`, loaders/provenance/training support and focused tests.

### Task 3: Fair matched-budget benchmark

Build a reusable benchmark module around Task 1 APIs. Each comparator receives identical final per-graph pair count and constraint; infeasible rows report failure explicitly. Support random, distance/degree/stellar-binding scores, real predictor saliency, provided RL scores, a train-only supervised selector and a structured/differentiable selection control appropriate to the supported model interface. Do not call heuristic/stub scores published trained methods. Include a full-graph reference, requested/actual physical and stored-edge counts, model/data/split hashes, predictor calls, synchronized runtime, prediction metrics and per-example outputs. Separate pretrained/frozen and retrained model labels. Add paired summary support and deterministic seeds/marks; test fairness, accounting and no parameter/gradient mutation during evaluation. Reconcile same-budget K=0/K>0 plumbing without an adaptation sweep.

### Task 4: Bounded feasible-selection pilot

Implement a reusable driver and CLI/config for three seeds, <=64 training halos, ten epochs and three feasible budgets. Compare repaired sampling (with explicit projection/accounting), common scaffold/residual selection, direct feasible sampling, and a supported structured selection control under comparable predictor-call accounting. Use actual likelihoods for sampled actions; train normalization on training data only. Add tiny synthetic problems with enumerated feasible masks and exact optima; bound ordered enumeration to <=8 pairs and length<=4. Save objective/regret, prediction errors, all budget counts, calls, time, duplicate masks and provenance. Run the small synthetic pilot and, if accessible, a clearly labeled legacy-halo development smoke; do not fabricate corrected real-data results. Test CLI caps, exact optima and separation of train/validation data.

### Task 5: Astronomy robustness infrastructure

Prepare manifest-driven real CAMELS-IllustrisTNG development and SIMBA external evaluation, including separate external-development/confirmation roles and simulation/initial-condition grouping. Fail closed for missing catalogs/units/fields; no download or synthetic substitute is implicit. Add configurable projected observations, missing members, physical noise, broader-richness selection and graph rebuilding. Never use target-derived R200c for normalization. Document units/observer conventions and transform provenance. Add a configuration and dry-run/audit CLI plus tests using explicitly synthetic fixtures. No external scientific results are claimed without actual catalog runs.

### Task 6: Integration, verification and delivery

Run affected focused tests after each task. Review each task for spec/quality and resolve important findings before dependent work. Run the complete feasible suite and lightweight notebook/CLI smokes; report actual failures/skips/environment limitations. Run and archive the capped pilot diagnostics. Update usage docs without publication claims; preserve old artifact labels. Create small commits, conduct final independent branch review, and report starting/final branch+HEAD, changed files, bugs/decisions, tests/results, smokes, commits, remaining GPU experiments, deferrals and final git status. Keep branch local; do not push/merge/open a PR.
