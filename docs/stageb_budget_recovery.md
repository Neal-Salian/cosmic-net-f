# Stage B exact-budget feasibility and policy recovery

The fix branch is `codex/fix-stageb-budget-feasibility`, based on
`aeb5dad65d681728d525fb78a4ab919b0432ab90`. It does not reinterpret archived
notebook outputs as executions of the edited source.

## Why training and validation succeeded

Stage A policy training retains its explicitly labelled legacy repaired-pair
environment. Policy selection evaluates validation graphs with the exact
decoder. Stage B subsequently asks that decoder for masks on training graphs,
whose topologies can have different feasible budgets. Validation success
therefore does not prove every training or test graph is feasible.

At fraction 0.4, 12 available physical pairs request `ceil(0.4 * 12) = 5`.
If the minimum edge cover has size 6, a symmetric five-pair graph with no
isolates is impossible. Self-loops do not cover physical isolates. The correct
decoder continues rejecting this request; it never increases five to six.

## Execution and comparison populations

`rls.notebook_budget.budget_cohort` checks only graph topology, the resolved
fraction and the constraint. It does not inspect targets, policy scores or
prediction errors. Its report includes every original graph ID, available,
requested and minimum pair counts, eligibility/reason and a physical-topology
hash. Actual selected pair counts remain null until masks are realized and
validated. Exact masks retain reciprocal/duplicate membership and architecture
loops, satisfy constraints, and retain exactly the requested physical pairs.

The original split manifest is unchanged. Execution populations are separate:

- B uses one feasible validation population for random controls and every
  policy-selection epoch. Insufficient validation/test cohorts fail before
  expensive training. Legacy Stage A training keeps its original population.
- Optional Stage B uses the explicitly recorded feasible training subset and
  validation subset. An empty training subset skips Stage B with a reason.
- B compares full/pruned/fine-tuned predictions on identical feasible test
  IDs, with exclusion counts and population labels. Physical retention and
  stored-column retention have separate fields.
- A/C keep original matched-benchmark failure rows and common-success
  populations. C's additional masks, physics and both coverage estimates use
  one aligned feasible diagnostic population. Empty optional diagnostics do
  not invent metrics.
- D resolves its own TTA fraction after restoring B's settings and applying
  explicit TTA overrides. Every K uses the same topology-defined validation
  or test population. Insufficient cohorts fail before adaptation.

`budget_eligibility.json` records these populations. B/C update realizations
from actual masks; D's TTA rows/histories record actual requested/final counts.
Preflight feasibility is not itself a realized selection. A's learned control
also explicitly excludes and records training graphs without physical pairs;
if none remain, it fails before teacher/training work.

Feasible-only evaluation is conditional on topology and may favor denser or
larger graphs. Report excluded IDs/counts and do not compare its headlines
directly with an older all-test population. C retains method-only/full-reference
evidence alongside common-population comparisons. No performance advantage is
established merely by successful execution of this fix.

## Recovery of failed Kaggle version 358545025

The selected policy, critic, logs and validation JSON were saved before Stage B
mask construction failed. Exact Notebook B version `v7` outputs were recovered
using the official Kaggle SDK. CLI 2.2.4's default output operation can resolve
an older successful output, so verify the returned session/version identity;
do not assume the default output belongs to the failed version.

The original policy SHA-256 is
`b1f41771c102bec644e0f84b0c002a228e5514321c768d8919e8beae3379f81d`.
Its selected epoch is 42 (zero-based); the log contains 60 completed epochs.
The original partial provenance remains `running`; it is not marked complete.

The original workflow did not export its final runtime configuration or fitted
reward scale. A new, separate reconstruction uses the exact original source,
immutable Git base, input bytes, split recipe and frozen backbone. Input hashes
were measured during recovery; they were not recorded by the original run.
Both inputs are byte-identical across all three available dataset versions.
The mounted historical version remains ambiguous without affecting those bytes.

Reconstruction checks the original backbone identity, validation IDs, all six
normalization buffers and a validation replay. The fitted scale is recomputed
from the original frozen full-graph training-RMSE recipe and explicitly labelled
reconstructed; CPU/GPU bitwise identity is not asserted. No policy/optimizer
training is repeated. Raw evidence and original scientific metrics remain
unchanged under `output/recovered-notebook-B-358545025/`.

## Rerunning on Kaggle

1. Use the reviewed fix commit, with identical code in A-D. Set the exact
   `EXPECTED_REPO_COMMIT`, explicit dataset inputs/units, and
   `ACKNOWLEDGE_LEGACY_INPUTS=True` for this legacy development campaign.
2. Use fresh output directories and preserve old output folders. Restart each
   kernel. If the Git fix is unpublished, deliver the verified local Git bundle
   privately and check out its commit before the standard checkout verification;
   do not run the old public branch and assume it contains the fix.
3. Run A to produce a source-compatible trained-control artifact and baseline
   rows. Preserve its complete output folder and manifest.
4. In B, set `RESUME_POLICY_DIR` to the verified folder containing
   `training_checkpoint.json` and its five immutable training artifacts.
   Resume into a new output directory. B verifies inputs, splits, backbone,
   training architecture/budget/config and artifact hashes, loads normalization
   buffers unchanged, records training versus evaluation source identities,
   and skips the policy-training loop. It then reruns controls/validation,
   optional Stage B and test evaluation on the declared exact-feasible cohorts.
5. B preserves the original training handoff in `recovered_training/` and writes
   a separate `recovery_record.json`. Only a newly completed evaluation campaign
   receives a complete Notebook B manifest. An old minimal `running` manifest
   alone is rejected. New training runs export `training_checkpoint.json`
   immediately after policy validation, before optional Stage B.
6. Run C with complete A/B folders and then D with complete B outputs. Keep the
   predeclared fractions/constraints and validation-only K selection. A negative
   result is valid evidence; do not tune on test outcomes.

Preserve code/source hashes, checkpoint hashes, reconstructed-metadata evidence,
budget reports, per-example successes/failures, validation decisions, training
and inference timings, result tables, manifests and output ZIPs. Recovery of
selected policy weights supports evaluation; optimizer/RNG states were not
saved, so bit-for-bit training continuation is unsupported.
