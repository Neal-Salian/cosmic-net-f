# Controlled research campaign readiness

This implementation starts from `f66bb570496e6062e755521512432d8476e7160e`
on `fix/research-pipeline-readiness`. Software tests and small synthetic
fixtures establish execution contracts, not astrophysical performance.

## Evidence and supported paths

- The archived full-GNN test result is R² 0.9075, RMSE 0.1167 dex, MAE
  0.0921 dex on 82 legacy test graphs. Notebook B's archived `f538726` run
  reports RL RMSE 0.134322 dex versus full RMSE 0.116717 dex: a 15.1% increase.
- The capped K4 experiment is a synthetic optimization diagnostic with eight
  unique validation graphs. Its original files are immutable. Three seeds
  do not establish superiority or astronomy transfer.
- A–D support an explicitly acknowledged **legacy-development** campaign,
  with common physical-pair evaluation. They cannot certify the historical
  catalog's target definition, the original backbone training split, or
  historical augmentation. Corrected transforms can change the old result.
- The audited projected full-graph baseline is a separate supported training
  path. See `audited_baseline.md`. It never fills absent dispersion or
  metallicity with invented observations.

## Before running A–D on Kaggle

1. Finish local review and verification, then make the verified fix branch
   available remotely through your normal authorized workflow. This work
   does not push. Upload the edited notebooks; old notebook copies contain
   old source even when their clone cell fetches newer code.
2. Record the exact final commit and set `EXPECTED_REPO_COMMIT` in every
   notebook. Use `REPO_BRANCH="fix/research-pipeline-readiness"` (or the
   actual reviewed branch after integration). Restart each kernel.
3. Attach the actual `best_model_augmented.pt` and exactly one matching CSV
   folder, or set `INPUT_DIR` explicitly. An LFS pointer is not a checkpoint.
   Ambiguous or missing inputs stop execution.
4. For historical development, set `ACKNOWLEDGE_LEGACY_INPUTS=True` in all
   four notebooks. This does not upgrade legacy evidence to corrected
   research. Set the same explicit `LEGACY_CSV_UNITS`, seed, graph settings
   and inputs throughout. The checked-in historical CSV uses positions and
   total radii in ckpc/h, log10 stellar Msun, and linear halo Msun. Already
   normalized files need their explicit schema; units are never guessed from
   magnitudes. API-exported normalized CSVs carry their own units marker.
5. Choose fractions and `evaluation_constraint` before looking at test
   results. Budgets mean `ceil(q * P)` distinct non-self physical pairs.
   Self-loops are architecture edges. Infeasible budgets are reported or
   rejected; they are not increased until a method works.
6. Choose a **fresh** `OUTPUT_DIR` per stage/run. The setup rejects existing
   directories. Preserve the input folders and every completed output.
7. Run A, then B, then C and D. A writes `learned_control.pt`, baseline
   summaries, per-example rows and a manifest. B writes the selected policy,
   validation record, logs and optional accepted Stage B artifact. C needs
   complete A and B folders; D needs the complete B folder. Set
   `A_ARTIFACT_DIR`/`B_ARTIFACT_DIR` explicitly if multiple runs are attached.
8. Each ZIP lives **inside** its stage output directory. Download the ZIP
   and preserve the complete folder/manifest. Source contents, input hashes,
   units, graph configuration, split IDs and backbone identity must match.

`SMOKE_TEST=True` caps graph counts/epochs and uses `fixture_smoke` labels;
all four stages must agree. Smoke weights/results are not scientific data.
No full notebook or expensive campaign was executed to validate these edits.

## Comparison contract

A/C use `rls.notebook_benchmark.matched_notebook_rows`, which delegates to
`evaluate_matched_budget`. All scorers receive the same graph, pair layout,
final budget, constraint and independent tie marks. Reciprocal/duplicate
stored copies share membership; loops do not consume the physical budget.
Failures and the intersection of successful graph IDs are saved. Headline
RMSE, MAE and R² use that common population; method-only population values
remain separately named. A full reference is stored per budget stratum.

Controls are random, distance, degree, stellar binding proxy, edge-feature
gradient×input and a **trained raw selector distilling train-only saliency**.
Normalization and teacher labels use training graphs only; label predictor
calls are recorded. This control is not attention, PGExplainer or an oracle.
The former untrained attention proxy and feature-scaling Gumbel control are
excluded from primary comparisons. C loads A's actual trained artifact and
evaluates every method together with supplied RL; it does not concatenate
incompatible old tables.

Stage A policy training retains the historical repaired PL action/objective,
explicitly distinguished from exact constrained inference. Validation uses
one fixed final training budget, with best actor and critic restoration.
Valid policies are retained even when validation improvement is absent;
negative results must remain visible in C. Stage B is exploratory and does
not replace the frozen primary reference.

D adapts a copy of the policy without targets, selects K on validation and
uses identical exact **final** budgets at K=0 and K>0. Adaptation rollouts
remain the labelled repaired PL environment. Time includes fresh embedding
extraction, policy/adaptation, decoding and retained-graph prediction; graph
construction is separately recorded. The primary matched evaluator similarly
includes scoring (including full embeddings and policy passes), decoder and
prediction. No speedup is inferred from edge retention. Unknown historical
RL training cost remains null rather than zero. Common preprocessing and
training cost must be included in any later end-to-end/amortized claim.

Physics alignment collapses reciprocal/duplicate copies first. Within-graph
rank tests are descriptive; physical pairs sharing nodes are not independent
experimental units. Use independent IC/halo groups for confirmatory inference.

## Physics, catalogs and limitations

The virial penalty remains a **proxy**, not a calibrated halo equilibrium
measurement. It preserves the squared-ratio penalty in its numerical range,
computes in log space, and saturates outside the configured natural-log
ratio bound (default 20, maximum 30). Saturated physics gradients are zero;
MSE still guides predictions. Zero lambda bypasses physics observations.
Clipping frequency is recorded and must be checked during real experiments.
`training.lambda_*` and `training.warmup_epochs` control the scheduler.

Strict periodic catalogs are rejected. Do not simply relabel a periodic box
as nonperiodic: use independently verified unwrapped halo coordinates with
a content-audited preparation record, or implement minimum-image geometry.
Legacy TNG API conversion supports snapshot 99 only. Raw API mass conversion
occurs once before grouping. Legacy TNG/CAMELS targets are not interchangeable;
C/D refuse the old GroupMass transfer route.

Remaining blockers include audited real catalog procurement, compatible
provenance-complete trained projected policy artifacts and a supported
projected reward/training design. Legacy intrinsic policy rewards require
unavailable intrinsic fields and cannot be used by inventing those fields.
Independent real-data repetitions, strong baseline comparisons, uncertainty
calibration, domain transfer and practical speedups remain unexecuted.

Inherited deployment architecture/preprocessing parity, random-weight fallback,
symbolic dimensional checking and explanation labels are outside this focused
pipeline fix. They must not support claims of production/scientific readiness.

## Upstream contribution

Git ancestry permits integration with upstream `main` at `486b74d`; this does
not establish GitHub fork-network membership. The upstream is private and
the personal public repository reports `fork:false`; authenticated
cross-repository comparison returned 404 during the audit. Verify the route
with the owner; do not change visibility/permissions as a workaround.
Scope an eventual PR as infrastructure/correctness work, with historical
results clearly distinguished from experiments still to run.
