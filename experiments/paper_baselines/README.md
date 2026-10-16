# Frozen-backbone baseline and robustness package

The executed analysis uses existing weights and saved artifacts only. It does
not train, download, merge, push, or create a PR. All 82 original test halos
are retained at q=0.4/0.6/0.8/1.0 with exact k=ceil(qP) unordered nonself
physical pairs, retained loops, and no physical isolates.

## Reproduce

From the repository root, use a Python environment with the repository's
PyTorch, PyG, NumPy, SciPy, pandas, matplotlib, pytest, PyYAML, python-dotenv,
and requests dependencies. Exact versions used are in the run manifest.
The local execution used CPU float32 and one PyTorch thread.

```sh
python3 -m experiments.paper_baselines.run \
  --artifact-root /path/to/stageb-fix-campaign-20261015 \
  --input-dir /path/to/recovered-notebook-B-358545025/inputs/v3 \
  --out experiments/paper_baselines/results/new-run
python3 -m experiments.paper_baselines.analyze \
  --results experiments/paper_baselines/results/new-run \
  --evidence paper/evidence
python3 -m experiments.paper_baselines.verify \
  --results experiments/paper_baselines/results/new-run \
  --evidence paper/evidence
python3 -m pytest experiments/paper_baselines/tests -q
python3 -m pytest -q
```

An output directory must be new. Missing inputs yield exit 2 and a
`not_executed` manifest; mismatched identities or scientific contracts yield
a `failed` manifest. Neither status claims scientific results. Successful
results require every configured halo. A failed partial run must never be
analyzed as complete. No external downloads or fallback weights are used.

The artifact root must contain completed A/B/C folders at
`{A,B,C}-v1-terminal-001/artifacts/cosmic-net-runs/notebook_{A,B,C}/`:

- A: provenance.json and learned_control.pt.
- B: provenance.json, policy.pt, and recovered_training/training_checkpoint.json.
- C: provenance.json, matched_per_example.json, results_table.csv, budget_eligibility.json.
- The input directory: tng100_clustered.csv and best_model_augmented.pt.

The supplied paths are read only. All expected original checksums are
validated against stage C/A/B provenance. The predictor's state hash must
match all historical records. Rebuilt graph hashes and replayed predictions
must also match. No sample can be dropped because a selector performs poorly.

## Experiments and outputs

`results/run-20261016-final-v2/` is the executed run. The pre-review run at `results/run-20261016-final` retains its complete
predictions and masks, but is superseded by v2 for gradient forward accounting
and recovered-checkpoint checksum validation. Earlier failed preflight
manifests are retained as diagnostic records, not as experiment results.
See `review.md` for the independent review and verified fixes.

- `historical_replay.csv`: all 2,624 original GPU predictions and CPU replays,
  with targets, signed residuals and pair counts.
- `new_predictions.csv`: every new test prediction, residual, mask identity,
  and separately labeled seed role. No per-seed winner selection.
- `validation_predictions.csv`: every validation candidate and halo, including
  losing hyperparameters. Three geometry and three hybrid candidates at 40%.
- `masks.jsonl.gz`: all new validation/test physical masks, scores, tie marks,
  seed families, and exact-constraint diagnostics. These `selected_pairs`
  are boolean vectors; the original C artifact uses a list of selected indices.
- `audit.json`, `eligibility.json`, `split_ids.json`, `halo_properties.csv`,
  `hyperparameters.json`, `config.json`, `checkpoint_metadata.json`:
  the audit trail and train/validation-only choices.
- `probability_ranking_checks.csv`: sigmoid versus logit RL ranking checks;
  equivalent masks are not counted as an extra competitor.
- `matched_budget_table.csv`, `paired_comparisons.csv`, `seed_metrics.csv`,
  `per_halo_comparisons.csv`, `stratified_comparisons.csv`: complete matched
  comparisons, all budgets, all predefined strata, all competitors.
- `manifest.json`: run source, input/output checksums, environment, seed roles,
  forward accounting, and explicit executed/not-executed experiment registry.
- `analysis_manifest.json`: analysis source and generated evidence checksums.

The manuscript table, plot, audit and results are under `paper/evidence/`.
Only booktabs is needed to include `baseline-comparison.tex` in a manuscript.

## Scientific definitions

Geometry mixes percentile ranks of negative 3D and negative XY separation.
The hybrid mixes that geometry score with the velocity-binding proxy. Each
mixing coefficient is chosen on all 81 validation halos at q=0.4 from
{0,0.5,1}, then locked across test budgets. Velocity softening is the median
train pair speed (at least 1 km/s), and gradient feature scales are train
standard deviations. No test label is read by a scorer or tuner.

The velocity-binding proxy is log((Mstar_i+Mstar_j)/r) -
log(delta_v²+v0²), with physical Mpc, km/s and positive stellar masses. It is
an inexpensive physical control, not a certified dark matter binding model.
Alternative gradients compare absolute signed grad×input sums and absolute
gradients weighted by train feature standard deviation against the existing
sum of absolute grad×input contributions.

The lazy decoder returns exactly the production direct decoder's ranked
mask: test candidate priorities from highest to lowest and accept the first
candidate admitting an exact feasible completion. Once enough slots remain
to cover every uncovered node individually, no matching call is required.
The existing exact feasibility predicate remains the authority. All 2,296
saved masks are compared before any new experiment runs. For stochastic
policy decoding, fresh Gumbel keys per action give softmax choice over all
feasible actions; this is not uniform sampling over feasible final masks.

Seed families are distinct:

- Policy training: only the available trained seed 42; no retraining.
- Random masks: 100–109; one random vector per seed/halo, coupled across budgets.
- Policy sampling: 200–209; fixed trained logits, new categorical actions.
- Tie marks: 300–302; fixed logits with alternative independent tie priorities.
- Bootstrap: 20261016, 10,000 halo-paired resamples. Not training replication.

Repeated-selector loss averages squared errors over seeds *within each halo*
before halo bootstrap. It does not average predictions into an ensemble or
count 820 mask/halo records as 820 independent halos. Seed RMSE mean, SD and
range describe mask variability separately. Reported 95% intervals are
exploratory and have no multiplicity adjustment. All validation-tertile
strata are exported, including unfavorable and empty strata.

The checkpoint is legacy_total_radius with uncertified target and original
predictor training provenance. Selector split disjointness is verified;
original predictor leakage cannot be ruled out. Only 7–10-node graphs are
available. CPU/cached-score sensitivity timings are not deployment latency
or GPU speedup measurements. Independent policy training seeds, corrected
astronomy weights, external populations, and fresh GPU latency trials are
explicitly not executed.
