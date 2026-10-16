# Statistical analysis of the completed A–D campaign

The saved predictions reproduce the starting results and the prior paired
analysis exactly. The evidence does **not** establish an RL advantage over
gradient × input or random selection, and the measured RL inference path is
slower than the full graph. All hypothesis tests here are exploratory.

## Evidence and independent verification

The source report is
`output/stageb-fix-campaign-20261015/scientific-analysis-20261016/Cosmic-Net-scientific-report-20261016.md`
in the original checkout. Official stage evidence comes from the corresponding
`{A,B,C,D}-v1-terminal-001/artifacts/cosmic-net-runs/notebook_{A,B,C,D}/`
directories. Exact original paths, byte counts and SHA-256 hashes are in
`inputs/manifest.json`; lossless snapshots are committed beside it.

The local scripts independently calculate metrics from targets and predictions,
validate graph IDs/content hashes, and reconstruct pair-to-column masks.

| Evidence | Verification | Limit |
|---|---|---|
| A: 2,296 per-halo records, 28 configurations | All RMSE, MAE, R², bias, residual standard deviation, retention and mean component timings agree with `baselines.csv` within 1.12e−16 | Repeated methods/budgets are not independent halos |
| C: 2,624 records, 32 configurations | Same checks against `results_table.csv`, within 1.12e−16; A/C graph identities and baseline RMSE agree | One fitted policy and one random test-mask realization |
| A/C masks | All 1,968/2,296 non-full selections have exact budgets, symmetric reciprocal columns, all loops retained and no physical isolates | No model inference was rerun |
| B frozen results | Agree with C within 3e−7, reflecting float32 versus float64 reductions | No saved B final per-halo prediction export |
| B optional fine-tuning | Six log epochs match provenance; epoch 2 passes the recorded 2% full-validation tolerance; 377/81/82 eligible cohorts | Fine-tuned test RMSE remains aggregate evidence |
| D adaptation | Four K rows; identical 82 graph IDs and exact counts; times and mean steps match histories within 1e−12; K=20 minimizes saved validation RMSE | Final test/validation predictions are absent, so aggregate accuracy cannot be independently reconstructed |
| Original 21 paired comparisons | RMSE differences, bootstrap endpoints/SE, MSE differences, leave-one-out limits, permutation p-values and both Holm corrections agree exactly (maximum difference 0) | Reproduction of a post hoc design does not make it confirmatory |

The catalog snapshot has 5,378 rows and 541 distinct halo groups. It matches
the execution CSV SHA-256
`6c148f2cae68abac7bc5c34d370ff0130a647b49e79c32880bcf36ecc0ab4d53`.
All A–D input/checkpoint identities, split IDs and frozen-backbone hashes agree.
Splits are disjoint: 378 training, 81 validation, 82 test halos. Test graphs have
15–45 physical pairs (median 45). The experiment seed is 42; the recovered
60-epoch policy selected zero-based epoch 42. The fitted policy was not retrained.
The optional Stage B exclusion is `tng100_0000005`; it does not remove test halos.

## Retention and experimental units

Let Pᵢ be the number of available distinct unordered non-self physical pairs in
halo i. At nominal retention q, every pruned method retains Bᵢ=ceil(qPᵢ).
Realized physical retention is Bᵢ/Pᵢ. All reported mean retentions are arithmetic
means of these per-halo fractions, not a ratio of pooled counts. Architecture
self-loops do not contribute to Pᵢ. Reciprocal columns represent one physical
pair. With two reciprocal columns per pair and Lᵢ loops retained, the stored
fraction is (2Bᵢ+Lᵢ)/(2Pᵢ+Lᵢ).

At nominal q=.4/.6/.8/1, the mean realized physical retentions are
40.30%/60.24%/80.26%/100%. At q=.4, the mean stored-column fraction is 46.69%.
The full reference always retains 100%, even when its saved record is tagged
with a nominal sweep q. It is never a matched pruned-budget comparator.
D uses q=.5, realized physical retention 51.01%, and stored-column retention
56.25%. Its q=.5 results must not be treated as matched to C's q=.4.

The sampling unit is one halo, n=82. The 2,624 C observations are repeated
measurements of those halos. Edges, orientations, budgets, bootstrap draws,
permutations and the ten random **validation** seeds are not extra test units.

## Estimands and deterministic statistical design

Residual eᵢ=predictionᵢ−targetᵢ is in dex (legacy log10 mass target). RMSE is
sqrt(mean(eᵢ²)); MAE is mean(|eᵢ|). The primary effect reported here is
ΔRMSE=RMSE_RL−RMSE_control, in dex; negative values favor RL. Percentage
improvement is 100(1−RMSE_RL/RMSE_control). This is a relative RMSE reduction,
not a percentage of halo mass. MAE differences and per-halo win fractions are
additional descriptive effects and estimate different quantities.

The existing comparison design is preserved without changing its test statistic:

1. Match the entire sets of halo IDs; sort IDs lexicographically; reject
   duplicates, mismatched targets, changed graph content and nonfinite values.
   Do not silently intersect populations or pool budgets.
2. For each q=.4,.6,.8, compare RL with controls in this exact order:
   degree, distance, gradient × input, full graph, random, stellar binding proxy,
   trained selector. This gives 21 contrasts; q=1 duplicates the full graph and
   is excluded from inference.
3. For contrast index j=0,…,20, take 50,000 **paired** halo bootstrap samples,
   using NumPy `default_rng(20261016+2j)`, SciPy `bootstrap`, `paired=True`,
   BCa, 95% two-sided confidence, batch 2,500. Both residual vectors receive
   the same resampled halo indices. These are pointwise, unadjusted intervals.
4. Use 99,999 within-halo label swaps of **squared errors**, SciPy
   `permutation_test(permutation_type='samples')`, RNG seed 20261017+2j,
   batch 2,500, and a two-sided **mean squared error difference** statistic.
   This tests exchangeability of the paired losses; its p-value is not obtained
   by permuting a nonlinear RMSE statistic. SciPy doubles the smaller tail and
   applies its +1 Monte Carlo correction. Resolution is 2/100,000=.00002.
5. Preserve both Holm families: seven controls at q=.4, and all 21 controls
   across the three pruned budgets. Each includes the full reference. Do not
   choose a more favorable family after seeing results. No replacement FDR
   correction or uncorrected significance designation is introduced.

See the primary [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
and [paired permutation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
Implementation and exact seeds are in `../scripts/analyze_results.py` and
`paired-comparisons.csv`. Reproduction used Python 3.13.3, NumPy 2.5.2 and
SciPy 1.18.1 (the exact Python patch version is also saved in `analysis.json`).

New relative-effect intervals use paired BCa bootstrap with the same index RNG
streams as the corresponding absolute effects, applying the relative statistic
to each paired resample. They do not alter the original tests or corrections.
Halo absolute-error win fractions have descriptive 95% Wilson intervals,
with ties included in the denominator; they are not additional significance
tests. All effects/intervals and ΔMAE are available in the machine-readable CSV.

## Starting numerical values

| Quantity | Independently recomputed |
|---|---:|
| Full RMSE | 0.114796698524 dex → 0.114797 |
| RL RMSE, q=.4 | 0.136160448590 dex → 0.136160 |
| Gradient × input RMSE | 0.136899632465 dex → 0.136900 |
| Trained selector RMSE | 0.147382198484 dex → 0.147382 |
| Random RMSE | 0.156571040459 dex → 0.156571 |
| RL − gradient RMSE | −0.000739183875 dex |
| 95% BCa interval | [−0.014507749282, +0.013581788713] dex |
| Paired permutation p | 0.9192 |
| RL − random RMSE | −0.020410591869 dex |
| 95% BCa interval | [−0.050979068906, +0.003193310050] dex |
| Paired permutation p | 0.1819 |
| Test halo count | 82 |

All supplied rounded values are verified. B's full RMSE is
0.114796645939 from float32 aggregation; C's independently saved predictions
give the value above. Both round to 0.114797. The new analysis does not overwrite
either source or substitute B reductions for the established C comparisons.

## Effects and interpretation

| RL minus control, q=.4 | ΔRMSE (dex) | Relative improvement (%) | 95% BCa relative interval (%) | Holm p, 7 | Holm p, 21 |
|---|---:|---:|---|---:|---:|
| Gradient × input | −0.000739 | +0.54 | [−10.58, +10.33] | .9192 | 1 |
| Random | −0.020411 | +13.04 | [−2.69, +27.32] | .5457 | 1 |
| Trained selector | −0.011222 | +7.61 | [+1.08, +14.28] | .15096 | .64158 |
| Degree | −0.015781 | +10.39 | [−5.59, +24.57] | .5457 | 1 |
| Distance | −0.033309 | +19.65 | [+5.59, +30.50] | .0775 | .279 |
| Stellar binding proxy | −0.071968 | +34.58 | [+23.59, +42.47] | .00028 | .00084 |
| Full reference (100%) | +0.021364 | −18.61 | [−30.48, −9.11] | .00132 | .0044 |

These relative intervals are unadjusted; an interval excluding zero does not
imply a multiplicity-adjusted discovery. The original MSE permutation test and
the RMSE BCa interval also target different functionals. Table 3 gives all 21
absolute-effect intervals and both preserved corrections.

At q=.4 RL is 18.61% worse in RMSE than the full graph. Relative to gradient,
RL improves absolute error on 42/82 halos and worsens it on 40/82; its MAE is
slightly worse (0.109011 versus 0.108918 dex). Removing one halo yields an
RL-minus-gradient RMSE difference between −0.003273 and +0.001172 dex, so
the point-estimate ranking is sensitive to individual halos. A non-significant
comparison is not evidence of equivalence or noninferiority; no equivalence
margin was specified. At q=.6 the strongest point control is random; at q=.8
distance outperforms RL in RMSE. This ranking is descriptive and was observed
on already-inspected test data.

After Holm correction, the q=.4 differences against the weak binding proxy
and against the full reference retain exploratory evidence. The latter favors
the full graph. Neither finding establishes a general advantage over strong
controls. There are no predeclared confirmatory results in this campaign.

## Runtime, fine-tuning and adaptation

C's mean q=.4 RL inference region is 0.440260 s/halo versus 0.007839 s/halo
for the full reference: 56.16× slower. RL scoring, exact decoding and retained
prediction are approximately 0.01182, 0.41823 and 0.00844 s/halo; total region
time includes additional overhead. Slowdowns at .6/.8/1 are 71.17/78.31/82.81×
relative to the respective saved full measurements. RL has two live predictor
forwards per halo, including the full embedding pass, compared with one for
the full reference. Graph construction (0.851806 s for the built split set),
data ingestion, setup and historical training are excluded. Timing samples
span different graphs, not repeated hardware trials. No timing significance
test or error bars are claimed. Exact accelerator model, peak memory and
historical RL training wall time are unavailable; unknown cost is not zero.

Stage B selected epoch 2 on validation: pruned RMSE 0.123302→0.122764,
full RMSE 0.117958→0.119724 (within the 2% tolerance). Frozen versus fine-tuned
test RMSE is 0.114797→0.118644 (full) and 0.136160→0.137266 (pruned).
This negative test result is retained. Primary C figures use the unchanged
frozen backbone, not the fine-tuned model.

D selected K=20 using validation RMSE 0.127205476165, slightly below K=10
(0.127205505967). Relative to K=0, test RMSE changes
0.131123229861→0.130391493440 dex, Δ=−0.000731736422 dex, a 0.558% point
improvement. Latency rises 0.468060→1.169164 s/halo (2.498×), with 6.195 mean
actual steps due to early stopping. K=10 and 20 are almost indistinguishable
on test. The validation choice is preserved. D's archive contains aggregate
CSV files and adaptation histories, but no final per-halo predictions/residuals.
No paired bootstrap, significance test, or accuracy uncertainty for D can be
calculated from the aggregate RMSE values. Final masks were also not exported;
count/constraint diagnostics are verifiable, their topology cannot be rechecked.

As descriptive diagnostics, saved physics alignment has mean/median per-halo
Spearman rho 0.376/0.410 and 78/82 positive. These are not causal evidence.
Thirty-draw MC-dropout intervals cover 100% of the 82 targets, with mean widths
1.9596 dex (full) and 1.9499 dex (RL). These predictive intervals are distinct
from confidence intervals for method differences and are not used as figure
error bars or evidence of calibrated uncertainty.

## Scope, missing inputs and results text

Halo resampling assumes independent/exchangeable halos, conditional on one
simulation volume, one fitted policy/backbone, graph construction and split.
Spatial dependence and shared simulation conditions can invalidate a simple
halo-population interpretation. Permutations additionally require the paired
loss labels to be exchangeable under the null; equal mean loss alone does not
guarantee this condition. No independent training-seed or volume replication,
multiple random **test** masks, external/OOD evaluation, or physics ablations
were saved. The dataset is explicitly `legacy_total_radius` with uncertified
target/acquisition/radius semantics and incomplete backbone training provenance.
Input hashes establish artifact identity, not astrophysical validity or absence
of historical backbone leakage.

Needed for stronger claims: raw B/D final predictions and D masks, independent
training seeds and random test masks, repeated synchronized hardware timings
with hardware identity/memory/training costs, and audited independent-volume
test data. No missing values were synthesized, and no paired significance tests
were derived from aggregate metrics. `../results.tex` is generated, manuscript-ready
results text preserving these qualifications.
