# Figure and table provenance

All figures use the official C matched benchmark with the frozen backbone.
D appears only in the separately labelled Figure 4(b) adaptation panel and
Table 5. No result was digitized from an image. The existing scientific report
was inspected before implementation; its original analysis code and statistical
results are preserved under `inputs/previous-analysis/` for independent comparison.

`inputs/manifest.json` lists every original absolute path, uncompressed SHA-256,
compressed SHA-256 and byte count. Its 24 snapshots decompress to the exact
source bytes. Source A/C per-halo JSON contains graph layouts and masks as well
as predictions; it is preserved in full rather than reduced to aggregate metrics.
Check snapshots with `analyze_results.py`; original-location preservation checks
and archive identities are in `source-audit.json`. No source notebook, artifact,
or original checkout file is modified by these scripts.

## Reproduction and numerical products

Run from the repository root, using `paper/scripts/requirements.txt`:

```bash
python3 paper/scripts/generate_figures.py
python3 paper/scripts/generate_tables.py
```

The first script recomputes `analysis.json`, `paired-comparisons.csv` (21 rows),
and `per-halo-error-differences.csv` (1,722 paired rows). The second rejects stale
analysis and writes tables and results prose. All inputs are local, committed
snapshots; default execution never reaches back into the original checkout.
Python/library versions and statistical seed streams are recorded in
`analysis.json`. Figures use Matplotlib 3.11.1, embedded TrueType fonts, vector
PDF and 400 dpi PNG. PDF timestamps are disabled for deterministic regeneration.
Lines connect measured retention points as a visual guide; no interpolated
experiment results or missing uncertainty estimates are manufactured.

## Figure 1 — prediction performance versus retention

Files: `../figures/figure_1_retention_performance.{pdf,png}`.
Sources: `inputs/C/matched_per_example.json.gz`, checked against
`inputs/C/results_table.csv.gz`. Transform: compute RMSE in dex over the same
82 halos for each method at nominal q=.4,.6,.8,1; plot against mean **realized
physical-pair** retention. Seven available pruned methods are shown. The dashed
full-graph RMSE is a separately labelled 100% reference, not a method measured
at 40% retention. No error bars: curves are point estimates for one fitted model.

Suggested caption: **Test prediction performance versus realized physical-pair
retention.** Each pruned method retains ceil(qPᵢ) distinct non-self pairs per halo,
with reciprocal columns and architecture loops preserved. Mean realized retention
differs slightly from nominal 40/60/80/100% due to integer rounding. All points
use 82 common test halos; the full graph retains 100%. Connecting lines are
guides. No training-seed uncertainty is available.

## Figure 2 — matched physical-pair budgets

Files: `../figures/figure_2_matched_budgets.{pdf,png}`.
Sources: same C predictions and aggregates. Transform: horizontal dot plots of
method RMSE at each nominal pruned budget (.4,.6,.8). Each panel labels nominal
q and mean realized physical retention. The dotted vertical line is RL RMSE in
that panel. X-axis scales differ between panels; within-panel method comparisons
are matched on both graph IDs and exact per-halo physical-pair counts.

Suggested caption: **Accuracy at matched physical-pair budgets.** All seven
methods use identical per-halo budgets and the same 82 test halos. Nominal and
realized retention are labelled separately. Dotted lines mark the RL point
estimate. Panels have different RMSE scales. The strongest baseline varies
with budget; apparent point differences are not significance claims.

## Figure 3 — degradation relative to the full graph

Files: `../figures/figure_3_degradation.{pdf,png}`.
Sources: C predictions, including each sweep's full-reference records.
Transform: RMSE(method,q)−RMSE(full,q), in dex, plotted versus realized physical
retention. This is a difference of aggregate RMSEs, not RMSE of prediction
differences or an absolute per-halo error difference. Zero is the full reference.
No uncertainty bands are added. Floating-point residual differences near q=1
are preserved in data and round to zero at figure precision.

Suggested caption: **RMSE degradation from the full-graph reference.** Positive
values indicate worse accuracy than the full graph on the same test halos.
Retained physical pairs exclude self-loops and reciprocal duplicates. All
methods approach full-graph accuracy at 100% retention; curves represent one
fitted-model realization.

## Figure 4 — accuracy versus measured inference latency

Files: `../figures/figure_4_accuracy_latency.{pdf,png}`.
Sources: C per-halo inference-region timings/predictions; D `tta_results.csv`,
`tta_validation.csv`, `tta_histories.json` and provenance.
Transform: (a) C nominal 40% method RMSE versus mean region time, plus the full
100% reference; logarithmic seconds/halo axis. (b) D nominal 50% results across
K=0,5,10,20, linear seconds/halo axis. K=20 is selected on validation; near-coincident
K=10/20 points are labelled together. The panels are separate budget experiments.
No accuracy or latency error bars are supported by these single-execution summaries.

Suggested caption: **Measured accuracy–latency tradeoff.** (a) At nominal 40%
retention, the full graph is both faster and more accurate than the RL inference
path. Region time includes embedding/scoring, exact decoding and prediction,
and excludes graph construction, setup and historical training. (b) At nominal
50% retention, validation-selected adaptation yields a small descriptive test
RMSE change at higher latency. Hardware is recorded as CUDA; exact accelerator
identity and repeated-run uncertainty were not saved. Panel axis scales differ.

## Figure 5 — paired per-halo absolute-error differences

Files: `../figures/figure_5_per_halo_differences.{pdf,png}`.
Sources: C raw targets/predictions, exported to
`per-halo-error-differences.csv`. Transform: at q=.4 calculate
|e_RL,i|−|e_control,i| in dex, against gradient × input, random and trained selector.
Every panel uses the same 82 halos in lexicographically sorted graph-ID order;
rank is an identifier, not a physical halo property. Blue/negative favors RL;
orange/positive favors control. All halo IDs and values remain in the CSV.
No per-halo confidence intervals are identifiable from one prediction pair.

Suggested caption: **Per-halo paired absolute-error differences at nominal 40%
retention.** Negative differences favor RL. The same ordered 82 halos appear
in all panels; the ordering does not encode mass or graph size. Win counts are
descriptive and differ from the RMSE comparison, which emphasizes squared errors.
D is excluded because its final per-halo predictions were not saved.

## Figure 6 — supported paired comparison intervals

Files: `../figures/figure_6_paired_confidence_intervals.{pdf,png}`.
Sources: C raw predictions, independently recomputed in `analysis.json` and
`paired-comparisons.csv`; checked exactly against the prior 21-contrast analysis.
Transform: ΔRMSE=RL−control and unadjusted paired 95% BCa endpoints for all seven
controls at .4/.6/.8. Shared x-axis; zero denotes no difference. Full graph is
marked † because it retains 100%, unlike the six matched pruned controls.
Only genuinely supported paired comparisons appear. No D or aggregate-only tests.

Suggested caption: **Exploratory paired RMSE differences with pointwise 95% BCa
intervals.** Intervals resample 82 halos together, using 50,000 draws. Negative
differences favor RL. †Full graph is a 100% reference. These are unadjusted
confidence intervals, not simultaneous intervals or indicators of corrected
significance. Table 3 gives the corresponding paired MSE permutation p-values
and both pre-existing Holm corrections. No confirmatory inference is claimed.

## Tables and text

All five table files are complete, full-width LaTeX `table*` environments using
`booktabs` and legible small type; labels are `tab:cosmic-1` through `tab:cosmic-5`.

| Product | Numerical evidence and transformation |
|---|---|
| `../tables/table_1_dataset_setup.tex` | Catalog byte identity/row/group counts; A–D provenance/splits; C test pair counts; B cohorts and selected policy epoch; analysis design |
| `../tables/table_2_primary_benchmark.tex` | Recomputed C RMSE for all 32 configurations, q=.4 MAE/R²; realized retentions in note; full reference explicitly labelled |
| `../tables/table_3_statistical_comparisons.tex` | All 21 independent recomputations of paired ΔRMSE/BCa CI, original MSE permutation statistic and Holm-7/Holm-21 families |
| `../tables/table_4_runtime.tex` | C q=.4 component/total timing means and live predictor-forward counts, ratios to the full reference; unknown training/memory/hardware data not replaced with zero |
| `../tables/table_5_adaptation.tex` | D saved aggregate test/validation metrics; histories independently verify times, steps and exact counts; original K=20 validation choice retained |
| `../results.tex` | Generated prose using the verified primary effects, C runtime and aggregate-only D change with explicit limits |

See `statistical-analysis.md` for assumptions, supported effects and absent inputs.
