Cosmic-Net scientific analysis — completed A–D legacy campaign, 16 October 2026

All four private Kaggle v1 runs completed. Their declared artifacts, source/input identities, dependencies, graph populations and exact budgets pass the cross-stage audit. This establishes a working legacy research pipeline; it does not establish a publishable RL advantage. No policy retraining, Git push, merge or PR was performed in this continuation.

The local branch is **codex/fix-stageb-budget-feasibility**, HEAD **c825bf57652005c80faff79cf4dfd58e3a9baa14**. The tracked worktree was clean; output/ and paper/ remain untracked and preserved. Fork remote research/readiness heads remain f66bb570496e6062e755521512432d8476e7160e and aeb5dad65d681728d525fb78a4ab919b0432ab90; the new fix branch is local only. Verified upstream HEAD remains main at 486b74dd6bbf6dc04583deae6be42fb997631dea.

**Execution and artifact evidence.**

| Stage | Official output session | Purpose | Evidence |
|---|---:|---|---|
| [A/v1](https://www.kaggle.com/code/nealsalian/cosmic-net-stageb-fix-a-20261015) | 358697902 | Full graph, heuristics and trained raw control | [manifest](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/A-v1-terminal-001/artifacts/cosmic-net-runs/notebook_A/provenance.json) · [ZIP](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/A-v1-terminal-001/artifacts/cosmic-net-runs/notebook_A/notebook_A_full_outputs.zip) |
| [B/v1](https://www.kaggle.com/code/nealsalian/cosmic-net-stageb-fix-b-20261015) | 358708561 | Recovered policy, validation and Stage B | [manifest](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/B-v1-terminal-001/artifacts/cosmic-net-runs/notebook_B/provenance.json) · [ZIP](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/B-v1-terminal-001/artifacts/cosmic-net-runs/notebook_B/notebook_B_full_outputs.zip) |
| [C/v1](https://www.kaggle.com/code/nealsalian/cosmic-net-stageb-fix-c-20261015) | 358713276 | Unified matched-budget benchmark and diagnostics | [manifest](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C/provenance.json) · [ZIP](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C/notebook_C_full_outputs.zip) |
| [D/v1](https://www.kaggle.com/code/nealsalian/cosmic-net-stageb-fix-d-20261015) | 358778452 | Validation-selected test-time adaptation | [manifest](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/D-v1-terminal-001/artifacts/cosmic-net-runs/notebook_D/provenance.json) · [ZIP](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/D-v1-terminal-001/artifacts/cosmic-net-runs/notebook_D/notebook_D_full_outputs.zip) |

[All-four artifact/metric audit](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/analysis-ABCD-official-20261016-001.json) reports valid=true with no errors. Server notebook code cells match the private execution copies. C’s official ZIP is byte-identical to the downloaded C ZIP. The generic A ZIP in Downloads is an older aeb5dad artifact and was not mixed into this campaign; C instead depends on the verified c825bf5 A manifest. D requires B’s policy/config/manifest plus the CSV and frozen backbone; A/C are comparison/analysis inputs, not dependencies of the TTA algorithm.

The supplied catalog contains 5,378 rows and 541 halo graphs, split 378/81/82. Every stage has the same source identity, CSV/checkpoint hashes, split IDs and frozen model state. Core input and policy identities are:

| Artifact | SHA-256 |
|---|---|
| CSV | 6c148f2cae68abac7bc5c34d370ff0130a647b49e79c32880bcf36ecc0ab4d53 |
| Frozen backbone file | 42764f0db204863b28672d6ec6e207443602de16dcfa926765a0429b3451c63f |
| Recovered selected policy file | b1f41771c102bec644e0f84b0c002a228e5514321c768d8919e8beae3379f81d |

**A and B: what was actually validated.**

A trained a raw selector by train-only saliency distillation and evaluated it alongside random, distance, degree, stellar-binding proxy and gradient×input controls. This is a trained control, not PGExplainer, attention, or an untrained selector. B reused the original 60-epoch policy checkpoint (selected epoch 42, zero-based). Its original weights/logs remain unchanged. Missing training metadata, including the fitted reward scale, were explicitly reconstructed and validation-replayed; this supports evaluation recovery, not bitwise optimizer/RNG continuation.

The Stage B failure was a genuinely infeasible TRAIN graph, tng100_0000005: P=12 available physical pairs, ceil(0.4P)=5 requested, minimum no-isolate edge cover=6. The low-level decoder correctly rejects five. Topology-only preflight records this graph as excluded from optional Stage B, leaving 377/378 TRAIN graphs. All 81 validation and 82 test graphs remain eligible. The original policy training population stays 378. Requested budgets were not enlarged.

Stage B completed six fine-tuning epochs and accepted its epoch-2 candidate using validation. Validation pruned RMSE changed 0.123302→0.122764; full validation RMSE changed 0.117958→0.119724, within the predeclared 2% tolerance. Test performance degraded: full 0.114797→0.118644 dex and pruned 0.136160→0.137266 dex. This negative generalization result is retained; C’s primary comparisons correctly use the unchanged frozen backbone. [B results](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/B-v1-terminal-001/artifacts/cosmic-net-runs/notebook_B/policy_results.csv)

B’s validation PASS is an execution/selection gate: RMSE below the random-mask validation mean at identical budgets/populations, with no isolates. It is not a significance test or evidence that the test policy outperforms the full GNN. Ten random validation seeds are not ten policy training seeds or ten random TEST realizations.

**C: verified accuracy at matched physical-pair budgets.**

C evaluated 32 configurations on the same 82 test-split halos with zero failures. All 2,296 non-full selections satisfy exact ceil(qP) counts, pair symmetry, preserved architecture loops and no-isolate constraints. The 2,624 raw method/budget records are repeated measurements of 82 halos, not 2,624 independent experimental units. The full graph is a separately labelled reference retaining all pairs.

| Method | RMSE q=40% | q=60% | q=80% | q=100% |
|---|---:|---:|---:|---:|
| Full graph | 0.114797 | 0.114797 | 0.114797 | 0.114797 |
| Random | 0.156571 | 0.128369 | 0.118196 | 0.114797 |
| Distance | 0.169469 | 0.135384 | 0.117740 | 0.114797 |
| Degree | 0.151941 | 0.132786 | 0.119166 | 0.114797 |
| Stellar binding proxy | 0.208129 | 0.147724 | 0.119173 | 0.114797 |
| Gradient × input | 0.136900 | 0.137050 | 0.122469 | 0.114797 |
| Trained raw selector | 0.147382 | 0.133855 | 0.125910 | 0.114797 |
| RL policy | 0.136160 | 0.127760 | 0.121663 | 0.114797 |

RMSE is in dex for the legacy log-mass targets. At 40%, full RMSE/MAE/R² are 0.114797/0.090928/0.910491; RL gives 0.136160/0.109011/0.874075. RL’s RMSE is 18.61% higher than full. Its point gain over gradient×input is only 0.54%, while its MAE is slightly worse (0.109011 versus 0.108918). At 60%, the gain over the strongest control (random) is 0.47%; at 80%, distance beats RL (0.117740 versus 0.121663). At 100%, methods recover the same full-graph accuracy within floating-point noise. [C full results](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C/results_table.csv) · [Per-halo raw evidence](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C/matched_per_example.json)

At nominal 40%, mean realized physical retention is 40.30%; stored-column retention is 46.69% because loops are retained and reciprocal copies are stored. Integer rounding is shared by every method on each graph.

![Accuracy and cost](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/scientific-analysis-20261016/accuracy-and-inference-cost.png)

**Is the 40% improvement statistically meaningful?**

The new analysis pairs predictions by graph ID and resamples halos together: 50,000 paired BCa bootstrap draws for RMSE differences and 99,999 within-halo label-swap permutations of squared errors, RNG seed 20261016. Statistical self-checks verify pairing, exact small-sample permutation behavior and Holm correction. Methods follow the official [SciPy paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) and [paired-samples permutation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) interfaces. These are post hoc exploratory tests conditional on independent/exchangeable halos and one fitted policy; they do not account for policy-seed or simulation-volume uncertainty.

| RL minus control at q=.4 | RMSE difference (dex) | Unadjusted 95% BCa CI | Paired p | Holm p, 7 contrasts | Holm p, 21 contrasts |
|---|---:|---|---:|---:|---:|
| Gradient × input | -0.000739 | [-0.014508, +0.013582] | 0.9192 | 0.9192 | 1 |
| Random | -0.020411 | [-0.050979, +0.003193] | 0.1819 | 0.5457 | 1 |
| Trained raw selector | -0.011222 | [-0.022461, -0.001826] | 0.03774 | 0.15096 | 0.64158 |
| Degree | -0.015781 | [-0.043872, +0.006982] | 0.26796 | 0.5457 | 1 |
| Distance | -0.033309 | [-0.057936, -0.008931] | 0.0155 | 0.0775 | 0.279 |
| Stellar binding proxy | -0.071968 | [-0.095711, -0.044928] | 4e-05 | 0.00028 | 0.00084 |
| Full graph | +0.021364 | [+0.011135, +0.032705] | 0.00022 | 0.00132 | 0.0044 |

Negative differences favor RL. The strongest-control interval spans both benefit and harm; RL improves absolute error on 42/82 halos versus gradient×input and worsens it on 40/82. Deleting one halo can reverse the sign of their RMSE gap. The apparent 13.04% gain over the single random test-mask realization also has a CI crossing zero and p=0.1819. Nominal evidence against the trained raw control/distance does not survive the seven-comparison correction. The binding heuristic is much weaker and RL beats it; that does not establish superiority over competitive controls. Full-graph accuracy is better in this conditional analysis. Lack of significance is not proof of equivalence. [Methods, assumptions, all 21 contrasts and leave-one-out sensitivity](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/scientific-analysis-20261016/paired-comparisons.json)

![Paired uncertainty](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/scientific-analysis-20261016/q40-paired-uncertainty.png)

**D: adaptation succeeded, but its effect is small and expensive.**

D uses requested q=.5, with 51.01% mean realized physical-pair retention after ceil rounding. Every K uses exactly the same per-graph final pair counts and all 82 test graphs; no isolates or budget excess occurred. D’s CSV keep_frac=56.25% is the stored-column fraction including loops, not physical-pair retention. D’s 50% results must not be presented as a matched-budget improvement over C’s 40% results.

| K | Validation RMSE | Test RMSE | Test MAE | Test R² | Seconds/halo | Actual mean steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.127539232 | 0.131123230 | 0.102724284 | 0.883220 | 0.468060 | 0.000 |
| 5 | 0.127231658 | 0.130803317 | 0.102795720 | 0.883789 | 1.034797 | 4.756 |
| 10 | 0.127205506 | 0.130391479 | 0.102651909 | 0.884520 | 1.166199 | 6.146 |
| 20 (selected) | 0.127205476 | 0.130391493 | 0.102651902 | 0.884520 | 1.169164 | 6.195 |

K=20 was selected on validation only. Relative to K=0, test RMSE improves 0.558% (-0.000731736 dex), at 2.498× latency. K=10/20 have essentially identical test results; the original validation choice is retained. Early stopping means K=20 runs about 6.20 actual steps. Adaptation receives graphs without target labels, operates on policy copies, and the frozen backbone hash remains unchanged. Adaptation rollouts retain the labelled legacy repaired action environment; final inference uses the exact decoder.

D exports aggregate metrics, counts and adaptation histories, but no per-halo final predictions or residuals. A paired significance test for its tiny accuracy change cannot be recovered from those exports. No statistical TTA-improvement claim is justified; future runs should save predictions, masks and paired residuals. No D failure, code fix, policy retraining or repeat of A–C was necessary. [D results](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/D-v1-terminal-001/artifacts/cosmic-net-runs/notebook_D/tta_results.csv) · [D histories](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/D-v1-terminal-001/artifacts/cosmic-net-runs/notebook_D/tta_histories.json)

![Adaptation accuracy and cost](/Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015/scientific-analysis-20261016/tta-accuracy-and-cost.png)

**Efficiency, physics and uncertainty limitations.**

C’s 40% RL inference region is 0.44026 s/halo versus 0.007839 s for the full reference: 56.16× slower. About 0.41823 s is exact decoding; scoring costs 0.01182 s and retained prediction 0.00844 s. Slowdown rises to about 71×/78×/83× at 60/80/100%. These timings include embedding extraction, two orientation policy passes, decoding and prediction. Graph construction (0.8518 s for the complete built split set), ingestion/setup and historical training are separate. Recovered RL training cost is unknown, not zero; the learned control’s 378 teacher calls are not its entire optimization cost. No amortized speedup or peak-memory improvement was measured.

The architecture also imposes a structural cost: RL needs a full-GNN embedding forward before a second, pruned prediction forward. Raw C rows record two live predictor forwards for RL versus one for the full reference. Optimizing the CPU decoder alone cannot establish a one-shot speedup over the same full GNN; a cheaper encoder, valid computation reuse or explicit offline/amortized use case is needed. See [metered RL scorer](/Users/nealsalian/Desktop/cosmic-net/rls/benchmark_scorers.py:415). Current test graphs have 15–45 physical pairs, median 45. The repository fetcher caps satellites at ten ([fetcher setting](/Users/nealsalian/Desktop/cosmic-net/scripts/fetch_tng_data.py:25)); large-graph scaling is untested.

Physics alignment is descriptive: mean per-halo Spearman rho=0.376, median=0.410, with 78/82 positive. Reciprocal copies are collapsed into physical pairs. Edges still share nodes, and stellar mass-product/distance is a proxy; within-graph Mann–Whitney p-values are not independent evidence for a causal dark-matter mechanism. No physics-reward/constraint ablation establishes which ingredient causes an accuracy gain.

The 30-draw MC-dropout intervals cover 100% of these 82 targets, with mean widths 1.9596 dex (full) and 1.9499 dex (RL). These broad, uncalibrated intervals do not establish useful 95% calibration. The Gaussian mean±1.96σ construction needs independent calibration and sharpness/NLL/coverage evaluation across mass/richness and OOD conditions.

**Scientific scope and publication verdict.**

The controlling label is legacy_total_radius. The catalog contract explicitly leaves the historical target definition unknown and identifies total-subhalo radius rather than audited type-4 stellar radius. Input/checkpoint hashes establish identity, not original acquisition semantics or absence of backbone training/test leakage. The current random halo split within one TNG volume does not provide independent-volume generalization. Only one fitted policy seed and one random TEST realization were evaluated. CAMELS/SIMBA and observation-shift evaluations were disabled. [Legacy catalog contract](/Users/nealsalian/Desktop/cosmic-net/data/loaders/tng_loader.py:258)

A corrected, content-audited projected full-graph training path exists separately. It requires real HDF5 manifests and independent initial-condition groups. Its projected inputs omit unmeasured dispersion/metallicity; the historical intrinsic-feature RL reward/trainer cannot be reused by matching tensor dimensions or zero-filling missing fields. A scientifically validated projected policy objective/trainer remains an implementation blocker. An alternative intrinsic-feature study needs an audited schema for every actual feature and a newly trained compatible backbone/policy. [Audited baseline and policy blocker](/Users/nealsalian/Desktop/cosmic-net/docs/audited_baseline.md)

GNN halo-mass inference is established prior work ([Villanueva-Domingo et al.](https://arxiv.org/abs/2111.08683)); RL sparsification ([SparRL](https://arxiv.org/abs/2112.01565)) and supervised neural sparsification ([NeuralSparse](https://proceedings.mlr.press/v119/zheng20d.html)) also predate this project. These are context, not matched reimplementations. A publishable contribution must identify and validate what exact constraints, physical objectives or astronomy transfer add beyond those ideas. Different published dex errors cannot be compared directly to this legacy split.

The project is a working ML engineering/research prototype with an executed legacy campaign. Methods and limitations can be drafted now. The current evidence supports neither a general RL superiority claim nor an efficiency claim, and is insufficient for a credible research submission. This does not prove the approach cannot work; it specifies which claims remain unsupported.

**Remaining work before a research submission, in priority order.**

1. Certify real catalog provenance, M200c/unit/radius semantics, periodic geometry and selection; use independent IC/volume development and external test groups. Train a fresh compatible full baseline with split/augmentation/checkpoint provenance. Resolve the projected-policy objective or complete intrinsic-feature schema before scientific RL training; never fabricate missing fields.
2. Predeclare the primary retention/strong comparator, meaningful effect or non-inferiority margin, hyperparameter search and held-out tests. Run multiple independent policy/backbone training seeds (a practical pilot is 5–10, with precision/power assessed), and multiple random TEST mask seeds. Current inspected test results are development evidence; reserve new independent volumes for confirmation. Report seed/volume-aware paired intervals and corrected comparisons. [RL evaluation guidance](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
3. Compare tuned learned controls, saliency and heuristics at identical final budgets, constraints and query/training-cost accounting. Add reward, physics, constraint/repair and feature ablations. Separate frozen-backbone selection from supervised backbone fine-tuning.
4. Redesign/profile the actual inference path before efficiency claims. Measure synchronized end-to-end latency, policy/embedding/decoder/transfer cost, memory and amortization on substantially larger graphs and repeated runs. A raw/cheap policy encoder is an essential control.
5. Evaluate real CAMELS/TNG→SIMBA transfer and held-out cosmology/feedback/IC variations with compatible targets/features; test richness, mass, observational noise/missingness and graph-size strata. Calibrate uncertainty on validation only and report interval sharpness and conditional coverage.
6. For TTA, export per-halo final predictions/masks/residuals and repeat independent seeds; compare all methods at the same 50% budget (or a predeclared common grid). Validate that uncertainty/physics reward improvement predicts accuracy rather than mere variance collapse. Select K/cost tradeoffs on validation with a practical tolerance; never choose K from test scores.

The existing legacy policy does not need to be retrained to preserve or analyze this campaign. New training is justified only for the new certified dataset/objective or explicitly designed replicates. The current completed artifacts remain immutable.

