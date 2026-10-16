# Executed RL ablation diagnostics

Executed locally on 16 October 2026. **No reward-ablation policy was retrained.** All scientific results below use the same recovered 60-epoch policy and frozen backbone. New training configurations are planned experiments only.

## Reproduction and scope

The original 40% test-cohort replay produced **0.136160320 dex RMSE on 82 halos**, reproducing the supplied 0.136160 dex result. This replay is separate from the common-cohort budget study below.

The budget study uses the same 80 TEST halos at every budget and for every decoder. Eligibility is the topology-only intersection over [0.2, 0.4, 0.6, 0.8]; all 82 original TEST IDs remain in the saved manifest. The excluded halos are `tng100_0000053` (minimum 6 pairs, .2 budget 3/15) and `tng100_0000080` (minimum 5 pairs, .2 budget 4/19). No halo was removed using predictions or labels.

The frozen full graph has **0.115423 dex RMSE** on these same 80 halos and remains the strongest accuracy reference. Its budget is 100%, so it is a reference rather than an equal-cost pruning comparison. All runs use deterministic frozen predictions, preserved self-loops, the same two-orientation scorer, no Stage B, and no TTA.

## Retention and policy behavior

Stochastic entries are mean ± sample SD of five separate single-mask inference RMSEs (seeds 7,19,43,71,101). They are not an ensemble and are not confidence intervals or independent training replicates. Marks and categorical draws are both reproducible within each draw seed.

| Requested pair retention | Realized mean pair retention | Mean stored-edge retention | Constrained rank RMSE | Stochastic RMSE mean ± SD | Unconstrained rank RMSE | Unconstrained isolate halos | Rank virial penalty |
|---|---|---|---|---|---|---|---|
| 20% | 0.202677 | 0.285622 | 0.244106 | 0.223931 ± 0.006008 | 0.707701 | 68/80 | 0.218863 |
| 40% | 0.402809 | 0.464926 | 0.137059 | 0.138932 ± 0.001531 | 0.230252 | 17/80 | 0.094043 |
| 60% | 0.602055 | 0.643454 | 0.128744 | 0.128962 ± 0.000953 | 0.137986 | 1/80 | 0.007584 |
| 80% | 0.802188 | 0.822758 | 0.122313 | 0.122483 ± 0.000415 | 0.122313 | 0/80 | 0.001527 |

RMSE units are dex. Every nominal rank/sample arm uses exactly `ceil(q P)` pairs on each halo. Constraints leave zero physical isolates in every constrained arm. Stored-edge retention includes all preserved self-loops and therefore exceeds physical-pair retention. More retained edges improve accuracy for this fixed policy, while the relative-virial deviation also decreases; the latter does not attribute the accuracy effect to the trained physics reward.

At low budgets the exact no-isolate decoder substantially changes the selected topology and observed error. This establishes dependence of this recovered policy on inference constraints in this cohort. It does not establish the effect of training with a connectivity reward. At .2, stochastic decoding has a lower mean RMSE than deterministic decoding in these five draws; at .4 it has a higher mean. These are exploratory observations, with no test-driven choice of deployment seed or budget and no significance claim.

Applying sigmoid to the **already orientation-averaged pair logits** before the same exact greedy decoder produced identical masks for 320/320 halo-budget cases. This checks a monotone ranking equivalence; it supplies no evidence for an additional policy mechanism. Averaging sigmoid outputs before pair/orientation aggregation is a different intervention and was not tested.

## Additive repair: nominal budget audit

The unconstrained top-k selection is followed by the repository additive no-isolate repair with fixed pair marks. An accuracy comparison at the nominal q is withheld if any halo exceeds that budget. No favorable subset of successfully repaired halos is selected.

| Nominal q | Budget-exceeding halos | Mean added pairs/halo | Mean final pair retention | Nominal-budget RMSE |
|---|---|---|---|---|
| 20% | 68/80 | 1.5750 | 0.241041 | withheld: unequal cost |
| 40% | 17/80 | 0.2375 | 0.408557 | withheld: unequal cost |
| 60% | 1/80 | 0.0125 | 0.602472 | withheld: unequal cost |
| 80% | 0/80 | 0.0000 | 0.802188 | 0.122313 |

## Exact comparison at the repaired per-halo cost

For this separate diagnostic, freeze each halo’s repaired final pair count without consulting its label, then run constraint-aware ranking at exactly that count. Both methods have zero isolates and equal counts per halo. The requested nominal fraction identifies the source of the cost schedule; the schedule is score/mark-dependent and is not the nominal-q protocol.

| Source nominal q | Matched mean pair retention | Constraint-aware RMSE | Select-then-repair RMSE |
|---|---|---|---|
| 20% | 0.241041 | 0.226435 | 0.223270 |
| 40% | 0.408557 | 0.136561 | 0.137196 |
| 60% | 0.602472 | 0.128885 | 0.129372 |
| 80% | 0.802188 | 0.122313 | 0.122313 |

These small cost-matched differences change sign across budgets. They do not support a general superiority claim for either selection mechanism, and neither policy was retrained for its decoder.

## Historical reward audit

The preserved training log contains 60 completed epochs. Its final epoch logs normalized prediction improvement -0.035881074, weighted virial term -0.049310798, weighted sparsity term -0.000584854, target pair retention 0.4, realized mean pair retention 0.414173119, and mean reward 0.914223275. This is final-epoch logging; the evaluated checkpoint was selected at epoch 42.

The virial term was nonzero and repair increased average retention. These observations establish that the corresponding computations were active, not that they improved generalization. The historical log has zero isolates in every epoch and reward extrema within [-10,10] without reaching the clip boundaries; the +1 connectivity contribution was therefore constant over these recorded rollouts and canceled in leave-one-out actor advantages. A separate critic is trained but not used by that actor baseline.

## Evidence limits and remaining work

- Prediction-only vs physics-aware reward, each trained coefficient removal, trained constraint interventions, specialized budgets, and coefficient sensitivities remain **planned**. Their scripts are runnable; no new trained outcomes are reported.
- These data do not alter the existing finding that RL superiority over gradient×input is statistically unsupported. This campaign did not rerun that saliency comparison.
- One recovered policy, one legacy input dataset, and one simulation volume support method diagnostics. They do not establish physical causality, calibrated mass inference, speedup, or robustness across training seeds/simulations.

## Reproducibility and artifact index

Run `python -m experiments.paper_ablations.report` to regenerate this document from completed outputs. The executable experiment and Kaggle instructions are in [the README](../../experiments/paper_ablations/README.md); the ranked training matrix is in [the specification](ablation-plan.md).

- [Per-arm metrics](../../experiments/paper_ablations/results/local-2026-10-16/summary.json)
- [Per-halo predictions, selected physical-pair indices, and realized budgets](../../experiments/paper_ablations/results/local-2026-10-16/predictions.json)
- [Full provenance, input/source checksums, splits, and completion hashes](../../experiments/paper_ablations/results/local-2026-10-16/run.json)
- [Topology eligibility](../../experiments/paper_ablations/results/local-2026-10-16/eligibility.json)
- [Ranking equivalence](../../experiments/paper_ablations/results/local-2026-10-16/mask_equivalence.json)
- [Historical log audit](../../experiments/paper_ablations/results/local-2026-10-16/historical_training_log.json)

Executed source tree fingerprint: `db9cedeb240a04e95f866e0fa79a3753c3d0535d62ef4cccd84e603890347750`; evaluator HEAD at execution: `c825bf57652005c80faff79cf4dfd58e3a9baa14`. Exact experiment-source hashes are saved because the harness was run before its final commit. Core RL files remain those of the pinned base. Kaggle packaging records the final committed experiment revision.

Recovered selected-policy SHA256: `b1f41771c102bec644e0f84b0c002a228e5514321c768d8919e8beae3379f81d`. Source archive identity is rechecked from bytes; no source policy file is opened for writing.

Runtime: Python 3.13.3, Torch 2.13.0, CPU, one thread. CUDA was unavailable locally; no expensive training was attempted. Software fixture tests perform tiny optimizer checks solely for validation.
