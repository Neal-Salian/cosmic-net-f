# Controlled RL ablation specification and execution plan

Base commit: `c825bf57652005c80faff79cf4dfd58e3a9baa14`. Branch: `codex/paper-rl-ablations`. Date: 16 October 2026. Scope: manuscript evidence from frozen-backbone pruning and reproducible follow-up training. The recovered policy must remain byte-identical. No merge, push, PR, or four-notebook rerun is part of this campaign.

## Scientific questions and registered comparisons

| Priority | Comparison | Retraining? | What it establishes | Status |
|---|---|---|---|---|
| 1 | `prediction_only` vs `prediction_physics` | Yes, both from paired fresh initialization | Incremental effect of relative-virial reward, with sparsity/connectivity rewards absent and optimizer regularizers held fixed | Planned; executable |
| 1 | `full` vs `no_virial` | Yes, both | Incremental physics contribution within the complete historical objective | Planned; executable |
| 2 | `full` vs `no_sparsity`, `no_connectivity` | Yes | Contributions of each reward term under legacy repaired training | Planned; executable |
| 2 | `full` vs `no_entropy`, `no_value` | Yes | Entropy regularization and separate critic negative control | Planned; executable |
| 2 | Exact constrained ranking vs unconstrained ranking | No | Dependence of an existing policy's predictive behavior on enforcing no-isolate coverage at the same count | Executed locally |
| 2 | Constraint-aware ranking vs ranking followed by additive repair | No for decoder diagnostic; yes for a trained constraint intervention | Selection behavior at fixed realized repair cost; nominal budgets are audited separately | Inference diagnostic executed; training planned |
| 3 | Greedy logit selection vs sigmoid probability ranking using the same exact decoder | No | Whether a monotone score transform changes selection; an equivalence check, not a new learned method | Executed locally |
| 3 | Deterministic constrained ranking vs stochastic constrained PL decoding | No | Sampling variability for one fixed trained policy and budget | Executed locally, five draw seeds |
| 3 | Retention .2, .4, .6, .8 | No for recovered-policy transfer; yes for budget-specialized policy training | Accuracy/physics response to inference compression versus effect of training at a different target | Inference executed; specialized training planned |
| 4 | Virial weight .1, .3, 1, 3 | Yes, paired seeds | Robustness to physics/prediction tradeoff | Planned; executable |

Prediction-only means reward weights `(w_acc,w_sp,w_conn,w_virial)=(1,0,0,0)`. It retains entropy .01 and critic loss .5 to isolate the reward change. Prediction+physics uses `(1,0,0,1)`. Full uses `(1,.5,1,1)`. `no_virial` retains the sparsity and connectivity rewards, so it differs from prediction-only. All single-term removals change exactly one coefficient. No result from changing evaluation weights or decoding will be described as a trained reward ablation.

The existing independently configurable offline loss terms are prediction, sparsity, connectivity, relative virial, entropy, and value loss. Reward clipping and training-derived error scaling are nonlinear/numerical controls that must be held fixed in the first stage. If clipping is frequent, prioritize a registered clipping sensitivity before interpreting constant reward terms. Uncertainty reward and TTA KL are outside this frozen offline experiment.

## Fixed protocol

- Keep the real legacy CSV and original backbone checksum fixed. The recovered policy is a read-only reference, never a warm start for a reward ablation.
- Keep seed-42 halo split membership/order fixed for all training seeds. Start each variant from the same initialization for its paired seed (42, 43, 44); use the same graph shuffle/rollout RNG law. Refit normalization on exactly the same TRAIN population only. Fix graph representation and two-orientation evaluation scoring.
- Use 60 epochs, four rollouts, Adam .001, batch 32, .9→target curriculum over 40 epochs, identical virial annealing, clipping, and error scale rules. Select the initialization or epoch with minimum VAL RMSE at the fixed final budget. Report the selected epoch even when initialization wins. Never choose variants or coefficients on TEST.
- Evaluate once per completed training configuration, using the frozen predictor, deterministic exact no-isolate decoding, fixed tie marks seed 42, preserved loops, and no TTA/Stage B. Save every halo's target, prediction, selected pair indices, requested/final counts, isolate count, and relative-virial diagnostic.
- For reward comparisons at .4, use the complete original TRAIN split with legacy repair, and the topology-feasible .4 VAL/TEST cohorts. For a budget curve or budget-specialized policies use the intersection feasible across **all** compared budgets on VAL and TEST; pass identical `--cohort-budgets .2 .4 .6 .8` to every run.
- To compare **training constraints**, use `--training-cohort exact_feasible` for both `legacy_repair` and `direct`. This fixes TRAIN membership before training. Direct PL masks candidate draws by exact completion feasibility and uses their actual ordered log probabilities. Do not compare a direct subset run against a legacy run trained on all halos as if only the decoder changed.
- Nominal fixed-q comparisons require equal realized pair counts for each halo. Additive repair failures are retained in the budget audit. The separate matched-repair-cost diagnostic first fixes each halo's repaired count using scores/marks without labels, then reruns constrained selection at exactly that count. It answers a cost-conditioned question, not a nominal-q question. No outcome-based subset is used.

## Interpretation and uncertainty

Primary endpoint: paired halo-level difference in squared prediction error, summarized as change in RMSE dex. Secondary: MAE, prediction bias, physical-pair retention, total stored-edge retention, no-isolate violations, number of components, and relative-virial preservation. Preserve full-graph and gradient×input controls on the identical evaluation cohort. The full graph is a different-budget accuracy reference, never a matched pruning competitor.

For trained results, report all seed results and paired uncertainty over independent halos and training seeds; do not count edges, categorical draws, or five inference seeds as independent training replicates. If deriving confidence intervals, resample paired halo IDs and preserve pairing across methods/seeds; disclose that one TNG volume does not support volume-level generalization. Coefficient sweeps use VAL for selection and a single locked TEST comparison. Treat the current inference tables as exploratory diagnostics; no new significance claim or RL-over-gradient×input claim is authorized by them.

## Execution ledger

- [x] Isolate the requested base in a new worktree and branch; verify the reconstructed tree hash against the commit after cloud-backed Git metadata stalled native checkout.
- [x] Audit pair training, scoring, reward, constraints, self-loops, and recovered provenance.
- [x] Write failing tests for budget mismatch, halo mismatch, replay, independent coefficient removal, immutable completion, and optimizer/RNG resume; implement and verify the harness.
- [x] Run only frozen-model diagnostics on local real inputs; reproduce 0.136160 dex at .4 on 82 TEST halos.
- [x] Run a common 80-halo budget/decoder study and publish raw per-halo outputs plus actual result tables.
- [x] Prepare a resumable single-configuration Kaggle script and offline committed-source exporter; keep new trained ablations unexecuted.
- [ ] Execute the ranked GPU retraining matrix; **no trained scientific ablation result exists yet**.

Review focus: invalid budget inflation, changing halo populations, mislabeled inference changes, accidentally loading a recovered policy as a fresh start, and resume after code/data/config changes. Automated tests cover these numerical and provenance contracts, including training parity/resume and recorded-output budget consistency. A final independent code review accompanies repository verification.

## Follow-up ranking for manuscript value

1. Run prediction-only, prediction+physics, full, and no-virial with paired seeds 42/43/44 at .4. This is the smallest matrix that tests the central physics-reward attribution both with and without other reward terms (12 runs). If only hours remain, complete all four at seed 42 and explicitly label them single-seed preliminary results.
2. Run no-connectivity/no-sparsity with the full control at the same seeds. Quantify action-wise reward variance and clipping frequency. Connectivity is likely constant after repair; sparsity is small but action-dependent through repair. Negative results are informative.
3. Compare direct constrained training to legacy repaired training on the same exact-feasible TRAIN cohort, controlling evaluation identically. This resolves the observed training/evaluation mismatch; the local inference study alone cannot.
4. Run no-entropy and no-value. Entropy can change exploration; the separate value network is an implementation negative control and should leave actor updates unchanged in paired deterministic execution. Investigate any difference before interpretation.
5. Sweep virial weights .1/.3/1/3 using VAL, then lock a comparison. Report prediction/virial tradeoff rather than optimizing only TEST RMSE. Run clipping/error-scale sensitivity if saturation is observed.
6. Train budget-specialized policies at .2/.4/.6/.8 on fixed splits and common evaluation cohorts. The completed transfer curve does not answer whether specialization repairs the low-budget failure.

Tomorrow's manuscript can use the completed decoder/retention diagnostics and their limitations. It should describe reward attribution as unresolved until these new training runs complete.

## Final validation

The full repository suite passed (816 passed, 13 skipped; dependency deprecation warnings remain). The exported Kaggle source passed real-input CPU preflight and 10 portable tests. The diagnostic rerun from committed source reproduced every prediction and metric exactly, and completed-run resume verified checksums without rerunning experiments. Independent review reported no remaining blocking findings. The recovered archive/policy hashes were rechecked unchanged. See [validation record](../../experiments/paper_ablations/validation.json) and the [ready-to-attach source ZIP](../../experiments/paper_ablations/kaggle-source-38b6509.zip).
