# Baseline robustness results

Executed locally on CPU, float32, one PyTorch thread, using the existing frozen backbone and all 82 test halos. Historical GPU predictions were replayed within an absolute tolerance of $10^{-4}$; this table uses CPU replay and new CPU predictions. No policy, predictor, or learned selector was retrained.

## Matched-budget results

| Selector | 40% | 60% | 80% | 100% |
|---|---:|---:|---:|---:|
| Full graph | 0.114797 | 0.114797 | 0.114797 | 0.114797 |
| RL (deterministic) | 0.136160 | 0.127760 | 0.121663 | 0.114797 |
| Gradient × input | 0.136900 | 0.137050 | 0.122469 | 0.114797 |
| Random (original mask) | 0.156571 | 0.128369 | 0.118196 | 0.114797 |
| Shortest distance | 0.169469 | 0.135384 | 0.117739 | 0.114797 |
| Degree | 0.151941 | 0.132786 | 0.119166 | 0.114797 |
| Stellar binding proxy | 0.208129 | 0.147724 | 0.119173 | 0.114797 |
| Learned saliency selector | 0.147382 | 0.133855 | 0.125910 | 0.114797 |
| Geometry (validation tuned) | 0.171646 | 0.137151 | 0.118279 | 0.114797 |
| Velocity binding proxy | 0.234806 | 0.170241 | 0.126666 | 0.114797 |
| Signed gradient × input | 0.136998 | 0.137028 | 0.122467 | 0.114797 |
| Scaled absolute gradient | 0.166084 | 0.133196 | 0.121127 | 0.114797 |
| Random (10 mask seeds) | 0.148600 | 0.125612 | 0.117768 | 0.114797 |
| RL (10 sampling seeds) | 0.138057 | 0.127866 | 0.121457 | 0.114797 |

At 40%, gradient × input is the strongest existing pruning competitor to RL: 0.136900 versus RL 0.136160. The RL minus gradient RMSE difference is -0.000739, with paired 95% interval [-0.014664, +0.013320]; this does not resolve an accuracy advantage. The signed-gradient control is close but does not improve the aggregate result.
Ten random-mask seeds give expected-single-mask RMSE 0.148600, below the original one-mask 0.156571. At 60%, repeated random (0.125612) is better in aggregate than RL (0.127760); at 80%, shortest-distance (0.117739) and repeated random (0.117768) are below RL (0.121663). These paired intervals include zero. RL is not uniformly best across budgets. All selectors coincide at full retention.

Fractions specify requested retention. Actual physical retention is ceil(qP)/P and is identical across selectors per halo. Full graph retains every pair and is a reference, not a matched pruning competitor. The CSV also gives MAE, bias and achieved retention.

## Additional controls and validation decisions

Geometry mixes average ranks of negative 3D and projected XY separation. Validation-only q=0.4 tuning selected alpha=0.5 (alpha=1 means 3D). The three alpha candidates were 0, 0.5, and 1; every validation halo and losing candidate is exported.
The physics control ranks log((stellar_mass_i+stellar_mass_j)/r) - log(delta_v²+v0²), with v0=371.849670 km/s fixed from train-only pair velocities. It is a stellar/velocity proxy, not a measurement of dark-matter binding energy.
The hybrid mixes geometric and velocity-binding percentile ranks; validation selected geometric weight 1. Both chosen weights were frozen before test evaluation and reused at every retention. The selected hybrid is exactly geometry (weight 1); its executions remain in the CSV but it is not counted as a separate headline competitor.
Alternative saliency uses (a) abs(sum_f grad_f × input_f), allowing signed feature cancellation before the absolute value, and (b) sum_f abs(grad_f) × train_std_f. The original uses sum_f abs(grad_f × input_f). All use the prediction gradient, never a test-label loss.
Sigmoid probability ranking reproduced the logit-ranked RL mask in 328/328 checks; it is recorded as an equivalent numerical control rather than an additional baseline.

## Randomness and replication

Policy training: one saved seed, 42; selected epoch comes from validation. No conclusion about stability across independently trained policies is supported.
Random masks: seeds 100–109, ten independent random score vectors per halo. The same vector is coupled across budgets. The exact feasibility decoder induces a constrained rank distribution; it is not uniform over feasible final masks.
Policy sampling: seeds 200–209, fixed saved orientation-averaged logits, fresh Gumbel draws at each action, exact completion feasibility. This samples the direct constrained PL decoder conditional on one trained policy; it is not independent training replication or the original legacy training sampler.
Tie marks: seeds 300–302 alter only independent tie priorities for deterministic RL. Baseline marks are held fixed in the mask/sampling experiments.
Bootstrap: seed 20261016, 10000 paired halo resamples; these intervals quantify halo sampling uncertainty, conditional on the supplied trained artifacts and evaluated mask seeds. They do not quantify policy-training uncertainty.
Repeated-selector RMSE is sqrt(mean_h mean_seed squared_error), an expected single-draw loss. Seed mean/SD/range are exported separately; predictions are never averaged into an ensemble. No mask seed is treated as another halo.

## Seed sensitivity at the primary budget

| Selector | Seed RMSE mean | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Random (10 mask seeds) | 0.148495 | 0.005890 | 0.136827 | 0.157937 |
| RL (10 sampling seeds) | 0.138047 | 0.001790 | 0.136059 | 0.140238 |
| RL (3 tie mark seeds) | 0.136160 | 0.000000 | 0.136160 | 0.136160 |

The range describes every evaluated seed; no best seed is promoted to the headline comparison. Tie-mark perturbations leave RL unchanged here. Sampling variance is conditional on the one available trained policy.

## Where RL is better or worse

| Competitor at 40% | RL − competitor RMSE | Paired 95% interval | RL lower / higher / tied halo loss |
|---|---:|---|---|
| Full graph | +0.021364 | [+0.010926, +0.032419] | 29 / 53 / 0 |
| Gradient × input | -0.000739 | [-0.014664, +0.013320] | 42 / 40 / 0 |
| Random (original mask) | -0.020411 | [-0.047045, +0.006529] | 40 / 42 / 0 |
| Shortest distance | -0.033309 | [-0.057284, -0.008111] | 49 / 33 / 0 |
| Degree | -0.015781 | [-0.040426, +0.009846] | 37 / 45 / 0 |
| Stellar binding proxy | -0.071969 | [-0.096721, -0.045394] | 60 / 22 / 0 |
| Learned saliency selector | -0.011222 | [-0.021834, -0.001091] | 45 / 37 / 0 |
| Geometry (validation tuned) | -0.035486 | [-0.062562, -0.007969] | 50 / 32 / 0 |
| Velocity binding proxy | -0.098646 | [-0.125687, -0.070653] | 64 / 18 / 0 |
| Hybrid (validation tuned) | -0.035486 | [-0.062562, -0.007969] | 50 / 32 / 0 |
| Signed gradient × input | -0.000838 | [-0.014784, +0.013220] | 42 / 40 / 0 |
| Scaled absolute gradient | -0.029924 | [-0.056176, -0.003164] | 42 / 40 / 0 |
| Random (10 mask seeds) | -0.012440 | [-0.033714, +0.010791] | 45 / 37 / 0 |
| RL (10 sampling seeds) | -0.001897 | [-0.004426, +0.000358] | 50 / 31 / 1 |
| RL (3 tie mark seeds) | +0.000000 | [-0.000000, +0.000000] | 0 / 0 / 82 |

Negative delta favors RL. The full CSV reports every competitor at every budget and every halo; no favorable subgroup is selected. Intervals are exploratory percentile intervals without multiplicity correction, so individual exclusions of zero are not confirmatory significance tests.

## Sensitivity and coverage

Strata were specified for node count, mean physical degree (2P/N), target log mass, and mean subhalo velocity dispersion. Cutpoints use validation tertiles; tied values go into the higher bin, and empty bins remain explicit. All 82 test halos occur exactly once across bins for every property, selector, and budget. Target-mass strata are descriptive outcome strata and never used for tuning or selection.
Test graph sizes span 7–10 nodes and 15–45 physical pairs. This narrow range cannot establish large-graph robustness. Degree and graph size are correlated; strata are descriptive, not controlled causal effects. See stratified_comparisons.csv for every bin, including unfavorable and empty bins.

## All primary-budget property strata

Every validation-tertile bin is shown for the strongest existing competitor (gradient × input) and repeated random. RMSE entries are descriptive; small or empty bins cannot establish a stable interaction.

| Property | Bin | Halos | RL RMSE | Gradient × input | Repeated random |
|---|---|---:|---:|---:|---:|
| nodes | low | 6 | 0.102455 | 0.075407 | 0.080331 |
| nodes | middle | 0 | — | — | — |
| nodes | high | 76 | 0.138472 | 0.140614 | 0.152696 |
| mean_degree | low | 20 | 0.166771 | 0.145865 | 0.202025 |
| mean_degree | middle | 0 | — | — | — |
| mean_degree | high | 62 | 0.124693 | 0.133880 | 0.126647 |
| target_log_mass | low | 18 | 0.086547 | 0.093269 | 0.072864 |
| target_log_mass | middle | 33 | 0.123537 | 0.134077 | 0.109511 |
| target_log_mass | high | 31 | 0.168657 | 0.159332 | 0.206304 |
| velocity_dispersion | low | 33 | 0.119915 | 0.099583 | 0.124756 |
| velocity_dispersion | middle | 19 | 0.123859 | 0.129540 | 0.140553 |
| velocity_dispersion | high | 30 | 0.158560 | 0.172310 | 0.175287 |

RL has higher error than gradient × input in the small-node, low-degree, high-target-mass, and low-velocity bins; it has lower error in the other nonempty gradient comparison bins. Against repeated random it is worse in the small-node and low/middle target-mass bins, and better in the remaining nonempty bins. These are descriptive partitions of the full cohort, not selected subpopulations or confirmatory interaction tests. Full all-method/all-budget strata and exact cutpoints are exported.

## Scientific and timing limits

The checkpoint is explicitly legacy_total_radius: input half_mass_radius is total rather than stellar radius, and target semantics and the original predictor training cohort are uncertified. Selector train/validation/test identities are disjoint, but this does not establish absence of leakage in the pre-existing backbone training. These experiments strengthen comparisons conditional on these weights; they do not certify corrected astronomy or external generalization.
Historical timings include scoring (with full-graph embeddings for RL and a gradient pass for saliency), constrained decoding, and masked prediction. The full reference is shared, graph construction is separate, and model loading/training is excluded. The reference CPU/NetworkX decoder dominates much of the original GPU region time. New seed sensitivity runs reuse saved scores; their wrapper/decoder times are diagnostics and cannot substantiate end-to-end speedups. New heuristic/gradient timing fields measure shared scoring work; they are not independent optimized deployment latencies. No repeated warmed GPU timing experiment was run.
Independent policy training seeds, a matching corrected-astronomy backbone, external populations, and new GPU latency runs are explicitly not executed in manifest.json. No new training was substituted for missing artifacts.

## Reproduction and artifacts

Run the commands in experiments/paper_baselines/README.md. The executed run is results/run-20261016-final-v2. Inputs and outputs have SHA-256 manifests; every new test/validation prediction, signed residual, mask, score, seed role, eligibility result and hyperparameter candidate is exported. Historical replay includes original and CPU predictions. The initial schema-validation failure is preserved as a failed manifest in results/run-20261016; it contains no claimed experiment results.
