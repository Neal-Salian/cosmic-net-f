# Cosmic-Net Research Strengthening Plan

> **For agentic workers:** When implementation is requested, use superpowers:subagent-driven-development or superpowers:executing-plans for the approved work package. This is a research roadmap, not authorization to implement every branch or launch training. Steps use checkbox syntax for tracking.

**Goal:** Establish a defensible astrophysical result and determine, through bounded experiments, whether Cosmic-Net supports a separate general ML or RL contribution.

**Architecture:** Build a reproducible evaluation layer and corrected data/backbone foundation, then compare selection mechanisms under explicit budgets and constraints. Run the astronomy study and one ML pilot on this shared foundation; expand only where evidence supports the claim.

**Tech Stack:** Existing Python, PyTorch, PyTorch Geometric, NumPy/pandas and notebooks; real TNG/CAMELS catalogs and selected public graph benchmarks. Record working environment versions rather than forcing older README pins.

**Spec:** [Proposed research design](../specs/2026-09-16-research-strengthening-design.md).

## Global constraints

- Planning only for the current request. Preserve the saved user Notebook B outputs. Do not modify training code, commit, push, download large datasets, or launch experiments as part of writing this plan.
- Assume Kaggle/single-GPU access and no fixed deadline until the user provides constraints.
- Use physical unordered pairs for budgets; report total stored edges separately.
- No silent budget increases, synthetic fallback in scientific runs, target-definition substitution, or test-set hyperparameter selection.
- Separate frozen-backbone, retrained-backbone and test-time-adaptation results.
- All novelty and performance propositions below are hypotheses. Failure to outperform a simpler method is a valid result and changes the paper claim.
- The existing 82 test halos have been inspected during development. Reserve fresh external data for final confirmation.

---

## 1. Recommended order and resource discipline

| Stage | Deliverable | Dependency | Decision |
|---|---|---|---|
| 0 | Auditable reference run and corrected claims inventory | None | Agree what is known versus proposed. |
| 1 | Correct invariants, physical schema and baseline training | Stage 0 | No major scientific experiment with unresolved validity failures. |
| 2 | Matched-budget evaluation and strong simple controls | Stage 1 | Establish whether learned selection adds value at all. |
| 3 | Small constrained-selection ML pilot | Enumeration after contract definition; post-hoc development pilot after Stage 1A; scientific training after Stages 1B/2 | Expand only if effects survive cost and budget controls. |
| 4 | Independent astrophysical transfer study | Data audit; Stage 2 | Select a scientific claim supported by independent evidence. |
| 5 | General ML benchmark study OR optional RL estimator pilot | Successful Stage 3 | Choose one primary methodological contribution. |
| 6 | Optional adaptation study | Reliable fixed-policy baselines | Proceed only if surrogate-error relationship is useful. |
| 7 | Locked final evaluation and manuscript package | Chosen route passes evidence review | Choose venue from the contribution actually established. |

Begin catalog feasibility/count audits while repairing the core, but do not wait for a large download to test the small selector hypothesis. Profile one run before estimating GPU hours. Use a three-seed pilot, shortlist methods, then a five-plus-seed final study; do not run a full Cartesian sweep of all rewards, backbones, datasets and budgets. Target the first decision report after Stages 0–3 before committing to a large study. Scheduling should follow measured throughput and data availability, not the old plan's submission dates.

## 2. Stage 0 — Freeze evidence and research claims

**Existing files:** `notebooks/notebook_B_train_policy.ipynb`; `rls/provenance.py`; `rls/notebook_workflow.py::run_context/complete_stage`; `README.md`; `rl-lit-review.md`; `rl-implementation-plan.md`.

**Proposed artifacts:** `outputs/research/<run_id>/manifest.json`, `predictions.parquet`, `metrics.json`, `environment.txt`; a claims-to-evidence table in the eventual manuscript folder.

- [ ] Archive the saved B run, checkpoint/dataset hashes, source commit plus dirty-file hashes, exact split IDs, configuration, seed and environment. Keep failed/rejected Stage B metadata.
- [ ] Reproduce the reference metrics within a declared numerical tolerance; record CPU/GPU differences instead of replacing saved results.
- [ ] Mark old smoke, pre-fix, synthetic and unmatched-budget results as development artifacts. Do not mix them with final tables.
- [ ] Verify each citation's actual venue, method and quantitative claim against its primary source. Correct the literature table's PGExplainer/NeuralSparse venue errors and remove unsupported claims that label-free adaptation guarantees OOD improvement.
- [ ] Replace manuscript/README assertions of physical discovery, speedup and automatic explanation with testable claims until their experiments exist.

**Exit:** every displayed scientific number has a traceable run, split and backbone; no numerical comparison combines validation and test or incompatible target definitions.

## 3. Stage 1 — Repair validity before optimizing scores

### 1A. Selection and evaluation contracts

**Modify later:** `rls/pair_policy.py::repair_pairs/pair_mask/pair_stats`, `rls/notebook_workflow.py::adaptation_trial/random_pair_reference/policy_metrics`, `rls/evaluate.py::build_results_table`, `config/config.yaml`.

**Tests:** `tests/test_pair_policy_repair.py`, `tests/test_relative_virial_reward.py`, `tests/test_virial_selfloop_contract.py`, `tests/test_tta.py`, `tests/test_notebook_pipeline.py`.

- [ ] Add the four-node counterexample from review: relabeling changes current repaired pair count from two to three. Add node permutation, edge-order, reversal and duplicate-storage checks.
- [ ] Specify physical-isolation semantics and reconcile the two failing legacy tests with that contract; preserve tests that expose real regressions.
- [ ] Implement a documented equivariant repair/reference construction, with randomized treatment of symmetric ties where necessary. This is a correctness fix, not the novel method.
- [ ] Report requested k, sampled k, final k, physical-pair retention, total retention, isolates and connected-component count for every method.
- [ ] Make K=0 and K>0 adaptation use identical requested and achieved budgets, or declare any infeasibility explicitly.
- [ ] Skip action-likelihood/entropy construction during deterministic inference when it is not used; retain exact ordered likelihood for training.
- [ ] Run the affected tests and notebook smoke workflows in the recorded supported environment. Treat unsupported CUDA tests separately from failures.

**Exit:** the relevant suite is green, permutations preserve the declared behavior, infeasible budgets are explicit, and no hidden repair-based budget advantage remains in comparisons.

### 1B. Physical data and original backbone

**Modify later:** `scripts/fetch_tng_data.py`, `data/loaders/tng_loader.py`, `data/loaders/camels_loader.py`, `data/loaders/base_loader.py::SubhaloData/split_data`, `graph/graph_builder.py`, `model/physics_loss.py`, `training/train.py`. Extract the augmentation in `kaggle/training-cosmic (4).ipynb` cell `65b8b13b` into a versioned, testable module such as `data/augmentation.py`.

**Tests:** `tests/test_loaders.py`, `tests/test_graph_builder.py`, `tests/test_physics_loss.py`; proposed `tests/test_augmentation.py` and `tests/test_catalog_contract.py`.

- [ ] Write a schema identifying M200c, stellar versus total radius, internal versus orbital velocity, physical/comoving position, h, a, missing values and completeness cuts. Validate sample rows against authoritative catalog fields.
- [ ] Audit CAMELS mass-field availability. If M200c is missing, obtain it from an appropriate catalog or define a separate task; do not silently call GroupMass M200c.
- [ ] Rotate positions and velocities together. Test invariant distances/norms/relative angles; treat projected separation using an explicit observer direction. Rebuild topology after jitter when its construction requires it.
- [ ] Remove the first-ten acquisition cap in the new scientific cohort, or replace it with a stated sampling/selection experiment; retain the original cohort for regression checks.
- [ ] Establish a corrected purely supervised backbone. Then ablate the current physics regularizer and any revised physically justified regularizer. Retrain when inputs or augmentation change; do not relabel an old checkpoint as corrected.
- [ ] Use disjoint simulation/halo/lineage groups, and ensure augmented copies and multiple views cannot cross splits. Save split IDs and hashes with every checkpoint.

**Exit:** a documented, auditable cohort and corrected full-model baseline exist. There is no predetermined RMSE the correction must reach: a worse but valid score is more useful than an invalid improvement.

## 4. Stage 2 — Establish whether graph selection adds value

**Modify later:** `rls/baselines.py`, `rls/evaluate.py`, `rls/notebook_workflow.py`, `scripts/multiseed_rl.py`, `scripts/generate_paper_plots.py`, notebooks A/B/C. Proposed small modules: `rls/benchmark.py` for run orchestration and `rls/statistics.py` for paired/block summaries.

**Benchmark contract:** freeze predictor and split; use q in `{0.25, 0.4, 0.5, 0.6, 0.8, 1.0}` where feasible, with q=0.5 the proposed primary astronomy endpoint. Record actual integer budgets and infeasible fractions. Show no-isolate and connected experiments separately. A random constrained sampler need not be uniform over all feasible masks: document its actual law.

- [ ] Fit stellar-mass/richness/velocity summary regressors and a permutation-invariant node-set model. These establish whether graph edges add information beyond inexpensive alternatives.
- [ ] Evaluate full GNN; random pairs; distance, degree and stellar-binding heuristics; gradient/perturbation scoring; differentiable selection; and the corrected RL policy.
- [ ] For shortlisted methods, add faithful adaptations of PGExplainer/NeuralSparse/SparRL relevant to the claim. For the constrained-selection ML claim, include an L2XGNN-style structured perturb-and-MAP control and an applicable structured relaxation; account for all-node coverage versus motif-selection differences. Record departures from the original method; do not label an untrained attention network or saliency stub as a trained published baseline.
- [ ] Equalize final physical-pair budget, structural constraint, backbone access and tuning/predictor-query budget. Use shared decoders to isolate scoring differences, and shared scorers to isolate decoder differences.
- [ ] Separate tests with full-GNN-derived selector embeddings from a cheap selector using raw inputs. Report the cost of every prerequisite computation.
- [ ] Save per-halo predictions/masks/metadata. Report paired RMSE differences and uncertainty, not only mean seed scores; cluster/block by simulation or related-object groups where appropriate.
- [ ] Measure synchronized end-to-end latency with warm-up, several batch sizes, peak memory and GNN calls. Include graph construction and selection; report cached and uncached scenarios separately.
- [ ] Plot error versus physical retention, and error versus measured runtime. Include bias by mass/richness and the full-model reference on every relevant plot.

**Pilot continuation rule:** continue learned selection if it beats the strongest simple matched-budget baseline at two adjacent budgets with consistent effect direction across three seeds, or gives a measured runtime benefit at equivalent error. This is a project resource-allocation rule, not a significance test or journal acceptance standard. If it fails, investigate one diagnosed mechanism; do not launch an unbounded reward sweep.

## 5. Stage 3 — One focused general-ML pilot

**Hypothesis:** under tight feasible budgets, direct feasible selection improves optimization regret per predictor evaluation by avoiding redundant executed masks caused by repair. Compare against structured perturb-and-MAP as well as repaired PL; establish whether any gain remains in wall time.

**Existing files:** `rls/policy.py::EdgePolicyNet`, `rls/pair_policy.py`, `rls/pair_training.py::train_pair_policy`, `rls/train_policy.py::prepare_graphs`.

**Proposed files:** `rls/constraints.py` (pair budget/coverage/feasibility), `rls/constrained_policy.py` (scaffold and direct-feasible decoders), `tests/test_constrained_policy.py`, `scripts/selector_pilot.py` (enumerated and empirical pilot). Names are proposed boundaries, not existing APIs.

- [ ] Enumerate small undirected graphs with at most six nodes, all feasible masks at selected k, and known optima for controlled rewards. Include disconnected inputs, unavailable connections, symmetry/ties and small/infeasible budgets. Limit exact ordered-path probability/gradient enumeration to at most eight pairs and action length four (at most 1,680 ordered paths per instance); use analytic and Monte Carlo checks beyond that. Fifteen-pair mask enumeration does not authorize fifteen-factorial path enumeration.
- [ ] Implement a fixed edge-cover/spanning-forest scaffold plus residual top-k/PL selection as the simple feasible baseline. Keep scaffold construction independent of learned logits and share its random marks across scorers. Record restrictions on its support; it may miss solutions that use a different scaffold. At k equal to scaffold size, report that no residual learning freedom remains.
- [ ] Prototype a direct-feasible sequential selector with a completion check. Verify action probabilities, exact ordered likelihoods and score-function gradients against enumeration before replacing the small-instance oracle with a scalable algorithm.
- [ ] Add explicit budget input to a small shared scorer. Compare raw-input pooled features with full target-GNN embeddings, keeping capacity and access distinctions visible.
- [ ] Compare corrected PL plus repair, scaffold plus residual selection, direct-feasible selection, and structured perturb-and-MAP/relaxation adapted from the closest literature. Include uniform-score/random controls for each relevant decoder. Audit the executed-mask support; differences in the action family can explain a gain independently of gradient estimation.
- [ ] Use both frozen synthetic predictors with known interaction structure and corrected halo development graphs. Keep tiny pilot instances near 10–15 physical pairs so unordered masks can be enumerated, and restrict ordered-action enumeration further when needed. Measure predictive error, oracle optimization regret, duplicate executed masks, constraint violations, budget distortion, support restriction, gradient variance, predictor calls and latency.
- [ ] Test interpolation to unseen budgets and size generalization. Fail closed when constraints are infeasible; do not count automatic budget relaxation as success.
- [ ] Cap the first trained halo screen at 64 prespecified training halos, a disjoint development-validation subset, ten epochs and three training seeds per shortlisted mechanism. Start with three feasible budgets. This is an optimization diagnostic only; increase scale after its cost/effect review rather than interpreting small-screen results as publication evidence.

**Go/no-go:** proceed if feasibility is correct, gains persist under matched resources and a focused literature comparison identifies a defensible difference. For the synthetic pilot, prespecify a practical screen such as at least 10% lower optimization regret at equal predictor calls or twice the query efficiency at a fixed regret target, with three-seed consistency and total time reported. These are project decision thresholds, not significance tests or publication standards; compare to the strongest structured baseline, not only random. If repair is rarely active, if a simple scaffold/structured relaxation matches performance, or if feasibility computation consumes the gain, use the simpler method in the astronomy study and drop the claim that RL is necessary. If only engineering invariants improve, report them as engineering.

## 6. Stage 4 — Build the astrophysics paper around one finding

**Question:** which galaxy relationships add transferable halo-mass information beyond aggregate galaxy properties, and how does that information change with richness, simulation physics and observational limitations?

Here, relationship information means predictive utility/inductive bias relative to controlled models, not information mathematically absent from the same raw galaxy features supplied to a node-set network.

**Existing files:** catalog loaders and graph builder; `rls/cross_sim.py`, `rls/notebook_workflow.py::local_camels_graphs/physics_diagnostics/coverage_rows`; notebooks C/D. Proposed `scripts/build_research_catalog.py` and `config/research_astro.yaml` for an explicit cohort/split manifest.

- [ ] Audit catalog availability and halo counts by mass/richness before choosing cohorts. Start with group/subhalo catalogs rather than particle snapshots. Do not assume the original high-mass cut gives enough objects in smaller CAMELS boxes.
- [ ] Use public CAMELS-IllustrisTNG and SIMBA z=0 catalogs as the preferred cross-family study, with source train/validation/test IDs fixed separately from external IDs. Treat CV cosmic-variance and LH parameter-variation sets separately. Exact catalog counts, stellar completeness threshold and common mass interval are determined by the schema/count audit and frozen before fitting; do not inherit the present TNG100 cuts automatically. Official [access](https://camels.readthedocs.io/en/latest/data_access.html) and [schema](https://camels.readthedocs.io/en/latest/subfind.html) pages document availability and corrected catalog releases, so preserve download date and hashes.
- [ ] Build source training/validation splits grouped by simulation/initial condition where multiple volumes exist. Reserve source-domain confirmation and external simulation-family confirmation separately. Keep related snapshots/initial conditions together where independence is claimed.
- [ ] Allocate external development and external confirmation simulations before exploration. Choose perturbation ranges, thresholds, baselines and the scientific hypothesis only using development simulations. Any confirmation set inspected to revise the method or claim becomes exploratory; obtain fresh confirmation or label the revised result accordingly.
- [ ] Use development variance and the available independent simulation counts for a prospective precision/detectable-effect assessment against the proposed 5% effect scale. Increase independent volumes or narrow the claim if the expected intervals are too wide. Repeated training seeds do not increase the number of independent universes/volumes.
- [ ] Use comparable target definitions, units and completeness cuts. Separate changes in simulation physics from changes in cosmology, mass support, resolution and richness; report these shifts rather than treating all as one OOD effect.
- [ ] Center positions with periodic minimum-image conventions and audit velocity conversions. Prefer common stellar/kinematic features for the main cross-suite experiment; treat metallicity and other subgrid-sensitive features as explicit ablations. Never use target-derived halo radius/mass to normalize the model inputs.
- [ ] Compare full 3D data with a separately defined projected/line-of-sight task. Add realistic measurement noise, missing faint members and interlopers as distinct controlled perturbations with declared ranges.
- [ ] Reproduce a HaloGraphNet-style baseline on the same cohort/features and split, plus aggregate regressors and a node-set model. Published headline errors from different datasets are context, not directly comparable benchmarks.
- [ ] Ablate physics reward, structural constraint, edge-feature groups and predictor physics regularization independently. Report the rewarded proxy and independent physical/predictive diagnostics; improvement of the rewarded proxy alone is circular evidence.
- [ ] Study retained-edge statistics conditional on distance, stellar mass and richness; use halo-level uncertainty and multiplicity correction for exploratory comparisons. Do not count reciprocal edges as independent observations.
- [ ] Test explanation stability under retraining, node relabeling and nuisance transformations. Add mask-only and selector-recomputation controls to detect information encoded in topology. Use controlled synthetic motifs for known-ground-truth checks; label these as synthetic checks, not astrophysical validation.
- [ ] Report calibration/coverage if uncertainty is claimed. Evaluate interval width and conditional error; do not infer valid OOD coverage from MC-dropout variance or an IID calibration procedure.

**Proposed substantive targets, fixed before confirmation:** at an approximately 50% physical-pair budget, aim for no more than 5% relative source-domain RMSE degradation versus the full predictor and at least 5% relative improvement over the strongest matched-budget baseline in a prespecified transfer setting. These are aspirational project thresholds, not predicted outcomes or publication rules. A different robust astronomical finding can support a paper even if these targets fail; revise the claim openly.

**Exit:** at least one independent astrophysical result survives strong controls, and limitations are documented. Failure of RL to win does not invalidate a useful finding about galaxy information or simulation dependence.

## 7. Stage 5 — Decide whether a standalone ML or RL paper is warranted

### General ML expansion, if Stage 3 succeeds

- [ ] Choose two non-astronomical task families aligned with the claim: an OOD regression setting such as GOOD-ZINC and a longer-range setting such as LRGB Peptides-struct; use a denser OGB task if efficiency/edge reduction is central. Check feasible budget ranges before committing to datasets.
- [ ] Use at least two distinct backbone families and at least five independent training seeds for shortlisted methods. Match pretrained-backbone resources across methods.
- [ ] Evaluate several budgets, graph-size shifts, frozen versus retrained predictors and actual runtime. Describe molecular pruning as computational message selection, not a chemically valid modified molecule.
- [ ] Establish a contribution beyond feasibility guarantees and familiar action masking: for example, a new efficient constraint mechanism with demonstrated general benefit, or a reproducible general finding about when constrained graph selection transfers.

### Optional pure-RL estimator pilot, a separate decision

- [ ] Write the precise objective over executed masks. For a latent ordered action a and fixed repair map T, distinguish p(a) from `P(m) = sum_{a:T(a)=m} p(a)`.
- [ ] On enumerated small instances, compare the current unbiased ordered-action estimator with conditioning on the executed mask. Check gradients against exact differentiation. The identity connecting conditional scores to the executed-mask gradient is a known starting point, not a novelty claim.
- [ ] Compare variance and optimization progress against equal-cost multi-sample/leave-one-out REINFORCE, established discrete gradient estimators and direct-feasible action masking. Count all extra likelihood/conditional-sampling work.
- [ ] Stop if the improvement only exists under exponentially expensive enumeration, if approximation bias is uncharacterized, or if equal-compute sampling closes the gap.
- [ ] Only after a useful scalable mechanism exists, evaluate other constrained subset tasks, such as sensor coverage and budgeted feature acquisition. A pure RL claim needs evidence beyond graph regression.

## 8. Stage 6 — Optional adaptation, only after testing the surrogate

**Existing files:** `rls/pair_tta.py`, `rls/tta.py`, `rls/notebook_workflow.py::adaptation_trial/validation_gates`, notebook D.

- [ ] On validation data only, generate candidate masks/updates and test whether uncertainty/physics surrogate improvement predicts lower true error. Report negative-transfer frequency and calibration changes.
- [ ] Compare no adaptation, random/local mask search, physics-only, uncertainty-only and a relevant GTrans-style adaptation baseline at the identical final budget and oracle-call allowance.
- [ ] Select stopping/fallback rules on validation. Any rule used on test must operate without labels. Explain whether each graph resets to the offline policy or shares adaptation state.
- [ ] Do not call a first-draw categorical KL a full-action trust region. Derive or measure the actual control before making stability guarantees.
- [ ] Stop this route if the surrogate rewards overconfidence or gains vanish after budget/query matching. No physics penalty automatically prevents reward exploitation.

## 9. Stage 7 — Confirmatory evaluation and manuscript

- [ ] Freeze the method, thresholds, budget grid, datasets/splits, primary endpoints, seeds and comparison protocol before the final run. Choose the primary strong comparator on development data; disclose all prespecified secondary comparisons and account for multiplicity rather than selecting a favorable test comparison.
- [ ] Re-run all shortlisted methods; include failed seeds and unsuccessful Stage B/TTA results in the record. Use paired uncertainty estimates and distinguish exploratory from confirmatory analyses.
- [ ] Produce six core artifacts: cohort/provenance table; accuracy–budget curve; accuracy–runtime curve; transfer/degradation results; mechanism ablations; stability/physical-interpretation checks. Add an estimator variance/cost figure only for an RL contribution.
- [ ] Write claims from results. Avoid claims of causal discovery, universal OOD safety, uniform feasible-mask sampling or inference acceleration unless explicitly supported.
- [ ] Choose the venue by the established contribution: ApJ/MNRAS for an astronomical finding; TMLR for a rigorous informative general ML study; a major ML conference if the methodological contribution and broad evidence justify it. Review current venue rules/deadlines at that time.
- [ ] Release configurations, environment, source revision, split manifests, per-example predictions, checkpoint metadata and data acquisition instructions consistent with source licenses. Keep smoke/synthetic artifacts visibly separate.

## 10. First executable work package after approval

Limit the first package to Stage 0, Stage 1A, the physical-schema/augmentation audit in Stage 1B, and the Stage 2 benchmark specification plus a bounded post-hoc development evaluation using existing scores. Its reviewable outputs are: a green relevant test suite, an invariance counterexample fixed under a declared tie policy, matched-budget development metric tables, an auditable schema, and a costed pilot configuration. Tiny-graph enumeration can proceed alongside the data audit; corrected-backbone training and the trained Stage 3 prototype follow that review. Large catalog acquisition, TTA sweeps and standalone RL development do not start automatically.

This plan is complete as a research roadmap. Each chosen coding package should receive its own implementation steps and regression tests once its experimental contract is accepted; unresolved research hypotheses must not be turned into promised implementation outcomes.
