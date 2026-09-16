# Cosmic-Net research direction: proposed design

Status: planning only. No implementation or training is authorized by this document. Prepared from the saved Notebook B run and source at `f538726` on 2026-09-16. Assumption: Kaggle/single-GPU access, no fixed submission deadline. These assumptions can be revised without changing the scientific controls below.

## Recommendation and scope

Build one reliable experimental foundation, then pursue an astrophysics paper while running one bounded ML-method feasibility study. Advance the ML route only if it produces a general result beyond ordinary constrained sampling and existing graph sparsification. A standalone RL paper is conditional on a new estimator, optimization result, or substantially better learning efficiency; using REINFORCE is not that result.

The central question is: **which graph relationships are sufficient for reliable prediction at a stated computational budget, and which remain useful under a change of data distribution?**

| Route | Proposed contribution | Main risk | Decision |
|---|---|---|---|
| Astrophysics | Identify transferable information about halo mass in sparse galaxy relationships, and establish when that information fails under simulation/observation changes. | Physics proxies or selection effects explain apparent gains. | Primary application study. |
| General ML | A selector conditioned on an explicit budget, respecting structural constraints and node relabeling, that improves a measured accuracy–cost tradeoff across tasks. | Existing methods already provide the proposed mechanism, or simpler selection performs as well. | One controlled feasibility study before expansion. |
| General RL | More efficient learning over constrained subsets, possibly reducing variance from multiple latent actions with the same executed result. | Action masking and Rao–Blackwellization are established; scalable new contribution may not exist. | Optional theory/estimator pilot, not a promised second paper. |

The old `rl-implementation-plan.md` asserts that label-free adaptation is the core novelty and implies that it will recover out-of-domain accuracy. Those statements are hypotheses, not established results. This design supersedes those research assumptions, without changing the historical file or its implementation record.

## Current evidence

Notebook B uses 541 halos (378/81/82), mostly complete ten-node graphs. Its test full/RL RMSEs are 0.116717/0.134322 dex and R² values are 0.907471/0.877453. Mean total edge retention is 0.479109. Validation RL RMSE is 0.120995 versus random 0.143022 ± 0.005000 across ten mask seeds. No matched random test result is displayed. Stage B rejects all candidates. The last targeted suite recorded 78 passed, two failed, one skipped.

This supports a working pruning experiment. It does not establish benchmark superiority, acceleration, physical discovery, adaptation efficacy, or complete software verification.

## Hypotheses and falsifiers

**H-A, astrophysics:** at matched final physical-pair budgets, relationship selection provides reproducible predictive utility beyond richness, stellar-mass summaries, velocity summaries, and simple geometric/binding heuristics; some of this advantage transfers to unseen simulation conditions and observational degradations. Pair relationships can be deterministic functions of the same galaxy positions/features given to a node-set model. A gain therefore establishes useful inductive bias under controlled capacity/resources, not newly created information or a proof that edges are necessary.

Falsifier: a pooled node-set/aggregate model or geometric selector matches the gains, selection patterns fail independent transfer, or the physics term only improves its own rewarded proxy. A negative answer can still be informative if the scientific mechanism and limits are established; it does not justify an RL superiority claim.

**H-M, ML:** under tight feasible budgets, direct feasible selection can reduce optimization regret per predictor evaluation relative to sample-then-repair, a simple feasible scaffold, and structured perturb-and-MAP. The proposed mechanism is that repair maps distinct sampled actions to the same executed graph and can waste learning effort. Measure this mechanism rather than assuming it. A useful result must survive total-time accounting and transfer beyond one synthetic reward.

Falsifier: gains disappear after matching realized budgets, encoder capacity, predictor calls, or inference cost; the approach merely reproduces an existing method. A correct feasible selector is a useful baseline even if this hypothesis fails.

**H-R, optional RL:** accounting for redundant latent selection paths can reduce gradient variance per unit computation and improve sample efficiency for constrained subset problems.

Falsifier: equal-cost multi-sample REINFORCE/leave-one-out is as good, bias is introduced without adequate characterization, or the estimator cannot scale beyond enumerated toy cases. The conditional-score/Rao–Blackwell identity itself is established mathematics, not the contribution. [Discrete Rao–Blackwellized gradients](https://proceedings.mlr.press/v97/liu19c.html)

## Selection contract

- A physical edge is one unordered non-self pair. Reciprocal storage and self-loops are implementation details; self-loops never consume the physical budget.
- Requested budget is `k = ceil(q * P)` for P available physical pairs. Record k and achieved k separately, with no hidden repair additions.
- The primary structural constraint is no isolated vertices; a connected-graph constraint is a separate experiment with a different feasible set.
- For a graph with no isolated vertices, the minimum no-isolate budget is its minimum edge-cover size, `|V| - maximum_matching_size`, not universally `ceil(|V|/2)`. A connected spanning subgraph requires a connected input and at least `|V|-1` pairs. Preserving each of c existing components needs at least `|V|-c` pairs.
- Report infeasible graph/budget combinations. Do not silently increase budgets, invent edges, or remove difficult examples from all summaries.
- Require node-relabeling equivariance with unique scores/features. In symmetric tie cases, an exact-budget deterministic equivariant choice may not exist: specify random tie handling and require equivariance in distribution. Never break ties using numeric node IDs without disclosing the loss of symmetry.
- A parameter-independent repair can define a valid sample-then-repair objective with the original ordered action likelihood. Do not mislabel that likelihood as the probability of the final unordered repaired mask.
- A sequential feasible policy must normalize over its actual legal actions. Ordinary invalid-action masking already has theoretical precedent. [Huang and Ontañón](https://arxiv.org/abs/2006.14171)

## Candidate ML architecture and mandatory controls

Use the existing full-embedding policy as a reference. Develop a cheap, permutation-equivariant selector using raw node/edge inputs, a pooled context and explicit budget input. The cheap selector must not require the target GNN's full-graph embeddings if an inference-efficiency claim is intended. Target predictor weights remain frozen in the primary experiment; joint/retrained models form a separately labeled comparison.

Start with a simple independent scaffold: an edge cover for no-isolates, or a spanning forest for component preservation, then select remaining pairs up to k. Its construction and tie behavior must be explicit. Treat this as a baseline. A more ambitious candidate samples feasible subsets directly, masking actions only when a completion within the remaining budget exists. Validate feasibility using exhaustive small instances before choosing a scalable completion algorithm.

Compare learned scorers under the same decoder where possible, and decoders with the same scorer where possible. Compare REINFORCE with a differentiable selection relaxation using equal tuning and predictor-query budgets. Changing both encoder and constraint handling at once cannot isolate the contribution.

The novelty boundary is narrow: learned graph sparsification is established by [NeuralSparse](https://proceedings.mlr.press/v119/zheng20d.html), RL sparsification by [SparRL](https://arxiv.org/abs/2112.01565), and learned frozen-model subgraphs by [PGExplainer](https://proceedings.nips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html).

Closer structured competitors are mandatory: [L2XGNN, 2024](https://link.springer.com/article/10.1007/s10994-024-06576-1) learns constrained subgraphs, including connected k-edge graphs, through combinatorial optimization and perturb-and-MAP. [Stochastic Softmax Tricks, 2020](https://proceedings.neurips.cc/paper/2020/hash/3df80af53dce8435cf9ad6c3e7a403fd-Abstract.html) covers differentiable structured sampling. [Mixture of Graphs, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/e7947b5e1d30864ebbe8714dbdd611d9-Paper-Conference.pdf) uses context-dependent sparsification experts. The [TopoBudget August 2026 preprint](https://arxiv.org/abs/2608.07903) already combines a connectivity backbone and remaining edge budget in another application. Therefore neither feasible selection, budget conditioning nor scaffold-plus-residual construction is assumed novel. Differences in frozen versus joint training and all-node coverage versus connected motifs must be stated and controlled.

The scaffold must be independent of learned parameters if training only scores the residual PL action. Use shared scaffold realizations across compared scorers. At the minimum feasible budget a fixed scaffold may leave no learnable residual action; this is a support restriction, not evidence that learning cannot help.

## Scientific validity and interpretation

Use a single declared target definition, initially log10(M200c/M_sun). Audit mass, radius, position, velocity, h and scale-factor conventions per catalog; convert any M_sun/h values using that catalog's recorded h. Do not normalize inputs by a catalog R200c derived from the target mass. Do not equate total subhalo half-mass radius with stellar half-mass radius or substitute another halo mass definition silently. Rotate positions and velocities together for rigid augmentation; jitter is a separate perturbation and requires appropriate edge reconstruction. Projection-dependent features need an explicit line of sight and are not expected to remain invariant under changing that line of sight.

Treat the existing energy ratio as a model regularizer until its relationship to physical quantities is demonstrated. Stellar internal dispersion, galaxy orbital velocities, stellar pair binding and total halo binding are distinct. Correcting a radius name alone does not validate the energy model. Establish a pure supervised backbone before adding any physically motivated prior.

Compare full 3D simulation information with an observation-oriented feature set (projected positions and line-of-sight velocities with a declared noise/selection model). These are separate inference problems with separate baselines. Audit completeness and available halo richness before selecting the final population; the current first-ten cap is not a representative survey selection function.

A selected computational subgraph is not automatically a physical causal explanation or a minimal set of observations. Full-input selectors can encode discarded information in the mask. This is distinct from train/test leakage. Include mask-only, selector-recomputation, randomization and perturbation controls before strong explanation claims. [Local selection leakage, ICML 2024](https://proceedings.mlr.press/v235/oosterhuis24a.html)

## Evaluation and paper decision

Use existing 541 halos for development, not as a fresh final confirmation set. New final splits must be fixed before results are inspected; use simulation/initial-condition grouping for cross-simulation claims and hold related snapshots/augmentations together. Reserve external development simulations separately from external confirmation simulations: using a domain to choose a hypothesis, shift, threshold or baseline makes it development data even without gradient updates. Fit all normalizers and learned reward scales on training data. Select hyperparameters on validation only; retain all seeds and failed runs.

Primary endpoints: absolute prediction error and relative error to the full predictor at realized pair budgets, plus actual inference time for an efficiency claim. Report R², bias/scatter, error by mass/richness/domain, physical/total keep, isolated vertices/components, repair additions, runtime, memory, and predictor calls as secondary endpoints. Pearson fidelity is descriptive, not accuracy preservation.

Use three pilot seeds and at least five independent training seeds for shortlisted confirmatory methods. Distinguish policy seeds, backbone seeds, random-mask seeds, and halo/volume sampling uncertainty. Use paired comparisons on common examples and hierarchical or block uncertainty estimates when simulations or related halos are the independent sampling units. Do not treat every edge or projected view as an independent datum. After the catalog audit, use development variance and independent-volume counts for a prospective precision/detectable-effect assessment. Five optimizer seeds cannot compensate for an inadequately small number of independent simulation volumes. Freeze the primary comparator on development data and report secondary comparisons with appropriate multiplicity handling.

Plan astrophysical transfer before claiming generic ML performance. General ML expansion should use two non-astronomical task families and at least two backbone families, selected for the proposed claim. Candidate resources are [GOOD](https://arxiv.org/abs/2206.08452), [LRGB](https://arxiv.org/abs/2206.08164), and [OGB](https://ogb.stanford.edu/docs/graphprop/). Check graph-density feasibility first; aggressive pruning of sparse molecular bonds may conflict with constraints and must be described as message selection.

Defer test-time adaptation until a validation-only study tests whether its surrogate predicts true error improvement at identical budgets. Lower MC-dropout disagreement can mean overconfidence. Frozen graph adaptation and structural alignment are already established directions. [GTrans](https://arxiv.org/html/2210.03561v2), [TSA](https://proceedings.mlr.press/v300/hsu26a.html)

Astrophysics venue fit depends on new astronomical insight: [ApJ scope](https://journals.aas.org/scope-statements/) and [MNRAS criteria](https://academic.oup.com/mnras/pages/General_Instructions). A technically rigorous, informative general ML study can also fit [TMLR's evidence-centered criteria](https://jmlr.org/tmlr/acceptance-criteria.html); SOTA and a novel algorithm are not universal publication requirements. No acceptance probability or submission date is assumed.
