# RL methodology audit

Audited base: `c825bf57652005c80faff79cf4dfd58e3a9baa14`. Audit date: 16 October 2026. This describes the recovered pair-PL experiment and the current exact-budget evaluator separately. Historical Bernoulli/top-k scaffolds are not interchangeable implementations.

## State and graph

Each halo is a one-step contextual decision. The frozen predictor supplies a full-graph node embedding `emb` and its mean `ctx`. For each stored directed edge, the policy consumes the two endpoint embeddings, five edge features, and graph context. Labels enter the training reward, never the policy input. Retention target, previous actions, and current degree are **not** explicit policy inputs. The decoder handles the requested budget.

The recovered predictor uses 64-dimensional embeddings. Node inputs are log stellar mass, log velocity dispersion, log half-mass radius, and log metallicity. Edge inputs are 3D distance, relative velocity magnitude, the cosine between relative position and velocity, signed log stellar-mass ratio, and separation in the xy plane. The checkpoint specifies radius graphs at 2 Mpc with self-loops. The builder also attempts to connect input isolates; no-isolate pruning constraints apply to the resulting topology. The xy feature means this is not a strictly rotation-invariant observational representation. See [graph builder](../../graph/graph_builder.py), [policy](../../rls/policy.py), and [graph preparation](../../rls/train_policy.py).

Inputs retain the `legacy_total_radius` research label: the existing CSV/weights do not certify an audited M200c target or corrected stellar-radius measurements. These experiments test pruning behavior on that fixed benchmark, not calibrated cosmological inference or cross-simulation generalization.

## Policy and actions

`EdgePolicyNet` projects edge features and endpoint/context embeddings to width 64, applies LeakyReLU(0.1), concatenates four projections (width 256), then applies a 256→64→64→1 MLP with LeakyReLU between layers. Endpoint and context projections share weights. Training-derived mean/standard deviation normalization is enabled, with scales floored at 1e-4 and standardized inputs clipped to [-10,10]. Final-layer weights start with standard deviation .01 and zero bias. A separate context→64→64→1 value MLP is trained.

The training action is an **ordered Plackett–Luce sample of distinct physical non-self pairs without replacement**. Directed logits are averaged over stored copies of each unordered pair. The sample contains `ceil(q * P)` pairs for P available physical pairs. The action likelihood is the sum of successive categorical log probabilities. It is not the probability of the final repaired mask: multiple sample orders and repair realizations can produce that mask. The graph-level decision horizon is one; this is REINFORCE, not PPO. Sequential draws implement sampling without replacement, rather than a learned multi-step graph-editing process. See [pair actions](../../rls/pair_policy.py) and [pair trainer](../../rls/pair_training.py).

## Reward and physics

For one halo, training uses

```
R = clip(w_acc * (|y_full-y| - |y_pruned-y|)/s
         - w_sp * min((q_realized-q_target)^2, 1)
         + w_conn * (1 if no physical isolates else -2)
         - w_virial(epoch) * V, -reward_clip, reward_clip)
V = [log(r_pruned) - log(r_full)]^2
r = 2 KE / |PE|
```

`compute_rewards` calls these errors RMSE, but the trainer calls it one halo at a time, so each is an absolute error. Scale `s` is fixed from full-backbone TRAIN RMSE with floor .05 dex; the recovered CPU reconstruction is 0.124135822057724 dex. It was not exported in the original GPU run and is not claimed bitwise identical. Defaults: `w_acc=1`, `w_sp=.5`, `w_conn=1`, `w_virial=1`, `reward_clip=10`.

The virial weight is zero at epochs 0–9, .1 at epoch 10, and reaches one at epoch 19 (zero-based indexing). Its annealing formula is `min(1,max(0,(epoch-10+1)/10))`. This is distinct from the retention curriculum.

`_graph_physics_terms` excludes self-loops. KE is `.5 sum_i (retained_degree_i/full_degree_i) Mstar_i sigma_i^2`; PE is `sum_retained_directed_edges G Mstar_u Mstar_v/r_uv`, with `G=4.302e-9`, positions in Mpc, masses in solar masses, velocity dispersions in km/s, and distance floor 1e-6. Bidirectional stored edges count twice in PE; that common convention cancels in the ratio relative to the same full graph. Ratios are clamped to [1e-12,1e12] for numerical finiteness. This is a stellar-mass and degree-weighted surrogate for preserving the full graph's ratio. It does not enforce equilibrium, recover dark-matter potential energy, or establish physical causation. The older one-sided absolute virial penalty remains in the module but is not used by pair training. See [reward functions](../../rls/rewards.py).

## Budgets, symmetry, isolation, and loops

| Mechanism | Recovered training | Exact evaluation |
|---|---|---|
| Budget unit | Distinct non-self physical pairs | Same |
| Requested count | `ceil(q P)` | Same |
| Symmetry | One pair action expands to all directed copies | Same |
| Isolation | Additive repair after sampling | Feasibility-aware selection during ranking/drawing |
| Budget after constraints | May exceed requested count | Exactly requested count or explicit infeasibility |
| Connectivity | No physical isolates; disconnected components allowed | `no_isolates`, not spanning connectivity |
| Self-loops | Always retained; not stochastic actions or physics interactions | Same |

Training repair visits pairs using independent exchangeable marks and adds an edge if an endpoint is uncovered. It can leave genuine input-topology isolates and cannot invent missing physical edges. The exact decoder uses maximum-matching-based completion feasibility for no-isolate coverage. Impossible graphs must be reported/excluded by topology-only preflight; the budget must not be enlarged. The required minimum is not always `ceil(N/2)`; star graphs are a counterexample. See [constraints](../../rls/constraints.py), [exact decoder](../../rls/constrained_policy.py), and [cohorts](../../rls/notebook_budget.py).

Training averages stored directed logits. The primary evaluator additionally averages policy outputs over both endpoint orientations while reversing the signed mass-ratio feature, then averages pair copies. It uses fixed per-graph exchangeable marks for ties. This orientation treatment is a training/evaluation difference that must be held fixed across ablations. Edge-order covariance does not make the endpoint-concatenation MLP intrinsically orientation invariant.

## Training and inference protocol

The original split has 378 TRAIN, 81 VAL, and 82 TEST halos; split seed 42. All 378 graphs are used in recovered legacy-repaired training. Policy training ran 60 epochs with Adam learning rate .001, graph batches of 32, four IID rollouts per graph, and separate actor/critic optimizers. The target retention anneals linearly .9→.4 over 40 epochs, then stays .4. Each epoch shuffles graphs. The predictor is frozen and in evaluation mode; embeddings and full predictions are cached.

For each action, the actor baseline is the average reward of the **other** rollouts on that halo. No batch-standardized advantage is used. The value MLP minimizes smooth-L1 loss to mean rollout reward, weighted .5; its output is **not used in the pair actor baseline**. The entropy coefficient .01 regularizes normalized entropy of the first categorical draw, not the full mask entropy. Actor and critic gradients are separately clipped to norm one. See [training implementation](../../rls/pair_training.py); the older general trainer's prose about a learned actor baseline does not describe this pair branch.

Checkpoint selection uses minimum exact-decoder validation RMSE at the final budget, including the untrained initialization. The recovered manifest records epoch 42 (zero-based) as selected after 60 completed epochs. The selected policy, not the final-epoch actor, is evaluated. The recovered provenance was reconstructed from original source revision `aeb5dad65d681728d525fb78a4ab919b0432ab90` and exact inputs; training was not resumed during recovery.

Frozen test inference computes full-graph embeddings, scores pairs with the two-orientation evaluator, selects exactly `ceil(qP)` feasible pairs, preserves loops, and runs the frozen GNN once on the selected graph. No label enters selection. This still requires a full-backbone embedding pass; compression is not evidence of end-to-end speedup. Stage-B predictor fine-tuning and label-free test-time adaptation are separate interventions and are disabled in these ablations. `w_unc` and TTA KL settings do not enter offline pair training. `min_keep_frac` is not used by the pair-PL branch.

## What can currently be attributed

The historical log demonstrates that a nonzero relative-virial penalty was evaluated and that the legacy repair typically increased retention. It cannot show that the penalty improved prediction. `w_conn` adds a constant when repair eliminates isolates, so its actor contribution cancels in leave-one-out advantages unless reward clipping changes this cancellation or isolation varies. Under an exact fixed count, the sparsity reward is also action-independent (including rounding residual); under additive repair it can vary. The detached, separate critic cannot change the actor through `value_coef` in this implementation. Those structural facts motivate negative controls; they do not substitute for empirical reward retraining.
