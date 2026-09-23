# Bounded feasible-selection pilot

The runner compares repaired PL, scaffold plus residual PL, direct feasible
selection, and exact finite feasible-mask Gibbs on frozen synthetic K4
predictors. It is an optimization diagnostic and supports no astrophysical or
general superiority claim. The structured control enumerates masks and is not
a scalable edgewise perturb-and-MAP method.

The approved campaign uses policy seeds `[0, 1, 2]`, data seed `1001`,
validation seed `2001`, 16 training examples (8 per family), 8 validation
examples (4 per family), ten epochs, physical-pair budgets `[2, 3, 4]`, four
independent reward rollouts, and CPU execution. Train-only input normalization
and a fixed train-only squared-target reward scale are shared by all methods.
Each method/seed/budget starts from a matched scorer initialization. Validation
uses deterministic rank/MAP selection with transported independent tie marks.
Validation labels do not tune the runner.

Run the capped campaign once into a new output directory:

```bash
python3 -m rls.feasible_pilot --config config/feasible_pilot.yaml \
  --output-dir outputs/feasible-pilot/capped-001
```

The runner rejects altered capped settings and an existing output path before
it creates the output directory or generates data. To exercise the path at
small scale, run:

```bash
python3 -m rls.feasible_pilot --smoke \
  --output-dir outputs/feasible-pilot/smoke-001
```

Smoke output is explicitly labeled and cannot substitute for the capped run.
Both modes write `summary.json`, `curves.json`, `val_per_example.json`, and
`provenance.json`. The summary records actual predictor calls and graph
evaluations by phase, elapsed time, duplicate masks among the actual training
rollouts, constraints, update norms, and the validation convention. The
provenance records source, data, split, configuration, predictor, and
normalization identities plus seed roles and IDs.

Validation rows record the negative squared-error objective, exhaustive oracle
objective and regret, signed/absolute prediction error, requested and final
physical retention, and selected versus available stored-edge retention.
Per-epoch curves report mean objective, mean squared error, and RMSE.

For the capped run, expected calls are 23,040 training graph forwards across
all methods, 2,880 validation forwards, 816 shared exhaustive-oracle forwards,
and 24 full-graph reference forwards. Oracle outputs are diagnostic only and
never enter policy optimization. The predictor meter slices the stored graph
edges physically for each selected mask. Each reported call is one frozen
predictor forward on one graph.

At `k=2`, the fixed no-isolate scaffold consumes the entire feasible budget,
so its residual policy has no learnable action and should have zero parameter
update. At `k=3` and `k=4`, residual choices remain and scorer updates are
possible. Repaired PL's likelihood is the ordered pre-repair action
likelihood, not the probability of its projected final mask. The structured
method uses the exact unordered Gibbs likelihood over all feasible masks.
Three seeds provide descriptive variation only; they do not establish
statistical significance or method superiority.
