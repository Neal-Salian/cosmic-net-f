# Audited projected full-graph baseline

The corrected baseline uses explicit content-audited astronomy HDF5 catalogs, measured projected observables, and supervised MSE. Audited partial records have no measured velocity dispersion or metallicity. No missing physical fields are fabricated.

Copy `config/audited-baseline.example.yaml`, supply a private manifest following `docs/astronomy_manifest.md`, and explicitly list the development catalog IDs. Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m training.full_graph_baseline \
  --config /absolute/path/to/audited-baseline.yaml \
  --output-dir /absolute/path/to/new-baseline-run
```

All selected catalogs must have role `development`. Scientific training requires `source.kind=local_hdf5`, `data.strict_research: true`, and `training.evidence_mode: scientific`. Selected hashes, shapes, units, epoch, membership and target semantics are checked before output creation. A legacy `catalog_contract` cannot bypass the audited source boundary. Unsupported graph schemas, physics losses, augmentation and invalid training settings fail before output creation too.

A catalog declares one independent initial-condition identity. Halos within that catalog cannot be randomly divided across train, validation and test. List enough genuinely independent development catalogs for nonempty whole-IC splits. The splitter allocates `floor(number_of_groups * train_ratio)` train groups, `floor(number_of_groups * val_ratio)` validation groups and the remainder to test. The four-IC example yields 2/1/1 groups; the default 0.7/0.15/0.15 ratios need at least seven groups. Related snapshots/views must declare the same IC/lineage ID. The code checks declarations and membership; it cannot establish independence from source physics. Grouped assignment covers every raw audited halo before observation or richness selection, including halos with no selected members. Rejected IDs and groups remain in the split manifest so evaluation can verify the complete raw catalog population. Observation and rejection reports identify their assigned split and IC. Selection must leave at least one model graph in each split before output creation or training.

Observation settings are `ObservationConfig` fields; graph settings are `ProjectedGraphConfig` fields. Node columns are the two centered sky coordinates, LOS velocity and log stellar-mass proxy. Missingness/noise is deterministic by catalog, halo, member, realization and seed. Graphs use fixed physical scales with no fitted normalization or held-out statistics. Splits preserve catalog-qualified halo IDs, and all observation/rejection reports are retained. The model trains on train graphs, restores the lowest validation MSE state, then predicts test membership. The local loop supports Adam/AdamW and no/cosine LR scheduling. R² is JSON null for constant/singleton targets; scatter uses population standard deviation.

Outputs are `projected_backbone.pt`, its SHA-256 sidecar, `resolved_config.json`, `split_manifest.json`, `training_history.json`, `heldout_predictions.json`, `observation_reports.json`, `rejected_halos.json`, `provenance.json` and `results.json`. The structured backbone embeds its complete model state/hash, resolved architecture/config hash, split hash, projected compatibility record, training observation payload/hash, each selected catalog contract/source hash, manifest identity and training source-code hashes. This is the backbone boundary consumed by `rls.astronomy_preflight` and `rls.astronomy_evaluation`. Keep the split file with the artifact. Development evaluation must choose an IC entirely in validation/test. The `scientific_result` flag remains false because training alone does not establish a controlled comparison result.

## Fixture evidence

Use `training.evidence_mode: fixture_smoke` for tiny integration fixtures. This is the only training mode opting into declared `synthetic_fixture` sources. Fixture contracts retain `synthetic: true` and `research_label: synthetic`; run/artifact provenance stays `fixture_smoke`. Explicit fixture mode retains that label even for a `local_hdf5` input. Scientific artifact loading rejects fixture-trained provenance. Fixture artifact metadata loading is allowed only in `evaluation_mode: fixture_smoke`. The astronomy evaluation catalog boundary continues rejecting `synthetic_fixture` sources; existing tiny evaluator fixtures use explicit local HDF5 and fixture evaluation mode. Release text cannot prove externally supplied data are real: correctly declare source kind and evidence mode.

Legacy `--smoke` still creates its separate seeded synthetic intrinsic-feature fixture. It is not an audited projected artifact or scientific experiment. No real-data or expensive training run was executed for implementation verification.

## Projected policy boundary and blocker

No policy is exported. The historical RL physics reward expects intrinsic 3D/dispersion statistics absent from audited projected records. A scientific paired policy needs a separately validated projected-observation objective/trainer. Fake dispersion, metallicity, zero physics penalties, random policy weights or historical intrinsic weights cannot establish that policy.

A future paired artifact must have `artifact_type: projected_policy`, complete `policy_state_dict`, positive provenance with matching catalog-contract/split identities, `backbone_state_sha256` equal to the exported backbone model-state hash, exact `projected_compatibility` and training observation payload/hash, complete `policy_config` also in hashed `resolved_config`, and ordered `policy_edge_feature_names` matching the projected edge schema. Its node embedding dimension must equal the backbone output dimension; its edge dimension must match the graph. Both states must load strictly into declared architectures. Preflight/evaluation require this pair; the baseline alone cannot execute a provided-policy comparison.
