# Astronomy evaluation preflight

`rls.astronomy_preflight.preflight_astronomy` performs an offline metadata and
content preflight before a projected astronomy evaluation. It selects one
manifest catalog and role, content-audits the local HDF5 source, loads it
through the strict catalog adapter, checks target/unit semantics, validates a
training split manifest's stored content hash, and checks that a projected
backbone checkpoint and policy artifact agree on feature schema, training
catalog, split, observation-transform payload, and backbone state. It also
checks train/validation/test split membership fields for internal uniqueness
and disjointness and requires initial-condition/lineage grouping.

The returned record sets `metadata_preflight_passed` when those checks pass.
It always sets `scientific_evaluation_allowed` and
`evaluation_execution_ready` to `false`: this function does not construct the
model, load either state dict into an architecture, build projected graphs,
test per-graph budget feasibility, or run inference. The later runner must do
those checks and use `rls.matched_benchmark.evaluate_matched_budget` for actual
comparisons. `evaluation_executed` is always false here.

Example caller setup (paths and values must come from the real experiment):

```python
from data.observations import ObservationConfig
from data.projected_graph import ProjectedGraphConfig
from rls.astronomy_preflight import preflight_astronomy
from rls.matched_benchmark import BudgetSpec

record = preflight_astronomy(
    manifest_path="config/astronomy_manifest.yaml",
    catalog_id="explicit-catalog-id",
    expected_role="external_development",
    split_manifest_path="outputs/projected/split_manifest.json",
    checkpoint_path="outputs/projected/backbone.pt",
    policy_path="outputs/projected/policy.pt",
    graph_config=ProjectedGraphConfig(method="radius", radius_mpc=2.0),
    observation_config=ObservationConfig(line_of_sight="z", seed=0),
    transform_config={"scenario": "declared-observation-case"},
    budget=BudgetSpec(keep_fraction=0.5, constraint="no_isolates"),
    evaluation_mode="scientific",
)
```

`ObservationConfig` is the source of truth for LOS, missingness, noise,
richness, and random seed. `transform_config` carries scenario labels only;
it cannot override those settings. Its output identity is computed from the
actual `ObservationConfig`. A `BudgetSpec` accepts an exact `pair_count` or
`keep_fraction`; a fraction resolves to a physical-pair count per graph only
once the projected graph and decoder run. The preflight reports the requested
specification and leaves achieved counts unavailable.

Scientific mode requires a manifest entry whose source kind is
`local_hdf5`, a successful content audit, and the requested role. The manifest
must declare target `log10(M200c/M_sun)` and supported field/unit semantics.
An entry declared `synthetic_fixture` is rejected. Tests that use temporary
synthetic files call `evaluation_mode="fixture_smoke"`; the output then marks
`fixture_smoke_ready=true` while keeping scientific execution disallowed.
That mode exists only to exercise the contract wiring. Text in the release
name never selects the mode.

Artifacts must be structured checkpoints with `artifact_type` equal to
`projected_backbone` or `projected_policy`, model weights under the existing
`model_state_dict` key or the policy `policy_state_dict` key, and embedded
`provenance`. The provenance needs a corrected-research catalog contract,
content-derived contract/config/model hashes, split hash, source revision and
source-code file identities. Both artifacts also need the projected
compatibility record from `data.projected_graph`, the shared training
observation-config identity, and matching split/catalog identity. The policy
must record the exact backbone state hash it was trained against. Bare legacy
state dicts, Git LFS pointer files, missing files, and artifacts without
projected semantic metadata fail closed. The existing legacy `policy.pt` and
3D checkpoints do not become projected artifacts by matching tensor widths;
they must be retrained/exported with the required semantic metadata.

The evaluation catalog HDF5 source is rehashed during audit and again at the
strict loader boundary. Its full file record is returned. Training source-file
records, revision, split identity and code-file identities are copied from
artifact provenance; this function does not rehash those historical source or
code paths. The embedded training observation-config payload is hashed and
bound to both artifacts and their resolved configuration. The projected
compatibility record is also bound to each resolved configuration, though the
actual model architecture and tensor shapes still require downstream loading.
`training_split_membership_verified=false` means the split file's embedded
hash and internal group/cluster/parent disjointness are checked, but its member
IDs are not checked against the training catalog rows. The record reports
these limits rather than treating hashes as proof of executed training or
evaluation.

The current repository does not contain projected trained artifacts or
audited real CAMELS/SIMBA catalogs. Therefore fixture tests exercise format
and rejection behavior only; no scientific preflight success or astronomy
result is claimed from those tests.
