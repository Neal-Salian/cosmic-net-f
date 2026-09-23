# Projected graph adapter

`data.projected_graph.build_projected_graph(observed_halo, config,
observation_report=None)` accepts a copied `HaloData` produced by
`data.observations.observe_halo`. It requires the `projected_observation_v1`
schema and the exact positive allowlist: projected position, LOS velocity and
stellar-mass proxy. It never calls legacy 3D feature accessors.

The fixed node schema is the two sky coordinates in Mpc divided by the declared
position scale, LOS velocity in km/s divided by the declared velocity scale,
and `log10(stellar_mass / stellar_mass_reference)`. Defaults are 1 Mpc,
1000 km/s and 10^10 M_sun. Projected distance and absolute LOS velocity
difference edge values use the same position and velocity scales; log mass
ratio is dimensionless. The projected 2D coordinates and single LOS velocity
component drive edge rebuilding. Radius and kNN edge construction,
tie inclusion, reciprocal edge coalescing and isolated-node repair reuse
`GraphBuilder` conventions. Edge features are projected distance, absolute
LOS velocity difference and log stellar-mass ratio; unknown names fail.

The graph stores only 2D `pos` and 1D `vel`; it does not store intrinsic
dispersion, radius, metallicity, hidden LOS position or transverse velocity.
The target remains in `y` for supervised use. It does not enter graph inputs,
edge construction or fixed normalization. `feature_mode`, graph schema,
observable list, LOS, units, scales and schema hash form a checkpoint
compatibility record. Use `checkpoint_compatibility_record` when saving a
model and `validate_checkpoint_compatibility` before inference. A same-width
legacy 3D checkpoint fails semantic compatibility.

`observe_halo` stores a deep copy of its complete transform report on the
observed halo, so the default graph call carries missingness, noise seed,
richness and rejection details. A caller may also pass `observation_report`
explicitly; the builder validates it against the copied observation metadata.
The copied observation metadata binds report identity to
catalog, halo, LOS, realization ID and seed; the builder rejects a mismatched
report, requires the full explicit report to equal the stored copy, and checks
its declared transform entries. Member-order invariance is
inherited from the stable-ID keyed observation transform; graph node order
follows input order.

When catalog metadata declares `initial_condition_id` or the shared catalog
contract's `volume_or_ic_group`, the graph exposes it as `initial_condition_id`
and `lineage_group` for grouped inference and matched-budget pairing. Catalog
provenance also retains `release`,
`selection_accounting`, role, source and contract fields when present.

## Astronomy loader integration boundary

The current strict catalog loader emits immutable `AstronomyHalo` and
`AstronomyMember` records with only target, position, velocity, stellar mass
and radius. The current `observe_halo` API accepts legacy `HaloData` with
additional required intrinsic fields. This adapter deliberately does not
manufacture those unavailable fields to bridge the two types. Until a
partial-observation transform entry point is added, audited astronomy loader
records cannot enter the Task 5b observation-to-graph pipeline directly. This
is a Task 5c integration item; it must preserve only declared observables and
must not create substitute intrinsic measurements.
