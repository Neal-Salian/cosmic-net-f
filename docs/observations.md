# Projected observation transform

`data.observations.observe_halo(halo, ObservationConfig(...))` handles the legacy complete-schema input. `data.astronomy_observations.observe_astronomy_halo(halo, ObservationConfig(...))` accepts immutable `AstronomyHalo` records returned by the strict audited astronomy loader. Both return a copied observation and a JSON-safe provenance report. A `None` copy means transformed richness fell outside the declared inclusive bounds; the report retains pre- and post-selection counts and rejection reasons. Inputs require stable catalog, halo, and member IDs.

The observation schema is `projected_observation_v1`. Select LOS `x`, `y`, or `z`; positions are physical Mpc and velocities are km/s. The LOS coordinate and transverse velocities are zeroed. Sky positions are centered on the retained-member sky centroid. The legacy complete-schema path clears its intrinsic dispersion, half-mass-radius, and metallicity attributes for compatibility. The partial astronomy path carries only observed ID, position, velocity, and stellar mass; it never fabricates unavailable dispersion or metallicity fields. Both exclude hidden quantities from the positive observable allowlist (`projected_position`, `los_velocity`, `stellar_mass_proxy`). Halo mass remains on the copied record strictly as the label; transforms, centering, noise, and selection do not read it. Target mass and target-derived `R200c` metadata do not affect observables or member selection.

Missingness is a stable-ID-keyed Bernoulli draw. Gaussian noise can be configured for projected-position Mpc, LOS velocity km/s, or stellar-mass dex. The same identity, seed, and realization replay regardless of member ordering. Nonpositive or nonfinite stellar mass and invalid physical units fail closed. The observation report records transform order, settings, catalog and halo IDs, LOS, and richness accounting. For strict astronomy records, source hash, role, initial-condition identity, release, selection accounting, and catalog contract stay in the copied halo's metadata; `build_projected_graph` carries those fields into `graph.catalog_provenance`.

Pass the result to `data.projected_graph.build_projected_graph` to rebuild edges from projected 2D coordinates and LOS velocity. The legacy `GraphBuilder` rejects projected copies.

Example for the legacy input:

```python
from data.observations import ObservationConfig, observe_halo

config = ObservationConfig(
    line_of_sight="z",
    seed=31,
    realization_id="mock-observation-0",
    missing_fraction=0.1,
    position_noise_mpc=0.02,
    los_velocity_noise_kms=5.0,
    min_richness=5,
)
observed_halo, transform_record = observe_halo(halo, config)
```

For an audited partial catalog, call `observe_astronomy_halo` from
`data.astronomy_observations`, then call
`build_projected_graph(observed_halo, observation_report=transform_record)`.
The copied record retains its `partial_observation_inputs_v1` source identity.
