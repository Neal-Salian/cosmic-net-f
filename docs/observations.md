# Projected observation transform

`data.observations.observe_halo(halo, ObservationConfig(...))` returns a copied `HaloData` and a JSON-safe provenance report. A `None` copy means the transformed member count failed the declared inclusive richness bounds; the report still records pre- and post-selection counts and rejection reasons. The minimum richness must be an integer of at least one; the optional maximum must be an integer. Inputs need stable `metadata.catalog_id`, `cluster_id`, and unique `subhalo_id` values.

The current schema is `projected_observation_v1`. Select LOS `x`, `y`, or `z`; positions are physical Mpc and velocities are km/s. The LOS coordinate and transverse velocities are zeroed in the observation payload. Sky positions are centered on the retained-member sky centroid. Intrinsic dispersion, half-mass radius, and metallicity are zeroed and excluded from the positive observable allowlist (`projected_position`, `los_velocity`, `stellar_mass_proxy`). The halo mass remains on the copied halo strictly as the label; transforms, centering, noise and selection do not read it. Target mass and any target-derived `R200c` metadata do not affect projected positions, LOS velocities, stellar-mass noise or member selection.

Missingness is a stable-ID-keyed Bernoulli draw. Gaussian noise may be configured in physical projected-position Mpc, LOS velocity km/s, or stellar-mass dex. The same identity, seed, and realization replay regardless of member ordering. Nonpositive/nonfinite stellar mass and invalid physical units fail closed. Transform order and settings are retained in report metadata. This module does not construct graph tensors; a later projected graph adapter must consume the schema and rebuild edges from transformed members.

Example:

```python
from data.observations import ObservationConfig, observe_halo

config = ObservationConfig(
    line_of_sight="z",
    position_units="Mpc",
    velocity_units="km/s",
    stellar_mass_units="M_sun",
    seed=31,
    realization_id="mock-observation-0",
    missing_fraction=0.1,
    position_noise_mpc=0.02,
    los_velocity_noise_kms=5.0,
    min_richness=5,
)
observed_halo, transform_record = observe_halo(halo, config)
```
