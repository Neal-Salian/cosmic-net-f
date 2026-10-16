# Data, target and policy feature definitions

Scope: the completed A–D campaign at `c825bf57652005c80faff79cf4dfd58e3a9baa14`.
Definitions below describe the actual legacy computation. They do not relabel
it as a corrected dataset. See [provenance-measurements.json](provenance-measurements.json)
for full machine evidence and [provenance-status.json](provenance-status.json)
for claim classifications.

## Dataset identity and extraction

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `data/raw/tng100_clustered.csv` | 1,068,424 | `6c148f2cae68abac7bc5c34d370ff0130a647b49e79c32880bcf36ecc0ab4d53` |
| `data/raw/tng100_clustered.h5` | 451,771 | `99d4b8e3b190e00a110c3253a6025f348648e749eff58418299ab6dae5050fa0` |
| `data/raw/checkpoint.csv` | 705,011 | `ef7e28e498668c326805e1f86331607a668aa8b2350ce6b89dcd608abbb64933` |

The CSV has **5,378 rows, 541 FoF group IDs and 5,378 unique subhalo IDs**.
The existing clustered CSV is the selected ingestion path in A–D; neither an
API fallback nor the synthetic/CAMELS adapters supply these campaign graphs.
Nearby `test_synthetic_*` files are unrelated fixtures, not extra campaign data.
All 18 HDF5 columns agree numerically with the CSV. This is a second encoding
of the same dataset, not an independent catalog. There are zero exact duplicate
CSV rows, zero repeated subhalo IDs and internally consistent group targets.
The extraction checkpoint has 5,000 rows/500 groups and overlaps in 4,970 IDs;
1,149 overlapping metallicity values differ (the final extraction applies
cleaning). Its other shared columns agree at the audit tolerance. It is an
intermediate overlapping catalog, not a valid external evaluation set.

`scripts/fetch_tng_data.py` declares TNG100-1 snapshot 99, h=0.6774. It requests
up to 1,200 massive central-subhalo candidates, fetches parent `Group_M_Crit200`,
sorts the successfully fetched candidates, takes up to 600 groups, requires
log mass >=12.5, caps fetched members at ten, drops rows missing critical
features and then requires >=3 surviving members. Actual final sizes are
515 groups with ten members, 21 with nine, four with eight and one with seven.
The loader's `n_halos=600` is not a guarantee of 600 CSV graphs, and loading a
clustered CSV does not reacquire all satellites.

Missing acquisition responses, failed-request/selection accounting and release
checksums prevent verifying completeness or a representative selection function.
The "top 600 most massive halos with all subhalos" description is invalid:
candidate selection is by central subhalo mass and membership is capped/cleaned.
The dataset metadata names `rusheelsharma/cosmic-net-tng100-clustered` and declares
CC0; the A–D attachment is `nealsalian/cosmicnet-data`. Matching hashes establish
byte identity across these local/campaign uses, not independent source licensing
or an authenticated API acquisition record.

## Prediction target

Current computational target, VERIFIED:

```text
M_csv = CSV halo_mass                 # declared linear Msun
y = log10(M_csv / Msun)               # one graph label
RMSE = sqrt(mean((prediction - y)^2)) # dex
```

The redundant `halo_mass_log` agrees with log10(`halo_mass`) throughout the
CSV, and the mass is constant within every FoF group. The final graph target
is `HaloData.log_halo_mass`, cast to float32; C's saved targets match that
construction exactly. Range before float32: **12.504518514850922 to
14.581092484003346 dex**. No target standardization or inverse-transform fit
is present in the current legacy path.

Intended physical target, PARTIALLY VERIFIED: the fetcher converts
`Group_M_Crit200 * 1e10 / h` to solar masses. The local raw groups 0 and 1
match **3.768138920412607e14** and **3.811469808735607e14 Msun**, respectively;
their GroupMass, M200m, M500c and top-hat alternatives differ. Raw coverage is
only two of 541 groups. Thus "legacy mass label intended as M200c, with partial
raw verification" is defensible; "all 541 masses are independently verified
M200c" is not.

M200c encloses **total matter** within a sphere averaging 200 times the critical
density at the epoch. It is not dark-matter-only mass and is not the FoF sum
`GroupMass` or bound `SubhaloMass`. Definitions and units follow the
[IllustrisTNG group-catalog specification](https://www.tng-project.org/data/docs/specifications/#sec2).
The loader deliberately retains `target_field=unknown` for unmarked legacy
CSVs; this audit does not silently upgrade that contract.

## Coordinates, node and edge features

The source catalog uses ckpc/h positions and radii. Current A–D declares
`LEGACY_CSV_UNITS`, converts lengths once by `1e-3 / 0.6774` and exponentiates
log stellar masses once. At snapshot 99, a≈1, converted lengths are physical
Mpc. Bulk subhalo velocities remain km/s peculiar velocities. These conversions
are verified computationally for the campaign and physically against the two
matched raw groups. Generalization to other epochs is not established by
these data. The older loader's magnitude heuristics are different; see
[backbone-audit.md](backbone-audit.md).

| Input | Actual construction | Interpretation / classification |
|---|---|---|
| Node x[0] | log10 stellar mass in Msun, from exponentiated CSV `stellar_mass` | Type-4 stellar mass; source/matches PARTIALLY VERIFIED across full catalog. |
| Node x[1] | log10 `vel_dispersion` in km/s | `SubhaloVelDisp`, 1D internal dispersion of all particle/cell types; not a measured stellar LOS dispersion. |
| Node x[2] | log10 converted `half_mass_radius` in Mpc, floor 1e-6 Mpc | Total subhalo half-mass radius. Stellar-radius interpretation INVALID. |
| Node x[3] | log10 CSV `metallicity`, floor 1e-10 | Gas metallicity; exporter already floors at 1e-6. Stellar interpretation INVALID. |
| Edge `distance` | norm of 3D endpoint position difference | Euclidean Mpc; no periodic minimum image. |
| Edge `delta_v` | norm of 3D endpoint bulk-velocity difference | km/s; not a LOS velocity difference. |
| Edge `cos_theta` | dot(delta position, delta velocity) / product of norms + epsilon | 3D relative-motion alignment; clipped to [-1,1]. |
| Edge `mass_ratio` | x[src,0] - x[dst,0] | Signed log10 stellar mass ratio, not a linear ratio. |
| Edge `proj_sep` | norm of x/y endpoint separation | Mpc in the fixed simulation xy plane, without periodic wrap. |

These definitions come from `SubhaloData.get_node_features` and
`GraphBuilder._compute_edge_features`; CSV `log_*` convenience columns are not
used as a separate fitted normalization. Raw field matching uses rtol 1e-5,
atol 1e-6 for rounded API feature exports, and rtol 1e-7 for linear targets.
The specification distinguishes total/type-4 radii, gas/star metallicity, bulk
velocity and internal dispersion. For example the group-1 central CSV radius
is 682.504 ckpc/h, matching total 682.503784, versus stellar 88.338547 ckpc/h.
These are material scientific differences, not just naming differences.

There is no literal halo target column in current x or edge_attr; features are
computed without consulting y. This is PARTIAL leakage assurance: total radius
and all-particle dispersion contain privileged simulation/DM information, the
selection uses mass, and the backbone's training population remains unknown.
Physically correlated simulation features are legitimate for an explicitly
intrinsic-feature study, but do not establish observable-only mass inference.
A feature-range assertion such as "stellar mass <14" cannot prove no leakage.

## Physical neighbors and split reliability

The graph is built **within each capped FoF member list**, not across the full
box. Current radius edges use a 3D Euclidean threshold <=2 Mpc, reciprocal
columns are coalesced, one self-loop per member is added, and isolated members
are joined bidirectionally to their nearest neighbor (inclusive ties), even
outside the threshold. `k_neighbors=8` does not control the selected radius
method. Retention counts unordered nonself pairs; self-loops are architectural
columns. Reciprocal messages are two stored columns of one physical pair.

This is a reproducible software definition but not a correct periodic physical
neighborhood. The box is 75,000 ckpc/h =110.717449 Mpc at this epoch. The audit
finds 215 threshold decisions differing under minimum images and five fallback
pairs outside the radius; see the main audit for split-specific impact.
Calling edges measured gravitational interactions is also unsupported: they
are geometric message-passing choices, and a stellar mass-product/r proxy
omits total halo mass, diffuse matter and unselected satellites.

The loader sorts graph IDs and applies `RandomState(42).permutation` to whole
halos, taking floor(541*0.7)=378 train, floor(541*0.15)=81 validation, remaining
82 test. Exact graph/member IDs are reproduced and agree with all A–D manifests;
no subhalo ID crosses splits. This verifies evaluation membership for these
bytes. It does not prove historical training membership, absence of spatial
dependence, or independent initial conditions. All halos share one box.
Current A–D does not augment its evaluation catalog. Historical augmentation
is train-only in notebook source, with unresolved executed lineage.

## Policy semantics and objective

The historical `EdgePolicyNet` consumes each five-column intrinsic edge record,
both frozen GNN endpoint embeddings, and a global mean embedding context.
The embeddings inherit all four intrinsic node features and full-graph message
passing. Current training code fits policy normalization on supplied TRAIN
graphs and fits the reward-error scale on TRAIN full-prediction residuals.
Targets legitimately enter offline policy reward; they are not policy input
columns. Recovery of policy training metadata supports evaluation of those
weights, not original backbone independence or exact optimizer/RNG continuation.

For a single graph, the accuracy reward is the improvement in absolute target
error divided by a fixed train-derived scale (0.124135822057724 dex in the C
configuration). Other terms penalize retention-target deviation and isolates,
and preserve a **relative stellar/internal-dispersion proxy**:

```text
KE(S) = 0.5 sum_i [retained_degree(i)/full_degree(i)] Mstar_i sigma_internal_i^2
PE(S) = G sum_(stored nonself i,j retained) Mstar_i Mstar_j / Euclidean_r_ij
penalty(S) = [ln(2 KE(S)/abs(PE(S))) - ln(2 KE(full)/abs(PE(full)))]^2
```

The implementation excludes self-loops but sums reciprocal stored columns in
PE. This cancels a constant orientation factor in the relative comparison;
it does not define a calibrated absolute binding energy. Stellar weighting,
internal all-particle dispersion, degree fractions, missing satellites and
unwrapped distances prevent a halo-equilibrium interpretation. Quantities are
dimensionally energy proxies when the declared units hold; calibrated physical
validity and a causal accuracy benefit remain UNRESOLVED.

D's label-free reward uses MC-dropout uncertainty reduction plus these
retention/connectivity/relative-proxy terms. Choosing K on validation is distinct
from proving the reward tracks accuracy; reduced predictive variance alone can
be misleading. Offline training uses repaired PL selections whereas primary C
inference uses exact final-budget constrained decoding. Their objectives and
action populations must be described separately.

The separate audited projected adapter uses two centered sky coordinates, LOS
bulk velocity and log stellar-mass proxy, with projected distance, absolute LOS
velocity difference and log mass ratio edges. Its objective-compatible
projected RL trainer/artifact is absent. Same tensor width is not semantic
compatibility. Do not substitute zero/fabricated dispersion or metallicity, or
reuse the intrinsic policy by renaming features.
