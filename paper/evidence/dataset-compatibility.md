# Cross-dataset compatibility — 16 October 2026

**No row below constitutes a completed external evaluation.** “Conditional” requires actual native fields, an audited training receipt and disjoint IC evidence. The current legacy CSV is not certified just because a potential external catalog uses the same number of features.

## Loader and catalog matrix

| Source | Repository support | Target / radius behavior | Volumes / epochs | Local status | Frozen intrinsic compatibility |
|---|---|---|---|---|---|
| IllustrisTNG clustered CSV | `data/loaders/tng_loader.py` | Unmarked CSV uses explicit legacy units; target remains uncertified; radius is total in overlap check | CSV identity is not established by config snapshot alone | 541-halo TNG100 snapshot-99 table | Current results are within one volume; original training transforms and selection open |
| TNG REST API | Same loader, configurable base URL | API normalizes M200c/type-4 fields in its supported route; some field fallbacks remain legacy behavior | `_require_supported_api_epoch` restricts supported normalization to snapshot 99; changing a number is not a verified epoch adapter | No separately audited cached external catalog | Stellar-radius API inputs cannot be passed as the total-radius frozen schema |
| Native TNG HDF5 | `data/astronomy_manifest.py`, `astronomy_audit.py`, `astronomy_loader.py` | Strict `Group_M_Crit200`, type-4 stellar mass and radius, declared h/a and field units | Explicit manifest simulation/snapshot/IC; strict manifest refuses periodic input without verified prepared geometry | Only sparse original-box chunks, not a full external catalog | Returns partial observation inputs, not legacy HaloData; needs matching projected artifacts |
| CAMELS IllustrisTNG HDF5 | `data/loaders/camels_loader.py` (legacy); generic strict native contract where fields match | Legacy path uses GroupMass, total radius, hardcoded h and weak defaults | Config suite/set/simulation/snapshot; default LH_0/090; cache key omits snapshot/set | No catalog found | Legacy adapter incompatible with certified M200c transfer; new package bypasses its download/fallback path |
| CAMELS SIMBA Subfind | Same suite parameter; generic strict schema conditional | Native fields need release-specific audit; legacy GroupMass is not M200c | CV/LH/1P identifiers configurable; does not prove independent phases | No catalog found | Conditional via native field/schema review, raw weight binding and disjoint ICs; later cross-simulation experiment |
| Standalone SIMBA / CAESAR / AHF / Rockstar | No dedicated semantic adapter | Different halo finder and target/feature definitions must be mapped and tested explicitly | No supported ready pipeline | None found | Reject; do not rename columns to force compatibility |
| Hugging Face CAMELS features | `CAMELSHFLoader` | Compact feature rows/defaults; no verified complete native scientific field contract; synthetic fallback possible | Configurable feature dataset does not establish native simulation lineage | Not present | Reject as scientific external input without a separate full semantic audit |
| TNG300 / TNG50 z=0 | Potential native-field support; external volume must be identified | Same native conventions are a starting point, not evidence of training compatibility | Distinct volume identity/IC must be verified; lower-resolution versions alone do not establish independence | No catalog found | TNG300 preferred for mass support; complete multi-file preparation still required |
| Other snapshots of TNG100 | Catalog/manifest can name epochs, legacy API normalization does not generally support them | Need a=1/(1+z), physical units, target at that epoch, membership evolution | Shared original IC/merger lineage | None found | Temporal-shift study, not independent volume |

Code support is not data availability. `get_camels_simulation_list` enumerates names; it does not inspect local files or prove which catalogs can be downloaded. `rls/cross_sim.py` deliberately rejects `require_real_data=True`; its opt-out permits explicitly labelled smoke plumbing, including random policy fallback. It cannot supply real transfer evidence.

## Semantic compatibility assertions

| Axis | Frozen intrinsic experiment requires | Rejection / interpretation |
|---|---|---|
| Target | Verified log10(M200c/Msun), native Group_M_Crit200; file identity and training target audit | GroupMass/FoF, M200m, M500c, total subhalo mass and log-vs-linear changes rejected |
| Mass convention | Native `1e10 Msun/h` converted by the catalog's own h; stellar component 4 | No hardcoded TNG h for CAMELS; no magnitude-based unit inference |
| Coordinates | Native periodic comoving ckpc/h, declared h/a/box; minimum-image per-halo unwrap; physical Mpc for model | Unknown frame, missing box, ambiguous extent or silently treating wrapped coordinates as Euclidean rejected |
| Velocities | Subhalo peculiar km/s, same definition used in training | No application of snapshot-particle sqrt(a) or GroupVel conversion to SubhaloVel |
| Node features | Ordered log stellar mass, log velocity dispersion, log **total** half-mass radius, log stellar metallicity; four columns | Stellar-radius or projected four-column replacement rejected even if shape matches |
| Edge features | Ordered distance [Mpc], relative speed [km/s], radial velocity cosine, signed log stellar-mass ratio, xy separation [Mpc] | Different order/scaling or projected-only features rejected; graph construction remains saved config |
| Population | Exact audited member extraction rule, minimum richness and source selection; target mass/feature support described | No silent change from a capped sample to thousands of native members; template deliberately unresolved |
| Cosmology | Header h/a must match declared epoch; parameters and release review hashed; shifts reported | Different cosmology is a documented evaluation shift, not corrected by refitting or silently assumed equal |
| Architecture | Saved model config, 4 node inputs / 5 edge inputs, strict tensor-state load; provided policy embedding dimensions and buffers | 128-d standalone policy cannot bind 64-d augmented backbone |
| Frozen artifacts | File hashes, policy-to-backbone receipt, all development ICs, split/code evidence; pre/post input/code/tensor hashes | Missing/LFS/synthetic artifacts, unknown lineage, shared ICs or mutated contents rejected |
| HDF5 provenance | Complete native one-file count/shape agreement and self-contained field storage | Partial chunks, fractional group indices, external/soft links, VDS and external raw storage rejected |

The [native TNG specification](https://www.tng-project.org/data/docs/specifications/) distinguishes spherical-overdensity mass, total/stellar radii, comoving lengths and subhalo velocity conventions. The local central-galaxy spot-check independently verifies only the target/radius values reported in the audit; it does not validate whole-catalog extraction.

CAMELS CV uses h=0.6711, Omega_m=0.3 and sigma8=0.8 in the [parameter documentation](https://camels.readthedocs.io/en/latest/parameters.html), whereas local TNG headers have h=0.6774 and Omega_m=0.3089. A same-family CAMELS evaluation changes more than volume. The parameters file and acquired header must be checked rather than treating these documentation values as proof of a file's identity.

[CAMELS Subfind documentation](https://camels.readthedocs.io/en/latest/subfind.html) describes release corrections affecting SIMBA/CV and other products and distinguishes native Subfind catalogs from snapshot/other-finder outputs. Record acquisition date, release/generation and source checksum. A synthetic cached fallback cannot be converted into real data by changing a manifest label.

## Existing artifact decisions

| Artifact | File SHA-256 | Decision |
|---|---|---|
| `kaggle/best_model_augmented.pt` | `42764f0db204863b28672d6ec6e207443602de16dcfa926765a0429b3451c63f` | Candidate frozen backbone; original training provenance still absent |
| Recovered notebook-B `policy.pt` | `b1f41771c102bec644e0f84b0c002a228e5514321c768d8919e8beae3379f81d` | Candidate 64-d normalized offline policy; require receipt binding |
| `outputs/rls/policy.pt` | `c28724f6c88fc95e972b9cbb808a556dac7cb9f426cda10ce7ba09d74004b180` | Reject for this backbone: 128-d node input |
| `outputs/checkpoints/best_model.pt` | LFS pointer, 134 bytes | Not model contents; reject |
| Any new projected/corrected model | None identified | Separate retraining and matched policy required; cannot claim frozen legacy transfer |
