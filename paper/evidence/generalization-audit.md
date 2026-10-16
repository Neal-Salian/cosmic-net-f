# Generalization audit — 16 October 2026

**Decision for the 17 October first draft: retain a within-volume claim only. No independent compatible astronomy dataset has been evaluated, and no CAMELS/SIMBA transfer result is available.** The fastest executable external pilot is one audited CAMELS IllustrisTNG CV volume at z=0 after resolving the existing training schema and member-selection provenance. Prefer TNG300-1 for cluster mass support when acquisition/consolidation is available. Neither should be presented as a completed result.

Audit base: `c825bf57652005c80faff79cf4dfd58e3a9baa14`. Work is isolated on `codex/paper-generalization`, in `/Users/nealsalian/.codex/worktrees/paper-generalization/cosmic-net`. Training/model/legacy loader code was not changed.

## What exists locally

Machine-readable evidence is in `experiments/paper_generalization/local-inventory.json` (32 files, content hashes, HDF5 attributes/shapes, checkpoint tensor dimensions and notebook provenance). Scope: repository data/raw, raw_hdf5, config, checkpoints, policies, Kaggle weight candidates and selected recovered B/C artifacts. A shallow Desktop/Downloads filename search found a duplicate TNG100 CSV, no CAMELS/SIMBA catalog; `downloads-discovery.json` records the duplicate hash. This is not a claim of a machine-wide search or of external data access.

| Local item | Actual contents and usability | Independence |
|---|---|---|
| `data/raw/tng100_clustered.csv` | 5,378 galaxy rows / 541 halos; usable by legacy TNG loader, with explicit unit declaration and unresolved extraction/backbone provenance | One TNG100 catalog |
| `data/raw/tng100_clustered.h5` | Same declared 541 halos / 5,378 rows; attrs identify TNG100-1, snapshot 99, h=0.6774 | Not an independent volume |
| `raw_hdf5/groupcat_99_0.hdf5` | One group (global group 0), 17,185 subhalos, native fields, Header.NumFiles=448 | Same snapshot/box; not a complete catalog |
| `raw_hdf5/groupcat_99_7.hdf5` | One group (global group 1), 14,157 subhalos; native M200c/type-4 and total radii present | Overlaps primary CSV group 1 |
| Other numbered raw files | Nine are header-only with zero groups/subhalos; `_4` is zero bytes | Neither extra volumes nor usable validation samples |
| `data/raw/checkpoint.csv` | Table exists; no independent catalog/IC contract | Uncertified derivative, not external validation |
| `test_synthetic_*`, synthetic pilot evidence | Explicit software/synthetic material | No astronomy validation |
| `config/astronomy_manifest.example.yaml` | Declaration example; no completed real source attestation | Not a dataset |
| CAMELS-TNG / CAMELS-SIMBA / TNG50 / TNG300 / other epochs | No usable native catalog found in audited locations | Missing dependencies |

Recovered notebook B and C provenance identifies the same CSV SHA-256 `6c148f2cae68abac7bc5c34d370ff0130a647b49e79c32880bcf36ecc0ab4d53` and policy split **378 train / 81 validation / 82 test**. Both mark `research_label=legacy_total_radius`, `target=legacy_uncertified`. Those 82 test halos are held-out rows in one volume. This audit did not rerun or independently certify their predictions or the original backbone's training split.

## Material semantic and provenance findings

1. **Radius and target, partial evidence.** `semantic-spot-check.json` matches CSV group 1 central 17185 to raw group 1: CSV halo mass and converted native M200c both equal `3.811469808735607e14 Msun`; CSV radius `682.504` matches native total radius `682.503784` ckpc/h, while the stellar type-4 radius is `88.338547`. This establishes one overlap, not all 541 target definitions.
2. **Member selection is unresolved and consequential.** CSV richness is 7–10: one halo has 7, four have 8, twenty-one have 9, and 515 have 10. Group 1 contains the first ten native IDs 17185–17194; the raw group contains 2,931 positive-stellar-mass subhalos. This is evidence consistent with truncation, not proof of the extraction algorithm. Evaluating every native galaxy would silently change the population. The package rejects an unverified or mismatched member-selection contract.
3. **Present loaders do not establish past training transforms.** The current TNG CSV loader explicitly converts raw ckpc/h positions/radii and logarithmic stellar masses. A checkpoint trained with another version may have seen different values. Replacing today's loader or radius semantics cannot retroactively repair a frozen model. Audit the original training notebook, feature scaling, sample extraction, augmentation and split.
4. **Real weights exist, but one default is unusable.** `kaggle/best_model_augmented.pt` is a real checkpoint, SHA `42764f0db204863b28672d6ec6e207443602de16dcfa926765a0429b3451c63f`, with four-node/five-edge input tensors and 64-dimensional output embeddings; it contains epoch/config/metrics/state and no embedded catalog/split provenance. `outputs/checkpoints/best_model.pt` is a 134-byte Git LFS pointer.
5. **Policies are not interchangeable.** The standalone `outputs/rls/policy.pt` expects 128-dimensional embeddings, incompatible with that backbone. Recovered B policy SHA `b1f41771c102bec644e0f84b0c002a228e5514321c768d8919e8beae3379f81d` has 64-dimensional normalized node/context buffers and five edge buffers. Its surrounding provenance names the correct input file but does not complete the original backbone semantic audit. Do not substitute B's `finetuned_gnn.pt` in a frozen-backbone experiment.
6. **Strict and legacy paths are distinct.** Generic strict astronomy loading enforces M200c and stellar type-4 radius, yields a partial observation schema and feeds projected graphs. A projected four-column graph has different meaning from the legacy intrinsic four-column graph. No compatible projected trained artifacts were identified. The new package preserves total-radius intrinsic semantics explicitly, and requires an audited receipt before real evaluation.

## Fastest defensible route

| Priority | Work before the draft | Scientific value / stopping rule |
|---|---|---|
| 1 | Finish exact backbone training/extraction/augmentation audit; reconcile original input transforms and selection with frozen policy metadata | Blocking regardless of which external dataset arrives. Leave receipt pending if evidence is incomplete |
| 2 | Acquire one complete current CAMELS IllustrisTNG CV_0 native Subfind catalog, parameters and IC lineage; execute the supplied fixed protocol | Small external pilot; same-family independent-volume evaluation with resolution/cosmology/volume shifts. No promise of enough cluster-mass halos |
| 3 | Obtain TNG300-1 z=0 complete required-field catalog and verified multi-file consolidation, preserving global membership offsets | Preferred cluster-support comparison; larger acquisition/adapter work, not implemented as a guessed chunk merge |
| 4 | Prespecify another disjoint CV for confirmation; then current-release SIMBA catalog with identical target/feature contract | Cross-simulation transfer only after compatible data and verified lineage; no completed result now |
| 5 | Multiple unseen physics/cosmology families with volume-level replication and no tuning on held-out domains | Needed for a broad domain-generalization claim; beyond a single-volume first-draft pilot |

This priority ordering is an engineering/scientific judgment from the local inventory, not a download or runtime benchmark. [CAMELS parameters](https://camels.readthedocs.io/en/latest/parameters.html) identify CV as fixed fiducial physics/cosmology; [TNG background](https://www.tng-project.org/data/docs/background/) describes the released volumes and TNG300's cluster statistics. Dataset acquisition and training provenance remain the uncertain critical path, not model inference.

## Claims that must remain distinct

| Claim | Minimum relevant evidence | Current status |
|---|---|---|
| Within-volume evaluation | Held-out halos within the original catalog, with split and backbone leakage audit | Existing 82-halo evaluation is of this type; original backbone provenance remains open |
| Independent-volume evaluation | Compatible catalog with IC/volume lineage disjoint from every development population | Not performed |
| Cross-simulation transfer | Frozen trained artifacts evaluated in an unseen simulation family with verified common target/features; disclose any matched phases | Not performed; the supplied protocol conservatively also requires disjoint ICs |
| Domain generalization | Prespecified unseen domains spanning simulation/physics/cosmology, held-out tuning and replicated volume-level evidence | Not established |

New random splits, HDF5 chunks, rotations, duplicate CSV/HDF5 encodings, resolutions sharing phases, or snapshots from the original box are not independent-volume replications. Repeated snapshots can be a separately labelled temporal-shift study after merger/IC leakage analysis.

Suggested draft wording: **“Evaluation is restricted to held-out halos from a single TNG catalog. Independent-volume and cross-simulation validation have not yet been completed. The source extraction semantics and original backbone provenance require further audit.”** Do not describe the software fixtures or prepared manifests as completed transfer.
