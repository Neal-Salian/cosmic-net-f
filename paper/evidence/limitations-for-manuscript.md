# Manuscript claim boundaries

## Wording supported now

> We evaluate an executed legacy Cosmic-Net sparsification pipeline on a
> byte-identified TNG100-derived catalog containing 5,378 selected subhalos in
> 541 FoF groups. The deterministic same-volume evaluation split contains
> 378 training, 81 validation and 82 test graphs. On the saved 82-graph test
> population, the frozen full-graph predictor achieves RMSE 0.114797 dex against
> the legacy logarithmic mass labels. The supplied RL selector achieves
> 0.136160 dex at nominal 40% retention of unordered nonself graph pairs
> (mean realized retention 40.2997%). These are development benchmark results;
> the original backbone's training/test separation is not independently
> established. The current evidence does not establish RL superiority.

> The mass labels were intended as total spherical-overdensity mass M200c.
> The extraction code and two available raw halo groups support that
> interpretation, but raw coverage is incomplete. Intrinsic inputs include
> total subhalo radius, internal all-particle velocity dispersion and gas
> metallicity. The graphs use Euclidean distances within capped FoF lists,
> self-loops and nearest-neighbor fallback, without periodic minimum images.
> These limitations preclude an observable-only interpretation or certification
> of the full catalog's physical-neighbor and target semantics.

These paragraphs are authored summaries of repository measurements, not quotations
from external literature. Cite the immutable experiment identifiers and archived
evidence with the eventual paper supplement.

## Required limits and their impact

| Issue | Classification | Consequence for the draft / what resolves it |
|---|---|---|
| Full-catalog M200c mapping/acquisition | PARTIALLY VERIFIED | Describe intent and two-group support. Obtain original release/API records or complete independently audited group targets before calling every label certified M200c. |
| Original backbone train/test membership | UNRESOLVED | No certified held-out accuracy or selection-free inference from these weights. Recover checkpoint-bound original membership/source evidence or train anew on audited independent groups. |
| Periodic neighborhood error | INVALID physical-radius claim | 215 candidate-radius decisions change under minimum images, including 15 across two test halos. Correct geometry consistently and rerun compatible training/evaluation in a new experiment. |
| Radius fallback beyond threshold | VERIFIED implementation; INVALID strict-radius description | Five stored training pairs exceed 2 Mpc. State radius-plus-isolate-fallback semantics; never claim all edges obey the threshold. |
| Intrinsic features labelled stellar/projected | INVALID | Total radius and internal all-particle dispersion carry privileged simulation information; metallicity is gas. Restrict claims to intrinsic-feature experiments or redesign features and retrain. |
| Historical preprocessing parity | INVALID | Archived loader and current loader differ materially; both score regimes reproduce under separate transforms. Tag each score with its actual preprocessing; do not claim a model-training improvement. |
| Augmentation physical consistency | INVALID | Position-only rotations retain unrotated velocities and original topology. Do not infer augmentation benefit; joint transforms/topology changes require a separate validated training run. |
| Repeated rows across final splits | VERIFIED absent for stable IDs | This is a useful consistency result, not assurance about unrecorded backbone training data or independent volumes. |
| Extraction checkpoint/HDF5 duplication | VERIFIED overlap | Treat as encodings/intermediates, never independent samples or extra test catalogs. |
| Selection function / missing satellites | UNRESOLVED completeness | Capped, cleaned massive-central candidate lists are not a complete or representative halo/galaxy survey. Obtain selection accounting; avoid richness/completeness claims. |
| Independent-volume generalization | INVALID as an established result | A random split within one TNG volume is development evidence; reserve genuinely independent ICs/volumes for confirmation. |
| Calibrated virial physics | INVALID | Relative stellar/internal-dispersion graph proxy does not measure total halo equilibrium. Describe the formula and scope; physical utility needs ablations and independent validation. |
| Projected policy / real observational transfer | INVALID as an established result | Current RL features/reward require intrinsic data. The separate projected baseline lacks a validated projected policy trainer/artifact. |
| RL advantage / accuracy preservation | INVALID as a general conclusion | Full is more accurate here; one learned seed and one test random-mask realization cannot establish general benefit. Do not choose a new primary comparator/budget after inspecting test results. |
| Efficiency or calibration | UNRESOLVED as general scientific claims | Edge retention alone does not imply speedup, and broad MC-dropout coverage does not establish calibration. Preserve measured costs and evaluate practical timing/calibration separately. |

An INVALID classification applies to the stated affirmative claim, not to the
entire research direction. A reproducible legacy calculation can coexist with
an invalid physical interpretation.

## Evidence to prioritize before stronger claims

1. Acquire real complete catalogs with field/release/checksum records, correct
   M200c definitions, coordinate/epoch/velocity units, stable membership and
   selection accounting. Verify unwrapping or implement consistent minimum
   images. Current raw files verify only 20 rows; nine chunks have no data and
   one is zero bytes.
2. Run the separate baseline configuration on independent IC groups with new
   weights and durable split/source/observation/checkpoint provenance. These
   results must have new identifiers and cannot inherit 0.114797 dex.
3. Choose an explicitly intrinsic-feature study or a genuinely projected
   observable study. The latter needs a justified projected RL objective and
   compatible artifacts; fabricated missing features are not a bridge.
4. Predeclare primary comparisons and practical tolerances, repeat backbone/
   policy/random-mask seeds, and use new independent test groups. Use relevant
   physics/feature/constraint controls before claiming a mechanism or benefit.

Methods, software validation and a candid negative/limited legacy result can
be drafted now. The unresolved target coverage and backbone provenance must
remain visible wherever the numerical scores are reported. No statement of
scientific certification is supported by this audit.
