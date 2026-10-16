# Cosmic-Net A–D scientific provenance audit

Audit date: 16 October 2026. Base: `c825bf57652005c80faff79cf4dfd58e3a9baa14`.
Branch: `codex/paper-provenance-audit`. Worktree:
`/Users/nealsalian/Desktop/cosmic-net-paper-provenance-audit`.

**Verdict: reproducible legacy-development measurements; no scientific certification.**
Current data/weights support a documented benchmark on the saved evaluation
population. They do not establish an independently held-out M200c prediction
result, an observational projected policy, or RL superiority.

The controlling machine-readable decisions are [provenance-status.json](provenance-status.json).
The measured hashes, complete reconstructed halo/member split lists, raw field
matches, geometry counts and per-stage artifact checks are in
[provenance-measurements.json](provenance-measurements.json). A VERIFIED label
applies only to its stated scope, and does not upgrade adjacent claims.

## Evidence hierarchy and preservation

1. Local CSV/checkpoint bytes match the A–D downloaded manifests. Each stage
   declares complete/full, the base commit, the same source identity, split IDs,
   graph settings and frozen backbone state. All declared artifact SHA-256
   values and recorded Python source hashes at the base commit were checked.
   The consumed campaign artifacts were rehashed unchanged at audit end.
2. C per-example targets and graph content hashes match reconstruction for
   both full and RL q=0.4 rows, with exactly the same 82 unique test halo IDs.
   Residual arithmetic independently reproduces full RMSE **0.11479669852430192**
   dex and RL **0.13616044859036275** dex. Mean realized RL retention is
   **0.4029971390520059** of distinct nonself physical pairs.
3. Current CPU frozen-model replay gives **0.11479662526768547** dex. The tiny
   difference is an environment/numerical difference; the saved C predictions
   are the historical score authority. No model training was performed.
4. Raw reference contents verify fields/targets only for **20 rows / two FoF
   groups**, leaving **5,358 rows / 539 groups** externally unmatched. Content
   matching against local files is not authentication against an official
   release checksum. Their headers identify TNG100 geometry/epoch but do not
   prove an unbroken acquisition chain.

The original checkout and its untracked `output/` and `paper/` were never
modified or cleaned. Those directories were read only for the requested
historical evidence. All audit code, new evidence, caches and tests run from
the isolated worktree. No existing notebook, weights, CSV, HDF5, plot or campaign
output was edited. No merge, rebase, push or PR is part of this audit.

## Major claim decisions

| Claim | Classification | Evidence and implication |
|---|---|---|
| A–D use the same catalog, weights and evaluation membership | VERIFIED | Hashes and exact ID lists agree across all four completed stages. |
| Model target is log10 of CSV linear `halo_mass` in declared solar masses | VERIFIED | Explicit ingestion, `HaloData.log_halo_mass`, graph `y`, and saved targets agree. |
| Every CSV target is physically M200c | PARTIALLY VERIFIED | Fetcher says `Group_M_Crit200`; groups 0/1 match exactly and alternatives differ. Other 539 groups lack raw references. |
| Target is dark-matter-only halo mass | INVALID | M200c is total enclosed matter; no DM-only extraction occurs. |
| `half_mass_radius` is the stellar type-4 radius | INVALID | Fetcher uses `halfmassrad`; raw matched values are total subhalo radii. |
| `metallicity` is stellar metallicity | INVALID | Fetcher uses `gasmetallicity`; matched values are gas metallicity (with floor). |
| All policy inputs are projected observables | INVALID | Full 3D distances/velocities and intrinsic all-particle dispersion/total radius feed the backbone and policy. |
| No exact duplicate rows or member reuse in the final CSV/splits | VERIFIED | 5,378 unique member IDs; zero full-row duplicates; no member overlap between reconstructed splits. |
| `checkpoint.csv` is an independent dataset | INVALID | 4,970 member IDs overlap with the final CSV. Never concatenate these as independent observations. |
| Raw files form a complete group catalog | INVALID | Two nonempty group chunks, nine empty chunks and one zero-byte invalid HDF5 file; header expects 448 chunks. |
| Current radius graph uses Euclidean Mpc plus isolate fallback and self-loops | VERIFIED | Actual reconstructed graphs, inclusive threshold and source trace. |
| Graph is exactly the physical 2 Mpc neighborhood of each member | INVALID | Periodic minimum-image geometry is absent; fallback can exceed 2 Mpc. |
| Historical preprocessing and current campaign preprocessing are identical | INVALID | Archived-loader diagnostic changes 1,389 member positions, 12 masses, one radius; six test graphs change edges. |
| Original training uses only train parents and physical augmentation | PARTIALLY VERIFIED / INVALID | Notebook splits before augmentation, but no durable training membership. Position-only rotation is physically inconsistent with retained velocity vectors. |
| Frozen backbone was independently held out from these test halos | UNRESOLVED | Checkpoint has no training dataset/source/split identities. Reproducing metrics cannot prove absence of training exposure. |
| Halo-random splits demonstrate independent-volume generalization | INVALID | All train/validation/test halos belong to one TNG100 realization. |
| RL reward measures calibrated halo virial equilibrium | INVALID | Degree-weighted stellar/internal-dispersion proxy compared with its full graph; not total halo dynamics. |
| Current data establish RL superiority | INVALID | Full RMSE is lower; one policy seed and inspected development test set do not establish general superiority. |

Every machine-readable major claim has one enum classification; the combined
augmentation row above is separated into two claims in the JSON file.

## Decisive geometry evidence

The CSV stores periodic-box positions in ckpc/h. The current loader explicitly
converts once to physical Mpc at a=1, h=0.6774. Radius construction uses
Euclidean distances, not minimum images. Comparing these on all unordered
member pairs yields **251 wrapped pairs** and **215 radius-neighbor decision
disagreements** across **14 halos**: train 180 decisions/11 halos, validation
20/one halo, test 15/two halos (`tng100_0000053`, `tng100_0000080`). These are
candidate-radius decisions, not a claim that exactly 215 final graph edges
would change after fallback. Separately the actual stored graphs contain
**five unordered pairs beyond 2 Mpc** in five training halos, caused by the
isolate-nearest-neighbor rule. Self-loops do not cure physical isolation.

This audit records the discrepancy and leaves the completed campaign intact.
A physical-neighbor study needs a new experiment with verified unwrapped
coordinates or consistent minimum-image construction AND edge/reward distances.
Changing topology alone while retaining old weights/scores is insufficient.

## Historical transform sensitivity

The loader archived at `0dfe85ac1df996f38c8ad793fd27fa7e19d0e687`
(source SHA-256 `dccb8214fa21e6f1e78bd114c24efe89ba6565d33d1b6361648d8c66e183eae5`)
first applies catalog-wide unit guesses, then per-row guesses. It converts
already normalized positions again above 100 Mpc and an already normalized
radius again above 1 Mpc; 12 stellar log masses below six fall into a linear
mass floor. These interacting stages matter: the 214 raw radii at/below one
ckpc/h are **not** 214 misconverted radii in that archived pipeline.

Running that actual archived loader on identical input bytes, with current
base-loader/graph/model code, produces test RMSE **0.11671666960606404** dex,
matching the notebook's rounded 0.1167 dex. This is a labelled hybrid diagnostic,
not a recovered training run. It links the old/new score difference to concrete
preprocessing sensitivity but cannot identify the historical training revision.
Details and checkpoint/log evidence: [backbone-audit.md](backbone-audit.md).

## Safe resolutions in this branch

- The legacy loader now rejects duplicate global subhalo IDs (including numeric
  aliases), fractional/negative/nonfinite IDs, nonpositive/nonfinite linear
  targets, conflicting member targets within a group, and disagreement between
  redundant linear/log target columns. Consistent values are unchanged.
- Non-mutating geometry and raw-reference checks have focused synthetic tests.
  Coverage counts only rows whose target and every compared field match;
  index alignment alone is not verification. Raw offsets use explicit
  `GroupFirstSub`, never a sum across missing or empty chunks.
- The exclusive-output audit CLI preserves inputs, verifies campaign artifacts,
  and emits portable source hashes plus full membership lists and measurements.
- A separate baseline experiment configuration uses the existing audited
  projected training boundary with new data and new weights. It remains blocked
  on real independent catalogs; no synthetic result substitutes for it.

## Reproduce

From the isolated repository root, use a previously nonexistent directory:

```sh
python3 -m scripts.audit_paper_provenance \
  --output-dir /absolute/path/to/new-audit-run \
  --campaign-root /Users/nealsalian/Desktop/cosmic-net/output/stageb-fix-campaign-20261015 \
  --replay
python3 -m pytest -q
```

The campaign folder is preserved external evidence, not committed data; without
it omit `--campaign-root` to reproduce catalog, split, raw and model diagnostics.
The command requires repository history including the pinned archived loader.
It refuses an existing output directory. The audit's local Python/PyTorch stack
differs from pinned historical requirements; exact versions are in measurements.
No GPU or external API access is used.

Further definitions: [data-and-target-definition.md](data-and-target-definition.md).
Draft-ready restrictions: [limitations-for-manuscript.md](limitations-for-manuscript.md).
