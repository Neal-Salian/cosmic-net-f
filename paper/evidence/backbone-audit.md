# Backbone provenance audit

## Identity established

The actual A–D frozen checkpoint is `kaggle/best_model_augmented.pt`, 3,349,165
bytes, SHA-256
`42764f0db204863b28672d6ec6e207443602de16dcfa926765a0429b3451c63f`.
It matches the supplied Kaggle v3 input and all four campaign manifests.
Its sorted name/shape/dtype/tensor-byte state hash is
`56907374794d9203b4b1873a3b930fff856aff427244d86e6a55742cded12034`.
These are distinct file/state identities, not interchangeable hashes.

The checkpoint contains exactly `epoch`, `model_state_dict`, `config`, `metrics`.
Epoch is 729 zero-based (730 displayed). Saved validation metrics are MSE
0.0138047831133008, RMSE 0.11749375611543655, MAE 0.09234894812107086,
R² 0.9240660667419434 and scatter 0.11765111237764359. Embedded config declares
seed 42, TNG100 snapshot 99, random-halo 0.7/0.15/0.15 ratios, a 4-node/5-edge
feature architecture with three 64-dimensional layers, mean pooling, dropout
0.05 and the 2 Mpc radius graph. It declares 1,000 epochs, patience 150,
AdamW learning rate 0.0002 and the older physics schedule ending at 0.25.
The base repository's different training settings are not the checkpoint's
historical training configuration.

`kaggle/best_model (2).pt` is a different epoch-828 checkpoint with optimizer/
scheduler state and different metrics. `outputs/checkpoints/best_model.pt` is
an LFS-managed alternate artifact. Neither filename establishes interchangeability
with the A–D backbone. Stage B's `finetuned_gnn.pt` is also a different state;
it cannot replace the frozen reference while retaining the primary full score.

## Historical source and logs

`kaggle/training-cosmic (4).ipynb` is the strongest available historical account:

- It clones the upstream repository without pinning a source commit, copies
  the Kaggle CSV into `data/raw`, writes its own config and rewrites
  `model/physics_loss.py` in a notebook cell. Therefore a Git revision alone
  would not necessarily reproduce the executed loss implementation.
- Cell 6 constructs sorted halo splits via `TNGLoader.split_data` and checks
  feature/target ranges. Those range assertions do not establish no leakage.
- Cell 7 loads/splits before augmentation, builds 378 train/81 validation/82
  test graphs, appends three views to each training graph, and constructs a
  fresh model with `build_model(cfg)` rather than loading a starting checkpoint.
  **1,512 =4*378** training graphs includes the originals; the printed "3x"
  describes added copies, not the total multiplication factor.
- It selects checkpoints by validation MSE, prints early stop at epoch 880,
  best validation MSE 0.0138 at displayed epoch 730, and test RMSE 0.1167,
  R² 0.9075, MAE 0.0921 and scatter 0.1174. Cell 8 repeats a final test.
  Epoch/validation values agree with checkpoint metadata. This numerical
  consistency is supporting evidence, not a cryptographic training record.
- Augmentation rotates and jitters positions, recomputes distance/alignment/
  projected-separation features on the **original topology**, but leaves
  velocity vectors unchanged. A joint spatial rotation should rotate both
  position and velocity; otherwise the alignment feature is changed
  artificially. Jitter can also violate the original radius threshold.
  Saved parent IDs, transforms, RNG continuation and augmentation manifests
  are absent. The separate modern `data/augmentation.py` rotates phase space
  jointly and records lineage; its correctness does not repair old weights.

The notebook was added in historical repository commits and is hashed in the
audit evidence. Embedded outputs record what was reported; they do not contain
training member IDs, input byte hashes, a pinned checkout or a signed acquisition
record. Later repository scripts (`main.py`, `training/train.py`) implement
reasonable split/training paths but cannot prove which code generated a file
whose training provenance was not stored.

## Preprocessing mismatch measured

The archived TNG loader at `0dfe85ac1df996f38c8ad793fd27fa7e19d0e687` is
available in Git. The audit executes its actual local-CSV path, with API
fallback unused, on the same hashed CSV. It finds changes relative to the
current explicit conversion path in **1,389 positions, 12 stellar masses and
one half-mass radius**. The reason is interacting catalog-wide and per-row
magnitude-based conversions, including double conversion of large already
normalized coordinates and a radius, and a low stellar-mass floor.

Using the archived loader with current graph/model code and the unchanged
checkpoint gives **0.11671666960606404 dex** on the reconstructed 82 test
halos, consistent with the original notebook. Six test graphs have changed
edges relative to the current campaign. Current CPU replay gives
**0.11479662526768547 dex**; C's saved arithmetic gives
**0.11479669852430192 dex**. This establishes meaningful preprocessing
sensitivity. The diagnostic deliberately combines archived loading with current
graph/base-loader implementations, and is not a bitwise historical training
reproduction. Its validation replay and exact environment are also stored in
[provenance-measurements.json](provenance-measurements.json).

Consequently the old 0.1167 and current 0.114797 numbers must carry different
preprocessing identities. Do not describe the latter as a new training
improvement, a controlled augmentation benefit or evidence that input/model
training semantics were unchanged. Replacing radii/targets/periodic geometry
needs new compatible model training and new results.

## Held-out verdict

**UNRESOLVED: independent held-out evaluation of the original backbone is not
demonstrable from the available record.** There is no positive evidence here
that test halos were trained on; the missing provenance equally prevents
certifying that they were excluded. Matching notebook counts, seed, epoch and
metrics increases confidence in the intended recipe without establishing
actual membership or the whole model-selection history.

Missing durable evidence includes training catalog hash/release, training source
revision plus notebook edits, train/validation/test IDs, checkpoint-bound split
identity, augmentation parent/transform lineage, and optimizer/RNG history.
Current A–D manifests bind downstream inputs and evaluation splits; they were
created after backbone training and cannot retroactively fill these gaps.

The present 378/81/82 evaluation split is reliably reconstructed for the current
CSV, but every halo is from the same simulation volume. Even perfect historical
halo exclusion would establish at most same-volume evaluation. The test set has
also been inspected repeatedly during development; new independent external
volumes/ICs are needed for confirmatory generalization claims.

Paper impact: report scores as legacy-development measurements with unknown
backbone training separation. Do not headline a certified unseen-halo M200c
accuracy, observational transfer, independent-volume generalization or an
augmentation causal effect from this checkpoint.

## Separate executable baseline experiment

This audit adds `config/paper-provenance-baseline.example.yaml` for a new
projected, MSE-only full-graph baseline through the existing audited trainer:

```sh
python3 -m training.full_graph_baseline \
  --config config/paper-provenance-baseline.example.yaml \
  --output-dir /absolute/path/to/new-paper-provenance-baseline-run
```

Supply a private manifest with real content hashes and four genuinely
independent development IC catalogs using the listed stable IDs. The example
allocates IC groups 2/1/1, requires correct M200c and measured observable
semantics, uses no physics proxy or augmentation, initializes new weights and
selects by validation MSE. It cannot run scientifically on the current one-box
CSV, incomplete raw chunks, dummy manifest or synthetic stand-ins. The existing
training entry point rejects invalid audited inputs before output creation,
and refuses any existing output directory. Tests cover that boundary; no real
training was executed here. Catalog procurement and geometry verification are
explicit remaining blockers, not optional prerequisites hidden in a runnable
example.

Verified halo-centered unwrapping with a content-audited preparation record is
required for currently supported nonperiodic catalogs; do not set a periodic
box to false simply to pass validation. Alternatively implement/review minimum
images consistently in a new experiment. Even the projected adapter adds
isolate-nearest-neighbor edges outside its nominal radius: report the graph as
radius-plus-fallback or separately implement and validate a strict-radius graph.

Expected new artifacts include `projected_backbone.pt`, source/catalog hashes,
split manifest, resolved configuration, training history, observation/rejection
reports and held-out predictions. Keep them in a fresh run directory. No
historical score transfers to this new feature/target/training contract. This
does not create a projected RL policy: a physically justified projected
objective/trainer remains a separate blocker. See `docs/audited_baseline.md`
and `docs/astronomy_manifest.md` for the existing interfaces.
