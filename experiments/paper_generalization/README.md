# Independent validation package — 16 October 2026

**No independent astronomy dataset was evaluated.** `run-status.json` records zero real external halos. `per_halo_results.csv` contains its header only. Pytest fixtures are software checks and never astronomy evidence.

This package prepares the fastest small native-catalog experiment: **CAMELS IllustrisTNG CV_0, snapshot 090**, one independent initial-condition volume after lineage verification. This is the same simulation family with different volume, resolution and cosmology; it is not a controlled change of volume alone. TNG300-1 at z=0 is preferable for cluster mass support but requires acquisition and verified consolidation of its multi-file catalog. Neither external input exists in the audited locations.

## Run

From the isolated repository root, using Python 3.12 and the packages in `validation-environment.txt`:

```bash
python -m experiments.paper_generalization.inventory \
  --root /absolute/path/to/local/cosmic-net \
  --output /new/path/local-inventory.json
python -m pytest -q tests/test_paper_generalization.py
python -m experiments.paper_generalization.run \
  --manifest /absolute/path/to/completed-cv0-manifest.json \
  --output /new/path/cv0-frozen
```

Use `camels-tng-cv0.example.json`, `training-receipt.example.json`, and `required-inputs.json` to prepare those inputs. Relative file paths resolve beside the containing manifest/receipt. Hash every required file. Replace the `PENDING` fields only with verified facts. `status=ready` requires an audited receipt; the shipped template exits **2**, prints **BLOCKED**, and creates no results directory. Output must be new. No network access, downloads, training, normalization fitting, policy selection, TTA or Stage B occur in the runner.

`--fixture-smoke` is for generated software fixtures only. It requires separate `fixture_smoke` manifest/receipt labels and a `synthetic_fixture` source. Its outputs always set `scientific_result=false`, including successful predictions. Tests keep those outputs in temporary directories.

## Compatibility boundary

The supported frozen model has four **intrinsic** node inputs and five ordered physical edge inputs; see `FEATURE_CONTRACT` in `contracts.py`. Total half-mass radius is intentional because the existing CSV uses that semantic. This is a separately audited intrinsic experiment, not the repository's `corrected_research` stellar-radius/projected-observation path. A four-column projected graph is incompatible with this model. GroupMass/FoF mass is not accepted as M200c. Native CAESAR, AHF, Rockstar, dark-matter-only and Hugging Face feature tables are unsupported.

A receipt must bind the exact backbone and raw policy state, the saved model/graph config, one complete development-catalog identity, the split, exact training code and a semantic audit. `training_catalogs` includes every population used to train, augment, select and validate both artifacts. This package supports **one** such catalog; more catalogs require extending the evidence mapping. Declaring an audited receipt is a scientific assertion requiring human source review; hashes prove identity, not truth of that assertion.

**Member selection is an unresolved blocker.** The CSV has 7–10 members per halo (515/541 have exactly 10). Its group 1 has only the first ten native subhalo IDs despite 2,931 native stellar members. This suggests truncation but does not prove the extraction algorithm. `population_contract` must match between training and evaluation. Supported explicit rules are `all_stellar_positive` and `native_first10_then_stellar_positive`, both with `min_members=3`. The latter takes the first ten rows of the complete native group's ordering, then removes zero stellar masses. Select neither until exact training extraction code is audited. If neither matches, stop and implement the verified rule with a regression test.

The adapter requires native self-contained HDF5 datasets, count/shape agreement, finite values, integer group membership and positive logarithmic inputs. External/soft links, VDS and external raw storage are rejected. Positions and radii become physical Mpc via `a/(1000 h)`; masses become Msun via `1e10/h`; SubhaloVel stays peculiar km/s. Positions are minimum-image unwrapped around the first selected member; ambiguous half-box extents fail. Missing, invalid or nonpositive required features never receive default scientific values.

## Prespecified independent validation protocol

1. **Freeze first.** Record the exact extraction semantics and all backbone/policy development ICs. Reconcile original backbone training/augmentation with the policy's 378/81/82 split. Verify policy binding against the original backbone; do not substitute `finetuned_gnn.pt`. Audit the historical position/radius transforms: a legacy checkpoint without provenance is not certified by today's loader.
2. **Choose before seeing predictions.** Use one complete CV_0 catalog, z=0, current release. Record global phase lineage (not just `CV_0`), parameters, halo finder and resolution. Check against all training/model-selection ICs, including aliases and snapshots. Prespecify CV_1 as confirmation before accessing its outcomes. A single volume is the minimum external pilot, not a cosmic-variance study.
3. **Match semantics and population.** Review every required input in the manifest. Match the verified member selection. Preserve the exact graph radius/k, self-loops, feature order/scaling, policy normalization buffers and budget from the audited training config. Keep all eligible external halos; export selection rejections. The template's q=0.4/no-isolates/direct is a candidate locked notebook setting, not a tuned external result.
4. **Describe support.** Compare halo mass, selected richness and each input-feature distribution before interpreting metrics. The CSV's observed target range is 12.5045–14.5811 dex; the receipt must supply the audited *training* range, not blindly copy this entire-catalog range. The runner exports mass-range exclusions as flags without dropping halos. A 25 Mpc/h box may have little cluster overlap. Report an all-eligible result and a prespecified common-mass-range descriptive analysis; do not remove hard cases after seeing predictions.
5. **Evaluate once, frozen.** Use full graph plus random, distance, degree and the provided policy at the same exact physical-pair budget. Random uses the fixed manifest seed. All masks go through the existing matched evaluator. Export full/pruned predictions, target, errors, requested/achieved pair counts, physical retention, feasibility and failure reasons. Model and policy tensor hashes and input/code hashes must stay unchanged.
6. **Analyze paired outcomes.** Report eligible/selected/rejected halos, success counts per method, joint-success count and every feasibility failure. Primary policy-vs-full change is paired delta squared error or RMSE in dex on identical halos; also report MAE, signed bias, fidelity to full prediction and achieved sparsity. Use paired halo bootstrap intervals within this catalog, explicitly conditional on this one volume; policy seeds, random seeds, projections and repeated snapshots are not independent halos or volumes. If adding volumes, resample at the volume level. A zero joint-success count is a failed evaluation and never a positive scientific result.
7. **Keep the claim narrow.** One independently sampled same-suite catalog supports an independent-volume *evaluation*. An unseen SIMBA suite with the same contract supports cross-simulation *transfer evaluation*. Neither a successful program exit nor good fidelity alone establishes accurate mass inference, superiority or broad domain generalization. Multiple prespecified unseen simulation/physics domains are needed for the latter claim, with no external tuning.

## Exports and limitations

A successful real run writes `results.json` (all rows, masks, summaries, source identities, training evidence, mass support, code hashes and rejections) and `per_halo_results.csv` (one row per halo/method including failures). Rows are labelled `audited_intrinsic_total_radius`; this does not relabel the original training artifact as corrected research. `scientific_result=true` means a real compatible external evaluation had joint successes; it is not a test of favorable generalization. `domain_generalization_established` remains false.

The package does not authenticate raw data release identities or resolve IC aliases automatically. The hashed schema/lineage review must establish them. The adapter intentionally refuses native multi-file TNG chunks; a future consolidation tool must preserve global SubhaloGrNr offsets, complete group membership and every source hash. Any corrected stellar-radius or projected-feature training requires separately trained compatible artifacts and a separate claim. This package never fabricates those weights.

Primary specifications: [TNG catalog semantics](https://www.tng-project.org/data/docs/specifications/), [TNG volumes and cosmology](https://www.tng-project.org/data/docs/background/), [CAMELS Subfind/release cautions](https://camels.readthedocs.io/en/latest/subfind.html), [CAMELS parameters and seeds](https://camels.readthedocs.io/en/latest/parameters.html). Consulted 16 October 2026. Current CAMELS documentation warns about corrected SIMBA/CV and other catalog products; record the actual acquired release before transfer.
