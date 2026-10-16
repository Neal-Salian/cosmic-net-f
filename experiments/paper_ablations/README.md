# Paper RL ablations

Local real-data diagnostics are complete. New reward/constraint **training runs remain planned**. No recovered checkpoint was changed. This package runs individual experiments without executing any notebook.

## Source and inputs

The base is exactly `c825bf57652005c80faff79cf4dfd58e3a9baa14`; new experiment code lives on `codex/paper-rl-ablations`. The base was not fetchable from the public GitHub remote when checked. Use the committed-source export below; do not clone a moving branch and assume it is the right experiment.

Required Kaggle dataset attachments:

| File | Required identity |
|---|---|
| `tng100_clustered.csv` | SHA256 `6c148f2cae68abac7bc5c34d370ff0130a647b49e79c32880bcf36ecc0ab4d53` |
| `best_model_augmented.pt` | SHA256 `42764f0db204863b28672d6ec6e207443602de16dcfa926765a0429b3451c63f` |
| `notebook_B_full_outputs.zip` | Original recovered-training archive, SHA256 `800c65be6fe3c02c96967c1db4fd4c77c6324d7df19f72898e60151f0659bf40` |
| Committed source ZIP | Generated below; contains `ABLATION_SOURCE.json` with the exact experiment commit and file checksums |

The archive must contain `recovered_training/{training_checkpoint.json,policy.pt,value_net.pt,training_log.csv,random_validation.csv,policy_validation.json}`. The loader checks the manifest digest, every recorded training artifact, input checksums, split IDs/order, and actual frozen-backbone state hash. It loads the recovered policy solely as a reference; new training builds a fresh policy and critic. No Stage-B weights are loaded. The archive and dataset attachments remain read-only.

After committing code locally, export a source attachment:

```bash
python -m experiments.paper_ablations.package_source --output /tmp/cosmic-rl-ablations-source.zip
```

This exports committed Git blobs, verifies their equality to working files, records the exact commit, and refuses overwrite. It excludes raw data, model weights, and result files. `ABLATION_SOURCE.json` is the authoritative commit pin for the exported source. Upload this ZIP as a Kaggle dataset attachment along with the three inputs above. This does not require pushing the branch.

## Kaggle setup

Enable a GPU for 60-epoch retraining. Unpack the source into a **fresh empty** `/kaggle/working/cosmic-rl-ablations` directory. Example setup cell (replace attachment locations with the actual paths shown by Kaggle):

```python
import json, os, pathlib, sys, zipfile
SOURCE_ZIP = pathlib.Path('/kaggle/input/rl-ablation-source/cosmic-rl-ablations-source.zip')
REPO = pathlib.Path('/kaggle/working/cosmic-rl-ablations')
REPO.mkdir(exist_ok=False)
with zipfile.ZipFile(SOURCE_ZIP) as z:
    z.extractall(REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
PIN = json.loads((REPO/'ABLATION_SOURCE.json').read_text())['commit']
print('Exact experiment commit:', PIN)
```

Use Kaggle's CUDA-enabled PyTorch, plus `torch-geometric`, NumPy, SciPy, pandas, h5py, NetworkX, PyYAML, requests, matplotlib, python-dotenv, and pytest. The portable torch graph builder avoids a torch-cluster dependency. Do not blindly install the repository's 2023 pinned CUDA stack over Kaggle's runtime. Verify imports and record versions in outputs; exact local versions are saved in the result provenance. Exact floating-point equality across different hardware/software is not promised. Missing dependencies can be attached as wheels for an offline session or installed in the setup cell with Internet enabled.

Run a configuration with an explicit immutable source pin:

```python
import subprocess
CSV = '/kaggle/input/cosmicnet-data/tng100_clustered.csv'
BACKBONE = '/kaggle/input/cosmicnet-data/best_model_augmented.pt'
RECOVERED = '/kaggle/input/recovered-policy/notebook_B_full_outputs.zip'
OUT = '/kaggle/working/rl-ablations/prediction_only-seed42-q04'
command = [sys.executable, '-m', 'experiments.paper_ablations.run',
    '--csv', CSV, '--backbone', BACKBONE, '--recovered-archive', RECOVERED,
    '--expected-commit', PIN, '--variant', 'prediction_only', '--seed', '42',
    '--epochs', '60', '--keep-fraction', '.4', '--device', 'cuda', '--output', OUT]
# Optional: first validate exact inputs/cohorts without training.
subprocess.run(command + ['--preflight-only'], check=True)
# Resume this preflight directory into real training.
subprocess.run(command + ['--resume'], check=True)
```

For the smallest useful matrix, run `prediction_only`, `prediction_physics`, `full`, and `no_virial` for paired seeds 42,43,44, each with a distinct output directory. Complete all configurations; do not select only successful improvements. Remaining registered variants: `no_sparsity`, `no_connectivity`, `no_entropy`, `no_value`, `virial_0.1`, `virial_0.3`, `virial_3.0`. The full configuration has virial weight one. Exact coefficient semantics and priority are in [the specification](../../paper/evidence/ablation-plan.md).

For retention-specialized training at .2/.4/.6/.8, give **every** run `--cohort-budgets .2 .4 .6 .8` and its own `--keep-fraction`. For a training-constraint experiment, compare `--training-decoder direct --training-cohort exact_feasible` to `--training-decoder legacy_repair --training-cohort exact_feasible`; do not use the original TRAIN population for only one arm. Direct feasibility checks can be slow; no runtime estimate is asserted.

## Outputs and resume

Each run saves `run.json` (source files/commit, config, seeds, input/checkpoint identities, original and execution split IDs, status and output checksums), `eligibility.json`, an atomic per-epoch `resume.pt` (actor, critic, both Adam states, Torch CPU/CUDA RNG, latest epoch, full history, selected best actor/critic), `selected_policy.pt`, `selected_value.pt`, `selected_validation.json`, `training_log.json`, `results.json`, and `per_halo_predictions.json`. Test predictions include exact budgets, physical isolates/components, selected pair indices, target, prediction, and relative-virial penalty. The full graph can be reconstructed from the pinned dataset and graph config.

`--resume` verifies the same source/data/config/population/device contract. An interrupted epoch restarts from the previous atomic epoch boundary, including optimizer and RNG state. The current loop's randomness comes from Torch; Python/NumPy are also initially seeded. A completed run is skipped only after all output checksums pass. A changed contract or corrupted completed artifact fails; use a new directory. The source archive must remain available unchanged, and Kaggle outputs (especially `resume.pt`) must be downloaded/attached when moving sessions. A weights-only policy file is not sufficient to resume optimization.

The selected checkpoint includes initialization as a validation candidate and may have `best_epoch=null`. Test evaluation occurs only after all requested epochs finish. A running/preflight/partial directory is not an ablation finding. Copying or loading the recovered weights never marks a reward condition trained.

## Reproduce inference diagnostics locally or on Kaggle CPU

```bash
python -m experiments.paper_ablations.diagnose \
  --csv /path/tng100_clustered.csv --backbone /path/best_model_augmented.pt \
  --recovered-archive /path/notebook_B_full_outputs.zip \
  --output /path/new-diagnostic-run
```

Defaults: .2/.4/.6/.8 retention, sampling seeds 7/19/43/71/101, fixed seed-42 deterministic marks, one CPU thread. This never updates policy or predictor weights. It saves every prediction/mask and budget report. Primary decoder/budget comparisons use a topology-only common feasible cohort. Nominal additive-repair accuracy is withheld if any halo exceeds its budget; the separate matched-realized-cost arms have explicit per-halo counts. Frozen full-graph accuracy uses the same halo cohort as a different-budget reference.

Run full repository validation with `pytest -q` in the Git checkout. The tests include exact numerical agreement with the existing trainer, interrupted-vs-uninterrupted optimizer/RNG replay, the separate critic negative control, budget mismatch rejection, coefficient isolation, saved result recomputation, and artifact integrity. They are software validation, not newly trained scientific ablations. In the exported Kaggle source, use `pytest -q tests/test_paper_ablations.py -k 'not recorded'`: recorded-result tests require the separate local result artifacts, and the repository's historical notebook tests require Git history that is not in the source ZIP. Run the real-input `--preflight-only` command before starting GPU training.

Local Git note: the original shared Git directory contains cloud-backed metadata that stalled normal commands. The isolated checkout was reconstructed from matching tracked blobs and verified against root tree `af1e1dec692a92b5eb132793981019135751089d`. An external metadata/object cache under `/tmp/cosmic-rl-ablations-*` allowed Git operations without changing original working files. This workaround is unrelated to model/data provenance and is unnecessary in the offline Kaggle ZIP.
