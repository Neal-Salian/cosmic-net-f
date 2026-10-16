# Validation record — 16 October 2026

No real external astronomy evaluation was executed. Successful software tests do not support generalization.

- New boundary/integration suite: **27 passed**. Tests generated tiny HDF5/model fixtures only; the executable fixture export retains `fixture_smoke`, `scientific_result=false` and `domain_generalization_established=false`.
- Full repository suite: **824 passed, 15 skipped, 2 warnings**. Warnings are the existing Torch JIT deprecation and PyG DataLoader deprecation. Skips remain skips, not validation successes.
- Real candidate checkpoint loadability: `artifact-load-check.json` records strict backbone and 64-dimensional normalized policy state loading, no predictions. Backbone state hash matches recovered B/C provenance (`56907374...`). This is not a training semantic audit.
- Original checkout verification: 224 locally hydrated tracked files matched the pinned base Git blob hashes. Cloud-only tracked files were left unread and unchanged; no original checkout content was edited.
- Python compilation, all package JSON parsing, header-only real-result CSV and no-output-on-blocked-template assertions passed.
- `git diff --cached --check` passed after normalizing the new CSV header to LF.
- Template CLI exited **2** with `BLOCKED: manifest is pending_inputs; supply audited real inputs and mark ready`; no result directory was created.
- Independent code review: no Critical findings. Reproduced and fixed two Important provenance gaps: external HDF5 backing data and incomplete multi-catalog evidence. Also rejected unknown lineage markers, captured/rechecked source code before/after inference and renamed inventory completeness to a header declaration. No unresolved review findings retained.

Commands, from the isolated worktree:

```bash
python -m pytest -q tests/test_paper_generalization.py
python -m pytest -q
python -m compileall -q experiments/paper_generalization
python -m experiments.paper_generalization.run \
  --manifest experiments/paper_generalization/camels-tng-cv0.example.json \
  --output /tmp/cosmic-generalization-real-cv0
```

The test Python executable was `/tmp/cosmic-generalization-venv/bin/python` (Python 3.12). `validation-environment.txt` pins the actual installed packages. The temporary environment is not required for using the package elsewhere; create an equivalent environment. No original virtual environment was changed.

For historical Git-object tests on this machine, final test invocations set:

```bash
export GIT_NO_REPLACE_OBJECTS=1
export GIT_COMMON_DIR=/tmp/cosmic-generalization-git-common
export GIT_OBJECT_DIRECTORY=/tmp/cosmic-generalization-objects.git/objects
export PATH=/tmp/cosmic-generalization-venv/bin:$PATH
```

These isolate reads from cloud-only metadata in the original Git repository. The object store combines locally readable exact objects with a read-only fetch of the existing public `feature/research-strengthening` history to supply missing ancestor objects. The task base remained exactly c825bf57652005c80faff79cf4dfd58e3a9baa14; no branch was merged or pushed. The app worktree operation failed on ref lookup, so the independent Git worktree was created directly. App attachment is unavailable for this manually created checkout.

Initial attempts: collection first reported missing `python-dotenv` in the new environment, then a full run waited in an existing `git show f66bb57:...` test against cloud-only metadata. Installing the dependency and using the isolated object view resolved those environmental issues; the final full suite completed. No test failures were suppressed or relabelled as astronomy results.
