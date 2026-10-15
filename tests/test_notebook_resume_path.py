"""A complete B rerun can reuse selected policy bytes without retraining."""
import hashlib
import importlib
import json

import torch

from tests.test_notebook_a_baselines import small_inputs
from tests.test_notebook_pipeline import run_notebook, single_thread


def test_notebook_b_recovery_skips_training_and_preserves_originals(
        tmp_path, monkeypatch, small_inputs, single_thread):
    monkeypatch.chdir(tmp_path)
    first = run_notebook('B', tmp_path)
    sparse = first['results_df'].set_index('method').loc['rl_policy']
    eligible = [row for row in first['budget_reports']['test']['graphs'] if row['eligible']]
    physical_keep = sum(row['realized_pair_count'] / row['available_pair_count']
                        for row in eligible) / len(eligible)
    assert sparse['mean_keep_frac'] == sparse['keep_frac'] == physical_keep
    assert sparse['mean_stored_keep_frac'] > physical_keep
    original = tmp_path / 'B' / 'smoke'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in original.iterdir() if p.is_file()}
    module = importlib.import_module('rls.train_policy')
    def forbidden(*args, **kwargs):
        raise AssertionError('A recovered policy must not enter the training loop')
    monkeypatch.setattr(module, 'train_policy', forbidden)
    second = run_notebook('B', tmp_path,
        OUTPUT_DIR=tmp_path / 'B-recovered', RESUME_POLICY_DIR=original)
    recovered = tmp_path / 'B-recovered' / 'smoke'
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.iterdir() if p.is_file()}
    assert (recovered / 'policy.pt').read_bytes() == (original / 'policy.pt').read_bytes()
    assert all(torch.equal(value, second['policy'].state_dict()[key])
               for key, value in first['policy'].state_dict().items())
    manifest = json.loads((recovered / 'provenance.json').read_text())['B']
    assert manifest['status'] == 'complete'
    assert manifest['recovered_policy']['status'] == 'policy_recovered'
    # The copied training manifest retains its original, training-only scope.
    training = json.loads((recovered / 'recovered_training/training_checkpoint.json').read_text())
    assert training['status'] == 'policy_training_complete'
    assert training['notebook_status'] == 'incomplete'
