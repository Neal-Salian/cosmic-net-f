"""Execute the notebook boundaries that previously assumed all budgets feasible."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rls.notebook_budget import budget_cohort, require_budget_cohort, record_budget_realizations
from rls.notebook_workflow import policy_masks
from rls.policy import EdgePolicyNet
from tests.test_notebook_budget_cohorts import cfg, make_graph, reported_graph

ROOT = Path(__file__).resolve().parents[1]


def cell(stage, name):
    path = next((ROOT / 'notebooks').glob(f'notebook_{stage}*.ipynb'))
    return next(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                if c.get('id') == name)


def graphs():
    return [make_graph(name, [(0, 1), (0, 2), (1, 2)], 3) for name in ('a', 'b')] + [reported_graph()]


@pytest.mark.parametrize('empty_train', [False, True])
def test_stage_b_uses_aligned_feasible_subset_or_explicitly_skips(tmp_path, empty_train):
    original = graphs()
    training = [original[-1]] if empty_train else original
    cohorts, reports = {}, {}
    for split, items in (('train', training), ('val', original)):
        cohorts[split], reports[split] = budget_cohort(items, cfg(), split=split)
    called = []
    def fine_tune(model, train, masks, **kwargs):
        called.append(True)
        assert [g['cluster_id'] for g in train] == ['a', 'b']
        assert [g['cluster_id'] for g in kwargs['val_graphs']] == ['a', 'b']
        assert len(train) == len(masks) == len(kwargs['val_masks']) == 2
        return [1.], dict(best_epoch=0, accepted=False, full_hist=[1.], pruned_hist=[1.])
    namespace = dict(RUN_STAGE_B=True, policy_quality={'verdict': 'PASS'}, SMOKE_TEST=False,
        budget_graphs=cohorts, budget_reports=reports, final_training_keep=.4,
        train_graphs=training, validation_graphs=cohorts['val'],
        policy=EdgePolicyNet(5, 4, hidden_dim=8), rls_cfg=cfg(), policy_masks=policy_masks,
        record_budget_realizations=record_budget_realizations,
        gnn=torch.nn.Linear(1, 1), copy=copy, fine_tune_gnn=fine_tune,
        stageb_epochs=1, device='cpu', np=np, pd=pd, torch=torch,
        record_backbone=lambda *args: {'stage': 'fixture'}, OUT=tmp_path, artifacts=[])
    exec(compile(cell('B', 'stage-b'), 'B:stage-b', 'exec'), namespace)
    assert namespace['train_graphs'] == training  # Original training population is preserved.
    assert namespace['stageb_info']['excluded_training_graph_ids'] == ['reported-12-5-6']
    if empty_train:
        assert not called
        assert namespace['stageb_model'] is None
        assert namespace['stageb_info']['skip_reason'] == 'no_exact_feasible_training_graphs'
    else:
        assert called
        assert all(r['realized_pair_count'] == r['requested_pair_count'] == 2
                   for r in reports['train']['graphs'] if r['eligible'])
        assert reports['train']['graphs'][-1]['realized_pair_count'] is None


def test_b_preflight_rejects_insufficient_validation_before_policy_construction(tmp_path):
    source = cell('B', 'train-policy').split('from rls.notebook_recovery import', 1)[0]
    namespace = dict(cfg={'rls': dict(cfg(), epochs=60)}, POLICY_OVERRIDES={},
        POLICY_EPOCHS=None, STAGEB_EPOCHS=None, SMOKE_TEST=False,
        prepared={'train': graphs(), 'val': [reported_graph()], 'test': graphs()},
        budget_cohort=budget_cohort, require_budget_cohort=require_budget_cohort,
        OUT=tmp_path, artifacts=[], json=json)
    with pytest.raises(ValueError, match='val.*0.*2'):
        exec(compile(source, 'B:preflight', 'exec'), namespace)
    report = json.loads((tmp_path / 'budget_eligibility.json').read_text())
    assert report['val']['excluded_graph_ids'] == ['reported-12-5-6']
    assert report['val']['graphs'][0]['requested_pair_count'] == 5


def test_d_preflight_uses_resolved_tta_override_before_any_adaptation(tmp_path):
    original = graphs()
    namespace = dict(TTA_OVERRIDES={'tta_target_sparsity': .5}, cfg={'rls': dict(cfg(),
        tta_mc_samples=2, tta_lr=.001)}, SMOKE_TEST=False,
        policy=EdgePolicyNet(5, 4, hidden_dim=8), val_graphs=original, test_graphs=original,
        budget_cohort=budget_cohort, require_budget_cohort=require_budget_cohort,
        OUT=tmp_path, json=json, artifacts=[])
    exec(compile(cell('D', 'tta-settings'), 'D:tta-settings', 'exec'), namespace)
    for split in ('val', 'test'):
        assert namespace['tta_budget_reports'][split]['requested_keep_fraction'] == .5
        assert namespace['tta_graphs'][split] == original
        assert namespace['tta_budget_reports'][split]['graphs'][-1]['requested_pair_count'] == 6


def test_c_keeps_primary_failures_and_aligns_secondary_masks(tmp_path, monkeypatch):
    import rls.notebook_benchmark as benchmark
    original = graphs()
    seen = []
    def matched(model, input_graphs, *args, **kwargs):
        seen.extend(g['cluster_id'] for g in input_graphs)
        return [dict(method='provided_rl', frac=.4, rmse=0., mae=0., r2=1.,
                     n_failed=1, common_n=2)], [dict(graph_id=g['cluster_id'],
                     status='failed' if g is original[-1] else 'success') for g in input_graphs]
    monkeypatch.setattr(benchmark, 'matched_notebook_rows', matched)
    monkeypatch.setattr(benchmark, 'load_learned_control', lambda *args: (None, None))
    namespace = dict(A_DIR=tmp_path, a_manifest={'fractions': [.4]},
        checked_artifact=lambda *args: tmp_path / 'unused', context={},
        cfg={'rls': cfg()}, copy=copy, gnn=None, split_graphs={'test': original},
        policy=EdgePolicyNet(5, 4, hidden_dim=8),
        save_paper_plots=lambda rows, out: pd.DataFrame(rows),
        OUT=tmp_path, json=json, artifacts=[], budget_cohort=budget_cohort,
        record_budget_realizations=record_budget_realizations,
        policy_masks=policy_masks, test_graphs=original, EVALUATE_STAGE_B=False)
    exec(compile(cell('C', 'evaluate'), 'C:evaluate', 'exec'), namespace)
    assert seen == ['a', 'b', 'reported-12-5-6']
    assert namespace['comparison_per_example'][-1]['status'] == 'failed'
    assert [g['cluster_id'] for g in namespace['diagnostic_graphs']] == ['a', 'b']
    assert len(namespace['test_masks']) == len(namespace['test_probabilities']) == 2
    assert namespace['diagnostic_budget_report']['graphs'][-1]['requested_pair_count'] == 5
