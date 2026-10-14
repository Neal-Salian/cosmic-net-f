"""The archived pilot is immutable; this pins the corrected future protocol."""
from collections import defaultdict

import torch

from rls.benchmark_predictor import MeteredPredictor
from rls.constraints import PhysicalPairLayout
from rls.feasible_pilot import _validate_method, decode
from rls.pilot_synthetic import build_datasets, SyntheticAdditivePredictor


def test_validation_of_unchanged_scorer_does_not_change_with_epoch():
    _, examples = build_datasets(n_train_add=1, n_train_int=0, n_val_add=1, n_val_int=0)
    models = {"additive": MeteredPredictor(SyntheticAdditivePredictor())}
    oracle = {r["graph_id"]: {2: {"se": 0.}} for r in examples}
    full = {r["graph_id"]: 0. for r in examples}
    rows = []
    for epoch in range(6):
        _validate_method(lambda g, layout, k: torch.zeros(len(layout.pairs)),
            "scaffold", 2, examples, oracle, full, models,
            {"constraint": "no_isolates", "structured_temperature": 1.},
            0, epoch, rows, defaultdict(float))
    assert len({r["prediction"] for r in rows}) == 1


def test_scaffold_action_count_excludes_deterministic_scaffold():
    _, examples = build_datasets(n_train_add=1, n_train_int=0, n_val_add=1, n_val_int=0)
    data = examples[0]["data"]
    layout = PhysicalPairLayout.from_edge_index(data.edge_index, 4)
    result = decode(layout, torch.ones(6), 2, "scaffold",
                    pair_marks=torch.arange(6), constraint="no_isolates", sample=False)
    assert result["final_count"] == 2
    assert result["sampled_count"] == 0
