"""Statistical guardrails: pairing, independent units, and physical budgets."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def analysis():
    path = Path(__file__).resolve().parents[1] / "paper/scripts/analyze_results.py"
    assert path.is_file(), "The reproducible statistical analysis is not implemented"
    spec = importlib.util.spec_from_file_location("paper_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows():
    return [{"graph_id": str(i), "target": 10., "method": "provided_rl",
             "pruned_prediction": 10. + i, "full_prediction": 10.}
            for i in range(3)]


def test_pairing_uses_ids_and_rejects_different_populations():
    a = analysis()
    ids, er, ec = a.align(rows(), list(reversed(rows())))
    assert ids == ["0", "1", "2"]
    np.testing.assert_array_equal(er, [0., 1., 2.])
    np.testing.assert_array_equal(er, ec)
    for bad in (rows()[:-1], rows() + [rows()[0]]):
        with pytest.raises(ValueError):
            a.align(rows(), bad)
    bad = rows()
    bad[0]["target"] = 11.
    with pytest.raises(ValueError):
        a.align(rows(), bad)
    bad = rows()
    bad[0]["pruned_prediction"] = float("nan")
    with pytest.raises(ValueError):
        a.align(rows(), bad)


def test_holm_controls_both_order_and_monotonicity():
    np.testing.assert_allclose(analysis().holm([.04, .01, .03]), [.06, .03, .06])


def test_permutation_swaps_within_halos_not_independent_samples():
    # Six +1 differences have 64 sign assignments: exactly 2 equally extreme.
    p = analysis().paired_permutation(np.ones(6), np.zeros(6), seed=42,
                                      n_resamples=np.inf)
    assert p == 2 / 64


def mask_row():
    return dict(method="random", layout_pairs=[[0, 1], [0, 2], [1, 2]],
                selected_pairs=[0, 2], layout_inverse=[0, 1, 0, 2, 1, 2],
                stored_real=[False, True, True, True, False, True, True, True, False],
                stored_mask=[True, True, False, True, True, True, False, True, True],
                available_pair_count=3, available_stored_columns=9,
                available_nonself_columns=6, available_loop_columns=3,
                requested_keep_fraction=.5, requested_pair_count=2, final_pair_count=2,
                physical_isolates=0, constraint_satisfied=True, budget_satisfied=True,
                physical_retention=2/3, stored_retention=7/9,
                total_kept_columns=7, retained_nonself_columns=4, loops_kept=3)


def test_masks_use_pair_indices_and_preserve_loops_and_symmetry():
    a = analysis()
    a.validate_mask(mask_row())
    for field, value in (("selected_pairs", [0, 0]),
                         ("selected_pairs", [0, 3]),
                         ("final_pair_count", 3),
                         ("physical_retention", 7/9)):
        bad = copy.deepcopy(mask_row())
        bad[field] = value
        with pytest.raises(ValueError):
            a.validate_mask(bad)
    for index in (0, 1):  # Remove a loop or one of a reciprocal pair.
        bad = copy.deepcopy(mask_row())
        bad["stored_mask"][index] = False
        with pytest.raises(ValueError):
            a.validate_mask(bad)
