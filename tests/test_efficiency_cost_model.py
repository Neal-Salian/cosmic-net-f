"""Break-even must charge setup and cannot invent savings when the GNN is slower."""
import pytest


def test_break_even_charges_setup_and_requires_strict_saving():
    from benchmarks.paper_efficiency.cost_model import break_even_reuses
    assert break_even_reuses(.01, .005, .003) == 6
    assert break_even_reuses(0., .005, .003) == 1
    assert break_even_reuses(1., .005, .005) is None
    assert break_even_reuses(1., .004, .005) is None
    # Decimal-looking coefficients must not round equality into a speedup.
    assert break_even_reuses(.3, .4, .1) == 2


@pytest.mark.parametrize('values', [(-1., 2., 1.), (1., float('nan'), 1.), (1., 2., -1.)])
def test_invalid_costs_rejected(values):
    from benchmarks.paper_efficiency.cost_model import break_even_reuses
    with pytest.raises(ValueError):
        break_even_reuses(*values)
