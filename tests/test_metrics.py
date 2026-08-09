import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.metrics import fidelity, stability_jaccard, physics_alignment, calibration_coverage

def test_fidelity_perfect_correlation():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = a * 2 + 0.5
    r = fidelity(a, b)
    assert abs(r - 1.0) < 1e-6

def test_stability_jaccard():
    m1 = torch.tensor([1, 1, 0, 0, 1])
    m2 = torch.tensor([1, 0, 0, 0, 1])
    j = stability_jaccard(m1, m2)
    assert abs(j - 2 / 3) < 1e-6

def test_physics_alignment_positive_spearman():
    torch.manual_seed(0)
    keep_prob = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    u_ij = torch.tensor([0.9, 0.7, 0.6, 0.3, 0.2, 0.1])  # monotone
    rho, p = physics_alignment(keep_prob, u_ij)
    assert rho > 0.5

def test_calibration_coverage():
    pred = torch.tensor([0.0, 1.0, 2.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    y = torch.tensor([0.2, 0.8, 5.0])
    cov = calibration_coverage(pred, std, y)
    assert abs(cov - 2 / 3) < 1e-6
