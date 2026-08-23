"""
Regression tests: SyntheticLoader must report synthetic provenance
(used_synthetic_fallback = True) from init, and rls/cross_sim.py's default
require_real_data=True must reject it — including the config-default path
where data.source is omitted entirely (get_loader defaults to 'synthetic').
Before this fix the attribute was absent, so getattr(..., False) let
synthetic-source OOD results pass the real-data guard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml

from data.loaders.synthetic_loader import SyntheticLoader


def _synthetic_cfg(tmp_path, drop_source=False):
    cfg = {
        "data": {
            "synthetic": {
                "features_path": str(tmp_path / "missing_features.pt"),
                "csv_path": str(tmp_path / "missing_meta.csv"),
            },
        },
        "seed": 42,
    }
    if not drop_source:
        cfg["data"]["source"] = "synthetic"
    return cfg


def test_synthetic_loader_reports_synthetic_from_init(tmp_path):
    loader = SyntheticLoader(_synthetic_cfg(tmp_path))
    assert loader.used_synthetic_fallback is True


def test_cross_sim_rejects_synthetic_source_by_default(tmp_path):
    from rls.cross_sim import evaluate_cross_sim
    cfg = _synthetic_cfg(tmp_path)
    cfg["graph"] = {"method": "knn", "k_neighbors": 4}
    with pytest.raises(AssertionError, match="synthetic"):
        evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                           out_dir=str(tmp_path / "out"))


def test_cross_sim_rejects_config_default_source_by_default(tmp_path):
    """data.source omitted -> get_loader defaults to 'synthetic'; the guard
    must still reject it."""
    from rls.cross_sim import evaluate_cross_sim
    cfg = _synthetic_cfg(tmp_path, drop_source=True)
    cfg["graph"] = {"method": "knn", "k_neighbors": 4}
    with pytest.raises(AssertionError, match="synthetic"):
        evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                           out_dir=str(tmp_path / "out"))
