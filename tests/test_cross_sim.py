import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from rls.cross_sim import evaluate_cross_sim

def test_cross_sim_runs(tmp_path):
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["source"] = "camels"
    cfg["data"]["camels"] = {"suite": "IllustrisTNG", "simulation": "LH_0",
                             "cache_dir": str(tmp_path / "camels_cache"),
                             "offline": True}
    cfg["graph"]["method"] = "knn"
    cfg["graph"]["k_neighbors"] = 4
    # synthetic fallback is generated when the HDF5 is missing (offline mode
    # skips the network download attempt entirely). This plumbing test opts
    # OUT of the real-data guard explicitly.
    out = evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                             out_dir=str(tmp_path / "out"),
                             require_real_data=False)
    assert os.path.exists(os.path.join(out, "cross_sim_results.csv"))

def test_cross_sim_rejects_synthetic_fallback_by_default(tmp_path):
    """Default require_real_data=True must refuse to produce OOD results from
    the loader's synthetic fallback (publish-gate regression test)."""
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["source"] = "camels"
    cfg["data"]["camels"] = {"suite": "IllustrisTNG", "simulation": "LH_0",
                             "cache_dir": str(tmp_path / "camels_cache"),
                             "offline": True}
    cfg["graph"]["method"] = "knn"
    cfg["graph"]["k_neighbors"] = 4
    with pytest.raises(AssertionError, match="synthetic"):
        evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                           out_dir=str(tmp_path / "out"))
