import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import pandas as pd
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
                             require_real_data=False,
                             allow_random_backbone=True)
    results_path = os.path.join(out, "cross_sim_results.csv")
    assert os.path.exists(results_path)
    rows = pd.read_csv(results_path)
    assert {"evaluation_mode", "research_label", "synthetic", "fallback"} <= set(rows.columns)
    assert set(rows["evaluation_mode"]) == {"smoke"}
    assert set(rows["research_label"]) == {"synthetic"}
    assert rows["synthetic"].all()
    assert rows["fallback"].all()

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
    with pytest.raises(ValueError, match="synthetic|audited|legacy"):
        evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                           out_dir=str(tmp_path / "out"))


def test_persisted_synthetic_cache_is_rejected_before_graph_or_model(tmp_path, monkeypatch):
    """A synthetic HDF5 cache remains synthetic when reopened by a new loader."""
    from data.loaders.camels_loader import CAMELSLoader
    from graph.graph_builder import GraphBuilder
    from model import model as model_module

    cfg = {
        "seed": 13,
        "data": {"source": "camels", "camels": {
            "suite": "IllustrisTNG", "simulation": "LH_0",
            "cache_dir": str(tmp_path / "camels_cache"), "offline": True,
        }},
        "graph": {"method": "knn", "k_neighbors": 4},
        "training": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    # Persist fallback data in one process/loader, then reopen the same cache
    # with a fresh loader whose in-memory flag starts false.
    first = CAMELSLoader(cfg)
    first._generate_synthetic_camels(first._get_cache_path())
    reopened = CAMELSLoader(cfg)
    assert reopened.used_synthetic_fallback is False

    side_effects = []
    monkeypatch.setattr(
        GraphBuilder, "build_graphs",
        lambda *args, **kwargs: side_effects.append("graph"),
    )
    monkeypatch.setattr(model_module, "build_model",
                        lambda *args, **kwargs: side_effects.append("model"))
    monkeypatch.setattr(model_module, "load_model",
                        lambda *args, **kwargs: side_effects.append("model"))

    with pytest.raises(ValueError, match="synthetic|audited"):
        evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                           out_dir=str(tmp_path / "out"))
    assert side_effects == []


def test_reopened_synthetic_cache_contract_is_truthful(tmp_path):
    from data.loaders.camels_loader import CAMELSLoader

    cfg = {"seed": 5, "data": {"camels": {
        "cache_dir": str(tmp_path / "camels_cache"), "offline": True,
    }}}
    initial = CAMELSLoader(cfg)
    initial._generate_synthetic_camels(initial._get_cache_path())
    reopened = CAMELSLoader(cfg)
    reopened.load_raw()

    contract = reopened.get_catalog_contract()
    assert reopened.used_synthetic_fallback is True
    assert contract.synthetic is True
    assert contract.fallback is True
    assert contract.research_label == "synthetic"


def test_unmarked_legacy_camels_cache_is_rejected_as_unverified(tmp_path, monkeypatch):
    """Legacy cache files lack proof of source/target semantics and fail closed."""
    import h5py
    from data.loaders.camels_loader import CAMELSLoader
    from graph.graph_builder import GraphBuilder
    from model import model as model_module

    cfg = {
        "seed": 21,
        "data": {"source": "camels", "camels": {
            "suite": "IllustrisTNG", "simulation": "LH_0",
            "cache_dir": str(tmp_path / "camels_cache"), "offline": True,
        }},
        "graph": {"method": "knn", "k_neighbors": 4},
        "training": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    loader = CAMELSLoader(cfg)
    cache_path = loader._generate_synthetic_camels(loader._get_cache_path())
    # Emulate a pre-marker legacy cache (including old generated files).
    with h5py.File(cache_path, "r+") as cache:
        del cache.attrs[loader.SYNTHETIC_PROVENANCE_ATTR]

    side_effects = []
    monkeypatch.setattr(
        GraphBuilder, "build_graphs",
        lambda *args, **kwargs: side_effects.append("graph"),
    )
    monkeypatch.setattr(model_module, "build_model",
                        lambda *args, **kwargs: side_effects.append("model"))
    monkeypatch.setattr(model_module, "load_model",
                        lambda *args, **kwargs: side_effects.append("model"))

    with pytest.raises(ValueError, match="audited|legacy|scientific"):
        evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                           out_dir=str(tmp_path / "out"), require_real_data=True)
    assert side_effects == []


def test_real_data_mode_fails_before_loader_for_non_camels_source(monkeypatch):
    """The legacy evaluator has no audit preflight for any configured source."""
    from data.loaders import base_loader

    side_effects = []
    monkeypatch.setattr(base_loader, "get_loader",
                        lambda cfg: side_effects.append("loader"))
    cfg = {"data": {"source": "synthetic"}}

    with pytest.raises(ValueError, match="audited|legacy"):
        evaluate_cross_sim(cfg, checkpoint=None, require_real_data=True)
    assert side_effects == []
