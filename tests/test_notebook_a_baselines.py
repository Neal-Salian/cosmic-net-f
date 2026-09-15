"""Notebook A: execute all local cells in order and check setup failure paths."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from model.model import build_model

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/notebook_A_setup_data_graphs_baselines.ipynb"


def cells():
    return {cell["id"]: "".join(cell["source"])
            for cell in json.loads(NOTEBOOK.read_text())["cells"]
            if cell["cell_type"] == "code"}


def execute(source, namespace, name="Notebook A"):
    exec(compile(source, name, "exec"), namespace)


@pytest.fixture
def small_inputs(tmp_path):
    folder = tmp_path / "input" / "nested" / "cosmicnet-data"
    folder.mkdir(parents=True)
    rows = []
    for halo in range(16):
        for node in range(3):
            rows.append(dict(subhalo_id=halo * 3 + node, group_id=halo,
                             stellar_mass=10.0 + 0.1 * node, vel_dispersion=100.0,
                             half_mass_radius=0.01, metallicity=0.02,
                             pos_x=halo + 0.01 * node, pos_y=0.0, pos_z=0.0,
                             vel_x=0.01 * node, vel_y=0.0, vel_z=0.0,
                             halo_mass=10.0 ** (12.0 + halo / 10)))
    pd.DataFrame(rows).to_csv(folder / "tng100_clustered.csv", index=False)
    cfg = {
        "graph": {"method": "radius", "radius_mpc": 2.0, "self_loops": True,
                  "edge_features": ["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"]},
        "model": {"node_features": 4, "hidden_dim": 8, "output_dim": 8,
                  "num_layers": 1, "dropout": 0.0},
    }
    model = build_model(cfg)
    torch.save({"config": cfg, "model_state_dict": model.state_dict()},
               folder / "best_model_augmented.pt")
    return folder


@pytest.mark.parametrize("device_name", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is not available")),
])
def test_run_all_and_reruns(tmp_path, monkeypatch, small_inputs, device_name):
    sources = cells()
    namespace = {}
    execute(sources["settings"], namespace)
    namespace.update(REPO_DIR=ROOT, SYNC_REPO=False, INSTALL_MISSING=False,
                     INPUT_ROOT=tmp_path / "input", DEVICE=device_name,
                     GRAPH_BACKEND="torch", SMOKE_TEST=True, GUMBEL_EPOCHS=1,
                     SMOKE_GRAPHS_PER_SPLIT=2, OUTPUT_DIR=tmp_path / "output")
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, source in sources.items():
            if name != "settings":
                execute(source, namespace, f"Notebook A: {name}")
        results = namespace["baseline_df"]
        assert len(results) == 6 * 6 + 2
        assert np.isfinite(results[["rmse", "r2", "keep_frac"]]).all().all()
        assert namespace["mask"].device == namespace["g"].edge_index.device
        assert namespace["gnn"].hidden_dim == 8
        assert all(p.grad is None for p in namespace["gnn"].parameters())
        assert any(p.grad is not None for p in namespace["gumbel_gnn"].parameters())
        out = tmp_path / "output" / "smoke"
        provenance = json.loads((out / "provenance.json").read_text())["A"]
        assert provenance["status"] == "complete"
        assert provenance["run_mode"] == "smoke"
        assert provenance["full_split_sizes"] == {"train": 11, "val": 2, "test": 3}
        assert provenance["graph_backend"] == "torch"
        assert provenance["dataset"]["checkpoint"]["path"] == str(small_inputs / "best_model_augmented.pt")
        assert provenance["backbone"]["sha256"] != provenance["gumbel_backbone"]["sha256"]
        assert not (tmp_path / "output" / "baselines.csv").exists()
        saved = pd.read_csv(out / "baselines.csv")
        execute(sources["gumbel-evaluate"], namespace)
        pd.testing.assert_frame_equal(saved, pd.read_csv(out / "baselines.csv"))
        execute(sources["device-check"], namespace)
        execute(sources["attention-smoke"], namespace)
    finally:
        torch.set_num_threads(threads)


def input_namespace():
    source = cells()["inputs"].split("CSV_PATH, CHECKPOINT_PATH =")[0]
    namespace = {"Path": Path}
    execute(source, namespace)
    return namespace


def test_input_discovery_rejects_ambiguity_and_lfs(tmp_path, small_inputs):
    locate = input_namespace()["locate_inputs"]
    assert locate(None, tmp_path / "input")[0].parent == small_inputs
    other = tmp_path / "input" / "other"
    other.mkdir()
    (other / "tng100_clustered.csv").write_text("a\n1\n")
    checkpoint = other / "best_model_augmented.pt"
    checkpoint.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(FileNotFoundError, match="Matching folders"):
        locate(None, tmp_path / "input")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        locate(other, tmp_path / "input")
    with pytest.raises(FileNotFoundError, match="Missing or empty"):
        locate(tmp_path / "missing", tmp_path / "input")


def test_input_validation_rejects_nonfinite(tmp_path, small_inputs):
    frame = pd.read_csv(small_inputs / "tng100_clustered.csv")
    frame.loc[0, "pos_x"] = np.inf
    frame.to_csv(small_inputs / "tng100_clustered.csv", index=False)
    with pytest.raises(ValueError, match="finite numeric"):
        execute(cells()["inputs"], {"Path": Path, "INPUT_DIR": small_inputs,
                                     "INPUT_ROOT": tmp_path / "input"})


def test_clone_skips_lfs_and_redacts_errors(tmp_path, monkeypatch):
    secret = "not-a-real-test-token"
    monkeypatch.setitem(sys.modules, "kaggle_secrets", SimpleNamespace(
        UserSecretsClient=lambda: SimpleNamespace(get_secret=lambda name: secret)))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert secret not in " ".join(command)
        env = kwargs["env"]
        assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert "--no-checkout" in command
        return SimpleNamespace(returncode=1, stderr="failure " + secret, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    source = cells()["checkout"].split("REPO_DIR, REPO_HEAD =")[0]
    import os
    namespace = {"Path": Path, "os": os, "subprocess": subprocess}
    execute(source, namespace)
    with pytest.raises(RuntimeError, match=r"failure \[redacted\]"):
        namespace["prepare_repository"](tmp_path / "checkout", "https://github.com/example/repo.git", "test")
    assert len(calls) == 1
    assert not (tmp_path / "checkout").exists()


@pytest.mark.parametrize("method", ["radius", "knn"])
def test_graph_fallback_preserves_neighborhood(monkeypatch, method):
    from graph.graph_builder import GraphBuilder
    source = cells()["graphs"].split("def validate_graph")[0]
    namespace = {"GraphBuilder": GraphBuilder, "torch": torch, "GRAPH_BACKEND": "auto"}
    execute(source, namespace)

    def unavailable(*args, **kwargs):
        raise ImportError("optional graph extension missing")

    monkeypatch.setattr(GraphBuilder, f"_build_{method}_edges", unavailable)
    builder = namespace["NotebookGraphBuilder"]({"graph": {"radius_mpc": 1.5, "k_neighbors": 1}})
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]])
    edges = getattr(builder, f"_build_{method}_edges")(positions, 3)
    pairs = set(map(tuple, edges.T.tolist()))
    expected = {(0, 1), (1, 0)}
    if method == "knn":
        expected |= {(1, 2), (2, 1)}
    assert pairs == expected
    assert builder.backend_used == "torch"
