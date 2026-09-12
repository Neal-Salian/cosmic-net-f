"""Execute Notebook A's baseline cells in the documented 15 -> 14 -> 13 order."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model.model import build_model


@pytest.mark.parametrize("device_name", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is not available")),
])
def test_baseline_cells_run_in_user_order(tmp_path, monkeypatch, device_name):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/notebook_A_setup_data_graphs_baselines.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    sources = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
               for c in cells if c["cell_type"] == "code"]

    def source_starting_with(prefix):
        return next(s for s in sources if s.startswith(prefix))

    device = torch.device(device_name)
    cfg = {
        "graph": {"edge_features": ["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"]},
        "model": {"node_features": 4, "hidden_dim": 8, "output_dim": 8,
                  "num_layers": 1, "dropout": 0.0},
        "rls": {"seed": 42},
    }
    gnn = build_model(cfg).to(device).eval()
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]])
    graphs = [Data(x=torch.rand(3, 4), edge_index=edge_index.clone(),
                   edge_attr=torch.rand(6, 5), pos=torch.rand(3, 3),
                   y=torch.tensor([12.0 + i])) for i in range(2)]
    namespace = {"cfg": cfg, "gnn": gnn, "device": device,
                 "test_loader": DataLoader(graphs, batch_size=2)}
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/rls").mkdir(parents=True)

    # Scorer/import setup, then the actual evaluation, smoke test and device check.
    for prefix in ["# Baseline masks", "# Evaluate baseline sparsifiers",
                   "b = next(iter(test_loader))", 'print("=== DEVICE CHECK ===")']:
        exec(compile(source_starting_with(prefix), f"Notebook A: {prefix}", "exec"), namespace)

    results = namespace["baseline_df"]
    assert len(results) == 6 * 6 + 1
    assert np.isfinite(results["rmse"]).all()
    assert (tmp_path / "outputs/rls/baselines.csv").is_file()
    assert namespace["mask"].device == namespace["g"].edge_index.device
    assert namespace["mask"].sum().item() == 3
    assert np.isfinite(namespace["pred"])
