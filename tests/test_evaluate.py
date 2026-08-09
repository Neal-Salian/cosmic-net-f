import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from rls.evaluate import build_results_table, save_paper_plots

def test_results_table_columns(tmp_path):
    torch.manual_seed(0)
    rows = build_results_table(
        preds_full=torch.randn(10), preds_policy=torch.randn(10),
        preds_gumbel=torch.randn(10), preds_random=torch.randn(10),
        targets=torch.randn(10),
        masks_policy=[torch.rand(20).round().bool() for _ in range(10)],
        masks_gumbel=[torch.rand(20).round().bool() for _ in range(10)],
        masks_random=[torch.rand(20).round().bool() for _ in range(10)],
    )
    df = pd.DataFrame(rows)
    required = {"method", "rmse", "r2", "scatter", "mean_keep_frac", "fidelity"}
    assert required.issubset(set(df.columns)), df.columns

def test_save_paper_plots_writes_files(tmp_path):
    torch.manual_seed(0)
    rows = build_results_table(
        preds_full=torch.randn(10), preds_policy=torch.randn(10),
        preds_gumbel=torch.randn(10), preds_random=torch.randn(10),
        targets=torch.randn(10),
        masks_policy=[torch.rand(20).round().bool() for _ in range(10)],
        masks_gumbel=[torch.rand(20).round().bool() for _ in range(10)],
        masks_random=[torch.rand(20).round().bool() for _ in range(10)],
    )
    out = tempfile.mkdtemp(dir=tmp_path)
    save_paper_plots(rows, out_dir=out)
    assert os.path.isfile(os.path.join(out, "results_table.csv"))
    assert os.path.isfile(os.path.join(out, "pareto_and_fidelity.png"))


def test_evaluate_tta_rows():
    """evaluate_tta returns one row per (mode, K); K=0 runs the FROZEN path
    (no adaptation) and must not touch the label-free TTA machinery."""
    from rls.policy import EdgePolicyNet
    from rls.evaluate import evaluate_tta

    torch.manual_seed(0)

    class MockGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.tensor(0.1))
        def predict_with_uncertainty(self, batch, n_samples=15):
            E = batch.edge_index.shape[1]
            return {"std": torch.tensor([1.0 / max(1, E)])}
        def forward(self, batch):
            n = batch.x.shape[0]
            return torch.rand(n, 1).sum(dim=0, keepdim=True), torch.randn(n, 4)

    def graph():
        n, e = 6, 12
        return {"x": torch.randn(n, 4), "edge_index": torch.randint(0, n, (2, e)),
                "edge_attr": torch.randn(e, 5), "y": torch.randn(1),
                "ctx": torch.randn(4), "emb": torch.randn(n, 4),
                "stellar_mass": torch.rand(n) * 1e10,
                "vel_disp": torch.rand(n) * 200,
                "half_mass_r": torch.rand(n) * 0.01, "pos": torch.randn(n, 3)}

    graphs = [graph() for _ in range(3)]
    policy = EdgePolicyNet(edge_dim=5, node_emb_dim=4, hidden_dim=8)
    cfg = {"tta_lr": 1e-3, "tta_steps": 3, "tta_mc_samples": 5, "tta_patience": 2,
           "min_keep_frac": 0.1, "entropy_coef": 0.01,
           "w_unc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 0.0, "seed": 0}
    rows = evaluate_tta(policy, graphs, MockGNN(), cfg, "cpu", ks=(0, 2))
    assert len(rows) == 2
    assert {r["mode"] for r in rows} == {"frozen", "tta"}
    for r in rows:
        assert {"mode", "K", "init", "rmse", "fidelity", "keep_frac",
                "mean_steps", "mean_time_s"} <= set(r)
    frozen_row = [r for r in rows if r["K"] == 0][0]
    assert frozen_row["mode"] == "frozen" and frozen_row["mean_steps"] == 0.0
