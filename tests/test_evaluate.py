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
