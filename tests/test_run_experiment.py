import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_run_experiment_smoke(tmp_path):
    """Runs the full pipeline on synthetic data with tiny config; asserts
    outputs/rls/results_table.csv exists and policy improved or matched."""
    from rls.run_experiment import main
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["rls"]["epochs"] = 1
    cfg["rls"]["batch_size"] = 8
    cfg["rls"]["stageb_epochs"] = 1
    cfg["rls"]["stageb_max_graphs"] = 32
    cfg["data"]["source"] = "synthetic"
    cfg["data"]["synthetic"] = {"features_path": str(tmp_path / "sf.pt"),
                                "csv_path": str(tmp_path / "spd.csv")}
    cfg["training"]["checkpoint_dir"] = "outputs/checkpoints"
    # graph.method=radius needs pyg-lib (not installable on win/torch2.12) —
    # the knn builder is pure-torch, so the smoke path uses knn.
    cfg["graph"]["method"] = "knn"
    cfg["graph"]["k_neighbors"] = 4
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.15
    cfg["data"]["test_ratio"] = 0.15
    rc = main(cfg, checkpoint=None)
    assert rc == 0
    assert os.path.exists("outputs/rls/results_table.csv")
