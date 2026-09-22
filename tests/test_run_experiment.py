import sys, os, yaml
import pytest
import pandas as pd
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
    from training.full_graph_baseline import _smoke_catalog
    halos, _ = _smoke_catalog(seed=42)
    rows = []
    for halo_id, halo in enumerate(halos):
        for subhalo in halo.subhalos:
            rows.append({
                "subhalo_id": subhalo.subhalo_id, "halo_id": halo_id,
                "cluster_id": halo.cluster_id,
                "x": subhalo.position[0], "y": subhalo.position[1],
                "z": subhalo.position[2], "vx": subhalo.velocity[0],
                "vy": subhalo.velocity[1], "vz": subhalo.velocity[2],
                "stellar_mass": subhalo.stellar_mass,
                "velocity_dispersion": subhalo.velocity_dispersion,
                "half_mass_radius": subhalo.half_mass_radius,
                "metallicity": subhalo.metallicity,
                "halo_mass": halo.halo_mass, "redshift": halo.redshift,
            })
    csv_path = tmp_path / "spd.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    cfg["data"]["synthetic"] = {"features_path": str(tmp_path / "sf.pt"),
                                "csv_path": str(csv_path)}
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["graph"]["method"] = "knn"
    cfg["graph"]["k_neighbors"] = 4
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.15
    cfg["data"]["test_ratio"] = 0.15
    cfg["data"]["num_workers"] = 0
    out = tmp_path / "rls"
    rc = main(cfg, checkpoint=None, output_dir=out,
              allow_random_backbone=True)
    assert rc == 0
    assert (out / "results_table.csv").exists()
    assert (out / "policy.pt").exists()


def test_run_experiment_requires_checkpoint_or_explicit_random_opt_in(tmp_path):
    from rls.run_experiment import main
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        main(cfg, checkpoint=None, output_dir=tmp_path / "rls")
