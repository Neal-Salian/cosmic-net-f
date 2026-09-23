import sys, os, yaml, json, math
import pytest
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _smoke_cfg(tmp_path):
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
    return cfg


def test_run_experiment_smoke(tmp_path):
    """Tiny 12-halo one-epoch StageA/B smoke with real matched evaluation."""
    from rls.run_experiment import main
    cfg = _smoke_cfg(tmp_path)
    out = tmp_path / "rls"
    rc = main(cfg, checkpoint=None, output_dir=out,
              allow_random_backbone=True)
    assert rc == 0
    assert (out / "results_table.csv").exists()
    assert (out / "policy.pt").exists()
    assert (out / "matched_summary.json").exists()
    assert (out / "matched_rows.json").exists()
    assert (out / "provenance.json").exists()
    summary = json.loads((out / "matched_summary.json").read_text())
    rows = json.loads((out / "matched_rows.json").read_text())
    prov = json.loads((out / "provenance.json").read_text())
    # Research label is explicit for random backbone, never faked real.
    assert prov["research_label"] == "synthetic_random_backbone_smoke"
    # Frozen backbone identity preserved; StageB used a deepcopy.
    assert prov["backbone_frozen"]["stage"] == "frozen"
    # Recompute RMSE/R2/fidelity/retention from saved rows; compare to summary.
    import numpy as np
    frozen_rows = [r for r in rows if r.get("backbone_stage", "frozen") == "frozen"]
    assert frozen_rows, "primary frozen rows missing"
    for s in [x for x in summary if x.get("backbone_stage", "frozen") == "frozen"]:
        own = [r for r in frozen_rows if r["method"] == s["method"] and r["status"] == "success"]
        if not own:
            continue
        preds = [r["pruned_prediction"] if r.get("pruned_prediction") is not None
                 else r["full_prediction"] for r in own]
        tgts = [r["target"] for r in own]
        rmse = float(np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(preds, tgts)])))
        assert abs(rmse - s["rmse"]) < 1e-6
        # retention from actual counts
        retains = [r["final_pair_count"] / r["available_pair_count"] for r in own]
        table = pd.read_csv(out / "results_table.csv")
        row = table[(table.method == s["method"]) & (table.backbone_stage == "frozen")]
        assert len(row) == 1
        assert abs(float(row.iloc[0]["mean_keep_frac"]) - float(np.mean(retains))) < 1e-9
    # Same requested pair count across scorers per graph (matched budget).
    from collections import defaultdict
    by_graph = defaultdict(list)
    for r in frozen_rows:
        if r["status"] == "success" and r["method"] != "full":
            by_graph[r["graph_id"]].append(r["requested_pair_count"])
    assert by_graph
    for gid, ks in by_graph.items():
        assert len(set(ks)) == 1, f"unmatched requested k for {gid}: {ks}"
    # RL vs random exact final counts graph-by-graph (both exact-budget paths).
    rl = {r["graph_id"]: r for r in frozen_rows if r["method"] == "provided_rl" and r["status"] == "success"}
    rnd = {r["graph_id"]: r for r in frozen_rows if r["method"] == "random" and r["status"] == "success"}
    common = set(rl) & set(rnd)
    assert common
    for gid in common:
        assert rl[gid]["final_pair_count"] == rl[gid]["requested_pair_count"]
        assert rnd[gid]["final_pair_count"] == rnd[gid]["requested_pair_count"]
    # Acceptance branch: artifact/label/hash agreement.
    if prov.get("stageb_accepted"):
        assert (out / "finetuned_gnn.pt").exists()
        assert prov["backbone_finetuned"]["stage"] == "stageB_finetuned"
        from data.provenance import sha256_file, model_state_hash
        import torch
        saved = torch.load(out / "finetuned_gnn.pt", map_location="cpu", weights_only=True)
        assert model_state_hash(saved) == prov["backbone_finetuned"]["sha256"]
        assert sha256_file(out / "finetuned_gnn.pt")["sha256"] == prov["finetuned_artifact"]["sha256"]
    else:
        assert not (out / "finetuned_gnn.pt").exists()
        assert "rejection" in prov


def test_run_experiment_rejection_saves_no_accepted_artifact(tmp_path, monkeypatch):
    """Controlled rejection branch: exercises real save code with tiny mocked outcome."""
    from rls.run_experiment import main
    import rls.stageb as stageb
    real = stageb.fine_tune_gnn

    def rejected(*args, **kwargs):
        hist, info = real(*args, **kwargs)
        info = dict(info)
        info["accepted"] = False
        info["best_epoch"] = -1
        return hist, info

    monkeypatch.setattr(stageb, "fine_tune_gnn", rejected)
    cfg = _smoke_cfg(tmp_path)
    out = tmp_path / "rls"
    rc = main(cfg, checkpoint=None, output_dir=out, allow_random_backbone=True)
    assert rc == 0
    assert not (out / "finetuned_gnn.pt").exists()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["stageb_accepted"] is False
    assert "rejection" in prov
    rows = json.loads((out / "matched_rows.json").read_text())
    assert any(r.get("backbone_stage", "frozen") == "frozen" for r in rows)
    assert not any(r.get("backbone_stage") == "stageB_finetuned" for r in rows)


def test_run_experiment_requires_checkpoint_or_explicit_random_opt_in(tmp_path):
    from rls.run_experiment import main
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        main(cfg, checkpoint=None, output_dir=tmp_path / "rls")


def test_lfs_checkpoint_fails_before_output_creation(tmp_path):
    from rls.run_experiment import main
    cfg = {"training": {"checkpoint_dir": str(tmp_path)},
           "data": {"source": "synthetic", "synthetic": {
               "features_path": str(tmp_path / "features.pt"),
               "csv_path": str(tmp_path / "catalog.csv")}}, "rls": {}}
    pointer = tmp_path / "model.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n" + "x" * 80)
    out = tmp_path / "new-output"
    with pytest.raises(ValueError, match="Git LFS pointer"):
        main(cfg, checkpoint=pointer, output_dir=out)
    assert not out.exists()


def test_declared_strict_group_is_copied_to_raw_graph_lineage():
    from rls.run_experiment import copy_declared_split_lineage
    from torch_geometric.data import Data
    graph = Data(cluster_id="cid-1", metadata={"volume_or_ic_group": "volume-7"})
    copy_declared_split_lineage([graph], "volume_or_ic_group")
    assert graph.lineage_group == "volume-7"
    from rls.matched_benchmark import _lineage_group
    assert _lineage_group(graph) == "volume-7"


def test_declared_strict_group_missing_fails_closed():
    from rls.run_experiment import copy_declared_split_lineage
    from torch_geometric.data import Data
    graph = Data(cluster_id="cid-1", metadata={})
    with pytest.raises(ValueError, match="volume_or_ic_group"):
        copy_declared_split_lineage([graph], "volume_or_ic_group")
