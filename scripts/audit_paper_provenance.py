"""Read-only legacy provenance measurements; writes only a NEW output directory.

Run as `python3 -m scripts.audit_paper_provenance --output-dir NEW_DIRECTORY`.
Optional --campaign-root points at the preserved A–D campaign download. This
does not retrain, correct the dataset, or certify historical training separation.
"""
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import types

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from data.loaders.tng_loader import TNGLoader
from data.provenance import hash_payload, sha256_file, validate_checkpoint_file
from data.scientific_validation import match_reference_group, periodic_pair_diagnostics
from graph.graph_builder import GraphBuilder
from rls.provenance import backbone_checksum


BASE = "c825bf57652005c80faff79cf4dfd58e3a9baa14"


@lru_cache(maxsize=1)
def _base_source_blobs():
    """Read only the pinned source blobs, once per audit process."""
    # Avoid traversing artifact trees: the historical identity includes only
    # source directories, direct data modules, config and A–D code cells.
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE,
        "rls", "model", "graph", "training", "data/loaders"], text=True).splitlines()
    paths += [f"data/{name}" for name in subprocess.check_output(
        ["git", "ls-tree", "--name-only", f"{BASE}:data"], text=True).splitlines() if name.endswith(".py")]
    paths += [f"notebooks/{name}" for name in subprocess.check_output(
        ["git", "ls-tree", "--name-only", f"{BASE}:notebooks"], text=True).splitlines()]
    paths.append("config/config.yaml")
    selected = []
    for path in sorted(set(paths)):
        code = path.endswith(".py") and path.split("/")[0] in {"rls", "data", "model", "graph", "training"}
        notebook = Path(path).match("notebooks/notebook_[ABCD]*.ipynb")
        if code or notebook or path == "config/config.yaml":
            selected.append(path)
    batch = subprocess.check_output(["git", "cat-file", "--batch"],
        input="".join(f"{BASE}:{path}\n" for path in selected).encode())
    offset, blobs = 0, {}
    for path in selected:
        end = batch.index(b"\n", offset)
        header = batch[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"missing source blob at base: {path}")
        size = int(header[2])
        blobs[path] = batch[end + 1:end + 1 + size]
        if len(blobs[path]) != size or batch[end + 1 + size:end + 2 + size] != b"\n":
            raise ValueError(f"incomplete Git source blob: {path}")
        offset = end + size + 2
    return blobs


def base_source_identity():
    """Reproduce notebook_workflow.source_identity from immutable Git blobs.

    Use base code rather than audit-worktree contents, which intentionally have
    new checks. This covers non-rls sources absent from per-stage source_files.
    """
    records = {}
    for path, blob in _base_source_blobs().items():
        notebook = Path(path).match("notebooks/notebook_[ABCD]*.ipynb")
        if notebook:
            records[path] = hash_payload([c["source"] for c in json.loads(blob)["cells"] if c["cell_type"] == "code"])
        else:
            records[path] = hashlib.sha256(blob).hexdigest()
    return hash_payload(records)


def raw_evidence(directory, frame):
    """Only complete single-group chunks with explicit first-subhalo IDs match.

    Missing/empty chunks prevent summing offsets. Multi-group chunks require a
    future explicit global group offset; they remain unverified here.
    """
    records, matches = [], []
    for path in sorted(Path(directory).glob("groupcat_99_*.hdf5")):
        record = sha256_file(path)
        records.append(record)
        if not h5py.is_hdf5(path):
            record["status"] = "INVALID_HDF5"
            continue
        with h5py.File(path) as source:
            header = source["Header"].attrs
            record["header"] = {key: float(header[key]) for key in (
                "Ngroups_ThisFile", "Nsubgroups_ThisFile", "Ngroups_Total",
                "Nsubgroups_Total", "NumFiles", "HubbleParam", "Time", "BoxSize")}
            if not int(header["Ngroups_ThisFile"]) or not int(header["Nsubgroups_ThisFile"]):
                record["status"] = "EMPTY_CHUNK"
                continue
            if int(header["Ngroups_ThisFile"]) != 1:
                record["status"] = "UNVERIFIED_GLOBAL_GROUP_OFFSET"
                continue
            group, sub = source["Group"], source["Subhalo"]
            member_groups = sub["SubhaloGrNr"][:]
            unique_groups = np.unique(member_groups)
            if len(unique_groups) != 1 or int(group["GroupNsubs"][0]) != len(member_groups):
                record["status"] = "UNVERIFIED_PARTIAL_GROUP"
                continue
            h = float(header["HubbleParam"])
            fields = {
                "half_mass_radius": sub["SubhaloHalfmassRad"][:],
                "vel_dispersion": sub["SubhaloVelDisp"][:],
                "metallicity": np.maximum(sub["SubhaloGasMetallicity"][:], 1e-6),
                "stellar_mass": np.log10(np.maximum(
                    sub["SubhaloMassType"][:, 4].astype(float) * 1e10 / h, 1e-300)),
            }
            for axis, name in enumerate("xyz"):
                fields[f"pos_{name}"] = sub["SubhaloPos"][:, axis]
                fields[f"vel_{name}"] = sub["SubhaloVel"][:, axis]
            reference = {"group_id": int(unique_groups[0]),
                         "first_subhalo_id": int(group["GroupFirstSub"][0]),
                         "member_group_ids": member_groups, "fields": fields,
                         "target_msun": float(group["Group_M_Crit200"][0]) * 1e10 / h}
            match = match_reference_group(frame, reference)
            match.update(source_file=str(path), first_subhalo_id=reference["first_subhalo_id"],
                         m200c_msun=reference["target_msun"],
                         alternative_group_masses_msun={key: float(group[key][0]) * 1e10 / h
                             for key in ("GroupMass", "Group_M_Mean200", "Group_M_Crit500", "Group_M_TopHat200")})
            central = frame[frame.subhalo_id == reference["first_subhalo_id"]]
            match["central_radius_comparison_ckpc_per_h"] = {
                "total": float(sub["SubhaloHalfmassRad"][0]),
                "stellar_type4": float(sub["SubhaloHalfmassRadType"][0, 4]),
                "csv": float(central.half_mass_radius.iloc[0]) if len(central) else None}
            if not match["matched_rows"]:
                record["status"] = "RAW_GROUP_ABSENT_FROM_CSV"
            elif len(match["fully_matching_subhalo_ids"]) == match["matched_rows"]:
                record["status"] = "MATCHED_CSV_GROUP_SELECTION"
            else:
                record["status"] = "CSV_CONTENT_MISMATCH"
            matches.append(match)
    covered = sorted({member for match in matches for member in match["fully_matching_subhalo_ids"]})
    return {"files": records, "groups": matches, "verified_row_count": len(covered),
            "unverified_row_count": len(frame) - len(covered),
            "identified_raw_group_ids": sorted({m["group_id"] for m in matches}),
            "verified_group_ids": sorted({m["group_id"] for m in matches
                if m["matched_rows"] > 0 and len(m["fully_matching_subhalo_ids"]) == m["matched_rows"]}),
            "complete_catalog_available": False,
            "scope": "content match to local reference files; no external release checksum authentication"}


def campaign_evidence(root, csv_hash, checkpoint_hash, state_hash, splits, graphs):
    """Verify saved identities, source bytes at BASE, artifacts and C arithmetic."""
    from rls.matched_benchmark import _graph_content_hash
    stages, errors, preserved = {}, [], []
    expected_source_identity = base_source_identity()
    for name in "ABCD":
        folder = root / f"{name}-v1-terminal-001/artifacts/cosmic-net-runs/notebook_{name}"
        manifest_path = folder / "provenance.json"
        manifest_record = sha256_file(manifest_path)
        preserved.append(manifest_record)
        manifest = json.loads(manifest_path.read_text())[name]
        checks = {"complete": manifest.get("status") == "complete",
                  "source_commit": manifest.get("repo", {}).get("head") == BASE,
                  "csv_identity": manifest["dataset"]["tng_csv"]["sha256"] == csv_hash,
                  "checkpoint_identity": manifest["dataset"]["checkpoint"]["sha256"] == checkpoint_hash,
                  "backbone_identity": manifest["backbone"]["sha256"] == state_hash,
                  "aggregate_source_identity": manifest.get("source_identity") == expected_source_identity,
                  "split_membership": manifest["split_cluster_ids"] == splits}
        artifacts = {}
        for filename, expected in manifest.get("artifacts", {}).items():
            record = sha256_file(folder / filename)
            artifacts[filename] = record
            checks[f"artifact:{filename}"] = record["sha256"] == expected["sha256"]
            preserved.append(record)
        source_checks = {}
        for filename, record in manifest.get("source_files", {}).items():
            source_bytes = _base_source_blobs().get(filename)
            source_checks[filename] = source_bytes is not None and hashlib.sha256(source_bytes).hexdigest() == record["sha256"]
        checks["recorded_python_source_at_base"] = bool(source_checks) and all(source_checks.values())
        stages[name] = {"manifest": manifest_record, "checks": checks,
                        "artifact_hashes": artifacts, "source_checks": source_checks,
                        "source_identity": manifest["source_identity"],
                        "research_label": manifest["research_label"]}
        errors.extend(f"{name}:{key}" for key, passed in checks.items() if not passed)
    cfolder = root / "C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C"
    rows = json.loads((cfolder / "matched_per_example.json").read_text())
    selected = [r for r in rows if r["requested_keep_fraction"] == .4
                and r["method"] in {"full", "provided_rl"} and r["status"] == "success"]
    if not selected or {r["method"] for r in selected} != {"full", "provided_rl"}:
        raise ValueError("expected saved C full/rl q=0.4 rows")
    graph_by_id = {str(g.cluster_id): g for g in graphs}
    metrics = {}
    for method in ("full", "provided_rl"):
        items = [r for r in selected if r["method"] == method]
        ids = [r["graph_id"] for r in items]
        if len(ids) != len(set(ids)) or set(ids) != set(splits["test"]):
            errors.append(f"C:{method}:test_membership")
        for row in items:
            graph = graph_by_id[row["graph_id"]]
            if _graph_content_hash(graph, row["graph_id"]) != row["graph_content_hash"]:
                errors.append(f"C:graph_content:{row['graph_id']}")
            if float(graph.y.item()) != row["target"]:
                errors.append(f"C:target:{row['graph_id']}")
        residuals = np.array([(r["full_prediction"] if method == "full" else r["pruned_prediction"])
                              - r["target"] for r in items])
        metrics[method] = {"n": len(items), "rmse_dex": float(np.sqrt(np.mean(residuals**2))),
                           "mean_physical_retention": float(np.mean([r["physical_retention"] for r in items]))}
    # Rehash all consumed campaign artifacts at the end, without writing to them.
    changed = [r["path"] for r in preserved if sha256_file(r["path"])["sha256"] != r["sha256"]]
    return {"root": str(root), "stages": stages,
            "expected_base_source_identity": expected_source_identity,
            "errors": sorted(set(errors)),
            "q40_saved_prediction_metrics": metrics, "preserved_artifacts_changed": changed}


def audit(args):
    root = Path(__file__).resolve().parents[1]
    csv = Path(args.csv).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    csv_record = sha256_file(csv)
    checkpoint_record = validate_checkpoint_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = yaml.safe_load((root / "config/config.yaml").read_text())
    config["model"] = deepcopy(checkpoint["config"]["model"])
    config["graph"] = deepcopy(checkpoint["config"]["graph"])
    config["data"]["tng"].update(clustered_file=str(csv), cache_dir=str(args.output_dir / "cache"),
                                clustered_units=TNGLoader.LEGACY_CLUSTERED_UNITS)
    loader = TNGLoader(config)
    halos = loader.load()
    split_halos = dict(zip(("train", "val", "test"), loader.split_data(halos)))
    split_ids = {name: [str(h.cluster_id) for h in items] for name, items in split_halos.items()}
    members = {name: sorted(s.subhalo_id for h in items for s in h.subhalos)
               for name, items in split_halos.items()}
    builder = GraphBuilder(config)
    graphs, geometry = [], []
    for halo in halos:
        graph = builder.build_graph(halo)
        graphs.append(graph)
        u, v = graph.edge_index.numpy()
        real = u < v
        distances = np.linalg.norm(graph.pos.numpy()[v[real]] - graph.pos.numpy()[u[real]], axis=1)
        geometry.append({"cluster_id": str(halo.cluster_id), "nodes": len(graph.x),
                         "pairs": int(real.sum()), "loops": int((u == v).sum()),
                         "beyond_radius_pairs": int((distances > builder.radius_mpc + 1e-6).sum()),
                         "periodic": periodic_pair_diagnostics(graph.pos.numpy(),
                             box_size=75000. * 1e-3 / .6774, radius=builder.radius_mpc)})
    frame = pd.read_csv(csv)
    raw = raw_evidence(args.raw_dir, frame)
    state_hash = backbone_checksum(checkpoint["model_state_dict"])
    measured = {"schema_version": 1, "audited_base_commit": BASE,
                "audit_source_files": [sha256_file(root / name) for name in (
                    "scripts/audit_paper_provenance.py", "data/scientific_validation.py", "data/loaders/tng_loader.py")],
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
                "inputs": {"csv": csv_record, "checkpoint": checkpoint_record},
                "catalog": {"rows": len(frame), "groups": len(halos),
                    "duplicate_full_rows": int(frame.duplicated().sum()),
                    "duplicate_subhalo_ids": int(frame.subhalo_id.duplicated().sum()),
                    "node_count_distribution": {str(k): int(v) for k, v in frame.groupby("group_id").size().value_counts().items()},
                    "log_target_range_dex": [float(frame.halo_mass_log.min()), float(frame.halo_mass_log.max())],
                    "raw_stellar_log_below_historical_threshold_6": int((frame.stellar_mass < 6).sum()),
                    "raw_radius_at_or_below_historical_threshold_1": int((frame.half_mass_radius <= 1).sum()),
                    "internally_consistent_identity_targets": True},
                "raw_reference": raw,
                "split_reconstruction": {"scope": "legacy evaluation membership, not historical training proof",
                    "algorithm": "sorted cluster IDs; numpy.random.RandomState(42).permutation; 0.7/0.15/remainder",
                    "cluster_ids": split_ids, "subhalo_ids": members,
                    "cluster_ids_sha256": hash_payload(split_ids),
                    "cross_split_member_overlap": {f"{a}:{b}": sorted(set(members[a]) & set(members[b]))
                        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}},
                "graph": {"settings": config["graph"], "per_halo": geometry,
                    "beyond_radius_pairs": sum(g["beyond_radius_pairs"] for g in geometry),
                    "wrapped_pairs": sum(g["periodic"]["wrapped_pair_count"] for g in geometry),
                    "periodic_radius_disagreements": sum(g["periodic"]["radius_membership_disagreements"] for g in geometry)},
                "checkpoint": {"keys": sorted(checkpoint), "epoch_zero_based": checkpoint.get("epoch"),
                    "metrics": checkpoint.get("metrics"), "model_state_sha256": state_hash,
                    "config": checkpoint.get("config"), "independent_heldout_demonstrable": False,
                    "missing_training_evidence": ["training dataset hash", "training source revision",
                        "training split IDs", "augmentation lineage", "optimizer/RNG continuation state"]}}
    notebook_path = root / "kaggle/training-cosmic (4).ipynb"
    notebook = json.loads(notebook_path.read_text())
    training_outputs = "".join("".join(output.get("text", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
        for output in cell.get("outputs", []) if output.get("output_type") == "stream")
    measured["historical_training_notebook"] = {
        **sha256_file(notebook_path),
        "code_cells_sha256": hash_payload([c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]),
        "stream_outputs_sha256": hashlib.sha256(training_outputs.encode()).hexdigest(),
        "reported_epoch_730": "Best val MSE: 0.0138 at epoch 730" in training_outputs,
        "reported_82_test_graphs": "N test graphs : 82" in training_outputs,
        "scope": "saved notebook source/output evidence; no training membership record"}
    hdf5_path, partial_path = csv.with_suffix(".h5"), csv.with_name("checkpoint.csv")
    if hdf5_path.is_file():
        with h5py.File(hdf5_path) as source:
            measured["clustered_hdf5"] = {**sha256_file(hdf5_path), "csv_column_matches": {
                name: name in source and source[name].shape == frame[name].shape and bool(np.allclose(
                    source[name][:], frame[name].to_numpy(), rtol=1e-12, atol=1e-12)) for name in frame}}
    if partial_path.is_file():
        partial = pd.read_csv(partial_path)
        indexed_partial, indexed_final = partial.set_index("subhalo_id"), frame.set_index("subhalo_id")
        common = indexed_partial.index.intersection(indexed_final.index)
        measured["extraction_checkpoint"] = {**sha256_file(partial_path), "rows": len(partial),
            "groups": int(partial.group_id.nunique()),
            "subhalo_id_overlap_with_final": len(common),
            "overlap_field_difference_counts": {name: int((~np.isclose(
                indexed_partial.loc[common, name], indexed_final.loc[common, name],
                rtol=1e-12, atol=1e-12)).sum()) for name in indexed_partial.columns if name in indexed_final}}
    # Run the actual archived loader source, not a guessed reimplementation of
    # its interacting magnitude heuristics. Only the local CSV path is used.
    # Current BaseDataLoader/GraphBuilder remain in use: this is a deliberately
    # labelled transform sensitivity diagnostic, not historical run recreation.
    archived_revision = "0dfe85ac1df996f38c8ad793fd27fa7e19d0e687"
    archived_bytes = subprocess.check_output(["git", "show", f"{archived_revision}:data/loaders/tng_loader.py"])
    archived_module = types.ModuleType("archived_tng_loader_for_diagnostic")
    exec(compile(archived_bytes, f"git:{archived_revision}:data/loaders/tng_loader.py", "exec"), archived_module.__dict__)
    archived_halos = archived_module.TNGLoader(config).load()
    old_members = {s.subhalo_id: s for h in archived_halos for s in h.subhalos}
    current_members = {s.subhalo_id: s for h in halos for s in h.subhalos}
    changed = {name: sum(not np.allclose(getattr(old_members[i], name), getattr(current_members[i], name),
                                       rtol=1e-5, atol=1e-10) for i in current_members)
               for name in ("position", "stellar_mass", "half_mass_radius")}
    measured["archived_transform_diagnostic"] = {
        "loader_revision": archived_revision, "loader_sha256": hashlib.sha256(archived_bytes).hexdigest(),
        "changed_members": changed,
        "scope": "archived loader on identical CSV; current base loader/graph code; not proof of training source"}
    if args.campaign_root:
        measured["campaign"] = campaign_evidence(args.campaign_root, csv_record["sha256"],
            checkpoint_record["sha256"], state_hash, split_ids, graphs)
    if args.replay:
        from model.model import build_model
        from torch_geometric.data import Batch
        model = build_model(config).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        by_id = {str(g.cluster_id): g for g in graphs}
        predictions = []
        with torch.no_grad():
            for identity in split_ids["test"]:
                graph = by_id[identity]
                prediction, _ = model(Batch.from_data_list([graph]))
                predictions.append({"cluster_id": identity, "target": float(graph.y.item()),
                                    "prediction": float(prediction.reshape(()))})
        errors = np.array([p["prediction"] - p["target"] for p in predictions])
        measured["local_full_replay"] = {"rmse_dex": float(np.sqrt(np.mean(errors**2))),
            "n": len(predictions), "scope": "current CPU replay; no model retraining", "predictions": predictions}
        old_graphs = {h.cluster_id: builder.build_graph(h) for h in archived_halos}
        old_residuals, changed_graphs = [], []
        with torch.no_grad():
            for identity in split_ids["test"]:
                old = old_graphs[identity]
                prediction, _ = model(Batch.from_data_list([old]))
                old_residuals.append(float(prediction.reshape(())) - float(old.y.item()))
                if not torch.equal(old.edge_index, by_id[identity].edge_index):
                    changed_graphs.append(identity)
        measured["archived_transform_diagnostic"].update(
            frozen_model_test_rmse_dex=float(np.sqrt(np.mean(np.square(old_residuals)))),
            test_graphs_with_changed_edges=changed_graphs)
        val_residuals = []
        with torch.no_grad():
            for identity in split_ids["val"]:
                graph = old_graphs[identity]
                prediction, _ = model(Batch.from_data_list([graph]))
                val_residuals.append(float(prediction.reshape(())) - float(graph.y.item()))
        measured["archived_transform_diagnostic"]["frozen_model_val_rmse_dex"] = float(
            np.sqrt(np.mean(np.square(val_residuals))))
    for name, expected in ((csv, csv_record), (checkpoint_path, checkpoint_record)):
        if sha256_file(name)["sha256"] != expected["sha256"]:
            raise ValueError(f"input changed while auditing: {name}")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/tng100_clustered.csv")
    parser.add_argument("--checkpoint", default="kaggle/best_model_augmented.pt")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_hdf5"))
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    measured = audit(args)
    output = args.output_dir / "measurements.json"
    output.write_text(json.dumps(measured, indent=2, allow_nan=False) + "\n")
    print(output)
    if measured.get("campaign", {}).get("errors") or measured.get("campaign", {}).get("preserved_artifacts_changed"):
        raise SystemExit("campaign evidence contradictions: inspect measurements.json")


if __name__ == "__main__":
    main()
