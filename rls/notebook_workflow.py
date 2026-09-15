"""Validated artifact handoffs and single-halo adapters for Notebooks B–D.

Training, masking and adaptation remain in their respective rls modules.
"""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from rls.evaluate import build_results_table
from rls.policy import build_policy
from rls.provenance import record_backbone, require_backbone_label
from rls.sparsify import eval_mask
from rls.train_policy import prepare_graphs
from rls.tta import adapt_at_test_time, tta_should_enable


def file_info(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def run_context(cfg, csv, checkpoint, graphs, backbone, smoke):
    return {
        "dataset": {"tng_csv": file_info(csv), "checkpoint": file_info(checkpoint)},
        "seed": int(cfg.get("seed", 42)), "config_used": copy.deepcopy(cfg),
        "split_cluster_ids": {name: [g.cluster_id for g in items]
                              for name, items in graphs.items()},
        "backbone": backbone, "run_mode": "smoke" if smoke else "full",
    }


def checked_artifact(folder, manifest, filename):
    path = Path(folder) / filename
    expected = manifest.get("artifacts", {}).get(filename, {}).get("sha256")
    if not expected or not path.is_file() or file_info(path)["sha256"] != expected:
        raise ValueError(f"Missing or changed artifact {path}. Rerun its source notebook and attach the complete output folder.")
    return path


def source_artifacts(explicit, root, stage, filename, context):
    """Require a completed, matching run; never silently select repo leftovers."""
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = []
        for path in sorted(Path(root).rglob("provenance.json")):
            manifest = json.loads(path.read_text()).get(stage, {})
            if manifest.get("status") == "complete" and (path.parent / filename).is_file():
                candidates.append(path.parent)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Set {stage}_ARTIFACT_DIR to the completed Notebook {stage} output folder. Matching folders: {candidates}")
    folder = candidates[0]
    path = folder / "provenance.json"
    manifest = json.loads(path.read_text()).get(stage, {}) if path.is_file() else {}
    if manifest.get("status") != "complete":
        raise ValueError(f"Notebook {stage} outputs are incomplete or from the old workflow; rerun Notebook {stage}.")
    for key in ("run_mode", "seed", "split_cluster_ids", "backbone"):
        if manifest.get(key) != context[key]:
            raise ValueError(f"Notebook {stage} {key} does not match this run. Use the same inputs and settings in A–D.")
    for key in ("tng_csv", "checkpoint"):
        if manifest.get("dataset", {}).get(key, {}).get("sha256") != context["dataset"][key]["sha256"]:
            raise ValueError(f"Notebook {stage} {key} checksum differs from this run.")
    if manifest.get("config_used", {}).get("graph") != context["config_used"]["graph"]:
        raise ValueError(f"Notebook {stage} graph configuration differs from this run.")
    checked_artifact(folder, manifest, filename)
    return folder, manifest


def load_saved_policy(folder, manifest, cfg, gnn, device):
    # Architecture AND inference decoder must match training.
    cfg["rls"] = copy.deepcopy(manifest["config_used"]["rls"])
    policy = build_policy(cfg, node_emb_dim=gnn.output_dim).to(device)
    path = checked_artifact(folder, manifest, "policy.pt")
    policy.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return policy.eval()


def prepared_splits(split_graphs, gnn, device):
    result = {}
    for name, graphs in split_graphs.items():
        for graph in graphs:
            for key in ("stellar_mass", "vel_disp", "half_mass_r", "pos"):
                value = getattr(graph, key, None)
                if value is None or not torch.isfinite(value).all():
                    raise ValueError(f"Missing or nonfinite physics feature {key}: {graph.cluster_id}")
        result[name] = prepare_graphs(DataLoader(graphs, batch_size=16, shuffle=False), gnn, device)
        for prepared, graph in zip(result[name], graphs):
            prepared["cluster_id"] = graph.cluster_id
    return result


def masked_batch(graph, mask, device):
    edges = graph["edge_index"].shape[1]
    if mask.dtype != torch.bool or mask.shape != (edges,):
        raise ValueError(f"Expected boolean mask of shape ({edges},), got {mask.shape}/{mask.dtype}")
    mask = mask.to(graph["edge_index"].device)
    return Batch.from_data_list([Data(x=graph["x"],
        edge_index=graph["edge_index"][:, mask], edge_attr=graph["edge_attr"][mask])]).to(device)


def predict_graph(model, graph, mask):
    model.eval()
    with torch.no_grad():
        pred, _ = model(masked_batch(graph, mask, next(model.parameters()).device))
    pred = pred.reshape(-1)
    if pred.numel() != 1 or not torch.isfinite(pred).all():
        raise ValueError(f"Expected one finite halo prediction: {graph.get('cluster_id', '?')}")
    return pred


def prediction_pair(model):
    def predict(graph, mask):
        return predict_graph(model, graph, torch.ones_like(mask)), predict_graph(model, graph, mask)
    return predict


def policy_masks(policy, graphs, cfg):
    policy.eval()
    masks, probabilities = [], []
    with torch.no_grad():
        for graph in graphs:
            probs = torch.sigmoid(policy(graph["edge_attr"], graph["emb"],
                                         graph["edge_index"], graph["ctx"])).squeeze(-1)
            if not torch.isfinite(probs).all():
                raise ValueError("Policy produced nonfinite probabilities.")
            masks.append(eval_mask(graph["edge_index"], probs, cfg))
            probabilities.append(probs)
    return masks, probabilities


def evaluate_masks(model, graphs, masks):
    if len(graphs) < 2 or len(graphs) != len(masks):
        raise ValueError("Evaluation requires at least two graphs and one mask per graph.")
    full, pruned, targets = [], [], []
    for graph, mask in zip(graphs, masks):
        full.append(predict_graph(model, graph, torch.ones_like(mask)).cpu())
        pruned.append(predict_graph(model, graph, mask).cpu())
        targets.append(graph["y"].reshape(1).cpu())
    return dict(full=torch.cat(full), pruned=torch.cat(pruned), targets=torch.cat(targets),
                masks=[m.cpu() for m in masks])


def result_rows(result, backbone, run_mode):
    rows = build_results_table(result["full"], result["pruned"], None, None,
                              result["targets"], result["masks"], None, None)
    require_backbone_label(rows, backbone)
    for row in rows:
        row.update(run_mode=run_mode, keep_frac=row["mean_keep_frac"])
    return rows


def merge_baselines(rows, folder, manifest):
    import pandas as pd
    frame = pd.read_csv(checked_artifact(folder, manifest, "baselines.csv"))
    required = {"method", "rmse", "r2", "keep_frac", "backbone_stage", "backbone_sha256", "run_mode"}
    if not required.issubset(frame.columns):
        raise ValueError("A's table lacks model provenance; rerun the updated Notebook A.")
    full = frame[frame.method == "full"]
    if len(full) != 1 or not np.isclose(full.iloc[0].rmse, rows[0]["rmse"], rtol=1e-4, atol=1e-6):
        raise ValueError("A and C full-graph results differ; check the split, checkpoint and graph backend.")
    baseline_rows = frame[frame.method != "full"].copy()
    baseline_rows["mean_keep_frac"] = baseline_rows["keep_frac"]
    # Preserve every source field, particularly the jointly trained Gumbel backbone.
    return rows + baseline_rows.to_dict(orient="records")


def physics_diagnostics(graphs, probabilities, masks):
    from scipy.stats import mannwhitneyu, spearmanr
    rows = []
    for graph, probs, mask in zip(graphs, probabilities, masks):
        u, v = graph["edge_index"]
        real = u != v
        binding = (graph["stellar_mass"][u] * graph["stellar_mass"][v] /
                   (graph["pos"][u] - graph["pos"][v]).norm(dim=1).clamp(min=1e-6))[real].cpu().numpy()
        p, kept = probs[real].detach().cpu().numpy(), mask[real].cpu().numpy()
        rho = float(spearmanr(p, binding).statistic) if len(p) > 1 and np.ptp(p) > 0 and np.ptp(binding) > 0 else None
        pvalue = float(mannwhitneyu(binding[kept], binding[~kept]).pvalue) if kept.any() and (~kept).any() else None
        rows.append({"cluster_id": graph.get("cluster_id"), "spearman_rho": rho,
                     "binding_mannwhitney_p": pvalue, "physical_edges": len(binding),
                     "note": "undefined statistics left empty" if rho is None or pvalue is None else ""})
    return rows


def coverage_rows(model, graphs, masks, n_samples):
    if n_samples < 2:
        raise ValueError("MC-dropout coverage requires at least two samples.")
    rows = []
    for mode in ("full", "rl_policy"):
        covered, widths = [], []
        for graph, mask in zip(graphs, masks):
            use = torch.ones_like(mask) if mode == "full" else mask
            with torch.no_grad():
                dist = model.predict_with_uncertainty(masked_batch(graph, use, next(model.parameters()).device), n_samples=n_samples)
            mean, std = dist["mean"].reshape(-1), dist["std"].reshape(-1)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError("Nonfinite MC-dropout estimate.")
            covered.append(float(((graph["y"] >= mean - 1.96 * std) & (graph["y"] <= mean + 1.96 * std)).float().mean()))
            widths.append(float((3.92 * std).mean()))
        rows.append(dict(method=mode, coverage_95=float(np.mean(covered)), mean_interval_width=float(np.mean(widths)), mc_samples=n_samples))
    model.eval()
    return rows


def adaptation_trial(policy, graphs, gnn, cfg, device, k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("TTA K must be a nonnegative integer.")
    masks, histories = [], []
    for index, graph in enumerate(graphs):
        # Comparable reproducible random streams across K; never pass labels.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(int(cfg.get("seed", 42)) + index)
            if k == 0:
                mask = policy_masks(policy, [graph], cfg)[0][0]
                info = {"steps_run": 0, "reward_hist": [], "step_diag": []}
            else:
                trial_cfg = dict(cfg, tta_steps=k)
                unlabeled = {key: value for key, value in graph.items() if key not in ("y", "cluster_id")}
                mask, info = adapt_at_test_time(policy, unlabeled, gnn, trial_cfg, device,
                    target_sparsity=cfg.get("tta_target_sparsity", cfg["target_sparsity_end"]))
        masks.append(mask)
        histories.append(dict(cluster_id=graph.get("cluster_id"), **info))
    result = evaluate_masks(gnn, graphs, masks)
    row = result_rows(result, record_backbone("frozen", gnn), "full")[1]
    row.update(K=k, mode="frozen" if k == 0 else "tta", mean_steps=float(np.mean([h["steps_run"] for h in histories])))
    return row, histories


def validation_gates(policy, graphs, gnn, cfg, device, ks):
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in ks):
        raise ValueError("TTA_KS must contain nonnegative integers.")
    rows = []
    for k in sorted(set([0] + list(ks))):
        row, _ = adaptation_trial(policy, graphs, gnn, cfg, device, k)
        row["enabled"] = k == 0 or tta_should_enable(rows[0]["rmse"], row["rmse"], cfg.get("tta_val_tol", 0.02))
        rows.append(row)
    selected = min((row for row in rows if row["enabled"]), key=lambda row: (row["rmse"], row["K"]))["K"]
    return rows, selected


def complete_stage(out, stage, context, artifacts, **extra):
    out = Path(out)
    path = out / "provenance.json"
    provenance = json.loads(path.read_text()) if path.exists() else {}
    provenance[stage] = dict(context, stage=stage, status="complete",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        artifacts={name: file_info(out / name) for name in artifacts}, **extra)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2))
    temporary.replace(path)
    # Archive only this run's declared outputs; stale optional artifacts stay out.
    import zipfile
    archive = out.parent / f"notebook_{stage}_{context['run_mode']}_outputs.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["provenance.json"] + list(artifacts):
            bundle.write(out / name, name)
    print(f"Notebook {stage} completed: {out}\nDownload: {archive}")
    return provenance[stage]


def local_camels_graphs(path, cfg, builder, validate, cache_dir, limit=None):
    """Read an explicitly supplied catalog without network or synthetic fallback."""
    import h5py
    from data.loaders.camels_loader import CAMELSLoader
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Set CAMELS_HDF5 to an attached real CAMELS group catalog before enabling RUN_CAMELS.")
    path = Path(path).resolve()
    with h5py.File(path, "r") as stream:
        required = ["Header", "Subhalo/SubhaloMass", "Subhalo/SubhaloPos", "Subhalo/SubhaloVel",
                    "Subhalo/SubhaloVelDisp", "Subhalo/SubhaloHalfmassRad", "Subhalo/SubhaloGrNr", "Group/GroupMass"]
        if any(key not in stream for key in required):
            raise ValueError("CAMELS catalog is incomplete or synthetic; supply the original group catalog with Header.")
        if not any(key in stream for key in ("Subhalo/SubhaloStellarMass", "Subhalo/SubhaloMassType")):
            raise ValueError("CAMELS catalog lacks stellar masses.")
        h_param = float(stream["Header"].attrs.get("HubbleParam", CAMELSLoader.H_PARAM))
        redshift = float(stream["Header"].attrs.get("Redshift", 0.0))
        if not np.isfinite(h_param) or h_param <= 0 or abs(redshift) > 1e-4:
            raise ValueError("Supply a z=0 CAMELS catalog with a valid Hubble parameter.")
        for key in required[1:]:
            if not np.isfinite(stream[key][:]).all():
                raise ValueError(f"Nonfinite CAMELS catalog field: {key}")

    class AttachedCatalog(CAMELSLoader):
        def _download_hdf5(self):
            return path

    local_cfg = copy.deepcopy(cfg)
    local_cfg["data"]["source"] = "camels"
    local_cfg["data"].setdefault("camels", {})["cache_dir"] = str(cache_dir)
    loader = AttachedCatalog(local_cfg)
    loader.H_PARAM = h_param
    halos = loader.load()
    if limit is not None:
        halos = halos[:limit]
    graphs = [builder.build_graph(halo) for halo in halos]
    for graph in graphs:
        validate(graph)
    if len(graphs) < 2:
        raise ValueError("CAMELS evaluation requires at least two usable halos.")
    return graphs
