"""Validated artifact handoffs and single-halo adapters for Notebooks B–D.

Training, masking and adaptation remain in their respective rls modules.
"""
import copy
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
from rls.pair_policy import policy_action, pair_stats, pair_mask
from data.provenance import sha256_file, hash_payload


def fresh_output_directory(path):
    """A new campaign directory is mandatory; never overwrite historical runs."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = []
    for directory in ("rls", "data", "model", "graph", "training"):
        paths.extend((root / directory).rglob("*.py"))
    records = {str(p.relative_to(root)): file_info(p)["sha256"] for p in sorted(paths)}
    records["config/config.yaml"] = file_info(root / "config/config.yaml")["sha256"]
    for path in sorted((root / "notebooks").glob("notebook_[ABCD]*.ipynb")):
        notebook = json.loads(path.read_text())
        records[str(path.relative_to(root))] = hash_payload(
            [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"])
    return hash_payload(records)


def file_info(path):
    return sha256_file(path)


def run_context(cfg, csv, checkpoint, graphs, backbone, smoke):
    return {
        "dataset": {"tng_csv": file_info(csv), "checkpoint": file_info(checkpoint)},
        "seed": int(cfg.get("seed", 42)), "config_used": copy.deepcopy(cfg),
        "split_cluster_ids": {name: [g.cluster_id for g in items]
                              for name, items in graphs.items()},
        "backbone": backbone, "run_mode": "smoke" if smoke else "full",
        "research_label": "fixture_smoke" if smoke else "legacy_total_radius",
        "source_identity": source_identity(),
        "evaluation_protocol": {"decoder": "direct", "constraint": cfg.get("rls", {}).get("evaluation_constraint", "no_isolates"),
            "clustered_units": cfg.get("data", {}).get("tng", {}).get("clustered_units"),
            "target": "legacy_uncertified"},
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
    for key in ("run_mode", "seed", "split_cluster_ids", "backbone", "source_identity", "evaluation_protocol"):
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
    quality = manifest.get("policy_validation", {})
    if (cfg["rls"].get("sparsity_mode") != "pair_pl" or quality.get("verdict") not in ("PASS", "FAIL")
            or not quality.get("budget_comparable")):
        raise ValueError("Policy lacks a budget-comparable validation record. Rerun Notebook B.")
    # FAIL means no improvement over the validation control, not an invalid
    # artifact. Retain negative results in C instead of selecting winners only.
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
        full = graph.get("reference_prediction")
        if full is None:
            full = predict_graph(model, graph, torch.ones_like(mask))
        return full, predict_graph(model, graph, mask)
    return predict


def policy_masks(policy, graphs, cfg, target_sparsity=None, return_diagnostics=False):
    from rls.budget_accounting import legacy_repair_report, resolve_requested_keep
    q = resolve_requested_keep(cfg, explicit=target_sparsity)
    policy.eval()
    masks, probabilities, diagnostics = [], [], []
    with torch.no_grad():
        for graph in graphs:
            if cfg.get("evaluation_decoder") == "direct":
                from rls.notebook_benchmark import exact_policy_mask
                mask, probs, diag = exact_policy_mask(policy, graph, cfg, q)
                masks.append(mask)
                probabilities.append(probs)
                diagnostics.append(diag)
                continue
            probs = torch.sigmoid(policy(graph["edge_attr"], graph["emb"],
                                         graph["edge_index"], graph["ctx"])).squeeze(-1)
            if not torch.isfinite(probs).all():
                raise ValueError("Policy produced nonfinite probabilities.")
            if cfg.get("sparsity_mode") == "pair_pl":
                mask, _, _, order = policy_action(policy, graph, cfg, keep=q)
            else:
                order = None
                mask = eval_mask(graph["edge_index"], probs, cfg, target_sparsity=q)
            masks.append(mask)
            probabilities.append(probs)
            diagnostics.append(legacy_repair_report(
                graph["edge_index"], mask, keep_fraction=q, order=order,
                num_nodes=len(graph["x"])))
    if return_diagnostics:
        return masks, probabilities, diagnostics
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
        for key, value in list(row.items()):
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    return rows


def policy_metrics(policy, graphs, gnn, cfg, target_sparsity=None):
    from rls.budget_accounting import resolve_requested_keep
    q = resolve_requested_keep(cfg, explicit=target_sparsity)
    masks, _, diagnostics = policy_masks(policy, graphs, cfg, target_sparsity=q,
                                         return_diagnostics=True)
    result = evaluate_masks(gnn, graphs, masks)
    row = result_rows(result, record_backbone("frozen", gnn), "full")[1]
    stats = [pair_stats(g["edge_index"], m.to(g["edge_index"].device), len(g["x"]),
                        requested_count=d["requested_pair_count"],
                        sampled_count=d["sampled_pair_count"])
             for g, m, d in zip(graphs, masks, diagnostics)]
    row.update(physical_pair_keep=float(np.mean([s["physical_pair_keep"] for s in stats])),
               total_edge_keep=row["mean_keep_frac"], physical_isolates=sum(s["physical_isolates"] for s in stats),
               requested_keep_fraction=q, decoder_kind=diagnostics[0]["decoder_kind"] if diagnostics else "legacy_repaired",
               final_pair_counts=[d["final_pair_count"] for d in diagnostics],
               requested_pair_counts=[d["requested_pair_count"] for d in diagnostics],
               graph_ids=[g.get("cluster_id") for g in graphs],
               budget_tag="exact_direct" if cfg.get("evaluation_decoder") == "direct" else "stageA_legacy_repaired")
    return row


def random_pair_reference(graphs, gnn, cfg, seeds=range(10), target_sparsity=None):
    from rls.budget_accounting import resolve_requested_keep
    q = resolve_requested_keep(cfg, explicit=target_sparsity)
    rows = []
    for seed in seeds:
        masks = []
        devices = [next(gnn.parameters()).device.index or 0] if next(gnn.parameters()).device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for graph in graphs:
                scores = torch.zeros(graph["edge_index"].shape[1], device=graph["edge_index"].device)
                if cfg.get("evaluation_decoder") == "direct":
                    from rls.constraints import PhysicalPairLayout, pair_budget
                    from rls.constrained_policy import select_pairs
                    from rls.notebook_benchmark import fixed_marks
                    layout = PhysicalPairLayout.from_edge_index(graph["edge_index"], len(graph["x"]))
                    k = pair_budget(layout, q)
                    selected = select_pairs(layout, torch.rand(len(layout.pairs), device=scores.device), k,
                        cfg.get("evaluation_constraint", "no_isolates"), method="direct", sample=False,
                        pair_marks=fixed_marks(layout, cfg.get("seed", 42), graph.get("cluster_id", "unknown")))
                    masks.append(layout.expand(selected.selected))
                else:
                    masks.append(pair_mask(graph["edge_index"], scores, q, sample=True)[0])
        row = result_rows(evaluate_masks(gnn, graphs, masks), record_backbone("frozen", gnn), "full")[1]
        row.update(seed=seed, requested_keep_fraction=q,
                   decoder_kind="exact_direct" if cfg.get("evaluation_decoder") == "direct" else "legacy_repaired",
                   budget_tag="stageA_legacy_repaired")
        if cfg.get("evaluation_decoder") == "direct":
            from rls.constraints import PhysicalPairLayout
            row.update(final_pair_counts=[int(PhysicalPairLayout.from_edge_index(g["edge_index"], len(g["x"])).collapse(m).sum()) for g,m in zip(graphs,masks)],
                       graph_ids=[g.get("cluster_id") for g in graphs])
        rows.append(row)
    if len(rows) < 2:
        raise ValueError("Random reference requires at least two seeds.")
    return rows


def policy_validation(metrics, random_rows):
    random_mean = float(np.mean([r["rmse"] for r in random_rows]))
    mq = metrics.get("requested_keep_fraction")
    rqs = [r.get("requested_keep_fraction") for r in random_rows]
    budgets_known = mq is not None and all(q is not None for q in rqs)
    comparable = budgets_known and all(q == mq for q in rqs)
    if metrics.get("decoder_kind") == "exact_direct":
        comparable = comparable and all(r.get("decoder_kind") == "exact_direct"
            and r.get("final_pair_counts") == metrics.get("final_pair_counts")
            and r.get("graph_ids") == metrics.get("graph_ids") for r in random_rows)
    passed = (comparable and np.isfinite(metrics["rmse"])
              and metrics["rmse"] < random_mean and metrics["physical_isolates"] == 0)
    criterion = ("Validation RMSE below random-pair mean at the same budget and graph population; "
                 "no physical isolates.")
    if not budgets_known:
        criterion += " Requested-budget metadata missing: comparability unknown."
    elif not comparable:
        criterion += " Requested budgets differ: tagged incomparable."
    return dict(verdict="PASS" if passed else "FAIL", selected_metrics=metrics,
                random_rmse_mean=random_mean, random_rmse_std=float(np.std([r["rmse"] for r in random_rows], ddof=1)),
                criterion=criterion, budget_comparable=comparable)


def merge_baselines(rows, folder, manifest):
    raise ValueError("Legacy table concatenation does not establish matched budgets or populations. "
                     "Use matched_notebook_rows to evaluate every method together.")


def physics_diagnostics(graphs, probabilities, masks):
    from scipy.stats import mannwhitneyu, spearmanr
    rows = []
    for graph, probs, mask in zip(graphs, probabilities, masks):
        from rls.constraints import PhysicalPairLayout
        layout = PhysicalPairLayout.from_edge_index(graph["edge_index"], len(graph["stellar_mass"]))
        chosen = layout.collapse(mask)
        if not torch.equal(layout.expand(chosen)[layout.real], mask[layout.real]):
            raise ValueError("Physics statistics require symmetric physical-pair masks")
        u, v = layout.pairs.T
        binding = (graph["stellar_mass"][u].double() * graph["stellar_mass"][v].double() /
                   (graph["pos"][u] - graph["pos"][v]).double().norm(dim=1).clamp(min=1e-6)).cpu().numpy()
        sums = probs.new_zeros(len(layout.pairs)).index_add(0, layout.inverse, probs[layout.real])
        counts = probs.new_zeros(len(layout.pairs)).index_add(0, layout.inverse, torch.ones_like(probs[layout.real]))
        p, kept = (sums / counts.clamp_min(1)).detach().cpu().numpy(), chosen.cpu().numpy()
        rho = float(spearmanr(p, binding).statistic) if len(p) > 1 and np.ptp(p) > 0 and np.ptp(binding) > 0 else None
        pvalue = float(mannwhitneyu(binding[kept], binding[~kept]).pvalue) if kept.any() and (~kept).any() else None
        rows.append({"cluster_id": graph.get("cluster_id"), "spearman_rho": rho,
                     "binding_mannwhitney_p": pvalue, "physical_edges": len(binding),
                     "note": "descriptive within-graph pair statistic; no independent graph-level significance claim"})
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


def adaptation_trial(policy, graphs, gnn, cfg, device, k, target_sparsity=None):
    import time
    from rls.budget_accounting import LEGACY_DECODER_KIND, resolve_requested_keep
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("TTA K must be a nonnegative integer.")
    q = resolve_requested_keep(cfg, explicit=target_sparsity)
    dev = torch.device(device) if isinstance(device, str) else device
    masks, histories, elapsed = [], [], []
    for index, graph in enumerate(graphs):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        start = time.perf_counter()
        graph = dict(graph)
        # Include a fresh full-backbone embedding pass in per-instance timing.
        if hasattr(gnn, "get_embeddings"):
            with torch.no_grad():
                graph["emb"] = gnn.get_embeddings(masked_batch(graph,
                    torch.ones(graph["edge_index"].shape[1], dtype=torch.bool,
                               device=graph["edge_index"].device), dev), embedding_point="pre_pooling")
                graph["ctx"] = graph["emb"].mean(dim=0)
        # Comparable reproducible random streams across K; never pass labels.
        with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
            torch.manual_seed(int(cfg.get("seed", 42)) + index)
            if k == 0:
                m, _, diags = policy_masks(policy, [graph], cfg, target_sparsity=q,
                                           return_diagnostics=True)
                mask = m[0]
                info = {"steps_run": 0, "reward_hist": [], "step_diag": [],
                        **diags[0]}
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                trial_cfg = dict(cfg, tta_steps=k)
                unlabeled = {key: value for key, value in graph.items() if key != "y"}
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mask, info = adapt_at_test_time(policy, unlabeled, gnn, trial_cfg, device,
                    target_sparsity=q)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        masks.append(mask)
        # Include the retained-graph prediction; aggregate accuracy is evaluated below.
        predict_graph(gnn, graph, mask)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed.append(time.perf_counter() - start)
        histories.append(dict(cluster_id=graph.get("cluster_id"), inference_time_s=elapsed[-1], **info))
    result = evaluate_masks(gnn, graphs, masks)
    row = result_rows(result, record_backbone("frozen", gnn), "full")[1]
    req = [h["requested_pair_count"] for h in histories if h.get("requested_pair_count") is not None]
    samp = [h["sampled_pair_count"] for h in histories if h.get("sampled_pair_count") is not None]
    final = [h["final_pair_count"] for h in histories if h.get("final_pair_count") is not None]
    row.update(K=k, mode="frozen" if k == 0 else "tta",
               mean_steps=float(np.mean([h["steps_run"] for h in histories])),
               requested_keep_fraction=q, decoder_kind="exact_direct" if cfg.get("evaluation_decoder") == "direct" else LEGACY_DECODER_KIND,
               mean_time_s=float(np.mean(elapsed)),
               timing_scope="full embedding+policy/adaptation+decoder+retained prediction; graph construction separate",
               final_pair_counts=final, graph_ids=[g.get("cluster_id") for g in graphs],
               requested_pair_count=float(np.mean(req)) if req else None,
               sampled_pair_count=float(np.mean(samp)) if samp else None,
               final_pair_count=float(np.mean(final)) if final else None,
               budget_excess_pair_count=float(np.mean(
                   [h["final_pair_count"] - h["requested_pair_count"] for h in histories
                    if h.get("final_pair_count") is not None and h.get("requested_pair_count") is not None])) if req and final else None,
               repair_added_pair_count=float(np.mean(
                   [h["final_pair_count"] - h["sampled_pair_count"] for h in histories
                    if h.get("final_pair_count") is not None and h.get("sampled_pair_count") is not None])) if samp and final else None)
    return row, histories


def validation_gates(policy, graphs, gnn, cfg, device, ks):
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in ks):
        raise ValueError("TTA_KS must contain nonnegative integers.")
    rows = []
    for k in sorted(set([0] + list(ks))):
        row, _ = adaptation_trial(policy, graphs, gnn, cfg, device, k)
        comparable = k == 0 or row["final_pair_counts"] == rows[0]["final_pair_counts"]
        row["budget_comparable"] = comparable
        row["enabled"] = k == 0 or (comparable and tta_should_enable(rows[0]["rmse"], row["rmse"], cfg.get("tta_val_tol", 0.02)))
        rows.append(row)
    selected = min((row for row in rows if row["enabled"]), key=lambda row: (row["rmse"], row["K"]))["K"]
    return rows, selected


def complete_stage(out, stage, context, artifacts, **extra):
    out = Path(out)
    path = out / "provenance.json"
    for record in context["dataset"].values():
        if not Path(record["path"]).is_file() or file_info(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("Campaign input changed during execution; no completion is published")
    provenance = json.loads(path.read_text()) if path.exists() else {}
    if provenance.get(stage, {}).get("status") == "complete":
        raise FileExistsError("Completed notebook outputs are immutable; select a fresh OUTPUT_DIR")
    provenance[stage] = dict(context, stage=stage, status="complete",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        artifacts={name: file_info(out / name) for name in artifacts}, **extra)
    # Record source contents as well as HEAD for local, uncommitted repairs.
    root = Path(__file__).resolve().parents[1]
    provenance[stage]["source_files"] = {str(p.relative_to(root)): file_info(p)
        for p in sorted((root / "rls").glob("*.py"))}
    temporary = path.with_suffix(".json.tmp")
    # Undefined correlations are represented as null, never nonstandard JSON.
    def finite_json(value):
        if isinstance(value, dict): return {k: finite_json(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    provenance = finite_json(provenance)
    temporary.write_text(json.dumps(provenance, indent=2, allow_nan=False))
    temporary.replace(path)
    # Archive only this run's declared outputs; stale optional artifacts stay out.
    import zipfile
    archive = out / f"notebook_{stage}_{context['run_mode']}_outputs.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive {archive}")
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
