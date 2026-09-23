"""End-to-end RL-Cosmic-Net experiment driver (Task 3c).

Primary comparison uses the shared matched-budget evaluator on unprepared
PyG test graphs. Stage A training/validation may retain legacy repaired
decoding, explicitly identified; the main frozen_backbone/scorer comparison
uses exact-budget decoding via rls.matched_benchmark.
"""
import copy
import os
import sys
import argparse
import csv
import yaml
import torch
import numpy as np


def _fresh_output_dir(path):
    from pathlib import Path
    p = Path(os.fspath(path))
    if p.exists():
        if p.is_file():
            raise FileExistsError(f"output_dir exists as a file: {p}")
        try:
            next(p.iterdir())
        except StopIteration:
            return p
        raise FileExistsError(
            f"output_dir exists and is nonempty: {p}. Refusing to overwrite.")
    return p


def _source_revision():
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                           capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _source_code_records(root, relpaths):
    from data.provenance import sha256_file
    from pathlib import Path
    out = []
    for rel in relpaths:
        p = Path(root) / rel
        try:
            if p.is_file():
                out.append({"path": rel, **sha256_file(p)})
            else:
                out.append({"path": rel, "status": "missing"})
        except Exception as exc:
            out.append({"path": rel, "status": f"unavailable: {exc}"})
    return out


TASK3C_SOURCE_FILES = (
    "rls/run_experiment.py", "rls/matched_benchmark.py",
    "rls/benchmark_scorers.py", "rls/benchmark_predictor.py",
    "rls/constraints.py", "rls/constrained_policy.py",
    "rls/policy.py", "rls/train_policy.py", "rls/pair_training.py",
    "rls/stageb.py", "rls/provenance.py", "rls/budget_accounting.py",
    "rls/evaluate.py", "rls/pair_tta.py", "rls/tta.py",
    "model/model.py", "graph/graph_builder.py",
    "data/provenance.py", "data/catalog.py", "data/splits.py",
    "data/augmentation.py",
    "data/loaders/base_loader.py", "data/loaders/synthetic_loader.py",
    "data/loaders/tng_loader.py", "data/loaders/camels_loader.py",
)


def model_device(model):
    """Device holding the model parameters (or buffers); cpu if none."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        for buf in model.buffers():
            if isinstance(buf, torch.Tensor):
                return buf.device
    except Exception:
        pass
    return torch.device("cpu")


def move_graph_to_device(graph, device):
    """Clone a PyG Data to device, preserving IDs/lineage/raw features."""
    cloned = graph.clone()
    return cloned.to(device)


def move_test_graphs_to_device(graphs, device):
    """Move raw test graphs; originals untouched, IDs/lineage preserved."""
    return [move_graph_to_device(g, device) for g in graphs]


def resolve_random_backbone_label(cfg, allow_random_backbone):
    """Truthful untrained-backbone label by actual data source."""
    source = str(cfg.get("data", {}).get("source", "unknown"))
    if source == "synthetic":
        return "synthetic_random_backbone_smoke", "synthetic_random_backbone_smoke"
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in source) or "unknown"
    label = f"untrained_random_backbone_smoke_{safe}"
    return label, label


def build_legacy_split_manifest(train_halos, val_halos, test_halos, seed, ratios):
    """Deterministic hash of actual returned split IDs/ratios/seed/method.

    Legacy BaseDataLoader.split_data path only; never claims grouped lineage.
    """
    from data.provenance import hash_payload
    payload = {
        "train_cluster_ids": [h.cluster_id for h in train_halos],
        "val_cluster_ids": [h.cluster_id for h in val_halos],
        "test_cluster_ids": [h.cluster_id for h in test_halos],
        "ratios": {"train": float(ratios[0]), "val": float(ratios[1]),
                   "test": float(ratios[2])},
        "seed": int(seed),
        "method": "BaseDataLoader.split_data_RandomState_permutation",
    }
    return {**payload, "manifest_sha256": hash_payload(payload)}


def copy_declared_split_lineage(graphs, group_key):
    """Copy strict split metadata to the direct attribute consumed by evaluator rows."""
    for graph in graphs:
        metadata = getattr(graph, "metadata", {}) or {}
        if group_key not in metadata or metadata[group_key] is None:
            raise ValueError(
                f"strict split group {group_key!r} missing for graph "
                f"{getattr(graph, 'cluster_id', '<unknown>')}"
            )
        graph.lineage_group = str(metadata[group_key])
    return graphs


def main(cfg=None, checkpoint=None, output_dir=None, allow_random_backbone=False):
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config/config.yaml")
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--allow-random-backbone", action="store_true")
        args = parser.parse_args()
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        checkpoint = args.checkpoint
        output_dir = args.output_dir
        allow_random_backbone = args.allow_random_backbone

    if output_dir is None:
        raise ValueError("output_dir must be explicit for research runs.")
    out_path = _fresh_output_dir(output_dir)
    # Fail missing checkpoint BEFORE any data-loader side effects.
    from model.model import build_model, load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_random = bool(checkpoint is None and allow_random_backbone)
    checkpoint_info = None
    if not use_random:
        if checkpoint is None:
            checkpoint = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
        from data.provenance import validate_checkpoint_file
        try:
            checkpoint_info = validate_checkpoint_file(checkpoint)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Required checkpoint does not exist: {checkpoint}") from exc
    out = os.fspath(out_path)
    os.makedirs(out, exist_ok=True)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed); np.random.seed(seed)
    rls_cfg = cfg["rls"]

    # 1. Data + graphs (strict grouped path vs legacy path).
    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder, build_dataloaders
    loader = get_loader(cfg)
    strict_research = bool(cfg.get("data", {}).get("strict_research", False))
    ratios = (float(cfg.get("data", {}).get("train_ratio", 0.7)),
              float(cfg.get("data", {}).get("val_ratio", 0.15)),
              float(cfg.get("data", {}).get("test_ratio", 0.15)))
    if strict_research:
        group_key = cfg.get("data", {}).get("split_group_key")
        if not group_key:
            raise ValueError("strict research requires data.split_group_key")
        from data.splits import grouped_split
        halos = loader.load()
        train_halos, val_halos, test_halos, grouped_manifest = grouped_split(
            halos, group_key=group_key, ratios=ratios, seed=int(seed))
        split_manifest = grouped_manifest
    else:
        train_halos, val_halos, test_halos = loader.split_data()
        split_manifest = build_legacy_split_manifest(
            train_halos, val_halos, test_halos, seed=int(seed), ratios=ratios)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_halos, val_halos, test_halos)

    # Raw test graphs with cluster_id/provenance intact (no prepare_graphs).
    builder = GraphBuilder(cfg)
    test_graphs_raw = builder.build_graphs(test_halos)
    if strict_research:
        copy_declared_split_lineage(test_graphs_raw, group_key)

    # 2. Frozen backbone (immutable for primary evaluation)
    if use_random:
        gnn = build_model(cfg).to(device)
        research_label, run_mode = resolve_random_backbone_label(
            cfg, allow_random_backbone)
        checkpoint_record = {"status": "untrained_random_backbone",
                             "research_label": research_label}
    else:
        research_label = cfg.get("data", {}).get(
            "research_label", "legacy_total_radius")
        gnn = load_model(
            checkpoint, cfg, device, research_label=research_label
        )
        run_mode = "full"
        checkpoint_record = {**checkpoint_info,
                             "research_label": research_label}
    gnn.eval()
    from rls.provenance import record_backbone
    frozen_record = record_backbone("frozen", gnn)
    frozen_hash_before = frozen_record["sha256"]

    # 3. Prepare policy-training graphs (node embeddings precomputed)
    from rls.train_policy import prepare_graphs
    graphs = prepare_graphs(train_loader, gnn, device)
    val_graphs = prepare_graphs(val_loader, gnn, device)

    # 4. Policy + value net + policy-gradient trainer
    from rls.policy import build_policy
    from rls.policy_gradient import PolicyGradientTrainer, ValueNet
    policy = build_policy(cfg, node_emb_dim=gnn.output_dim).to(device)
    value_net = ValueNet(gnn.output_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=rls_cfg["lr"])
    vopt = torch.optim.Adam(value_net.parameters(), lr=rls_cfg["lr"])
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, rls_cfg)

    # 5. GNN adapter for rollouts: given (graph_dict, mask) -> preds.
    #    Honors use_gnn so StageB evaluation never reuses a stale model.
    from torch_geometric.data import Data, Batch

    def gnns(graph, mask, use_gnn=None):
        model = gnn if use_gnn is None else use_gnn
        with torch.no_grad():
            m = mask.to(device)
            assert m.dtype == torch.bool, f"expected bool mask, got {m.dtype}"
            d_full = Data(x=graph["x"], edge_index=graph["edge_index"],
                          edge_attr=graph["edge_attr"])
            pred_full, _ = model(Batch.from_data_list([d_full]))
            d_pr = Data(x=graph["x"], edge_index=graph["edge_index"][:, m],
                        edge_attr=graph["edge_attr"][m])
            pred_pruned, _ = model(Batch.from_data_list([d_pr]))
        return pred_full.view(-1), pred_pruned.view(-1)

    # 6. Train policy (Stage A, legacy repaired decoding explicitly identified)
    from rls.train_policy import train_policy
    log_rows = []
    def log_fn(epoch, target_sp, loss):
        log_rows.append([epoch, target_sp, float(loss)])
        print(f"[epoch {epoch}] target_sparsity={target_sp:.3f} loss={float(loss):.4f}")
    losses = train_policy(trainer, graphs, gnns, rls_cfg, device,
                          epochs=rls_cfg["epochs"], log_fn=log_fn)
    torch.save(policy.state_dict(), os.path.join(out, "policy.pt"))
    torch.save(value_net.state_dict(), os.path.join(out, "value_net.pt"))
    with open(os.path.join(out, "training_log.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "target_sparsity", "loss"])
        w.writerows(log_rows)

    # 7. Stage B on a deepcopy; acceptance guard preserved as is.
    from rls.sparsify import eval_mask
    masks_tr = []
    for g in graphs:
        with torch.no_grad():
            p = torch.sigmoid(policy(g["edge_attr"], g["emb"].to(device),
                                     g["edge_index"], g["ctx"].to(device))).squeeze(-1)
        m = eval_mask(g["edge_index"], p, rls_cfg)
        masks_tr.append(m)
    from rls.stageb import fine_tune_gnn, edge_dropout_masks
    stageb_epochs = int(rls_cfg.get("stageb_epochs", 10))
    stageb_lr = float(rls_cfg.get("stageb_lr", 1e-4))
    stageb_max = rls_cfg.get("stageb_max_graphs")
    stageb_augment = rls_cfg.get("stageb_augment", "policy")
    stageb_graphs = graphs if stageb_max is None else graphs[:int(stageb_max)]
    stageb_masks = masks_tr if stageb_max is None else masks_tr[:int(stageb_max)]
    if stageb_augment == "dropout":
        stageb_masks = edge_dropout_masks(
            stageb_graphs, float(rls_cfg.get("stageb_dropout_frac", 0.2)))
    elif stageb_augment == "both":
        stageb_graphs = list(stageb_graphs) + list(stageb_graphs)
        stageb_masks = list(stageb_masks) + edge_dropout_masks(
            stageb_graphs[:len(stageb_masks)],
            float(rls_cfg.get("stageb_dropout_frac", 0.2)))
    val_stageb = val_graphs if stageb_max is None else val_graphs[:int(stageb_max)]
    from rls.sparsify import eval_mask as _eval_mask
    val_masks = []
    for g in val_stageb:
        with torch.no_grad():
            pv = torch.sigmoid(policy(g["edge_attr"], g["emb"].to(device),
                                      g["edge_index"], g["ctx"].to(device))).squeeze(-1)
        val_masks.append(_eval_mask(g["edge_index"], pv, rls_cfg))
    stageb_gnn = copy.deepcopy(gnn)
    history, stageb_info = fine_tune_gnn(
        stageb_gnn, stageb_graphs, stageb_masks, epochs=stageb_epochs, lr=stageb_lr,
        device=device, val_graphs=val_stageb, val_masks=val_masks,
        patience=int(rls_cfg.get("stageb_patience", 3)),
        full_tol=float(rls_cfg.get("stageb_full_tol", 0.02)))
    print(f"[stageB] best_epoch={stageb_info['best_epoch']} "
          f"stopped_early={stageb_info['stopped_early']} "
          f"prefinetune_full_rmse={stageb_info['prefinetune_full']:.4f} "
          f"best_full={stageb_info['best_full']} best_pruned={stageb_info['best_pruned']}")
    accepted = bool(stageb_info.get("accepted", False))
    # Frozen model must be unchanged by StageB (deepcopy isolation).
    frozen_record_after = record_backbone("frozen", gnn)
    assert frozen_record_after["sha256"] == frozen_hash_before, \
        "frozen backbone mutated during StageB"
    stageb_record = None
    finetuned_path = os.path.join(out, "finetuned_gnn.pt")
    if accepted:
        torch.save(stageb_gnn.state_dict(), finetuned_path)
        stageb_record = record_backbone("stageB_finetuned", stageb_gnn)
        # Declared artifact must match actual saved state.
        from data.provenance import sha256_file, model_state_hash
        actual_state = model_state_hash(stageb_gnn.state_dict())
        assert stageb_record["sha256"] == actual_state
        file_hash = sha256_file(finetuned_path)["sha256"]
    else:
        file_hash = None
        if os.path.exists(finetuned_path):
            raise AssertionError("rejected StageB must not leave a finetuned artifact")

    # 8. Matched-budget evaluation on raw test graphs (frozen primary).
    from rls.matched_benchmark import BudgetSpec, evaluate_matched_budget
    from rls.benchmark_scorers import PairScorer
    from data.provenance import build_run_provenance, model_state_hash, sha256_file
    keep_q = float(rls_cfg.get("target_sparsity_end", 0.4))
    budget = BudgetSpec(keep_fraction=keep_q, constraint="no_isolates", decoder="direct")
    edge_feature_names = list(cfg.get("graph", {}).get("edge_features",
        ["distance", "delta_v", "cos_theta", "mass_ratio", "proj_sep"]))
    scorers = [PairScorer(name="random"),
               PairScorer(name="distance"),
               PairScorer(name="degree"),
               PairScorer(name="provided_rl", policy=policy,
                          edge_feature_names=edge_feature_names)]
    # Honest provenance: unknown catalog stays unknown, never inferred real.
    contract = None
    try:
        contract = test_halos[0].metadata.get("catalog_contract") if test_halos else None
    except Exception:
        contract = None
    if isinstance(contract, dict) and contract:
        catalog_contract = copy.deepcopy(contract)
        source_files = list(contract.get("source_files", []))
    else:
        catalog_contract = {"source": cfg.get("data", {}).get("source", "unknown"),
                            "research_label": research_label,
                            "status": "unknown_no_contract",
                            "note": "full catalog/source unknown, not inferred real"}
        source_files = []
    try:
        aug_manifest = getattr(train_loader, "augmentation_manifest", None)
    except Exception:
        aug_manifest = None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_code_files = _source_code_records(root, TASK3C_SOURCE_FILES)
    provenance = build_run_provenance(
        source_files=source_files,
        source_code_files=source_code_files,
        catalog_contract=catalog_contract, split_manifest=split_manifest,
        resolved_config=copy.deepcopy(cfg), source_revision=_source_revision(),
        graph={"backend": "portable_torch",
               "method": cfg.get("graph", {}).get("method"),
               "tie_policy": "inclusive"},
        augmentation_manifest=aug_manifest if isinstance(aug_manifest, dict) else None,
        research_label=research_label,
    )
    provenance["rng_roles"] = {"torch_seed": int(seed), "numpy_seed": int(seed),
                               "cuda_seed": int(seed), "data_loader_seed": int(seed),
                               "model_init_seed": int(seed)}
    provenance["run_mode"] = run_mode
    provenance["stage_a_decoder"] = "legacy_repaired"
    try:
        provenance["policy_state_hash"] = model_state_hash(policy.state_dict())
    except Exception:
        pass
    # Kaggle/CUDA: raw graphs are CPU; move to the evaluating model device
    # before each matched evaluation, preserving IDs/lineage/raw features.
    # provided_rl shares the same device via the policy on `device`.
    frozen_graphs = move_test_graphs_to_device(test_graphs_raw, model_device(gnn))
    policy_dev = model_device(policy)
    assert model_device(gnn) == policy_dev, "policy must share the frozen model device"
    summary_frozen, rows_frozen = evaluate_matched_budget(
        gnn, frozen_graphs, scorers, budget, seed=int(seed),
        provenance=provenance, backbone="frozen")
    all_summaries = [{"backbone_stage": "frozen", **s} for s in summary_frozen]
    all_rows = list(rows_frozen)
    if accepted:
        stageb_graphs_eval = move_test_graphs_to_device(
            test_graphs_raw, model_device(stageb_gnn))
        summary_b, rows_b = evaluate_matched_budget(
            stageb_gnn, stageb_graphs_eval, scorers, budget, seed=int(seed),
            provenance=provenance, backbone="stageB_finetuned")
        all_summaries.extend([{"backbone_stage": "stageB_finetuned", **s} for s in summary_b])
        all_rows.extend(rows_b)

    import json
    def _json_safe(v):
        if isinstance(v, dict):
            return {k: _json_safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        if isinstance(v, (np.floating,)):
            f = float(v)
            return f if np.isfinite(f) else None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, torch.Tensor):
            return _json_safe(v.detach().cpu().tolist())
        return v
    with open(os.path.join(out, "matched_summary.json"), "w") as f:
        json.dump(_json_safe(all_summaries), f, indent=2, allow_nan=False)
    with open(os.path.join(out, "matched_rows.json"), "w") as f:
        json.dump(_json_safe(all_rows), f, indent=2, allow_nan=False)
    # Compatibility results_table.csv from actual summary only (no fake numbers).
    import pandas as pd
    compat = []
    rows_by_method = {}
    for r in rows_frozen:
        rows_by_method.setdefault((r["method"],), []).append(r)
    for s in summary_frozen:
        method_rows = [r for r in rows_frozen if r["method"] == s["method"] and r["status"] == "success"]
        if method_rows:
            retains = [r["final_pair_count"] / r["available_pair_count"]
                       for r in method_rows if r.get("available_pair_count")]
            mean_keep = float(np.mean(retains)) if retains else None
        else:
            mean_keep = None
        compat.append({
            "method": s["method"],
            "rmse": s.get("rmse"),
            "r2": s.get("r2"),
            "scatter": s.get("std_residual"),
            "mean_keep_frac": mean_keep,
            "fidelity": s.get("fidelity_full_pred"),
            "backbone_stage": "frozen",
            "backbone_sha256": frozen_record["sha256"],
            "research_label": research_label,
            "run_mode": run_mode,
            "n_success": s.get("n_success"),
            "n_failed": s.get("n_failed"),
        })
    pd.DataFrame(compat).to_csv(os.path.join(out, "results_table.csv"), index=False)
    prov_out = {"backbone_frozen": frozen_record,
                "stageb": {k: v for k, v in stageb_info.items()
                           if k not in ("full_hist", "pruned_hist")},
                "stageb_accepted": accepted,
                "sparsity_mode": rls_cfg.get("sparsity_mode", "penalty"),
                "stage_a_decoder": "legacy_repaired",
                "stageb_augment": stageb_augment,
                "research_label": research_label,
                "run_mode": run_mode,
                "checkpoint": checkpoint_record,
                "split": split_manifest,
                "budget": budget.config_dict(),
                "provenance": provenance}
    if accepted:
        prov_out["backbone_finetuned"] = stageb_record
        prov_out["finetuned_artifact"] = {"path": "finetuned_gnn.pt",
                                          "sha256": file_hash,
                                          "stage": "stageB_finetuned"}
    else:
        prov_out["rejection"] = {"frozen_sha256": frozen_record["sha256"],
                                 "reason": "stageB acceptance guard rejected; no finetuned artifact emitted"}
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(_json_safe(prov_out), f, indent=2, allow_nan=False)
    print(f"DONE -> {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
