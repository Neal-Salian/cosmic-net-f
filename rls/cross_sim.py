"""Legacy cross-simulation smoke plumbing for the existing CAMELS loader.

This path does not validate an audited scientific manifest or compatible
research artifacts. Real-data evaluation belongs to the separate audited
astronomy workflow; this legacy function is available only with its explicit
smoke-mode opt-out."""
import os
import torch
import pandas as pd

def evaluate_cross_sim(cfg, checkpoint, max_halos=200, out_dir="outputs/rls",
                       require_real_data=True, allow_random_backbone=False):
    if require_real_data:
        raise ValueError(
            "The legacy cross-simulation evaluator has no positive audited "
            "scientific manifest or compatible artifact preflight for any "
            "source. Use the audited astronomy evaluation path for real-data "
            "work, or pass require_real_data=False for an explicitly labeled "
            "smoke run."
        )
    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder
    from model.model import load_model, build_model
    from rls.policy import build_policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(cfg)
    halos = loader.load()[:max_halos]
    contract = loader.get_catalog_contract()
    if hasattr(contract, "to_dict"):
        contract = contract.to_dict()
    elif not isinstance(contract, dict):
        contract = {}
    # Include both the loader's runtime signal and its persisted catalog
    # contract. The latter survives process boundaries and is written into
    # every result row so opt-out plumbing runs remain visibly labeled.
    synthetic = bool(getattr(loader, "used_synthetic_fallback", False)
                     or contract.get("synthetic", False))
    fallback = bool(getattr(loader, "used_synthetic_fallback", False)
                    or contract.get("fallback", False))
    research_label = str(contract.get(
        "research_label", "synthetic" if synthetic else "unknown"
    ))
    gb = GraphBuilder(cfg)
    graphs = gb.build_graphs(halos)

    if checkpoint is None and allow_random_backbone:
        gnn = build_model(cfg).to(device)
    else:
        if checkpoint is None:
            checkpoint = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Required checkpoint does not exist: {checkpoint}")
        gnn = load_model(
            checkpoint, cfg, device,
            research_label=cfg.get("data", {}).get(
                "research_label", "legacy_total_radius"),
        )
    gnn.eval()
    policy = build_policy(cfg, node_emb_dim=gnn.output_dim).to(device)
    policy_path = os.path.join(out_dir, "policy.pt")
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(
            policy_path, map_location=device, weights_only=True
        ))
    else:
        print("[cross_sim] policy.pt not found; using randomly-initialized "
              "policy (smoke mode)")
    policy.eval()

    rows = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            if g.edge_index.shape[1] == 0:
                continue
            # CosmicNetGNN.forward/get_embeddings require a PyG Batch (.batch
            # attribute) — wrap single Data objects (model.py:323-326).
            from torch_geometric.data import Batch
            b = Batch.from_data_list([g])
            emb = gnn.get_embeddings(b, embedding_point="pre_pooling")
            from torch_geometric.nn import global_mean_pool
            ctx = global_mean_pool(emb, b.batch).squeeze(0)  # [out]
            p = torch.sigmoid(policy(g.edge_attr, emb, g.edge_index, ctx)).squeeze(-1)
            # FIX (audit P0-2, Sep 2026): mode-aware symmetric mask.
            from rls.sparsify import eval_mask
            m = eval_mask(g.edge_index, p, cfg)
            pred_full, _ = gnn(b)
            g_pr = g.clone()
            g_pr.edge_index = g.edge_index[:, m]
            g_pr.edge_attr = g.edge_attr[m]
            pred_pruned, _ = gnn(Batch.from_data_list([g_pr]))
            rows.append({"cluster_id": getattr(g, "cluster_id", "?"),
                         "y": float(g.y.cpu()), "pred_full": float(pred_full.cpu()),
                         "pred_pruned": float(pred_pruned.cpu()),
                         "keep_frac": float(m.float().mean().cpu()),
                         "evaluation_mode": "smoke",
                         "research_label": research_label,
                         "synthetic": synthetic,
                         "fallback": fallback})
    columns = ["cluster_id", "y", "pred_full", "pred_pruned", "keep_frac",
               "evaluation_mode", "research_label", "synthetic", "fallback"]
    df = pd.DataFrame(rows, columns=columns)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "cross_sim_results.csv"), index=False)
    return out_dir
