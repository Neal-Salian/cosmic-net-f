"""Cross-simulation OOD evaluation: policy trained on TNG, tested on CAMELS.

The headline generalization claim of the paper. Uses the repo's CAMELSLoader
(HDF5 download or synthetic fallback) and the SAME frozen policy + GNN."""
import os
import torch
import pandas as pd

def evaluate_cross_sim(cfg, checkpoint, max_halos=200, out_dir="outputs/rls",
                       require_real_data=True):
    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder
    from model.model import load_model, build_model
    from rls.policy import build_policy
    from rls.sparsify import hard_mask, repair_connectivity
    from rls.run_experiment import _ensure_graph_builder_works

    _ensure_graph_builder_works()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(cfg)
    halos = loader.load()[:max_halos]
    if require_real_data:
        # The CAMELS loader silently generates synthetic data when the HDF5 is
        # missing; synthetic OOD results must never be reported as real ones.
        # require_real_data=False is the explicit opt-out for plumbing tests.
        assert getattr(loader, "used_synthetic_fallback", False) is False, (
            "CAMELS loader used its synthetic fallback - synthetic OOD results "
            "are not publishable. Cache the real HDF5 first (or pass "
            "require_real_data=False for plumbing tests only).")
    gb = GraphBuilder(cfg)
    graphs = gb.build_graphs(halos)

    if checkpoint is None:
        checkpoint = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(checkpoint):
        gnn = load_model(checkpoint, cfg, device)
    else:
        print("[cross_sim] checkpoint not found; using random backbone "
              "(smoke mode)")
        gnn = build_model(cfg).to(device)
    gnn.eval()
    policy = build_policy(cfg, node_emb_dim=gnn.output_dim).to(device)
    policy_path = os.path.join(out_dir, "policy.pt")
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=device))
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
            m = hard_mask(p, 0.1)
            m = repair_connectivity(g.edge_index, m)
            pred_full, _ = gnn(b)
            g_pr = g.clone()
            g_pr.edge_index = g.edge_index[:, m]
            g_pr.edge_attr = g.edge_attr[m]
            pred_pruned, _ = gnn(Batch.from_data_list([g_pr]))
            rows.append({"cluster_id": getattr(g, "cluster_id", "?"),
                         "y": float(g.y.cpu()), "pred_full": float(pred_full.cpu()),
                         "pred_pruned": float(pred_pruned.cpu()),
                         "keep_frac": float(m.float().mean().cpu())})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "cross_sim_results.csv"), index=False)
    return out_dir
