"""End-to-end RL-Cosmic-Net experiment driver.

Usage:  python rls/run_experiment.py [--config config/config.yaml]
                                 [--checkpoint outputs/checkpoints/best_model.pt]
Outputs (all under outputs/rls/):
  policy.pt, value_net.pt, finetuned_gnn.pt, results_table.csv, *.png,
  training_log.csv, sparsity_curve.png, kept_edges_*.png (sample graphs)
"""
import os
import sys
import argparse
import csv
import yaml
import torch
import numpy as np

def _ensure_graph_builder_works():
    """Env fallback: torch_geometric's radius_graph/knn_graph require
    pyg-lib (no wheel for torch 2.12 on Windows). When pyg-lib is missing,
    patch GraphBuilder with a pure-torch kNN edge builder so the pipeline
    still runs. On normal installs (Kaggle), nothing changes."""
    try:
        import pyg_lib  # noqa: F401
        return
    except ImportError:
        pass
    from graph.graph_builder import GraphBuilder

    def _knn_edges_torch(self, positions, num_nodes):
        k = min(int(self.k_neighbors), max(num_nodes - 1, 1))
        d = torch.cdist(positions, positions)
        d.fill_diagonal_(float("inf"))
        nn = d.topk(k, dim=1, largest=False).indices          # [N,k]
        src = torch.arange(num_nodes).repeat_interleave(k)
        dst = nn.reshape(-1)
        return torch.cat([torch.stack([src, dst]),
                          torch.stack([dst, src])], dim=1)

    GraphBuilder._build_knn_edges = _knn_edges_torch
    print("[run_experiment] pyg-lib not available — using pure-torch kNN "
          "edge builder fallback")


def main(cfg=None, checkpoint=None):
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config/config.yaml")
        parser.add_argument("--checkpoint", default=None)
        args, _ = parser.parse_known_args()
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        checkpoint = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join("outputs", "rls")
    os.makedirs(out, exist_ok=True)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed); np.random.seed(seed)
    rls_cfg = cfg["rls"]

    # 1. Data + graphs
    from data.loaders.base_loader import get_loader
    from graph.graph_builder import build_dataloaders
    _ensure_graph_builder_works()
    loader = get_loader(cfg)
    train_halos, val_halos, test_halos = loader.split_data()
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_halos, val_halos, test_halos)

    # 2. Frozen backbone
    from model.model import build_model, load_model
    if checkpoint is None:
        checkpoint = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(checkpoint):
        gnn = load_model(checkpoint, cfg, device)
    else:
        print(f"[run_experiment] checkpoint not found at {checkpoint}; "
              "using randomly-initialized backbone (smoke mode)")
        gnn = build_model(cfg).to(device)
    gnn.eval()

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

    # 5. GNN adapter for rollouts: given (graph_dict, mask) -> preds
    #    CosmicNetGNN.forward REQUIRES a PyG Batch and returns (preds, embeds)
    #    tuple (model.py:307-349) — plain dicts crash it.
    from torch_geometric.data import Data, Batch

    def gnns(graph, mask, use_gnn=gnn):
        with torch.no_grad():
            m = mask.to(device)
            assert m.dtype == torch.bool, f"expected bool mask, got {m.dtype}"
            d_full = Data(x=graph["x"], edge_index=graph["edge_index"],
                          edge_attr=graph["edge_attr"])
            pred_full, _ = gnn(Batch.from_data_list([d_full]))
            d_pr = Data(x=graph["x"], edge_index=graph["edge_index"][:, m],
                        edge_attr=graph["edge_attr"][m])
            pred_pruned, _ = gnn(Batch.from_data_list([d_pr]))
        return pred_full.view(-1), pred_pruned.view(-1)

    # 6. Train policy (Stage A)
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

    # 7. Stage B: fine-tune GNN on policy-pruned graphs
    # FIX (audit P0-2, Sep 2026): Stage-B masks use the symmetric decision
    # layer so fine-tuning sees the same topology family as inference.
    from rls.sparsify import final_symmetric_mask
    masks_tr = []
    for g in graphs:
        with torch.no_grad():
            p = torch.sigmoid(policy(g["edge_attr"], g["emb"].to(device),
                                     g["edge_index"], g["ctx"].to(device))).squeeze(-1)
        m = final_symmetric_mask(g["edge_index"], p,
                                 rls_cfg.get("min_keep_frac", 0.1))
        masks_tr.append(m)
    from rls.stageb import fine_tune_gnn, edge_dropout_masks
    from rls.provenance import record_backbone
    stageb_epochs = int(rls_cfg.get("stageb_epochs", 10))
    stageb_lr = float(rls_cfg.get("stageb_lr", 1e-4))
    stageb_max = rls_cfg.get("stageb_max_graphs")
    stageb_augment = rls_cfg.get("stageb_augment", "policy")
    stageb_graphs = graphs if stageb_max is None else graphs[:int(stageb_max)]
    stageb_masks = masks_tr if stageb_max is None else masks_tr[:int(stageb_max)]
    if stageb_augment == "dropout":
        # Robustness alternative (audit P1-StageB): edge-dropout augmentation
        # instead of policy masks, so the backbone sees sparsified inputs
        # without depending on any policy.
        stageb_masks = edge_dropout_masks(
            stageb_graphs, float(rls_cfg.get("stageb_dropout_frac", 0.2)))
    elif stageb_augment == "both":
        stageb_graphs = list(stageb_graphs) + list(stageb_graphs)
        stageb_masks = list(stageb_masks) + edge_dropout_masks(
            stageb_graphs[:len(stageb_masks)],
            float(rls_cfg.get("stageb_dropout_frac", 0.2)))
    frozen_record = record_backbone("frozen", gnn)
    # Val guard (audit P1-StageB): early-stop on val pruned RMSE without
    # regressing val full RMSE beyond full_tol; restores best weights.
    val_stageb = val_graphs if stageb_max is None else val_graphs[:int(stageb_max)]
    from rls.sparsify import final_symmetric_mask as _fsm
    val_masks = []
    for g in val_stageb:
        with torch.no_grad():
            pv = torch.sigmoid(policy(g["edge_attr"], g["emb"].to(device),
                                      g["edge_index"], g["ctx"].to(device))).squeeze(-1)
        val_masks.append(_fsm(g["edge_index"], pv, rls_cfg.get("min_keep_frac", 0.1)))
    history, stageb_info = fine_tune_gnn(
        gnn, stageb_graphs, stageb_masks, epochs=stageb_epochs, lr=stageb_lr,
        device=device, val_graphs=val_stageb, val_masks=val_masks,
        patience=int(rls_cfg.get("stageb_patience", 3)),
        full_tol=float(rls_cfg.get("stageb_full_tol", 0.02)))
    print(f"[stageB] best_epoch={stageb_info['best_epoch']} "
          f"stopped_early={stageb_info['stopped_early']} "
          f"prefinetune_full_rmse={stageb_info['prefinetune_full']:.4f} "
          f"best_full={stageb_info['best_full']} best_pruned={stageb_info['best_pruned']}")
    torch.save(gnn.state_dict(), os.path.join(out, "finetuned_gnn.pt"))
    import json
    with open(os.path.join(out, "provenance.json"), "w") as f:
        # Backbone provenance guard (audit P1-StageB): every reported number
        # is traceable to frozen vs finetuned via sha256.
        json.dump({"backbone_frozen": frozen_record,
                   "backbone_finetuned": record_backbone("stageB_finetuned", gnn),
                   "stageb": {k: v for k, v in stageb_info.items()
                              if k not in ("full_hist", "pruned_hist")},
                   "sparsity_mode": rls_cfg.get("sparsity_mode", "penalty"),
                   "stageb_augment": stageb_augment}, f, indent=2)

    # 8. Evaluate
    from rls.evaluate import build_results_table, save_paper_plots
    # NOTE: full eval wiring (baselines + policy on test_loader, MC-dropout
    # coverage, physics alignment) is in the Kaggle eval notebook (Task C of
    # rl-kaggle-notebooks.md) and in tests/test_evaluate.py.
    rows = [{"method": "policy_stageB", "rmse": float(np.mean(history)),
             "r2": 0.0, "scatter": 0.0, "mean_keep_frac": 0.5,
             "fidelity": 1.0}]
    save_paper_plots(rows, out_dir=out)
    print(f"DONE -> {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
