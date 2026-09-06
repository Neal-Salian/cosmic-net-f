"""Multi-seed RL harness (audit P2, Sep 2026): repeat offline policy training +
evaluation over >=5 seeds; report mean +/- std (and bootstrap CI) for RMSE,
R2, keep_frac at matched operating points. A script, not notebook cells, so
the numbers are reproducible outside Kaggle.

Usage:
  python scripts/multiseed_rl.py --config config/config.yaml \
      --seeds 42 43 44 45 46 --epochs 60 --out outputs/rls/multiseed.json

Writes JSON with per-seed rows (each carrying backbone_stage/sha256 and the
policy.pt md5 per the provenance.json convention) plus summarize_multiseed
CI records. Compare sparsity_mode=penalty vs topk_scheduled by running twice
with --rls-override sparsity_mode=topk_scheduled.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    import torch
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-graphs", type=int, default=None)
    ap.add_argument("--rls-override", nargs="*", default=[],
                    help="KEY=VALUE pairs merged into cfg['rls'], e.g. "
                         "sparsity_mode=topk_scheduled")
    ap.add_argument("--out", default="outputs/rls/multiseed.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.rls_override:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cfg["rls"][k] = v
    if args.epochs is not None:
        cfg["rls"]["epochs"] = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from data.loaders.base_loader import get_loader
    from graph.graph_builder import build_dataloaders
    from rls.run_experiment import _ensure_graph_builder_works
    _ensure_graph_builder_works()
    loader = get_loader(cfg)
    train_halos, val_halos, test_halos = loader.split_data()
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_halos, val_halos, test_halos)

    from model.model import build_model, load_model
    ckpt = args.checkpoint or os.path.join(
        cfg["training"]["checkpoint_dir"], "best_model.pt")
    seed_rows = []
    for s in args.seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        gnn = (load_model(ckpt, cfg, device) if os.path.exists(ckpt)
               else build_model(cfg).to(device))
        gnn.eval()
        from rls.train_policy import prepare_graphs, train_policy
        from rls.policy import build_policy
        from rls.policy_gradient import PolicyGradientTrainer, ValueNet
        from rls.provenance import record_backbone
        graphs = prepare_graphs(train_loader, gnn, device)
        if args.max_graphs is not None:
            graphs = graphs[:args.max_graphs]
        test_graphs = prepare_graphs(test_loader, gnn, device)
        policy = build_policy(cfg, node_emb_dim=gnn.output_dim).to(device)
        value_net = ValueNet(gnn.output_dim).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=cfg["rls"]["lr"])
        vopt = torch.optim.Adam(value_net.parameters(), lr=cfg["rls"]["lr"])
        trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, cfg["rls"])

        from torch_geometric.data import Data, Batch

        def gnns(graph, mask, use_gnn=gnn):
            with torch.no_grad():
                m = mask.to(device)
                d_full = Data(x=graph["x"], edge_index=graph["edge_index"],
                              edge_attr=graph["edge_attr"])
                pred_full, _ = use_gnn(Batch.from_data_list([d_full]))
                d_pr = Data(x=graph["x"], edge_index=graph["edge_index"][:, m],
                            edge_attr=graph["edge_attr"][m])
                pred_pruned, _ = use_gnn(Batch.from_data_list([d_pr]))
            return pred_full.view(-1), pred_pruned.view(-1)

        train_policy(trainer, graphs, gnns, cfg["rls"], device,
                     epochs=cfg["rls"]["epochs"])
        # evaluate: frozen symmetric policy mask on the test graphs
        from rls.sparsify import final_symmetric_mask
        from rls.evaluate import _rmse, _r2
        pf, pp, ys, keeps = [], [], [], []
        with torch.no_grad():
            for g in test_graphs:
                gd = {k: v.to(device) for k, v in g.items()
                      if isinstance(v, torch.Tensor)}
                p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"],
                                         gd["edge_index"], gd["ctx"])).squeeze(-1)
                m = final_symmetric_mask(
                    gd["edge_index"], p, cfg["rls"].get("min_keep_frac", 0.1))
                full, pruned = gnns(gd, m)
                pf.append(full.cpu())
                pp.append(pruned.cpu())
                ys.append(gd["y"].view(-1).cpu())
                keeps.append(m.float().mean().item())
        pf, pp, ys = torch.cat(pf), torch.cat(pp), torch.cat(ys)
        rec = record_backbone("frozen", gnn)
        seed_rows.append({
            "seed": s, "method": "rl_policy",
            "rmse": _rmse(pp, ys), "r2": _r2(pp, ys),
            "mean_keep_frac": float(np.mean(keeps)),
            "full_rmse": _rmse(pf, ys), "full_r2": _r2(pf, ys),
            "backbone_stage": rec["stage"], "backbone_sha256": rec["sha256"],
        })
        print(f"[seed {s}] rmse={seed_rows[-1]['rmse']:.4f} "
              f"r2={seed_rows[-1]['r2']:.4f} "
              f"keep={seed_rows[-1]['mean_keep_frac']:.3f}")

    from rls.evaluate import summarize_multiseed
    summary = summarize_multiseed([[r] for r in seed_rows])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_seed": seed_rows, "summary": summary,
                   "sparsity_mode": cfg["rls"].get("sparsity_mode", "penalty")},
                  f, indent=2)
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
