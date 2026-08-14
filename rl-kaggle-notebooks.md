# RL-Cosmic-Net — Kaggle GPU Notebooks

Four notebooks, each with copy-paste cells. **The repo is private**, so each
notebook clones it using a GitHub **Personal Access Token (PAT)** stored in a
Kaggle Secret. Data + model weights are **committed to the repo**, so no
separate dataset upload is required.

> **One-time Kaggle setup:**
> 1. Add a Kaggle Secret: Notebook → **Add-ons → Secrets → Add new secret**,
>    name `GITHUB_PAT`, value = a GitHub PAT with `repo` scope (GitHub →
>    Settings → Developer settings → Personal access tokens → Fine-grained or
>    classic, with read access to the private repo).
> 2. Create each notebook with **GPU accelerator (T4/P100)** and **Internet ON**.

> **Model facts (verified from the checkpoints' embedded config + state dict):**
> `kaggle/best_model_augmented.pt` and `kaggle/best_model (2).pt` were BOTH
> trained at **`hidden_dim=64`, `output_dim=64`**, `node_features=4`,
> `edge_features=5`, `num_layers=3`. (`best_model_augmented.pt` = 729 epochs,
> recorded test `rmse 0.117 / r2 0.924`; `best_model (2).pt` = 828 epochs, val
> `rmse 0.144 / r2 0.885`.) `load_model` reads the checkpoint's own config, so
> these notebooks never hardcode a dim that could drift.
>
> All notebooks import the committed `rls/` package (no inline helper copies).
> The `rls` package is REINFORCE + learned baseline + entropy (honestly named,
> NOT PPO), real PyG `Batch` wrapping, `(preds, embeds)` tuple unpacking, and
> gradients that flow to the policy.

---

# NOTEBOOK A — Setup, Data, Graphs, Baselines

```python
# CELL 1: Install dependencies (match Kaggle's preinstalled torch; internet ON)
import torch, sys, subprocess
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(args))
# torch-geometric (>=2.5) auto-installs pyg-lib / torch-cluster / torch-scatter /
# torch-sparse for the preinstalled torch. Do NOT install the repo's pinned
# torch==2.1.0 requirements.txt (it would downgrade CUDA torch and break GPU).
pip("torch-geometric")
pip("pyyaml", "scipy", "pandas", "matplotlib", "seaborn", "h5py")
print("deps OK")
```

```python
# CELL 2: Clone the PRIVATE repo via PAT stored in a Kaggle Secret
import subprocess, os
from kaggle_secrets import UserSecretsClient
PAT = UserSecretsClient().get_secret("GITHUB_PAT")
if not PAT:
    raise RuntimeError("Missing Kaggle Secret 'GITHUB_PAT'")
if not os.path.exists("/kaggle/working/cosmic-net"):
    subprocess.check_call(["git", "clone",
        f"https://{PAT}@github.com/Rusheel86/cosmic-net.git",
        "/kaggle/working/cosmic-net"])
os.chdir("/kaggle/working/cosmic-net")
print("cloned. data:", os.listdir("data/raw"))
print("checkpoints:", [f for f in os.listdir("kaggle") if f.endswith(".pt")])
```

```python
# CELL 3: Load config (dims come from the checkpoint, not from config.yaml)
import sys, yaml, torch, numpy as np
sys.path.insert(0, ".")
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["source"] = "tng"
cfg["data"]["num_workers"] = 0          # Kaggle/Win-safe
cfg["data"]["batch_size"] = 16
cfg["model"]["mc_samples"] = 30
# rls config is now committed in config.yaml; leave it (or override below)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| source:", cfg["data"]["source"])
```

```python
# CELL 4: Load halos, build graphs, split
from data.loaders.base_loader import get_loader
from graph.graph_builder import build_dataloaders
loader = get_loader(cfg)
halos = loader.load()
print("total halos:", len(halos))
train_halos, val_halos, test_halos = loader.split_data()
print(f"train={len(train_halos)} val={len(val_halos)} test={len(test_halos)}")
torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])   # AFTER split_data (it resets RNG)
train_loader, val_loader, test_loader = build_dataloaders(cfg, train_halos, val_halos, test_halos)
b0 = next(iter(test_loader))
print("graph:", b0.x.shape, b0.edge_index.shape, b0.edge_attr.shape, "y:", b0.y.shape)
```

```python
# CELL 5: Load frozen backbone (load_model reads the checkpoint's own dims)
from model.model import load_model
ckpt = "kaggle/best_model_augmented.pt"
gnn = load_model(ckpt, cfg, device)
gnn.eval()
print("loaded", ckpt, "| hidden_dim:", gnn.hidden_dim, "output_dim:", gnn.output_dim)
with torch.no_grad():
    single = b0.get_example(0).to(device)   # a real single graph, correctly indexed
    pred, _ = gnn(single)
print("smoke prediction:", float(pred.view(-1)[0]))
```

```python
# CELL 6: Baselines + policy sparsification, imported from the committed rls package
from rls import (random_mask, degree_mask, distance_mask, mass_ratio_mask,
                 gradient_saliency_mask, attention_topk_mask,
                 hard_mask, repair_connectivity)
import torch_geometric.data as pg

def pred_for_mask(graph, mask):
    """Frozen-GNN prediction on a (sub)graph. mask: bool tensor [E]."""
    g = pg.Data(x=graph.x, edge_index=graph.edge_index[:, mask],
                edge_attr=graph.edge_attr[mask])
    g.batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=g.x.device)
    with torch.no_grad():
        pred, _ = gnn(g)          # forward returns (predictions, embeddings)
        return pred.view(-1)[0]
```

```python
# CELL 7: Evaluate baselines on the test set -> Pareto curve
import numpy as np, pandas as pd
from model.physics_loss import MetricsComputer

def run_baseline(name, mask_fn, frac):
    preds, targets, keeps = [], [], []
    for b in test_loader:
        b = b.to(device)
        for i in range(b.num_graphs):
            g = b.get_example(i)
            pos = g.pos if hasattr(g, "pos") else g.x[:, :3]
            if name == "grad_saliency":
                mask = mask_fn(g.edge_index, g.edge_attr, pos, frac,
                               model=gnn, x=g.x)
            else:
                mask = mask_fn(g.edge_index, g.edge_attr, pos, frac)
            mask = mask.to(device)
            preds.append(pred_for_mask(g, mask).item())
            targets.append(g.y.item())
            keeps.append(mask.float().mean().item())
    preds, targets = torch.tensor(preds), torch.tensor(targets)
    m = MetricsComputer.compute_all(preds, targets)
    return {"method": name, "frac": frac, "rmse": m["rmse"],
            "r2": m["r2"], "scatter": m["scatter"], "keep_frac": np.mean(keeps)}

results = []
fractions = [0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
for frac in fractions:
    for name, mask_fn in [("random", random_mask), ("degree", degree_mask),
                          ("distance", distance_mask), ("mass_ratio", mass_ratio_mask),
                          ("grad_saliency", gradient_saliency_mask)]:
        results.append(run_baseline(name, mask_fn, frac))

# full-graph reference (once, not per-fraction)
full_preds, full_targets = [], []
for b in test_loader:
    b = b.to(device)
    for i in range(b.num_graphs):
        g = b.get_example(i)
        full_preds.append(pred_for_mask(g, torch.ones(g.edge_index.shape[1], dtype=torch.bool, device=device)).item())
        full_targets.append(g.y.item())
full_m = MetricsComputer.compute_all(torch.tensor(full_preds), torch.tensor(full_targets))
results.append({"method": "full", "frac": 1.0, "rmse": full_m["rmse"],
                "r2": full_m["r2"], "scatter": full_m["scatter"], "keep_frac": 1.0})

baseline_df = pd.DataFrame(results)
import os; os.makedirs("outputs/rls", exist_ok=True)
baseline_df.to_csv("outputs/rls/baselines.csv", index=False)
print(baseline_df.to_string(index=False))
print("\nFull-graph reference RMSE:", round(full_m["rmse"], 4))
```

```python
# CELL 8: Save baselines plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4.5))
for name, grp in baseline_df[baseline_df.method != "full"].groupby("method"):
    plt.plot(grp.keep_frac, grp.rmse, "o-", label=name)
plt.axhline(full_m["rmse"], color="k", ls="--", label="full graph")
plt.xlabel("mean keep fraction"); plt.ylabel("RMSE (dex)"); plt.legend(); plt.grid(alpha=0.3)
plt.savefig("outputs/rls/baselines_pareto.png", dpi=150, bbox_inches="tight")
print("saved outputs/rls/baselines_pareto.png")
```

---

# NOTEBOOK B — Offline Policy Training (Stage A) + GNN Fine-tune (Stage B, optional)

*(Cells 1–5 identical to Notebook A. Start from Cell 6 below.)*

```python
# CELL 6: Precompute frozen node embeddings + graph contexts (committed adapter)
from rls.train_policy import prepare_graphs
train_graphs = prepare_graphs(train_loader, gnn, device)
val_graphs   = prepare_graphs(val_loader,   gnn, device)
test_graphs  = prepare_graphs(test_loader,  gnn, device)
print("train graphs:", len(train_graphs), "| ctx dim:", train_graphs[0]["ctx"].shape)
```

```python
# CELL 7: Policy + value net + PG trainer (REINFORCE + learned baseline, not PPO)
from rls.policy import build_policy
from rls.policy_gradient import PolicyGradientTrainer, ValueNet
rls = cfg["rls"]
policy = build_policy(cfg, node_emb_dim=gnn.output_dim).to(device)
value_net = ValueNet(gnn.output_dim).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=rls["lr"])
vopt = torch.optim.Adam(value_net.parameters(), lr=rls["lr"])
trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, rls)
```

```python
# CELL 8: GNN adapter — full + pruned predictions for a graph dict
def gnns_adapter(g, mask):
    def run(edge_index, edge_attr):
        d = pg.Data(x=g["x"], edge_index=edge_index, edge_attr=edge_attr)
        d.batch = torch.zeros(d.x.shape[0], dtype=torch.long, device=d.x.device)
        with torch.no_grad():
            pred, _ = gnn(d)      # forward returns (predictions, embeddings)
            return pred.view(-1)
    mask = mask.to(device)
    pred_full = run(g["edge_index"], g["edge_attr"])
    pred_pruned = run(g["edge_index"][:, mask], g["edge_attr"][mask])
    return pred_full, pred_pruned
```

```python
# CELL 9: Policy-gradient training loop (Stage A) — ~20-30 min on T4
# REINFORCE + learned baseline (one-step MDP: no GAE, no importance ratio).
# Gradients flow to the policy: the policy forward runs OUTSIDE torch.no_grad().
from rls.train_policy import train_policy
log_rows = []
def log_fn(epoch, target_sp, loss):
    log_rows.append([epoch, target_sp, loss])
    print(f"epoch {epoch}: target_sparsity={target_sp:.3f} loss={loss:.4f}")
losses = train_policy(trainer, train_graphs, gnns_adapter, rls, device,
                      epochs=rls["epochs"], log_fn=log_fn)
import os; os.makedirs("outputs/rls", exist_ok=True)
torch.save(policy.state_dict(), "outputs/rls/policy.pt")
torch.save(value_net.state_dict(), "outputs/rls/value_net.pt")
pd.DataFrame(log_rows, columns=["epoch", "target_sparsity", "loss"]).to_csv(
    "outputs/rls/training_log.csv", index=False)
print("Stage A done.")
```

> **PITFALL NOTE:** if the `keep` fraction stays pinned at the `min_keep_frac`
> floor for 5+ epochs, the policy is collapsing — raise `w_conn` to 2.0 and
> restart. If val RMSE explodes (>0.25 dex), the sparsity curriculum is too
> aggressive — slow the anneal (`sparsity_anneal_epochs` 40 -> 50).

```python
# CELL 10: Stage B (optional) — fine-tune GNN on policy-pruned graphs
from rls.sparsify import hard_mask, repair_connectivity
from rls.stageb import fine_tune_gnn
masks = []
with torch.no_grad():
    for g in train_graphs:
        p = torch.sigmoid(policy(g["edge_attr"].to(device), g["emb"].to(device),
                                 g["edge_index"].to(device), g["ctx"].to(device))).squeeze(-1)
        masks.append(repair_connectivity(g["edge_index"].to(device),
                                         hard_mask(p, rls["min_keep_frac"])))
history = fine_tune_gnn(gnn, train_graphs, masks,
                        epochs=rls.get("stageb_epochs", 10), lr=1e-4, device=device)
torch.save(gnn.state_dict(), "outputs/rls/finetuned_gnn.pt")
print("Stage B done -> outputs/rls/finetuned_gnn.pt")
```

```python
# CELL 11: Verify final policy (+ fine-tuned GNN) on the test set
from model.physics_loss import MetricsComputer
preds_pol, preds_full, targets, keeps = [], [], [], []
with torch.no_grad():
    for g in test_graphs:
        gd = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
        p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"], gd["edge_index"], gd["ctx"])).squeeze(-1)
        m = repair_connectivity(gd["edge_index"], hard_mask(p, rls["min_keep_frac"]))
        preds_pol.append(gnns_adapter(gd, m)[1].item())
        preds_full.append(gnns_adapter(gd, torch.ones(gd["edge_index"].shape[1], dtype=torch.bool, device=device))[0].item())
        targets.append(gd["y"].view(-1)[0].item())
        keeps.append(m.float().mean().item())
preds_pol = torch.tensor(preds_pol); preds_full = torch.tensor(preds_full); targets = torch.tensor(targets)
mp, mf = MetricsComputer.compute_all(preds_pol, targets), MetricsComputer.compute_all(preds_full, targets)
print(f"FULL  graph: RMSE={mf['rmse']:.4f} R2={mf['r2']:.4f}")
print(f"RL    pruned: RMSE={mp['rmse']:.4f} R2={mp['r2']:.4f} keep={np.mean(keeps):.3f}")
fid = np.corrcoef(preds_pol.numpy(), preds_full.numpy())[0, 1]
print(f"fidelity (Pearson): {fid:.4f}")
```

---

# NOTEBOOK C — Physics Validation, Cross-Sim, Plots

*(Cells 1–5 of A, then B's cells 6–9, then:)*

```python
# CELL 12: Physics alignment — binding energy vs keep probability
from scipy.stats import spearmanr, mannwhitneyu
from rls.rewards import virial_ratio_pruned
from rls.train_policy import _graph_physics_terms
G = 4.302e-9
rhos, pvals, virial_ratios, keeps, dists_kept, dists_dropped = [], [], [], [], [], []
with torch.no_grad():
    for g in test_graphs:
        gd = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
        p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"], gd["edge_index"], gd["ctx"])).squeeze(-1)
        m = repair_connectivity(gd["edge_index"], hard_mask(p, rls["min_keep_frac"]))
        u, v = gd["edge_index"]
        stellar = gd["stellar_mass"]; pos = gd["pos"]
        r = torch.norm(pos[u] - pos[v], dim=1).clamp(min=1e-6)
        u_ij = G * stellar[u] * stellar[v] / r
        rho, pv = spearmanr(p.cpu().numpy(), u_ij.cpu().numpy())
        rhos.append(rho); pvals.append(pv)
        ke, pe = _graph_physics_terms(gd, gd["edge_index"], m)
        virial_ratios.append(virial_ratio_pruned(ke, pe).item())
        keeps.append(m.float().mean().item())
        dists_kept.extend(r[m].cpu().tolist()); dists_dropped.extend(r[~m].cpu().tolist())
print(f"Spearman rho vs U_ij: mean={np.mean(rhos):.3f} (p median={np.median(pvals):.2e})")
print(f"virial ratio (pruned): median={np.median(virial_ratios):.3f}")
mw = mannwhitneyu(dists_kept, dists_dropped, alternative="less")
print(f"kept edges shorter than dropped? p={mw.pvalue:.2e}")
```

```python
# CELL 13: Uncertainty calibration — MC-dropout coverage before/after pruning
def coverage(model, graphs, device, n_samples=30, policy_pruned=False):
    covered, total = 0, 0
    with torch.no_grad():
        for g in graphs:
            gd = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
            if policy_pruned:
                p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"], gd["edge_index"], gd["ctx"])).squeeze(-1)
                m = repair_connectivity(gd["edge_index"], hard_mask(p, rls["min_keep_frac"]))
                d = pg.Data(x=gd["x"], edge_index=gd["edge_index"][:, m], edge_attr=gd["edge_attr"][m])
            else:
                d = pg.Data(x=gd["x"], edge_index=gd["edge_index"], edge_attr=gd["edge_attr"])
            d.batch = torch.zeros(d.x.shape[0], dtype=torch.long, device=device)
            out = model.predict_with_uncertainty(d, n_samples=n_samples)
            lo, hi = out["mean"] - 1.96 * out["std"], out["mean"] + 1.96 * out["std"]
            covered += int(((gd["y"] >= lo) & (gd["y"] <= hi)).item()); total += 1
    return covered / max(1, total)

print("full-graph 95% CI coverage:", round(coverage(gnn, test_graphs, device), 3))
print("pruned 95% CI coverage:", round(coverage(gnn, test_graphs, device, policy_pruned=True), 3))
```

```python
# CELL 14: Cross-simulation OOD — TNG-trained policy on CAMELS
# NOTE: requires the REAL CAMELS HDF5 (downloads ~100 MB from Flatiron).
# The loader's synthetic fallback is NOT publishable and is guarded here.
import yaml
from data.loaders.base_loader import get_loader
from graph.graph_builder import GraphBuilder
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import global_mean_pool

cfg2 = yaml.safe_load(open("config/config.yaml"))
cfg2["data"]["source"] = "camels"
cfg2["data"]["camels"] = {"suite": "IllustrisTNG", "simulation": "LH_0",
                          "cache_dir": "/kaggle/working/camels_cache"}
cfg2["data"]["num_workers"] = 0
try:
    loader2 = get_loader(cfg2)
    halos2 = loader2.load()
    assert getattr(loader2, "used_synthetic_fallback", False) is False, \
        "synthetic CAMELS fallback is not publishable"
    gb2 = GraphBuilder(cfg2)
    graphs2 = gb2.build_graphs(halos2[:100])
    predsF, predsP, ys = [], [], []
    with torch.no_grad():
        for g in graphs2:
            g = g.to(device)
            gb = PyGBatch.from_data_list([g])
            emb = gnn.get_embeddings(gb, embedding_point="pre_pooling")
            ctx = global_mean_pool(emb, gb.batch).squeeze(0)
            p = torch.sigmoid(policy(g.edge_attr, emb, g.edge_index, ctx)).squeeze(-1)
            m = repair_connectivity(g.edge_index, hard_mask(p, rls["min_keep_frac"]))
            pf, _ = gnn(gb)
            dp = pg.Data(x=g.x, edge_index=g.edge_index[:, m], edge_attr=g.edge_attr[m])
            dp.batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
            pp, _ = gnn(dp)
            predsF.append(pf.view(-1)[0].item()); predsP.append(pp.view(-1)[0].item()); ys.append(g.y.item())
    mf2 = MetricsComputer.compute_all(torch.tensor(predsF), torch.tensor(ys))
    mp2 = MetricsComputer.compute_all(torch.tensor(predsP), torch.tensor(ys))
    print(f"CROSS-SIM (TNG->CAMELS): full RMSE={mf2['rmse']:.4f} | RL-pruned RMSE={mp2['rmse']:.4f}")
    pd.DataFrame({"y": ys, "pred_full": predsF, "pred_pruned": predsP}).to_csv(
        "outputs/rls/cross_sim_results.csv", index=False)
except Exception as e:
    print("cross-sim failed/skipped (real HDF5 required):", e)
```

```python
# CELL 15: Final results table + paper plots + download
rows = []
bl = pd.read_csv("outputs/rls/baselines.csv")
for _, r in bl.iterrows():
    rows.append({"method": r.method, "rmse": r.rmse, "r2": r.r2,
                 "scatter": r.scatter, "keep_frac": r.keep_frac, "fidelity": float("nan")})
rows.append({"method": "rl_policy", "rmse": mp["rmse"], "r2": mp["r2"],
             "scatter": mp["scatter"], "keep_frac": float(np.mean(keeps)),
             "fidelity": fid})
res = pd.DataFrame(rows)
res.to_csv("outputs/rls/results_table.csv", index=False)
print(res.to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name in ["full", "random", "degree", "distance", "mass_ratio", "rl_policy"]:
    sub = res[res.method == name]
    if not sub.empty:
        axes[0].plot(sub.keep_frac, sub.rmse, "o-", label=name)
axes[0].set_xlabel("mean keep fraction"); axes[0].set_ylabel("RMSE (dex)")
axes[0].set_xlim(1.05, -0.05); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].scatter(targets, preds_full, s=15, alpha=0.5, label="full")
axes[1].scatter(targets, preds_pol, s=15, alpha=0.5, marker="x", label="rl-pruned")
axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], "k--")
axes[1].set_xlabel("true log M_halo"); axes[1].set_ylabel("predicted"); axes[1].legend()
fig.tight_layout(); fig.savefig("outputs/rls/paper_figures.png", dpi=200)
import shutil
shutil.make_archive("/kaggle/working/rls_outputs", "zip", "outputs/rls")
print("saved outputs/rls/* + /kaggle/working/rls_outputs.zip")
```

---

# NOTEBOOK D — RL at Inference (TTA): the headline results

*(Cells 1–5 of A, then B's cells 6–9 — you need the trained offline policy.
Do NOT run Stage B for TTA: TTA adapts the policy around the frozen backbone.)*

The method: at inference, for EACH input graph, a copy of the policy takes K
policy-gradient steps against a **label-free** reward (MC-dropout uncertainty
reduction + sparsity + connectivity + virial), then emits the final mask.
Labels are used ONLY for the final evaluation RMSE — never in the reward.
FROZEN mode (K=0) is the ablation.

```python
# CELL D1: Import TTA engine (committed) + label-free reward
from rls.tta import adapt_at_test_time, mc_std
from rls.evaluate import evaluate_tta
import copy, time
```

```python
# CELL D2: In-domain TTA — frozen (K=0) vs TTA (K ablation) on the TNG test set
rls = cfg["rls"]
tta_rows = []
for K in [0, 5, 10, 20]:
    tta_cfg = dict(rls); tta_cfg["tta_steps"] = K
    preds, fulls, ys, keeps, steps, secs = [], [], [], [], [], []
    for g in test_graphs:
        gd = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
        t0 = time.time()
        if K == 0:  # FROZEN mode
            with torch.no_grad():
                p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"], gd["edge_index"], gd["ctx"])).squeeze(-1)
                m = repair_connectivity(gd["edge_index"], hard_mask(p, rls["min_keep_frac"]))
            info = {"steps_run": 0, "reward_hist": []}
        else:
            m, info = adapt_at_test_time(policy, gd, gnn, tta_cfg, device,
                                         init="offline",
                                         target_sparsity=rls["tta_target_sparsity"])
        secs.append(time.time() - t0)
        preds.append(gnns_adapter(gd, m)[1].item())
        fulls.append(gnns_adapter(gd, torch.ones(gd["edge_index"].shape[1], dtype=torch.bool, device=device))[0].item())
        ys.append(gd["y"].view(-1)[0].item())
        keeps.append(float(m.float().mean())); steps.append(info["steps_run"])
    preds, fulls, ys = np.array(preds), np.array(fulls), np.array(ys)
    row = {"mode": "frozen" if K == 0 else "tta", "K": K,
           "rmse": float(np.sqrt(((preds - ys) ** 2).mean())),
           "fidelity": float(np.corrcoef(preds, fulls)[0, 1]),
           "keep_frac": float(np.mean(keeps)), "mean_steps": float(np.mean(steps)),
           "mean_time_s": float(np.mean(secs))}
    tta_rows.append(row)
    print(row)
tta_df = pd.DataFrame(tta_rows)
tta_df.to_csv("outputs/rls/tta_indomain.csv", index=False)
frozen_rmse = tta_df.loc[tta_df.K == 0, "rmse"].iloc[0]
print("frozen (K=0) RMSE:", frozen_rmse, "(this is the TTA ablation baseline)")
```

```python
# CELL D3: OOD TTA — TNG-trained policy on CAMELS, frozen vs TTA (THE headline)
# Requires REAL CAMELS HDF5 (synthetic fallback is NOT publishable).
cfg2 = yaml.safe_load(open("config/config.yaml"))
cfg2["data"]["source"] = "camels"
cfg2["data"]["camels"] = {"suite": "IllustrisTNG", "simulation": "LH_0",
                          "cache_dir": "/kaggle/working/camels_cache"}
cfg2["data"]["num_workers"] = 0
try:
    loader2 = get_loader(cfg2)
    halos2 = loader2.load()
    assert getattr(loader2, "used_synthetic_fallback", False) is False, \
        "synthetic CAMELS fallback is not publishable"
    gb2 = GraphBuilder(cfg2)
    graphs2 = gb2.build_graphs(halos2[:100])
    camels_graphs = []
    with torch.no_grad():
        for g in graphs2:
            g = g.to(device)
            gb = PyGBatch.from_data_list([g])
            emb = gnn.get_embeddings(gb, embedding_point="pre_pooling")
            ctx = global_mean_pool(emb, gb.batch).squeeze(0)
            n = g.x.shape[0]
            camels_graphs.append({
                "x": g.x, "edge_index": g.edge_index, "edge_attr": g.edge_attr, "y": g.y,
                "ctx": ctx, "emb": emb,
                "stellar_mass": getattr(g, "stellar_mass", torch.ones(n, device=device) * 1e10),
                "vel_disp": getattr(g, "vel_disp", torch.ones(n, device=device) * 100),
                "half_mass_r": getattr(g, "half_mass_r", torch.ones(n, device=device) * 0.01),
                "pos": getattr(g, "pos", torch.zeros(n, 3, device=device)),
            })
    ood_rows = []
    for mode in ["frozen", "tta"]:
        preds, ys = [], []
        for g in camels_graphs:
            gd = {k: v for k, v in g.items() if isinstance(v, torch.Tensor)}
            if mode == "frozen":
                with torch.no_grad():
                    p = torch.sigmoid(policy(gd["edge_attr"], gd["emb"], gd["edge_index"], gd["ctx"])).squeeze(-1)
                    m = repair_connectivity(gd["edge_index"], hard_mask(p, rls["min_keep_frac"]))
            else:
                m, info = adapt_at_test_time(policy, gd, gnn, rls, device, init="offline",
                                             target_sparsity=rls["tta_target_sparsity"])
            preds.append(gnns_adapter(gd, m)[1].item()); ys.append(float(gd["y"].view(-1)[0]))
        preds, ys = np.array(preds), np.array(ys)
        row = {"mode": mode, "rmse": float(np.sqrt(((preds - ys) ** 2).mean())),
               "r2": float(1 - ((preds - ys) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())}
        ood_rows.append(row); print("CAMELS", row)
    pd.DataFrame(ood_rows).to_csv("outputs/rls/tta_camels.csv", index=False)
    print("HEADLINE: TTA recovers", round(ood_rows[0]["rmse"] - ood_rows[1]["rmse"], 4),
          "dex of the frozen policy's OOD degradation")
except Exception as e:
    print("CAMELS OOD skipped/failed:", e)
    print("If this is the synthetic fallback, do NOT put these numbers in the paper.")
```

```python
# CELL D4: TTA plots + reward trajectories + save
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(tta_df.K.astype(str), tta_df.rmse, color=["gray"] + ["steelblue"] * 3)
axes[0].set_xlabel("TTA steps K (0 = frozen)"); axes[0].set_ylabel("test RMSE (dex)")
axes[0].set_title("In-domain: TTA vs frozen")
axes[1].bar(tta_df.K.astype(str), tta_df.mean_time_s, color=["gray"] + ["darkorange"] * 3)
axes[1].set_xlabel("TTA steps K"); axes[1].set_ylabel("adaptation time (s/graph)")
axes[1].set_title("TTA cost")
fig.tight_layout(); fig.savefig("outputs/rls/tta_results.png", dpi=200)
import shutil
shutil.make_archive("/kaggle/working/rls_outputs", "zip", "outputs/rls")
print("saved outputs/rls/tta_*.csv + tta_results.png — download rls_outputs.zip")
```

> **TTA pitfalls (plan Part 4):** if TTA's test RMSE is WORSE than frozen while
> Δunc is positive, the uncertainty reward is being gamed → raise
> `w_virial`/`w_conn` or lower K. Tune (K, tta_lr, w_unc) on the VAL split only.
> K=0 must reproduce the frozen numbers exactly (same policy, same mask).
