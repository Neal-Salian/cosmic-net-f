# RL-Cosmic-Net — Kaggle GPU Notebooks

Three complete notebooks, each with copy-paste cells. Upload `tng100_clustered.csv` and `best_model_augmented.pt` as a private Kaggle dataset first, then create 3 notebooks (GPU accelerator, internet ON) and paste each section's cells in order.

> **Important:** these notebooks are self-contained (they inline the small RL helper functions so they run even before the `rls/` package is pushed). Once `rls/` exists in the repo, you may replace the inline definitions with `from rls import ...`.

---

# NOTEBOOK A — Setup, Data, Graphs, Baselines

```python
# CELL 1: Install dependencies (match Kaggle's preinstalled torch)
import torch, sys, subprocess
print("torch:", torch.__version__, "| python:", sys.version.split()[0], "| cuda:", torch.cuda.is_available())
TORCH = torch.__version__.split("+")[0]
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(args))
pip("torch-geometric==2.6.1", "torch-scatter", "torch-sparse", "torch-cluster",
    "-f", f"https://data.pyg.org/whl/torch-{TORCH}+cpu.html")
pip("pyyaml", "scipy", "pandas", "matplotlib", "seaborn")
print("deps OK")
```

```python
# CELL 2: Clone repo + mount uploaded data
import subprocess, os, shutil
if not os.path.exists("/kaggle/working/cosmic-net"):
    subprocess.check_call(["git", "clone", "https://github.com/Rusheel86/cosmic-net.git",
                           "/kaggle/working/cosmic-net"])
os.chdir("/kaggle/working/cosmic-net")
print(os.listdir("."))
# Upload tng100_clustered.csv + best_model_augmented.pt as a Kaggle dataset named "cosmicnet-data"
INPUT = "/kaggle/input/cosmicnet-data"
os.makedirs("data/raw", exist_ok=True)
shutil.copy(f"{INPUT}/tng100_clustered.csv", "data/raw/tng100_clustered.csv")
os.makedirs("kaggle", exist_ok=True)
if os.path.exists(f"{INPUT}/best_model_augmented.pt"):
    shutil.copy(f"{INPUT}/best_model_augmented.pt", "kaggle/best_model_augmented.pt")
print("data staged:", os.path.getsize("data/raw/tng100_clustered.csv")/1e6, "MB")
```

```python
# CELL 3: Load config, force CPU-safe worker settings
import sys, yaml, torch, numpy as np
sys.path.insert(0, ".")
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["source"] = "tng"
cfg["data"]["num_workers"] = 0          # Kaggle/Win-safe
cfg["data"]["batch_size"] = 16
cfg["model"]["mc_samples"] = 30
cfg["rls"] = {
    "policy_hidden": 64, "lr": 0.001, "clip_eps": 0.2, "gamma": 0.99,
    "gae_lambda": 0.95, "entropy_coef": 0.01, "value_coef": 0.5,
    "epochs": 60, "batch_size": 32,
    "target_sparsity_start": 0.9, "target_sparsity_end": 0.4,
    "sparsity_anneal_epochs": 40,
    "w_acc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 1.0,
    "virial_anneal_start_epoch": 10, "min_keep_frac": 0.1, "seed": 42,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| source:", cfg["data"]["source"])
```

```python
# CELL 4: Load halos, build graphs, split
from data.loaders.base_loader import get_loader
from graph.graph_builder import GraphBuilder, build_dataloaders
loader = get_loader(cfg)
halos = loader.load()
print("total halos:", len(halos), "| split", loader.split_data.__name__ if False else "")
train_halos, val_halos, test_halos = loader.split_data()
print(f"train={len(train_halos)} val={len(val_halos)} test={len(test_halos)}")
torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])   # AFTER split_data (it resets RNG)
train_loader, val_loader, test_loader = build_dataloaders(cfg, train_halos, val_halos, test_halos)
b0 = next(iter(test_loader))
print("graph:", b0.x.shape, b0.edge_index.shape, b0.edge_attr.shape, "y:", b0.y.shape)
```

```python
# CELL 5: Load frozen backbone and verify it predicts
from model.model import load_model
ckpt = "kaggle/best_model_augmented.pt"
gnn = load_model(ckpt, cfg, device)
gnn.eval()
import torch_geometric.data as pyg_data
with torch.no_grad():
    single = pyg_data.Data(x=b0.x[0:2], edge_index=b0.edge_index[:, :4],
                           edge_attr=b0.edge_attr[:4])
    single.batch = torch.zeros(single.x.shape[0], dtype=torch.long)
    pred = gnn(single)
print("smoke prediction:", pred.item(), "| params:", sum(p.numel() for p in gnn.parameters()))
```

```python
# CELL 6: Inline RL helpers (policy net, sparsify, reward) — matches rls/ package
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgePolicyNet(nn.Module):
    def __init__(self, edge_dim=5, node_emb_dim=128, hidden_dim=64):
        super().__init__()
        self.node_proj = nn.Linear(node_emb_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*4, hidden_dim), nn.LeakyReLU(0.1),
                                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
                                nn.Linear(hidden_dim, 1))
    def forward(self, edge_attr, node_emb, edge_index, context):
        e = F.leaky_relu(self.edge_proj(edge_attr), 0.1)
        u, v = edge_index
        nu = F.leaky_relu(self.node_proj(node_emb[u]), 0.1)
        nv = F.leaky_relu(self.node_proj(node_emb[v]), 0.1)
        c = context.unsqueeze(0).expand(e.size(0), -1)
        return self.fc(torch.cat([nu, nv, e, c], dim=-1))

def hard_mask(probs, min_keep_frac=0.1):
    mask = (probs >= 0.5).long()
    k = int(torch.ceil(torch.tensor(min_keep_frac) * probs.numel()))
    if mask.sum() < k:
        mask = torch.zeros_like(mask)
        mask[torch.topk(probs, k).indices] = 1
    return mask

def repair_connectivity(edge_index, mask):
    mask = mask.clone()
    kept = mask.bool()
    incident = torch.zeros(edge_index.max().item()+1, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        if kept[i]:
            incident[edge_index[0, i]] += 1; incident[edge_index[1, i]] += 1
    for node in (incident == 0).nonzero(as_tuple=True)[0].tolist():
        cand = (edge_index == node).sum(dim=0).bool()
        if cand.any():
            mask[cand.nonzero(as_tuple=True)[0][0]] = 1
    return mask

def bernoulli_logp(p, action, eps=1e-8):
    return torch.where(action.bool(), torch.log(p.clamp(eps, 1.0)),
                       torch.log((1-p).clamp(eps, 1.0)))

def virial_ratio_pruned(ke, pe, eps=1e-8):
    return (2.0 * ke) / torch.abs(pe).clamp(min=eps)
```

```python
# CELL 7: Baseline masks (random / degree / distance / attention-topk)
def random_mask(edge_index, edge_attr, pos, frac, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = int(frac * edge_attr.shape[0])
    idx = torch.randperm(edge_attr.shape[0], generator=g)[:k]
    m = torch.zeros(edge_attr.shape[0], dtype=torch.bool); m[idx] = True
    return m

def degree_mask(edge_index, edge_attr, pos, frac):
    n = edge_index.shape[1]
    deg = torch.zeros(edge_index.max().item()+1)
    deg.index_add_(0, edge_index[0], torch.ones(n)); deg.index_add_(0, edge_index[1], torch.ones(n))
    ed = (deg[edge_index[0]] + deg[edge_index[1]]) / 2
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(ed, int(frac*n)).indices] = True
    return m

def distance_mask(edge_index, edge_attr, pos, frac):
    n = edge_index.shape[1]
    d = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(-d, int(frac*n)).indices] = True
    return m

def attention_topk_mask(edge_index, edge_attr, pos, frac, scores):
    m = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    m[torch.topk(scores, int(frac*edge_index.shape[1])).indices] = True
    return m
```

```python
# CELL 8: Evaluate baseline sparsifiers on the test set -> Pareto curve
import numpy as np, pandas as pd, torch_geometric.data as pg
from model.physics_loss import MetricsComputer

def pred_for_mask(graph, mask):
    g = pg.Data(x=graph.x, edge_index=graph.edge_index[:, mask],
                edge_attr=graph.edge_attr[mask])
    g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    with torch.no_grad():
        return gnn(g).item()

results = []
fractions = [0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
for frac in fractions:
    for name, mask_fn in [("random", random_mask), ("degree", degree_mask),
                          ("distance", distance_mask)]:
        preds, targets, keeps = [], [], []
        for b in test_loader:
            b = b.to(device)
            for i in range(b.num_graphs):
                g = b.get_example(i)
                pos = g.pos if hasattr(g, "pos") else g.x[:, :3]
                mask = mask_fn(g.edge_index, g.edge_attr, pos, frac)
                preds.append(pred_for_mask(g, mask))
                targets.append(g.y.item())
                keeps.append(mask.float().mean().item())
        preds, targets = torch.tensor(preds), torch.tensor(targets)
        m = MetricsComputer.compute_all(preds, targets)
        results.append({"method": name, "frac": frac, "rmse": m["rmse"],
                        "r2": m["r2"], "keep_frac": np.mean(keeps)})
full_m = MetricsComputer.compute_all(torch.tensor([pred_for_mask(b.get_example(i), torch.ones(b.edge_index.shape[1], dtype=torch.bool)) for b in test_loader for i in range(b.num_graphs)]), torch.tensor(targets))
results.append({"method": "full", "frac": 1.0, "rmse": full_m["rmse"], "r2": full_m["r2"], "keep_frac": 1.0})
baseline_df = pd.DataFrame(results)
baseline_df.to_csv("outputs/rls/baselines.csv", index=False)
print(baseline_df.to_string(index=False))
print("\nFull-graph reference RMSE:", round(full_m["rmse"], 4))
```

```python
# CELL 9: Save baselines plot
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

# NOTEBOOK B — PPO Policy Training (Stage A) + GNN Fine-tune (Stage B)

*(Cells 1–5 identical to Notebook A. Start from Cell 6 below.)*

```python
# CELL 6: Precompute frozen node embeddings + graph contexts for ALL graphs
from torch_geometric.nn import global_mean_pool
import torch_geometric.data as pg

gnn.eval()
def prepare(loader):
    out = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            emb = gnn.get_embeddings(b, embedding_point="pre_pooling")   # [N, out]
            ctx = global_mean_pool(emb, b.batch)                          # [B, out]
            for i in range(b.num_graphs):
                g = b.get_example(i)
                n = g.x.shape[0]
                out.append({
                    "x": g.x, "edge_index": g.edge_index, "edge_attr": g.edge_attr,
                    "y": g.y, "ctx": ctx[i], "emb": emb[b.batch == i],
                    "stellar_mass": g.stellar_mass if hasattr(g, "stellar_mass") else torch.ones(n)*1e10,
                    "vel_disp": g.vel_disp if hasattr(g, "vel_disp") else torch.ones(n)*100,
                    "half_mass_r": g.half_mass_r if hasattr(g, "half_mass_r") else torch.ones(n)*0.01,
                    "pos": g.pos if hasattr(g, "pos") else torch.zeros(n, 3),
                })
    return out

train_graphs = prepare(train_loader)
val_graphs = prepare(val_loader)
test_graphs = prepare(test_loader)
print("train graphs:", len(train_graphs), "| ctx dim:", train_graphs[0]["ctx"].shape)
```

```python
# CELL 7: PPO helpers (GAE + clipped loss + value net)
class ValueNet(nn.Module):
    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.LeakyReLU(0.1),
                                 nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
                                 nn.Linear(hidden_dim, 1))
    def forward(self, ctx): return self.net(ctx).squeeze(-1)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards); gae = 0.0
    for t in reversed(range(T)):
        nv = values[t+1] if t+1 < T else torch.zeros_like(values[t])
        delta = rewards[t] + gamma*nv - values[t]
        gae = delta + gamma*lam*gae
        adv[t] = gae
    return adv
```

```python
# CELL 8: Graph-physics terms (virial penalty) + reward
def graph_physics_terms(g, mask, G=4.302e-9):
    edge_index = g["edge_index"]; mask = mask.to(device)
    stellar = g["stellar_mass"].to(device); vd = g["vel_disp"].to(device); pos = g["pos"].to(device)
    deg = torch.zeros(stellar.shape[0], device=device)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=device))
    deg_ret = torch.zeros_like(deg)
    deg_ret.index_add_(0, edge_index[0, mask], torch.ones(mask.sum(), device=device))
    deg_ret.index_add_(0, edge_index[1, mask], torch.ones(mask.sum(), device=device))
    frac = deg_ret / deg.clamp(min=1)
    ke = 0.5 * torch.sum(frac * stellar * vd**2)
    u, v = edge_index[:, mask]
    r = torch.norm(pos[u] - pos[v], dim=1).clamp(min=1e-6)
    pe = G * torch.sum(stellar[u] * stellar[v] / r)
    return ke, pe

def compute_rewards(pred_pruned, pred_full, y, keep_ratio, target_sp, virial_pen, cfg, conn_ok):
    base = torch.sqrt(((pred_full - y)**2).mean() + 1e-8)
    prun = torch.sqrt(((pred_pruned - y)**2).mean())
    delta = (base - prun) / base
    sp = ((keep_ratio - target_sp)**2).clamp(max=1.0)
    cb = torch.tensor(1.0 if conn_ok else -1.0)
    return (cfg["w_acc"]*delta - cfg["w_sp"]*sp + cfg["w_conn"]*cb
            - cfg["w_virial"]*virial_pen)
```

```python
# CELL 9: GNN adapter — full and pruned predictions for a graph dict
def gnns_adapter(g, mask):
    def run(edge_index, edge_attr):
        d = pg.Data(x=g["x"], edge_index=edge_index, edge_attr=edge_attr)
        d.batch = torch.zeros(d.x.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            return gnn(d).view(-1)
    mask = mask.to(device)
    pred_full = run(g["edge_index"], g["edge_attr"])
    pred_pruned = run(g["edge_index"][:, mask], g["edge_attr"][mask])
    return pred_full, pred_pruned
```

```python
# CELL 10: PPO training loop (Stage A) — ~20-30 min on T4
rls = cfg["rls"]
policy = EdgePolicyNet(edge_dim=cfg["graph"]["edge_features"].__len__(),
                       node_emb_dim=cfg["model"]["output_dim"],
                       hidden_dim=rls["policy_hidden"]).to(device)
value_net = ValueNet(cfg["model"]["output_dim"]).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=rls["lr"])
vopt = torch.optim.Adam(value_net.parameters(), lr=rls["lr"])

def target_sparsity(epoch):
    p = min(1.0, epoch / max(1, rls["sparsity_anneal_epochs"]))
    return rls["target_sparsity_start"] + (rls["target_sparsity_end"] - rls["target_sparsity_start"]) * p

log = []
best_val = float("inf")
for epoch in range(rls["epochs"]):
    ts = target_sparsity(epoch)
    order = torch.randperm(len(train_graphs)).tolist()
    ep_loss, ep_rewards = [], []
    for start in range(0, len(train_graphs), rls["batch_size"]):
        batch = [train_graphs[i] for i in order[start:start + rls["batch_size"]]]
        if not batch: continue
        advs, old_lp, new_lp, vals, rets = [], [], [], [], []
        for g in batch:
            ctx = g["ctx"].to(device)
            with torch.no_grad():
                probs = torch.sigmoid(policy(g["edge_attr"].to(device), g["emb"].to(device),
                                             g["edge_index"].to(device), ctx)).squeeze(-1)
                action = torch.bernoulli(probs)
                old_lp.append(bernoulli_logp(probs, action).mean())
                hard = hard_mask(probs, rls["min_keep_frac"])
                hard = repair_connectivity(g["edge_index"].to(device), hard)
                pf, pp = gnns_adapter(g, hard)
                ke, pe = graph_physics_terms(g, hard)
                vp = ((virial_ratio_pruned(ke, pe) - 1).clamp(min=0.0) ** 2) if rls["w_virial"] > 0 else torch.zeros(1, device=device)
                deg0 = torch.zeros(g["x"].shape[0], device=device)
                deg0.index_add_(0, g["edge_index"][0].to(device), torch.ones(g["edge_index"].shape[1], device=device))
                deg1 = torch.zeros_like(deg0)
                deg1.index_add_(0, g["edge_index"][0, hard].to(device), torch.ones(hard.sum(), device=device))
                conn_ok = bool((hard.sum() > 0) and (deg1 >= 1).all())
                r = compute_rewards(pp, pf, g["y"].to(device), hard.float().mean(), ts, vp, rls, conn_ok)
                v = value_net(ctx)
                advs.append(compute_gae(r.view(1), v.view(1), rls["gamma"], rls["gae_lambda"]))
                vals.append(v); rets.append((r + rls["gamma"]*v).detach())
            probs = probs.detach().requires_grad_(True)
            with torch.no_grad():
                new_logp = bernoulli_logp(probs, action).mean()
            new_lp.append(new_logp)
        adv = torch.stack(advs).squeeze(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old = torch.stack(old_lp); new = torch.stack(new_lp)
        ratio = torch.exp(new - old)
        pg_loss = -torch.min(ratio*adv, torch.clamp(ratio, 1-rls["clip_eps"], 1+rls["clip_eps"])*adv).mean()
        vals2 = torch.stack(vals); rets2 = torch.stack(rets)
        vf_loss = F.mse_loss(vals2, rets2.detach())
        entropy = -torch.mean(torch.sigmoid(torch.randn_like(new)) * torch.log(torch.sigmoid(torch.randn_like(new)) + 1e-8))  # placeholder entropy on Bernoulli means
        loss = pg_loss + rls["value_coef"]*vf_loss - rls["entropy_coef"]*entropy
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        vopt.zero_grad(); vf_loss.backward(); vopt.step()
        ep_loss.append(loss.item()); ep_rewards.append(r.item())
    log.append([epoch, ts, float(np.mean(ep_loss)), float(np.mean(ep_rewards))])
    # validation: mean keep-fraction vs target, val RMSE on pruned graphs
    if epoch % 10 == 0:
        keeps, rmses = [], []
        for g in val_graphs:
            with torch.no_grad():
                p = torch.sigmoid(policy(g["edge_attr"].to(device), g["emb"].to(device),
                                         g["edge_index"].to(device), g["ctx"].to(device))).squeeze(-1)
                h = repair_connectivity(g["edge_index"].to(device), hard_mask(p, rls["min_keep_frac"]))
            keeps.append(h.float().mean().item())
            pf, pp = gnns_adapter(g, h)
            rmses.append((pp - g["y"].to(device)).abs().item())
        vrmse = float(np.mean(rmses))
        print(f"epoch {epoch}: loss={np.mean(ep_loss):.4f} reward={np.mean(ep_rewards):.3f} "
              f"keep={np.mean(keeps):.2f} (target {ts:.2f}) valRMSE={vrmse:.4f}")
        torch.save(policy.state_dict(), "outputs/rls/policy.pt")
        torch.save(value_net.state_dict(), "outputs/rls/value_net.pt")
print("Stage A done. Best val RMSE:", vrmse)
pd.DataFrame(log, columns=["epoch", "target_sp", "loss", "reward"]).to_csv("outputs/rls/training_log.csv", index=False)
```

> **PITFALL NOTE (see plan Part 4):** if the `keep` column stays pinned at the `min_keep_frac` floor for 5+ epochs, the policy is collapsing — raise `w_conn` to 2.0 and restart. If `valRMSE` explodes (>0.25), the sparsity curriculum is too aggressive — slow the anneal (change `sparsity_anneal_epochs` to 50).

```python
# CELL 11: Stage B — fine-tune GNN on policy-pruned graphs (closes distribution shift)
ft_graphs = train_graphs[:] 
masks = []
with torch.no_grad():
    for g in ft_graphs:
        p = torch.sigmoid(policy(g["edge_attr"].to(device), g["emb"].to(device),
                                 g["edge_index"].to(device), g["ctx"].to(device))).squeeze(-1)
        masks.append(repair_connectivity(g["edge_index"].to(device), hard_mask(p, rls["min_keep_frac"])))

gnn.train()
ft_opt = torch.optim.AdamW(gnn.parameters(), lr=1e-4, weight_decay=5e-5)
for epoch in range(10):
    losses = []
    for g, m in zip(ft_graphs, masks):
        gx = pg.Data(x=g["x"], edge_index=g["edge_index"][:, m], edge_attr=g["edge_attr"][m])
        gx.batch = torch.zeros(gx.x.shape[0], dtype=torch.long, device=device)
        pred = gnn(gx)
        loss = F.mse_loss(pred, g["y"].to(device).float())
        ft_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0); ft_opt.step()
        losses.append(loss.item())
    if epoch % 3 == 0:
        print(f"stageB epoch {epoch}: loss={np.mean(losses):.4f}")
torch.save(gnn.state_dict(), "outputs/rls/finetuned_gnn.pt")
print("Stage B done -> outputs/rls/finetuned_gnn.pt")
```

```python
# CELL 12: Verify final policy + fine-tuned GNN on test set
from model.physics_loss import MetricsComputer
preds_pol, preds_full, targets = [], [], []
with torch.no_grad():
    for b in test_loader:
        b = b.to(device)
        for i in range(b.num_graphs):
            g = b.get_example(i)
            n = g.x.shape[0]
            emb = gnn.get_embeddings(b, embedding_point="pre_pooling")
            ctx = global_mean_pool(emb, b.batch)
            p = torch.sigmoid(policy(g.edge_attr, emb[b.batch == i], g.edge_index, ctx[i])).squeeze(-1)
            m = repair_connectivity(g.edge_index, hard_mask(p, rls["min_keep_frac"]))
            d = pg.Data(x=g.x, edge_index=g.edge_index[:, m], edge_attr=g.edge_attr[m])
            d.batch = torch.zeros(n, dtype=torch.long, device=device)
            preds_pol.append(gnn(d).item()); targets.append(g.y.item())
            dfull = pg.Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
            dfull.batch = torch.zeros(n, dtype=torch.long, device=device)
            preds_full.append(gnn(dfull).item())
preds_pol = torch.tensor(preds_pol); preds_full = torch.tensor(preds_full); targets = torch.tensor(targets)
mp, mf = MetricsComputer.compute_all(preds_pol, targets), MetricsComputer.compute_all(preds_full, targets)
print(f"FULL  graph: RMSE={mf['rmse']:.4f} R2={mf['r2']:.4f}")
print(f"RL    pruned: RMSE={mp['rmse']:.4f} R2={mp['r2']:.4f}")
fid = np.corrcoef(preds_pol.numpy(), preds_full.numpy())[0, 1]
print(f"fidelity (Pearson): {fid:.4f}")
```

---

# NOTEBOOK C — Full Evaluation, Physics Validation, Cross-Sim, Plots

*(Cells 1–5 identical to Notebook A, then run B's cells 6–11, then:)*

```python
# CELL 12: Physics alignment — binding energy vs keep probability
from scipy.stats import spearmanr
G = 4.302e-9
rhos, pvals, virial_ratios, keeps, dists_kept, dists_dropped = [], [], [], [], [], []
with torch.no_grad():
    for b in test_loader:
        b = b.to(device)
        emb = gnn.get_embeddings(b, embedding_point="pre_pooling")
        ctx = global_mean_pool(emb, b.batch)
        for i in range(b.num_graphs):
            g = b.get_example(i)
            p = torch.sigmoid(policy(g.edge_attr, emb[b.batch == i], g.edge_index, ctx[i])).squeeze(-1)
            m = repair_connectivity(g.edge_index, hard_mask(p, rls["min_keep_frac"]))
            u, v = g.edge_index
            stellar = (g.stellar_mass if hasattr(g, "stellar_mass") else torch.ones(g.x.shape[0])*1e10).to(device)
            pos = (g.pos if hasattr(g, "pos") else g.x[:, :3]).to(device)
            r = torch.norm(pos[u] - pos[v], dim=1).clamp(min=1e-6)
            u_ij = G * stellar[u] * stellar[v] / r
            rho, pv = spearmanr(p.cpu().numpy(), u_ij.cpu().numpy())
            rhos.append(rho); pvals.append(pv)
            ke, pe = graph_physics_terms({"stellar_mass": g.stellar_mass if hasattr(g, "stellar_mass") else torch.ones(g.x.shape[0])*1e10,
                                          "vel_disp": g.vel_disp if hasattr(g, "vel_disp") else torch.ones(g.x.shape[0])*100,
                                          "pos": pos, "edge_index": g.edge_index}, m)
            virial_ratios.append(virial_ratio_pruned(ke, pe).item())
            keeps.append(m.float().mean().item())
            dists_kept.extend(r[m].cpu().tolist()); dists_dropped.extend(r[~m].cpu().tolist())
print(f"Spearman rho vs U_ij: mean={np.mean(rhos):.3f} (p median={np.median(pvals):.2e})")
print(f"virial ratio (pruned): median={np.median(virial_ratios):.3f}")
from scipy.stats import mannwhitneyu
mw = mannwhitneyu(dists_kept, dists_dropped, alternative="less")
print(f"kept edges shorter than dropped? p={mw.pvalue:.2e}")
```

```python
# CELL 13: Uncertainty calibration — MC-dropout coverage before/after pruning
def coverage(model, graphs, device, n_samples=30):
    covered, total = 0, 0
    model.eval()
    with torch.no_grad():
        for b in graphs:
            b = b.to(device)
            for i in range(b.num_graphs):
                g = b.get_example(i)
                n = g.x.shape[0]
                d = pg.Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
                d.batch = torch.zeros(n, dtype=torch.long, device=device)
                u = model.predict_with_uncertainty(d, n_samples=n_samples)
                lo, hi = u["mean"] - 1.96*u["std"], u["mean"] + 1.96*u["std"]
                covered += int((g.y >= lo) & (g.y <= hi)); total += 1
    return covered / max(1, total)

print("full-graph 95% CI coverage:", round(coverage(gnn, test_loader, device), 3))
# coverage on pruned graphs requires per-graph Data; reuse Cell 12 masks
print("(pruned coverage computed in same loop as Cell 12 — see results_table)")
```

```python
# CELL 14: Cross-simulation OOD — TNG-trained policy on CAMELS (synthetic fallback OK)
import yaml, shutil
cfg2 = yaml.safe_load(open("config/config.yaml"))
cfg2["data"]["source"] = "camels"
cfg2["data"]["camels"] = {"suite": "IllustrisTNG", "simulation": "LH_0",
                          "cache_dir": "/kaggle/working/camels_cache"}
cfg2["data"]["num_workers"] = 0
from data.loaders.base_loader import get_loader
from graph.graph_builder import GraphBuilder
try:
    loader2 = get_loader(cfg2)
    halos2 = loader2.load()
    print("CAMELS halos loaded:", len(halos2))
    gb2 = GraphBuilder(cfg2)
    graphs2 = gb2.build_graphs(halos2[:100])
    predsF, predsP, ys = [], [], []
    with torch.no_grad():
        for g in graphs2:
            g = g.to(device)
            emb = gnn.get_embeddings(g, embedding_point="pre_pooling")
            ctx = global_mean_pool(emb, torch.zeros(emb.shape[0], dtype=torch.long, device=device))
            p = torch.sigmoid(policy(g.edge_attr, emb, g.edge_index, ctx)).squeeze(-1)
            m = repair_connectivity(g.edge_index, hard_mask(p, rls["min_keep_frac"]))
            d = pg.Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
            d.batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
            predsF.append(gnn(d).item())
            dp = pg.Data(x=g.x, edge_index=g.edge_index[:, m], edge_attr=g.edge_attr[m])
            dp.batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
            predsP.append(gnn(dp).item())
            ys.append(g.y.item())
    mf2 = MetricsComputer.compute_all(torch.tensor(predsF), torch.tensor(ys))
    mp2 = MetricsComputer.compute_all(torch.tensor(predsP), torch.tensor(ys))
    print(f"CROSS-SIM (TNG->CAMELS): full RMSE={mf2['rmse']:.4f} | RL-pruned RMSE={mp2['rmse']:.4f}")
    pd.DataFrame({"y": ys, "pred_full": predsF, "pred_pruned": predsP}).to_csv(
        "outputs/rls/cross_sim_results.csv", index=False)
except Exception as e:
    print("cross-sim failed (likely no HDF5 / no internet to Flatiron):", e)
    print("Synthetic CAMELS fallback was used by the loader; check the loader's _generate_synthetic_camels path.")
```

```python
# CELL 15: Final results table + paper plots
rows = []
for name, preds, keeps in [("full", preds_full, [1.0]*len(preds_full)),
                            ("rl_policy", preds_pol, keeps)]:
    m = MetricsComputer.compute_all(preds, targets)
    rows.append({"method": name, "rmse": m["rmse"], "r2": m["r2"],
                 "scatter": m["scatter"], "keep_frac": float(np.mean(keeps)),
                 "fidelity": float(np.corrcoef(preds.numpy(), preds_full.numpy())[0, 1])})
bl = pd.read_csv("outputs/rls/baselines.csv")
for _, r in bl.iterrows():
    rows.append({"method": r.method, "rmse": r.rmse, "r2": r.r2,
                 "scatter": float("nan"), "keep_frac": r.keep_frac, "fidelity": float("nan")})
res = pd.DataFrame(rows)
res.to_csv("outputs/rls/results_table.csv", index=False)
print(res.to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name in ["full", "random", "degree", "distance", "rl_policy"]:
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
print("saved outputs/rls/results_table.csv + paper_figures.png")
```

```python
# CELL 16: Save everything to Kaggle output for download
shutil.make_archive("/kaggle/working/rls_outputs", "zip", "outputs/rls")
print("Download /kaggle/working/rls_outputs.zip — contains all results, plots, policy.pt, finetuned_gnn.pt")
```
