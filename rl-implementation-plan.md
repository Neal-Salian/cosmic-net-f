# RL-Cosmic-Net: Physics-Informed RL Graph Sparsification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a reinforcement-learned edge-selection policy that prunes noisy edges from halo graphs before GNN inference, producing (a) equal-or-better halo-mass predictions, (b) a physically meaningful "minimal skeleton" that serves as the model's intrinsic explanation, and (c) a publishable contribution for the **IEEE BigData 2026 conference** (submission Aug 21, notification Oct 24 — primary target, verified) and/or the NeurIPS 2026 ML4PS workshop (2026 CFP not yet published — unverified), then an ApJ/MNRAS journal paper.

**Architecture:** A lightweight per-edge policy network π_θ (MLP over edge features + frozen GNN node embeddings + graph context) outputs keep/drop probabilities. Trained with a **policy gradient (REINFORCE with a learned baseline + entropy bonus — honestly named, NOT PPO)** against a reward combining (1) relative RMSE improvement over the full-graph prediction, (2) a sparsity curriculum term, and (3) a virial-theorem consistency penalty. The GNN backbone (`best_model_augmented.pt`) stays frozen during policy training; a short Stage-B fine-tune of the GNN on policy-pruned graphs closes the train/test distribution shift. At inference: one policy forward pass → hard mask → sparsified graph → GNN prediction. The pruned graph *is* the explanation.

**Tech Stack:** PyTorch 2.1, PyTorch Geometric 2.4 (repo pins), numpy, pandas, matplotlib; new package `rls/` added to the existing cosmic-net repo. GPU optional for most tasks (Kaggle T4/P100 for policy training sweeps and final runs).

---

## PART 0 — PLAN REVIEW (read this first)

### 0.1 Verdict on the rl.md plan

| Plan item | Verdict | Reason |
|---|---|---|
| Frozen backbone `best_model_augmented.pt` as feature extractor | ✅ KEEP | Correct RL-scope decision; prevents gradient divergence. |
| RL edge-selection policy (REINFORCE/PPO) | ✅ KEEP (policy gradient, honestly named) | Novel vs PTDNet (WSDM'21, RL edge dropping, no physics) and GSAT (ICML'22, differentiable, no hard sparsity). Name it "policy gradient (REINFORCE + baseline)" — not PPO (see Phase 1 row). |
| Reward = accuracy + sparsity + virial | ✅ KEEP, but REWORK | Accuracy must be *relative improvement* (per-graph RMSE is dominated by model error, giving a noisy reward); sparsity must be a *curriculum* (constant weight collapses policy); virial must be a *penalty* with a hard feasibility clamp, not a reward term (see 0.3). |
| "Sparsified graph IS the explanation" | ✅ KEEP as the paper's core claim | Must be validated with fidelity, stability, and physics-alignment metrics (Part 3). |
| Phase 0 baselines (random, attention top-k, Gumbel) | ✅ KEEP + add degree-drop and distance-drop | The Gumbel-softmax baseline is the single most important one — it is the "why RL?" control. |
| Phase 1 REINFORCE w/ moving baseline | ✅ KEEP, HONESTLY NAMED | Drop the "PPO" label. Each graph is a one-step MDP, so GAE degenerates to `reward − value` and the clipped-importance ratio can never fire (old/new log-probs come from the same forward pass → ratio ≡ 1). REINFORCE with a learned baseline + entropy bonus is the statistically equivalent, defensible choice — and a reviewer who catches "PPO" in the paper against this code will not trust anything else. |
| Baseline numbers (0.137 dex, R² 0.965, 541 halos 432/109) | ⚠️ NOT REPRODUCIBLE FROM GIT | `config.yaml` has `n_halos: 500` and the loader drops halos below `min_subhalos_per_halo: 3` (tng_loader.py:600) → strictly <500 halos, cannot yield 541; the 0.7/0.15/0.15 split of 541 = 379/81/81, not 432/109. No results artifact is committed and wandb is disabled. **Task 0 must commit the exact config + a baseline_metrics.json before any RL work.** |
| Existing explainer pathway (`explain/explainer.py`, default method `pgexplainer`) | ⚠️ ADD BASELINE ROW | The repo already ships an edge-importance pipeline against this same frozen model. It is gradient saliency *mislabeled* PGExplainer (real PyG PGExplainer is imported but unused; `train_explainer` is a no-op stub). The paper needs a **gradient-saliency baseline row** (cheap — reuse `explain/explainer.py`) and, if time permits, the real `torch_geometric.explain.PGExplainer`. If the RL policy does not beat both, that IS the finding ("RL wins over the standard explainer baseline, here's why") — a better paper than the one currently planned. |
| Phase 2 CAMELS OOD + U_ij binding-energy validation | ✅ KEEP | This is the physics novelty that reviewers at ML4PS and ApJ will care about. |
| Phase 3 MUTAG/OGB standard benchmarks | ⚠️ CUT for workshop; keep as ICLR stretch | Not required for ML4PS; a small ZINC/ogbg-molhiv sweep is enough if ICLR main track is attempted. |
| NeurIPS AI4Science Workshop (Aug 29) then ICLR 2027 (Sep 25) | ⚠️ REPLACE | ICLR 2027 decisions (Dec 16 2026) miss the Nov-1-2026 notification constraint; ML4PS 2026 CFP is NOT yet published (Aug 29 was inferred from the 2025 pattern — unverified). **Primary target: IEEE BigData 2026** (verified: submit Aug 21, notify Oct 24, camera-ready Nov 14, conf Dec 14–17 Phoenix, 10 pages IEEE 2-col incl. refs). Path: IEEE BigData → workshop (if CFP appears) → journal (ApJ/MNRAS) → ICLR 2028 if the method wins on CAMELS-scale data. |

### 0.2 What SOTA research changes (verified citations in `rl-lit-review.md`)

1. **PTDNet / "Learning to Drop" (Luo et al., WSDM 2021)** — already did RL (policy gradient) edge dropping for GNN robustness with nuclear-norm regularization. Your paper MUST cite it and the "why RL" section must differentiate: PTDNet uses joint training (explainer + GNN together) and targets node classification on noisy graphs; RL-Cosmic-Net works on a **frozen pretrained backbone** (plug-in sparsifier, works with any checkpoint), adds **physics constraints**, and targets **graph-level regression + interpretability**.
2. **GSAT (Miao et al., ICML 2022)** — the strongest differentiable alternative. It cannot enforce exact sparsity budgets, needs joint retraining, and its kept subgraph is stochastic-soft. Your hard-decision + frozen-backbone story is the differentiator. It is also your best "interpretability metric" comparison on standard datasets.
3. **GIB / DIR / CIGA** — the invariant-rationale family. Relevant for the OOD claim: your cross-simulation (TNG→CAMELS) test is where these methods would shine; cite them as the motivation that "pruned structure → generalization".
4. **Cranmer et al. (NeurIPS 2020) "Discovering Symbolic Models from Deep Learning with Inductive Biases"** — the direct precursor of your symbolic-regression distillation step; cite as prior art for GNN→equation distillation.
5. **LaSR (Grayeli et al., NeurIPS 2024)** — current SR SOTA on Feynman (72/100 exact vs PySR 59/100). Optional upgrade for the journal paper: use LaSR/LLM-SR on the *pruned-graph features* to rediscover the virial/Faber-Jackson relations. Not required for the workshop.
6. **HaloGraphNet (Villanueva-Domingo et al., ApJ 2022)** — your benchmark baseline: ~0.2 dex scatter, R² 0.96–0.97 on CAMELS CV. Your current 0.137 dex RMSE on TNG already beats it on that metric; the paper must report both numbers and explain dataset differences honestly.
7. **PGExplainer (Luo et al., NeurIPS 2020)** — the standard parameterized edge-mask explainer. You MUST benchmark against it: `explain/explainer.py` already names it as the repo's default method, but its implementation is gradient saliency, not the real algorithm (the real PyG `PGExplainer` import sits unused; `train_explainer` is a stub). The paper needs (a) a gradient-saliency row reusing the repo explainer as-is, and (b) ideally the true PGExplainer via `torch_geometric.explain`. This is the "why RL at all?" control — same frozen model, same metrics.

### 0.3 Reworked reward design (the "improve it further" part)

```
r_t = α · ΔRMSE_rel + β · curriculum_sparsity + γ · connectivity_bonus − δ · virial_penalty
```
- `ΔRMSE_rel = (RMSE_full − RMSE_pruned) / RMSE_full` — **relative**, per-graph, can be negative (policy must not just keep everything).
- `curriculum_sparsity = (keep_ratio − target_t)² ` with target annealed from 0.9 → 0.4 over training — starts easy (learn what to keep), ends at the paper's headline sparsity.
- `connectivity_bonus` = +1 if every node retains ≥ 1 edge else −1 — prevents degenerate "empty graph" collapses.
- `virial_penalty = max(0, (virial_ratio_pruned − 1))² ` where `virial_ratio_pruned` uses pairwise PE over kept edges and KE scaled by retained interaction fraction (definition in Task 4). Annealed in with the sparsity curriculum.
- Trainer: **REINFORCE with a learned baseline (honest naming — this is NOT PPO)**. Each graph is a one-step MDP, so GAE/clipped-importance add nothing (at horizon 1, GAE ≡ `reward − value`). Loss: `−(mean(adv·log π)) + 0.5·MSE(value, reward) − 0.01·H(π)` with `adv = reward − value.detach()`. Adam lr 1e-3, entropy bonus 0.01. The paper says "policy gradient", never "PPO".

### 0.3b Virial reconciliation (mandatory before the paper)

- The repo's training-time `VirialLoss` (`model/physics_loss.py`) defines the ratio as `2·KE/(G·M_pred²/R_half)` — a monopole approximation using the **predicted** total mass. The RL reward uses a pairwise `2·KE_retained/(G·Σ_kept M_i·M_j/r_ij)`. Two definitions in one paper need one narrative:
  1. State explicitly why the reward uses the pairwise, data-side ratio: the monopole depends on the model's own prediction, which is circular for a pruning justification.
  2. Report **both** ratios on the full graph in a calibration note (they should agree to ~10% on virialized halos; if they don't, say so).
  3. **CRITICAL confound:** `PE_retained` shrinks mechanically as edges are dropped (and `KE_retained` via the kept-degree fraction) — both terms track the sparsity *level*, not which edges were kept. A virial ratio in [0.9, 1.1] may be an artifact of pruning to ~50% and would hold for random/degree/distance baselines too. Therefore: **report the virial ratio for EVERY baseline at matched sparsity** (Part 3 table). The RL claim is only "same virial ratio at equal or better accuracy/fidelity" — never "RL achieved virial consistency".

### 0.4 Benchmarks to chase (targets table)

| Benchmark | Metric | Target / comparison |
|---|---|---|
| TNG test accuracy | RMSE, R², scatter (dex) | ≤ 0.137 dex at ≥ 50% sparsity (no degradation vs full graph); HaloGraphNet ref ~0.2 dex |
| Cross-sim OOD (TNG→CAMELS) | RMSE, R² | Pruned model ≥ full model; report both suites (IllustrisTNG, SIMBA) |
| Sparsity–accuracy Pareto | RMSE vs keep-fraction {0.1,0.25,0.4,0.6,0.8,1.0} | RL policy dominates random/degree/distance/attention/Gumbel baselines |
| Physics validity | Virial ratio of pruned graphs; Spearman(ρ, U_ij); Mann-Whitney on kept vs dropped edge distance | virial_ratio ∈ [0.9, 1.1] **reported for every baseline at matched sparsity** (otherwise the metric proves nothing); ρ ≥ 0.5; kept edges significantly shorter (p < 0.05) |
| Explainability baselines | Fidelity, stability, U_ij alignment | **Gradient saliency (repo `explain/explainer.py`) and PGExplainer (PyG) are mandatory rows**, same frozen model, same metrics — the "why RL?" control. Optional if behind: gradient saliency only. |
| Degenerate-policy guard | Spearman ρ vs U_ij | `mass_ratio` is already an input edge feature, so a policy that learns "keep high-mass-ratio edges" partially correlates with U_ij (M_i·M_j) *by construction*. Report the **mass-ratio top-k baseline** at every sparsity level to prove RL beats the trivial shortcut; gate the headline claim on ρ ≥ 0.5. |
| Interpretability | Fidelity (corr pruned vs full preds, Pearson ≥ 0.98); stability (Jaccard of kept edges across 3 seeds ≥ 0.8); runtime speedup | Report all |
| Uncertainty | 95% MC-dropout coverage before/after pruning | Coverage change ≤ 2 pts |
| Generality (ICLR stretch) | ogbg-molhiv / ZINC | Same framework, short sweep |

---

## PART 1 — FILE STRUCTURE

New package `rls/` (sibling of `model/`, `graph/`), plus tests:

```
rls/
├── __init__.py            # exports
├── policy.py              # EdgePolicyNet, build_policy
├── rewards.py             # compute_rewards (relative RMSE + curriculum + connectivity + virial), virial_ratio_pruned
├── policy_gradient.py     # REINFORCE + learned baseline (NOT PPO): compute_advantages, compute_pg_loss, PolicyGradientTrainer
├── sparsify.py            # apply_policy (hard mask, min-keep clamp, connectivity repair), sparsify_loader
├── baselines.py           # random, degree-drop, distance-drop, mass-ratio-drop, gradient-saliency, attention-topk, gumbel-mask
├── metrics.py             # fidelity, stability, sparsity_stats, physics_alignment (U_ij Spearman, MW test), calibration
├── evaluate.py            # full eval suite → CSV + PNG plots (paper-ready)
├── stageb.py              # GNN fine-tune on policy-pruned graphs (distribution-shift fix; OPTIONAL under triage)
├── cross_sim.py           # TNG→CAMELS OOD (only with real cached CAMELS data; synthetic fallback is NOT publishable)
└── run_experiment.py      # end-to-end driver
tests/
├── test_policy.py
├── test_rewards.py
├── test_policy_gradient.py
└── test_metrics.py
```

Modify: `config/config.yaml` (add `rls:` section), `requirements.txt` (add nothing new — pure PyTorch). No changes to existing model/graph/data code (frozen backbone means we reuse `CosmicNetGNN.forward(batch_data)` and `GraphBuilder` as-is).

Existing interfaces this plan relies on (from REPOWISE.md / repo):
- `CosmicNetGNN(config)`; `forward(batch_data, return_embeddings=False) -> preds [B]`; `get_embeddings(batch_data, embedding_point='pre_pooling') -> (N, out_dim)`; `predict_with_uncertainty(batch_data, n_samples)`.
- `batch_data` fields: `.x [N,4]`, `.edge_index [2,E]`, `.edge_attr [E,5]`, `.batch [N]`, `.y [B]`, `.stellar_mass`, `.vel_disp`, `.half_mass_r`.
- `load_model(checkpoint_path, config, device)` handles both dict-key and raw state-dict checkpoints.
- Config keys: `model.hidden_dim=256`, `model.output_dim=128`, `graph.edge_features` (5), `model.mc_samples=30`.

**Kaggle note (IMPORTANT):** `data/raw/tng100_clustered.csv` and `kaggle/best_model_augmented.pt` are gitignored — they exist locally but are NOT on GitHub. For Kaggle runs: upload both as a private Kaggle dataset and mount with `/kaggle/input/<ds>/`. The `rl-kaggle-notebooks.md` file contains the three ready-to-paste notebooks.

---

## PART 2 — TASKS (TDD, bite-sized)

### Task 0: Commit the reproducible baseline (fix the divergence FIRST)

**Why:** the reported baseline (0.137 dex, R² 0.965, 541 halos split 432/109) is **not reproducible from git**. `config.yaml` has `n_halos: 500` with `min_subhalos_per_halo: 3` (loader drops halos, tng_loader.py:600) → strictly <500 halos; the 0.7/0.15/0.15 split of 541 is 379/81/81, not 432/109; no results artifact is committed; wandb is disabled. Do this before ANY RL work — otherwise you can't defend the numbers you're claiming to beat.

- [ ] **Step 1:** Re-run the baseline evaluation with the committed config on the current test split. Write `outputs/rls/baseline_metrics.json`:
```json
{
  "config": "config/config.yaml",
  "n_halos_configured": 500,
  "n_halos_actual": 541,
  "split": {"train": 432, "val": 81, "test": 109},
  "rmse_dex": 0.137, "r2": 0.965, "scatter_dex": 0.10,
  "checkpoint": "kaggle/best_model_augmented.pt"
}
```
- [ ] **Step 2:** Reconcile: if the real run used different `n_halos`/split than committed, either update `config.yaml` to match the run that produced 541 halos, or record the exact divergence in the JSON. The JSON + config, not memory, are the paper's baseline.
- [ ] **Step 3:** Commit: `git add config/config.yaml outputs/rls/baseline_metrics.json && git commit -m "fix(rls): commit reproducible baseline config + metrics"`
- [ ] **Step 4:** Extend `tests/test_rewards.py` (or a new `tests/test_baseline.py`) to assert `baseline_metrics.json` exists and its `n_halos_actual` matches what `split_data()` actually produces — a guard against future config drift.

---

### Task 1: Add `rls/` package skeleton + config section

**Files:**
- Create: `rls/__init__.py`
- Create: `rls/policy.py`
- Modify: `config/config.yaml` (append `rls:` block)
- Test: `tests/test_policy.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from rls.policy import EdgePolicyNet

def make_cfg():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("rls", {})
    return cfg

def test_policy_output_shapes():
    torch.manual_seed(0)
    cfg = make_cfg()
    N, E, D = 8, 20, 5
    emb_dim = cfg["model"]["output_dim"]
    net = EdgePolicyNet(edge_dim=D, node_emb_dim=emb_dim, hidden_dim=64)
    edge_attr = torch.randn(E, D)
    node_emb = torch.randn(N, emb_dim)
    edge_index = torch.randint(0, N, (2, E))
    ctx = torch.randn(emb_dim)
    logits = net(edge_attr, node_emb, edge_index, ctx)
    assert logits.shape == (E, 1)
    p = torch.sigmoid(logits)
    assert (p > 0).all() and (p < 1).all()

def test_policy_equivariant_to_node_ordering_within_edge():
    torch.manual_seed(0)
    cfg = make_cfg()
    net = EdgePolicyNet(edge_dim=5, node_emb_dim=128, hidden_dim=64)
    N, E = 8, 6
    edge_attr = torch.randn(E, 5)
    node_emb = torch.randn(N, 128)
    edge_index = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,6]])
    ctx = torch.randn(128)
    l1 = net(edge_attr, node_emb, edge_index, ctx)
    perm = torch.tensor([5,4,3,2,1,0])
    l2 = net(edge_attr[perm], node_emb, edge_index[:, perm], ctx)
    assert torch.allclose(l1[perm], l2, atol=1e-5)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python -m pytest tests/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rls'`

- [ ] **Step 3: Create the package and implementation**

`rls/__init__.py`:
```python
from rls.policy import EdgePolicyNet, build_policy
from rls.rewards import compute_rewards, virial_ratio_pruned
from rls.policy_gradient import PolicyGradientTrainer
from rls.sparsify import apply_policy, sparsify_batch
from rls.baselines import (random_mask, degree_mask, distance_mask,
                           mass_ratio_mask, gradient_saliency_mask,
                           attention_topk_mask, gumbel_edge_mask)
from rls.metrics import (fidelity, stability_jaccard, sparsity_stats,
                         physics_alignment, calibration_coverage)
from rls.evaluate import run_evaluation
from rls.stageb import fine_tune_gnn
```

`rls/policy.py`:
```python
"""Per-edge keep/drop policy network (inductive, graph-size agnostic)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePolicyNet(nn.Module):
    def __init__(self, edge_dim=5, node_emb_dim=128, hidden_dim=64):
        super().__init__()
        self.node_proj = nn.Linear(node_emb_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr, node_emb, edge_index, context):
        """Logits per edge. context: [emb_dim] graph-level pooled embedding.

        Edge ordering must be permutation-covariant: permuting (edge_attr,
        edge_index) together permutes logits identically.
        """
        e = F.leaky_relu(self.edge_proj(edge_attr), 0.1)          # [E,H]
        u, v = edge_index
        nu = F.leaky_relu(self.node_proj(node_emb[u]), 0.1)       # [E,H]
        nv = F.leaky_relu(self.node_proj(node_emb[v]), 0.1)       # [E,H]
        c = context.unsqueeze(0).expand(e.size(0), -1)            # [E,H]
        h = torch.cat([nu, nv, e, c], dim=-1)                     # [E,4H]
        return self.fc(h)                                          # [E,1]


def build_policy(cfg, node_emb_dim=None):
    rls_cfg = cfg.get("rls", {})
    return EdgePolicyNet(
        edge_dim=len(cfg.get("graph", {}).get("edge_features", ["distance", "delta_v",
                                                                 "cos_theta", "mass_ratio",
                                                                 "proj_sep"])),
        node_emb_dim=node_emb_dim or cfg["model"]["output_dim"],
        hidden_dim=rls_cfg.get("policy_hidden", 64),
    )
```

`config/config.yaml` — append:
```yaml
rls:
  policy_hidden: 64
  lr: 0.001
  entropy_coef: 0.01
  value_coef: 0.5
  epochs: 60
  batch_size: 32
  target_sparsity_start: 0.9
  target_sparsity_end: 0.4
  sparsity_anneal_epochs: 40
  w_acc: 1.0
  w_sp: 0.5
  w_conn: 1.0
  w_virial: 1.0
  virial_anneal_start_epoch: 10
  min_keep_frac: 0.1
  seed: 42
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_policy.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add rls/ tests/test_policy.py config/config.yaml
git commit -m "feat(rls): add edge policy network and config"
```

---

### Task 2: Rewards (relative RMSE + sparsity curriculum + connectivity + virial)

**Files:**
- Create: `rls/rewards.py`
- Test: `tests/test_rewards.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from rls.rewards import compute_rewards, virial_ratio_pruned

def test_relative_rmse_reward_sign():
    y = torch.tensor([10.0, 11.0, 12.0])
    pred_full = torch.tensor([10.3, 10.8, 12.2])
    pred_pruned_better = torch.tensor([10.1, 10.9, 12.1])
    pred_pruned_worse = torch.tensor([10.8, 10.2, 12.7])
    r_better = compute_rewards(pred_pruned_better, pred_full, y,
                               keep_ratio=0.5, target_sparsity=0.5,
                               virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                        "w_conn": 1.0, "w_virial": 1.0},
                               connectivity_ok=True)
    r_worse = compute_rewards(pred_pruned_worse, pred_full, y,
                              keep_ratio=0.5, target_sparsity=0.5,
                              virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                       "w_conn": 1.0, "w_virial": 1.0},
                              connectivity_ok=True)
    assert r_better > 0
    assert r_worse < 0

def test_sparsity_curriculum_guides_keeps():
    # At target_sparsity=0.5, keeping 0.5 of edges should give sparsity term 0;
    # keeping 1.0 should give a negative term.
    r_at_target = compute_rewards(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                                  keep_ratio=0.5, target_sparsity=0.5,
                                  virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                           "w_conn": 1.0, "w_virial": 1.0},
                                  connectivity_ok=True)
    r_keeps_all = compute_rewards(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                                  keep_ratio=1.0, target_sparsity=0.5,
                                  virial_penalty=0.0, cfg={"w_acc": 1.0, "w_sp": 0.5,
                                                           "w_conn": 1.0, "w_virial": 1.0},
                                  connectivity_ok=True)
    assert r_at_target > r_keeps_all

def test_virial_penalty():
    p_bad = virial_ratio_pruned(ke_retained=2.0, pe_retained=0.5)   # ratio 8.0 -> penalty
    p_good = virial_ratio_pruned(ke_retained=1.0, pe_retained=1.0)  # ratio 1.0 -> 0
    assert p_bad > 0 and p_good == 0

def test_connectivity_penalty():
    r_conn = compute_rewards(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                             keep_ratio=0.5, target_sparsity=0.5, virial_penalty=0.0,
                             cfg={"w_acc": 1.0, "w_sp": 0.5, "w_conn": 1.0, "w_virial": 1.0},
                             connectivity_ok=False)
    assert r_conn < 0
```

- [ ] **Step 2: Run tests, verify FAIL** (`ModuleNotFoundError: No module named 'rls.rewards'`)

- [ ] **Step 3: Implement `rls/rewards.py`**

```python
"""Reward design for RL edge sparsification.

r = w_acc * dRMSE_rel + w_sp * (keep_ratio - target)^2 + w_conn * conn_bonus
    - w_virial * virial_penalty

All terms are per-graph scalars; the trainer averages over the batch.
"""
import torch


def virial_ratio_pruned(ke_retained, pe_retained, eps=1e-8):
    """Dimensionless virial ratio of the pruned graph: 2*KE_retained / |PE_retained|.

    KE_retained = 0.5 * sum_i (deg'(i)/deg(i)) * M_*,i * sigma_i^2
    PE_retained = G * sum_{(i,j) in E'} M_*,i * M_*,j / r_ij   (absolute value used)
    Returns the ratio (1.0 = virialized).
    """
    pe_retained = torch.clamp(torch.abs(pe_retained), min=eps)
    return (2.0 * ke_retained) / pe_retained


def virial_penalty(ke_retained, pe_retained):
    ratio = virial_ratio_pruned(ke_retained, pe_retained)
    return torch.clamp(ratio - 1.0, min=0.0) ** 2


def _rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def compute_rewards(pred_pruned, pred_full, targets, keep_ratio, target_sparsity,
                    virial_penalty, cfg, connectivity_ok=True):
    """Vectorized over a batch of graphs (one entry per graph)."""
    base_rmse = _rmse(pred_full, targets) + 1e-8
    pruned_rmse = _rmse(pred_pruned, targets)
    delta_rel = (base_rmse - pruned_rmse) / base_rmse          # positive = better

    sparsity_term = (keep_ratio - target_sparsity) ** 2        # 0 at target, + away from it
    sparsity_term = sparsity_term.clamp(max=1.0)

    conn_bonus = torch.where(connectivity_ok, torch.ones_like(delta_rel),
                             -torch.ones_like(delta_rel))

    r = (cfg["w_acc"] * delta_rel
         - cfg["w_sp"] * sparsity_term
         + cfg["w_conn"] * conn_bonus
         - cfg["w_virial"] * virial_penalty)
    return r
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/rewards.py tests/test_rewards.py && git commit -m "feat(rls): physics-informed reward function"`

---

### Task 3: Policy-gradient trainer (REINFORCE + learned baseline + entropy)

**Files:**
- Create: `rls/policy_gradient.py`
- Test: `tests/test_policy_gradient.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.policy_gradient import compute_advantages, compute_pg_loss

def test_advantages_are_reward_minus_baseline():
    rewards = torch.tensor([1.0, 0.0, -0.5])
    values = torch.tensor([0.5, 0.4, 0.3])
    adv = compute_advantages(rewards, values)
    assert adv.shape == rewards.shape
    # one-step MDP: no bootstrapping, no GAE — advantage is reward - value
    assert torch.allclose(adv, rewards - values, atol=1e-5)

def test_pg_loss_encourages_positive_advantage_actions():
    # Same positive advantage; the action with HIGHER sampled logp must
    # yield a LOWER (less negative is "higher") pg loss contribution.
    adv = torch.tensor([1.0])
    logp_good = torch.tensor([-0.2])
    logp_bad = torch.tensor([-1.5])
    l_good = compute_pg_loss(logp_good, adv, torch.zeros(1), torch.ones(1),
                             entropy=torch.zeros(1))[0]
    l_bad = compute_pg_loss(logp_bad, adv, torch.zeros(1), torch.ones(1),
                            entropy=torch.zeros(1))[0]
    assert l_good.item() < l_bad.item()

def test_pg_loss_clips_negative_advantage():
    # Negative advantage must DECREASE the prob of the sampled action.
    adv = torch.tensor([-1.0])
    logp = torch.tensor([-0.5])
    l = compute_pg_loss(logp, adv, torch.zeros(1), torch.zeros(1),
                        entropy=torch.zeros(1))[0]
    assert l.item() > 0
```

- [ ] **Step 2: Run tests, verify FAIL** (`ModuleNotFoundError: No module named 'rls.policy_gradient'`)

- [ ] **Step 3: Implement `rls/policy_gradient.py`**

```python
"""Policy-gradient training: REINFORCE with a learned baseline + entropy bonus.

WHY NOT PPO: each graph is a one-step MDP (the decision is a single mask
sample; no sequential credit assignment). At horizon 1, GAE degenerates to
reward - value and a clipped-importance ratio would be exactly 1 forever
(old and new log-probs come from the same forward pass before any update).
PPO's machinery therefore adds nothing here; REINFORCE with a baseline is
the statistically equivalent, honest choice — and the paper says exactly
this. If a multi-step MDP is ever added, swap in real PPO (rollout buffer
+ K update epochs) at this boundary."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    """Critic/baseline: graph context embedding -> scalar state value."""

    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ctx):
        return self.net(ctx).squeeze(-1)


def compute_advantages(rewards, values):
    """Advantage = reward - baseline. One-step MDP: no GAE, no bootstrap."""
    return rewards - values.detach()


def compute_pg_loss(logp, advantages, values, rewards,
                    entropy=0.0, value_coef=0.5, entropy_coef=0.01):
    """logp: [T] log-prob of sampled action per graph (mean over its edges).

    Returns (loss, pg_loss, vf_loss, entropy). advantages/values/rewards: [T].
    """
    pg_loss = -(advantages.detach() * logp).mean()
    vf_loss = F.mse_loss(values, rewards.detach())
    loss = pg_loss + value_coef * vf_loss - entropy_coef * entropy
    return loss, pg_loss, vf_loss, entropy


class PolicyGradientTrainer:
    """Trains EdgePolicyNet over a frozen GNN backbone.

    Each 'episode' is one mini-batch of graphs: policy samples masks, GNN
    predicts on pruned graphs, reward is computed, one gradient step happens
    (no replay buffer — one-step MDPs need no importance sampling)."""

    def __init__(self, policy, value_net, optimizer, value_optimizer, cfg):
        self.policy = policy
        self.value_net = value_net
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer
        self.cfg = cfg

    def target_sparsity(self, epoch):
        s, e = self.cfg["target_sparsity_start"], self.cfg["target_sparsity_end"]
        n = max(1, self.cfg["sparsity_anneal_epochs"])
        p = min(1.0, epoch / n)
        return s + (e - s) * p
```

Note: rollouts and the update loop live in the training script (Task 5) so they can run with or without W&B; `PolicyGradientTrainer` holds the reusable math. The Bernoulli log-prob utilities are added here:

```python
def bernoulli_logp(p, action):
    """p: [E] probs; action: [E] in {0,1}. Returns [E] log probs."""
    eps = 1e-8
    return torch.where(action.bool(),
                       torch.log(torch.clamp(p, eps, 1.0)),
                       torch.log(torch.clamp(1 - p, eps, 1.0)))


def bernoulli_entropy(p):
    """H(Bernoulli(p)) per edge: -(p log p + (1-p) log(1-p)). [E]."""
    eps = 1e-8
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def sample_actions(p):
    """Straight-through-free Bernoulli sampling."""
    return torch.bernoulli(p)
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/policy_gradient.py tests/test_policy_gradient.py && git commit -m "feat(rls): policy-gradient losses + baseline value net (honest naming, not PPO)"`

---

### Task 4: Sparsification (hard mask + min-keep clamp + connectivity repair)

**Files:**
- Create: `rls/sparsify.py`
- Test: `tests/test_rewards.py` (extend) — use `tests/test_sparsify.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.sparsify import hard_mask, repair_connectivity

def test_hard_mask_respects_min_keep():
    probs = torch.full((10,), 0.05)          # wants to drop almost everything
    mask = hard_mask(probs, min_keep_frac=0.2)
    assert mask.sum() >= 2                   # >= 0.2*10
    # keeps the highest-probability edges
    assert mask[probs.argmax()] == 1

def test_repair_connectivity_no_isolated_nodes():
    N, E = 5, 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # path graph
    mask = torch.tensor([1, 1, 0, 0])        # node 4 loses its only edge
    repaired = repair_connectivity(edge_index, mask)
    assert repaired.sum() == mask.sum()      # repair adds exactly the needed edges
    # every node must have degree >= 1
    deg = torch.zeros(N)
    for i in range(repaired.shape[0]):
        if repaired[i]:
            deg[edge_index[0, i]] += 1
            deg[edge_index[1, i]] += 1
    assert (deg >= 1).all()
```

- [ ] **Step 2: Run tests, verify FAIL**

- [ ] **Step 3: Implement `rls/sparsify.py`**

```python
"""Hard decision layer: probabilities -> binary masks with guarantees."""
import torch


def hard_mask(probs, min_keep_frac=0.1):
    """Threshold at 0.5, then force-keep the top-k highest-prob edges so the
    kept fraction is at least min_keep_frac."""
    mask = (probs >= 0.5).long()
    k_min = int(torch.ceil(torch.tensor(min_keep_frac) * probs.numel()))
    if mask.sum() < k_min:
        top = torch.topk(probs, k_min).indices
        mask = torch.zeros_like(mask)
        mask[top] = 1
    return mask


def repair_connectivity(edge_index, mask):
    """Guarantee every node has >= 1 incident kept edge.

    For each isolated node, force-keep its nearest-neighbor edge (first in
    edge_index order among its incident edges is a deterministic fallback;
    nearest by distance is handled by the caller via edge reordering)."""
    mask = mask.clone()
    kept = mask.bool()
    incident = torch.zeros(edge_index.max().item() + 1, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        if kept[i]:
            incident[edge_index[0, i]] += 1
            incident[edge_index[1, i]] += 1
    isolated = (incident == 0).nonzero(as_tuple=True)[0]
    for node in isolated.tolist():
        cand = (edge_index == node).sum(dim=0).bool()
        if cand.any():
            first = cand.nonzero(as_tuple=True)[0][0]
            mask[first] = 1
    return mask
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/sparsify.py tests/test_sparsify.py && git commit -m "feat(rls): hard masking with connectivity repair"`

---

### Task 5: Training loop (frozen backbone rollouts + policy-gradient updates)

**Files:**
- Create: `rls/train_policy.py` (runs on CPU or GPU; checkpointing every 10 epochs; W&B optional)
- Test: `tests/test_policy_gradient.py` (extend with a smoke end-to-end train on 8 synthetic graphs) + `tests/test_train_policy.py` (real-adapter test)

- [ ] **Step 1: Write the failing tests**

```python
def test_end_to_end_policy_train_smoke():
    """Trains 2 PG epochs on 8 tiny synthetic graphs; asserts no NaNs and
    finite losses."""
    import sys, os, torch, yaml, copy
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rls.policy import build_policy
    from rls.policy_gradient import PolicyGradientTrainer, ValueNet
    from rls.train_policy import train_policy

    torch.manual_seed(0)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["rls"]["epochs"] = 2
    cfg["rls"]["batch_size"] = 8
    # tiny graphs via random tensors (policy + pg only; no GNN needed)
    graphs = [{"x": torch.randn(5, 4), "edge_index": torch.randint(0, 5, (2, 12)),
               "edge_attr": torch.randn(12, 5), "y": torch.randn(1),
               "ctx": torch.randn(128)} for _ in range(8)]
    policy = build_policy(cfg)
    value_net = ValueNet(128)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, cfg["rls"])
    losses = train_policy(trainer, graphs, None, cfg["rls"], device="cpu",
                          epochs=2, log_fn=None)
    assert all(torch.isfinite(l) for l in losses)
```

```python
def test_prepare_graphs_real_pyg_data():
    """The adapter MUST work on real PyG Batch objects (this is the
    integration point the plan previously shipped broken)."""
    import sys, os, torch, yaml
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch_geometric.data import Data, Batch, DataLoader
    from rls.train_policy import prepare_graphs
    from rls.policy import build_policy
    from rls.policy_gradient import ValueNet

    torch.manual_seed(0)
    datalist = []
    for _ in range(4):
        n = 5
        datalist.append(Data(
            x=torch.randn(n, 4),
            edge_index=torch.randint(0, n, (2, 10)),
            edge_attr=torch.randn(10, 5),
            y=torch.randn(1),
            stellar_mass=torch.rand(n) * 1e10,
            vel_disp=torch.rand(n) * 200,
            half_mass_r=torch.rand(n) * 0.01,
            pos=torch.randn(n, 3),
        ))
    loader = DataLoader(datalist, batch_size=2)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    from model.model import build_model
    gnn = build_model(cfg)
    graphs = prepare_graphs(loader, gnn, "cpu")
    assert len(graphs) == 4
    assert all(set(g.keys()) >= {"x", "edge_index", "edge_attr", "y", "ctx",
                                 "stellar_mass", "vel_disp", "half_mass_r", "pos"}
               for g in graphs)
    assert all(g["ctx"].shape[0] == cfg["model"]["output_dim"] for g in graphs)
    assert all(g["edge_attr"].shape[0] == g["edge_index"].shape[1] for g in graphs)
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: No module named 'rls.train_policy'`)

- [ ] **Step 3: Implement `rls/train_policy.py`**

```python
"""End-to-end policy-gradient training of the edge policy over a frozen GNN.

Signature: train_policy(trainer, graphs, gnns, cfg, device, epochs, log_fn)

graphs: list of dicts with x, edge_index, edge_attr, y, ctx (+ physics attrs).
gnns:   None (smoke mode) or callable (graph_dict, mask) -> (pred_full, pred_pruned).
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from rls.policy_gradient import (compute_advantages, compute_pg_loss,
                                 bernoulli_logp, bernoulli_entropy, sample_actions)
from rls.sparsify import hard_mask, repair_connectivity
from rls.rewards import compute_rewards, virial_penalty


def _graph_physics_terms(graph, edge_index, mask, G=4.302e-9):
    """ke_retained, pe_retained per graph (see rewards.py docstring)."""
    stellar = graph["stellar_mass"]  # [N]
    vel_disp = graph["vel_disp"]     # [N]
    pos = graph["pos"]               # [N,3]
    deg = torch.zeros(stellar.numel())
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    deg_ret = torch.zeros_like(deg)
    deg_ret.index_add_(0, edge_index[0, mask], torch.ones(mask.sum()))
    deg_ret.index_add_(0, edge_index[1, mask], torch.ones(mask.sum()))
    frac = deg_ret / deg.clamp(min=1)
    ke = 0.5 * torch.sum(frac * stellar * vel_disp ** 2)
    u, v = edge_index[0, mask], edge_index[1, mask]
    r = torch.norm(pos[u] - pos[v], dim=1)
    pe = torch.sum(G * stellar[u] * stellar[v] / r.clamp(min=1e-6))
    return ke, pe


def prepare_graphs(loader, gnn, device):
    """Adapter: PyG loader -> list of per-graph dicts for policy training.

    Uses batch.get_example(i) (the same pattern as explain/explainer.py's
    explain_batch) so per-graph tensors keep their original indices — NO
    fragile boolean-mask remapping. Precomputes frozen node embeddings and
    the graph context once (GNN stays frozen and in eval mode)."""
    graphs = []
    gnn.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for i in range(batch.num_graphs):
                g = batch.get_example(i).to(device)
                b = Batch.from_data_list([g])
                emb = gnn.get_embeddings(b, embedding_point="pre_pooling")  # [N, out]
                ctx = global_mean_pool(emb, b.batch)                        # [1, out]
                graphs.append({
                    "x": g.x,
                    "edge_index": g.edge_index,
                    "edge_attr": g.edge_attr,
                    "y": g.y.view(1),
                    "ctx": ctx[0],
                    "stellar_mass": getattr(g, "stellar_mass", None),
                    "vel_disp": getattr(g, "vel_disp", None),
                    "half_mass_r": getattr(g, "half_mass_r", None),
                    "pos": getattr(g, "pos", None),
                })
    return graphs


def train_policy(trainer, graphs, gnns, cfg, device="cpu", epochs=60, log_fn=None):
    """One epoch = one pass over the graphs; batch = cfg['batch_size'] graphs.

    gnns: callable (graph_dict, mask) -> (pred_full, pred_pruned) or None
    in smoke mode, in which case rewards use random preds.
    """
    policy, value_net = trainer.policy, trainer.value_net
    losses = []
    for epoch in range(epochs):
        target_sp = trainer.target_sparsity(epoch)
        order = torch.randperm(len(graphs)).tolist()
        for start in range(0, len(graphs), cfg["batch_size"]):
            batch_graphs = [graphs[i] for i in order[start:start + cfg["batch_size"]]]
            if len(batch_graphs) == 0:
                continue
            losses.append(_update_batch(trainer, batch_graphs, gnns, cfg,
                                        target_sp, device))
        if log_fn:
            log_fn(epoch, target_sp, losses[-1])
    return losses


def _update_batch(trainer, batch_graphs, gnns, cfg, target_sp, device):
    """One policy-gradient step on a mini-batch of graphs.

    Everything the policy must NOT differentiate through (mask repair, GNN
    forward, rewards, baseline) runs under no_grad. Only the sampled log
    probs and the entropy term carry policy gradients."""
    policy, value_net = trainer.policy, trainer.value_net
    policy = policy.to(device)
    value_net = value_net.to(device)

    total_logp = []
    total_ent = []
    total_values = []
    total_rewards = []

    for g in batch_graphs:
        g = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
        probs = torch.sigmoid(policy(g["edge_attr"], g["ctx"],
                                     g["edge_index"], g["ctx"])).squeeze(-1)
        action = sample_actions(probs)
        logp = bernoulli_logp(probs, action)
        ent = bernoulli_entropy(probs).mean()
        with torch.no_grad():
            mask = hard_mask(probs, min_keep_frac=cfg.get("min_keep_frac", 0.1))
            mask = repair_connectivity(g["edge_index"], mask)
            pred_full, pred_pruned = gnns(g, mask) if gnns else (torch.randn(1), torch.randn(1))
            ke, pe = _graph_physics_terms(g, g["edge_index"], mask)
            vp = virial_penalty(ke, pe) if cfg.get("w_virial", 0) > 0 else torch.zeros(1).to(device)
            conn_ok = bool((mask.sum() > 0) and _no_isolated(g["edge_index"], mask))
            keep_ratio = mask.float().mean().item()
            rew = compute_rewards(pred_pruned, pred_full, g["y"], keep_ratio,
                                  target_sp, vp, cfg, connectivity_ok=conn_ok)
            val = value_net(g["ctx"])
        total_logp.append(logp.mean())
        total_ent.append(ent)
        total_values.append(val)
        total_rewards.append(rew)

    logp = torch.stack(total_logp)
    entropy = torch.stack(total_ent).mean()
    values = torch.stack(total_values)
    rewards = torch.stack(total_rewards)

    adv = compute_advantages(rewards, values)
    loss, pg_loss, vf_loss, ent = compute_pg_loss(
        logp, adv, values, rewards,
        entropy=entropy, value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"])

    trainer.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    trainer.optimizer.step()
    trainer.value_optimizer.zero_grad()
    vf = F.mse_loss(values, rewards.detach())
    vf.backward()
    trainer.value_optimizer.step()
    return loss.detach().cpu()


def _no_isolated(edge_index, mask):
    deg = torch.zeros(edge_index.max().item() + 1)
    deg.index_add_(0, edge_index[0, mask], torch.ones(mask.sum()))
    deg.index_add_(0, edge_index[1, mask], torch.ones(mask.sum()))
    return bool((deg >= 1).all())
```

> **INTEGRATION NOTE (was the #1 crash risk):** `prepare_graphs` above now uses `batch.get_example(i)` — verified against the repo's own working pattern in `explain/explainer.py:explain_batch` (explainer.py:439-457) and the Kaggle notebook adapters. `gnns` in `run_experiment.py` (Task 10) must build a real PyG `Data`/`Batch` and unpack the `(preds, embeddings)` tuple that `CosmicNetGNN.forward` returns (model.py:307-349) — do NOT pass plain dicts. The real-adapter test in Step 1 exercises this path; the old smoke-only testing was exactly the "TDD theater" hole.

- [ ] **Step 4: Run tests, verify PASS** (smoke trains 2 epochs, losses finite; adapter test passes on real PyG Data)

- [ ] **Step 5: Commit** — `git add rls/train_policy.py tests/test_policy_gradient.py tests/test_train_policy.py && git commit -m "feat(rls): policy-gradient training loop + tested real-data adapter"`

---

### Task 6: Baselines (random, degree-drop, distance-drop, attention top-k, Gumbel)

**Files:**
- Create: `rls/baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.baselines import (random_mask, degree_mask, distance_mask,
                           mass_ratio_mask, attention_topk_mask)

def test_masks_have_correct_fraction():
    torch.manual_seed(0)
    N, E = 10, 40
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 5)
    pos = torch.rand(N, 3)
    f = 0.5
    for mask_fn in [random_mask, degree_mask, distance_mask, mass_ratio_mask,
                    attention_topk_mask]:
        m = mask_fn(edge_index, edge_attr, pos, f)
        assert m.sum() == int(f * E), f"{mask_fn.__name__} fraction wrong"

def test_distance_mask_prefers_short_edges():
    N, E = 10, 40
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 5)
    pos = torch.rand(N, 3)
    m = distance_mask(edge_index, edge_attr, pos, 0.5)
    kept_dists = torch.norm(pos[edge_index[0][m]] - pos[edge_index[1][m]], dim=1)
    dropped_dists = torch.norm(pos[edge_index[0][~m]] - pos[edge_index[1][~m]], dim=1)
    assert kept_dists.mean() < dropped_dists.mean()

def test_mass_ratio_mask_prefers_high_mass_ratio():
    N, E = 10, 40
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.rand(E, 5)
    edge_attr[:, 3] = torch.linspace(0, 1, E)   # mass_ratio is feature index 3
    pos = torch.rand(N, 3)
    m = mass_ratio_mask(edge_index, edge_attr, pos, 0.5)
    assert edge_attr[m, 3].mean() > edge_attr[~m, 3].mean()
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement `rls/baselines.py`**

```python
"""Non-RL sparsification baselines (Phase 0 of the paper)."""
import torch


def random_mask(edge_index, edge_attr, pos, frac, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = edge_attr.shape[0]
    k = int(frac * n)
    idx = torch.randperm(n, generator=g)[:k]
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m


def degree_mask(edge_index, edge_attr, pos, frac):
    """Keep the highest-degree edges (hubs are informative)."""
    n = edge_index.shape[1]
    deg = torch.zeros(edge_index.max().item() + 1)
    deg.index_add_(0, edge_index[0], torch.ones(n))
    deg.index_add_(0, edge_index[1], torch.ones(n))
    edge_deg = (deg[edge_index[0]] + deg[edge_index[1]]) / 2
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(edge_deg, k).indices] = True
    return m


def distance_mask(edge_index, edge_attr, pos, frac):
    """Keep the shortest edges (gravity acts locally)."""
    n = edge_index.shape[1]
    dists = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(-dists, k).indices] = True
    return m


def mass_ratio_mask(edge_index, edge_attr, pos, frac, mass_ratio_idx=3):
    """Keep the highest mass_ratio edges (feature index 3) — the DEGENERATE
    shortcut policy. mass_ratio is an input edge feature, so this baseline
    bounds what a trivial 'keep massive pairs' policy achieves; RL must beat
    it, and the U_ij alignment claim is meaningless without this row."""
    n = edge_index.shape[1]
    scores = edge_attr[:, mass_ratio_idx]
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(scores, k).indices] = True
    return m


def gradient_saliency_mask(edge_index, edge_attr, pos, frac, model=None):
    """Top-k edges by gradient-saliency importance (the repo's existing
    'pgexplainer' pathway: explain/explainer.py `_explain_pgexplainer`).
    model: callable(data) -> (pred, _) used for the backward pass; falls
    back to distance when model is None (used by tests). This is the
    mandatory 'why RL?' control — same frozen model, same metrics."""
    if model is None:
        return distance_mask(edge_index, edge_attr, pos, frac)
    n = edge_index.shape[1]
    # gradient w.r.t. node features, edge importance = (|g_src| + |g_dst|) / 2
    x = edge_attr.new_ones(edge_index.max().item() + 1, 4, requires_grad=True)
    from torch_geometric.data import Data, Batch
    pred, _ = model(Batch.from_data_list([Data(x=x, edge_index=edge_index,
                                               edge_attr=edge_attr)]))
    pred.backward()
    g = x.grad.abs().sum(dim=1)
    scores = (g[edge_index[0]] + g[edge_index[1]]) / 2
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(scores, k).indices] = True
    return m


def attention_topk_mask(edge_index, edge_attr, pos, frac, model=None):
    """Top-k edges by a scalar edge-importance score from a light MLP.
    Falls back to distance when model is None (used by tests)."""
    if model is None:
        return distance_mask(edge_index, edge_attr, pos, frac)
    n = edge_index.shape[1]
    scores = model(edge_attr).squeeze(-1)
    k = int(frac * n)
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(scores, k).indices] = True
    return m


class GumbelEdgeMask(torch.nn.Module):
    """Differentiable edge-mask control (the 'why not Gumbel?' baseline).

    Trained end-to-end with the GNN unfrozen: sigmoid(logits) mask with
    straight-through Gumbel-Softmax sampling. This is the direct
    differentiable alternative to the RL policy."""

    def __init__(self, edge_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr, hard=False, tau=0.5):
        logits = self.net(edge_attr).squeeze(-1)      # [E]
        logits = torch.stack([-logits, logits], dim=-1)  # [E,2] keep/drop
        u = torch.rand_like(logits) + 1e-8
        g = -torch.log(-torch.log(u))
        soft = torch.softmax((logits + g) / tau, dim=-1)
        if hard:
            idx = soft.argmax(dim=-1)
            return (torch.nn.functional.one_hot(idx, 2).float() - soft).detach() + soft
        return soft[:, 1]
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/baselines.py tests/test_baselines.py && git commit -m "feat(rls): sparsification baselines incl. gumbel control"`

---

### Task 7: Metrics (fidelity, stability, physics alignment, calibration)

**Files:**
- Create: `rls/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.metrics import fidelity, stability_jaccard, physics_alignment, calibration_coverage

def test_fidelity_perfect_correlation():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = a * 2 + 0.5
    r = fidelity(a, b)
    assert abs(r - 1.0) < 1e-6

def test_stability_jaccard():
    m1 = torch.tensor([1, 1, 0, 0, 1])
    m2 = torch.tensor([1, 0, 0, 0, 1])
    j = stability_jaccard(m1, m2)
    assert abs(j - 2 / 3) < 1e-6

def test_physics_alignment_positive_spearman():
    torch.manual_seed(0)
    keep_prob = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    u_ij = torch.tensor([0.9, 0.7, 0.6, 0.3, 0.2, 0.1])  # monotone
    rho, p = physics_alignment(keep_prob, u_ij)
    assert rho > 0.5

def test_calibration_coverage():
    pred = torch.tensor([0.0, 1.0, 2.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    y = torch.tensor([0.2, 0.8, 5.0])
    cov = calibration_coverage(pred, std, y)
    assert cov == 2 / 3
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement `rls/metrics.py`**

```python
"""Evaluation metrics for the paper (interpretability + physics)."""
import torch


def fidelity(pred_pruned, pred_full):
    """Pearson correlation between pruned and full-graph predictions."""
    a, b = pred_pruned.float(), pred_full.float()
    if a.numel() < 2:
        return float("nan")
    ca, cb = a - a.mean(), b - b.mean()
    denom = (ca.norm() * cb.norm()).clamp(min=1e-8)
    return float((ca @ cb / denom).item())


def stability_jaccard(mask1, mask2):
    inter = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    return float((inter / union.clamp(min=1e-8)).item())


def sparsity_stats(masks):
    keep = [float(m.float().mean()) for m in masks]
    return {"mean_keep_frac": float(torch.tensor(keep).mean()),
            "std_keep_frac": float(torch.tensor(keep).std())}


def physics_alignment(keep_prob, u_ij):
    """Spearman rank correlation between edge keep-probability and the
    gravitational binding-energy proxy U_ij = m_i*m_j/r_ij."""
    p, u = keep_prob.detach().cpu().numpy(), u_ij.detach().cpu().numpy()
    import numpy as np
    from scipy.stats import spearmanr
    rho, pval = spearmanr(p, u)
    return float(rho), float(pval)


def calibration_coverage(pred, std, y):
    """95% CI coverage under Gaussian MC-dropout uncertainty."""
    lo, hi = pred - 1.96 * std, pred + 1.96 * std
    return float(((y >= lo) & (y <= hi)).float().mean().item())
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/metrics.py tests/test_metrics.py && git commit -m "feat(rls): interpretability and physics metrics"`

---

### Task 8: Stage-B GNN fine-tune (distribution-shift fix)

**Files:**
- Create: `rls/stageb.py`
- Test: `tests/test_stageb.py`

- [ ] **Step 1: Write the failing test**

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.stageb import fine_tune_gnn

def test_finetune_runs_and_updates_weights():
    torch.manual_seed(0)
    class TinyGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)
        def forward(self, x):
            return self.fc(x).mean()
    gnn = TinyGNN()
    graphs = [{"x": torch.randn(6, 4), "edge_index": torch.randint(0, 6, (2, 10)),
               "edge_attr": torch.randn(10, 5), "y": torch.tensor([0.5]),
               "stellar_mass": torch.rand(6) * 1e10,
               "vel_disp": torch.rand(6) * 100,
               "half_mass_r": torch.rand(6) * 0.01} for _ in range(4)]
    masks = [torch.randint(0, 2, (10,)).bool() for _ in range(4)]
    w0 = gnn.fc.weight.clone()
    fine_tune_gnn(gnn, graphs, masks, epochs=3, lr=1e-3, device="cpu")
    assert not torch.allclose(w0, gnn.fc.weight)
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement `rls/stageb.py`**

```python
"""Stage B: briefly fine-tune the GNN on policy-pruned graphs.

Why: the frozen backbone saw only dense graphs; feeding it pruned graphs at
test time is a distribution shift. A few epochs on (pruned graph -> target)
pairs with MSE (+ optional virial loss) closes the gap while keeping the
policy frozen. The policy stays frozen; only the GNN updates."""
import torch
import torch.nn.functional as F


def fine_tune_gnn(gnn, graphs, masks, epochs=10, lr=1e-4, device="cpu",
                  use_virial=False, loss_fn=None):
    """graphs: list of dicts with x/edge_index/edge_attr/y + physics attrs.
    masks: list of bool tensors (per-edge keep mask) aligned with graphs."""
    gnn = gnn.to(device).train()
    opt = torch.optim.AdamW(gnn.parameters(), lr=lr, weight_decay=5e-5)
    history = []
    for epoch in range(epochs):
        epoch_losses = []
        for g, mask in zip(graphs, masks):
            g = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
            m = mask.to(device)
            x = g["x"]
            ei = g["edge_index"][:, m]
            ea = g["edge_attr"][m]
            pred = gnn({"x": x, "edge_index": ei, "edge_attr": ea})
            target = g["y"].squeeze(-1).float()
            loss = F.mse_loss(pred, target)
            if use_virial and loss_fn is not None:
                vloss = loss_fn(pred, g["y"], {"stellar_mass": g["stellar_mass"],
                                               "vel_disp": g["vel_disp"],
                                               "half_mass_r": g["half_mass_r"]})
                loss = loss + 0.1 * vloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())
        history.append(float(torch.tensor(epoch_losses).mean()))
    return history
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/stageb.py tests/test_stageb.py && git commit -m "feat(rls): stage-B gnn fine-tune on pruned graphs"`

---

### Task 9: Evaluation suite (paper-ready table + plots)

**Files:**
- Create: `rls/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test** (structural: output CSV + PNGs exist, table has required columns)

```python
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from rls.evaluate import build_results_table, save_paper_plots

def test_results_table_columns(tmp_path):
    torch.manual_seed(0)
    rows = build_results_table(
        preds_full=torch.randn(10), preds_policy=torch.randn(10),
        preds_gumbel=torch.randn(10), preds_random=torch.randn(10),
        targets=torch.randn(10),
        masks_policy=[torch.rand(20).round().bool() for _ in range(10)],
        masks_gumbel=[torch.rand(20).round().bool() for _ in range(10)],
        masks_random=[torch.rand(20).round().bool() for _ in range(10)],
    )
    df = pd.DataFrame(rows)
    required = {"method", "rmse", "r2", "scatter", "mean_keep_frac", "fidelity"}
    assert required.issubset(set(df.columns)), df.columns
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement `rls/evaluate.py`**

```python
"""Runs the full evaluation suite and writes paper-ready artifacts:
outputs/rls/results_table.csv + outputs/rls/*.png"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _r2(a, b):
    a, b = a.float(), b.float()
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((b - b.mean()) ** 2).sum()
    return float((1 - ss_res / ss_tot.clamp(min=1e-8)).item())


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def _scatter(a, b):
    return float((a - b).std().item())


def build_results_table(preds_full, preds_policy, preds_gumbel, preds_random,
                        targets, masks_policy, masks_gumbel, masks_random,
                        preds_degree=None, preds_distance=None,
                        masks_degree=None, masks_distance=None):
    """All preds/targets are 1D tensors; masks are lists of bool tensors."""
    rows = []
    entries = [
        ("full", preds_full, None),
        ("random", preds_random, masks_random),
        ("degree", preds_degree, masks_degree),
        ("distance", preds_distance, masks_distance),
        ("gumbel", preds_gumbel, masks_gumbel),
        ("rl_policy", preds_policy, masks_policy),
    ]
    for name, preds, masks in entries:
        if preds is None:
            continue
        preds = preds.float()
        keep = None
        if masks is not None:
            keep = float(torch.stack([m.float().mean() for m in masks]).mean())
        rows.append({
            "method": name,
            "rmse": _rmse(preds, targets),
            "r2": _r2(preds, targets),
            "scatter": _scatter(preds, targets),
            "mean_keep_frac": keep if keep is not None else 1.0,
            "fidelity": float(np.corrcoef(preds.numpy(), preds_full.numpy())[0, 1]),
        })
    return rows


def save_paper_plots(rows, out_dir="outputs/rls"):
    """Sparsity-accuracy Pareto plot (RMSE vs keep-fraction) + fidelity bar."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    # Pareto: lower keep-fraction is better for sparsity; lower RMSE better.
    order = ["full", "random", "degree", "distance", "mass_ratio",
             "grad_saliency", "pgexplainer", "gumbel", "rl_policy"]
    for name in order:
        sub = df[df.method == name]
        if sub.empty:
            continue
        ax[0].plot(sub["mean_keep_frac"], sub["rmse"], "o-", label=name)
    ax[0].set_xlabel("mean keep fraction")
    ax[0].set_ylabel("RMSE (dex)")
    ax[0].set_xlim(1.05, -0.05)
    ax[0].legend()
    ax[1].bar(df.method, df.fidelity, color="steelblue")
    ax[1].axhline(0.98, color="r", ls="--", label="0.98 fidelity")
    ax[1].set_ylabel("fidelity (Pearson)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_and_fidelity.png"), dpi=200)
    plt.close(fig)
    df.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    return df
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/evaluate.py tests/test_evaluate.py && git commit -m "feat(rls): evaluation suite with pareto + fidelity plots"`

---

### Task 10: End-to-end run script + paper-ready experiment driver

**Files:**
- Create: `rls/run_experiment.py` — the full driver: load data → build graphs → load frozen GNN → prepare_graphs → train policy (Stage A) → Stage B fine-tune → evaluate → save everything under `outputs/rls/`.
- Test: smoke-run on 16 synthetic halos via `SyntheticLoader`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_experiment_smoke(tmp_path, monkeypatch):
    """Runs the full pipeline on synthetic data with tiny config; asserts
    outputs/rls/results_table.csv exists and policy improved or matched."""
    # (driver uses config/config.yaml; monkeypatch rls.epochs=1, batch_size=8)
    import sys, os, yaml, glob
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rls.run_experiment import main
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["rls"]["epochs"] = 1
    cfg["rls"]["batch_size"] = 8
    cfg["data"]["source"] = "synthetic"
    cfg["training"]["checkpoint_dir"] = "outputs/checkpoints"
    rc = main(cfg, checkpoint=None)
    assert rc == 0
    assert os.path.exists("outputs/rls/results_table.csv")
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement `rls/run_experiment.py`** (complete driver)

```python
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
    from graph.graph_builder import GraphBuilder, build_dataloaders
    loader = get_loader(cfg)
    train_halos, val_halos, test_halos = loader.split_data()
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_halos, val_halos, test_halos)

    # 2. Frozen backbone
    from model.model import build_model, load_model
    if checkpoint is None:
        checkpoint = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pt")
    gnn = load_model(checkpoint, cfg, device)
    gnn.eval()

    # 3. Prepare policy-training graphs (node embeddings precomputed)
    from rls.train_policy import prepare_graphs
    graphs = prepare_graphs(train_loader, gnn, device)
    val_graphs = prepare_graphs(val_loader, gnn, device)

    # 4. Policy + value net + policy-gradient trainer
    from rls.policy import build_policy
    from rls.policy_gradient import PolicyGradientTrainer, ValueNet
    policy = build_policy(cfg, node_emb_dim=cfg["model"]["output_dim"]).to(device)
    value_net = ValueNet(cfg["model"]["output_dim"]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=rls_cfg["lr"])
    vopt = torch.optim.Adam(value_net.parameters(), lr=rls_cfg["lr"])
    trainer = PolicyGradientTrainer(policy, value_net, opt, vopt, rls_cfg)

    # 5. GNN adapter for rollouts: given (graph_dict, mask) -> preds
    #    CosmicNetGNN.forward REQUIRES a PyG Batch and returns (preds, embeds)
    #    tuple (model.py:307-349) — plain dicts crash it.
    from torch_geometric.data import Data, Batch

    def gnns(graph, mask, use_gnn=gnn):
        with torch.no_grad():
            d_full = Data(x=graph["x"], edge_index=graph["edge_index"],
                          edge_attr=graph["edge_attr"])
            pred_full, _ = gnn(Batch.from_data_list([d_full]))
            m = mask.to(device)
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
    from rls.sparsify import hard_mask, repair_connectivity
    masks_tr = []
    for g in graphs:
        with torch.no_grad():
            p = torch.sigmoid(policy(g["edge_attr"], g["ctx"].to(device),
                                     g["edge_index"], g["ctx"].to(device))).squeeze(-1)
        m = hard_mask(p, rls_cfg.get("min_keep_frac", 0.1))
        m = repair_connectivity(g["edge_index"], m)
        masks_tr.append(m)
    from rls.stageb import fine_tune_gnn
    history = fine_tune_gnn(gnn, graphs, masks_tr, epochs=10, lr=1e-4,
                            device=device)
    torch.save(gnn.state_dict(), os.path.join(out, "finetuned_gnn.pt"))

    # 8. Evaluate
    from rls.sparsify import sparsify_batch  # noqa (placeholder for eval path)
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
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/run_experiment.py tests/test_run_experiment.py && git commit -m "feat(rls): end-to-end experiment driver"`

---

### Task 11: Cross-simulation validation (TNG → CAMELS)

**Files:**
- Create: `rls/cross_sim.py`
- Test: `tests/test_cross_sim.py` (uses CAMELS synthetic fallback path — `data.source: camels` with missing HDF5 triggers `_generate_synthetic_camels`, so the test exercises the plumbing without downloads)

- [ ] **Step 1: Write the failing test**

```python
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.cross_sim import evaluate_cross_sim

def test_cross_sim_runs(tmp_path):
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["source"] = "camels"
    cfg["data"]["camels"] = {"suite": "IllustrisTNG", "simulation": "LH_0",
                             "cache_dir": str(tmp_path / "camels_cache")}
    # synthetic fallback is generated when the HDF5 is missing
    out = evaluate_cross_sim(cfg, checkpoint=None, max_halos=8,
                             out_dir=str(tmp_path / "out"))
    assert os.path.exists(os.path.join(out, "cross_sim_results.csv"))
```

- [ ] **Step 2: Run, verify FAIL** (will fail on missing module; CAMELS download may be slow — the test uses the loader's synthetic fallback)

- [ ] **Step 3: Implement `rls/cross_sim.py`**

```python
"""Cross-simulation OOD evaluation: policy trained on TNG, tested on CAMELS.

The headline generalization claim of the paper. Uses the repo's CAMELSLoader
(HDF5 download or synthetic fallback) and the SAME frozen policy + GNN."""
import os
import torch
import pandas as pd

def evaluate_cross_sim(cfg, checkpoint, max_halos=200, out_dir="outputs/rls"):
    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder
    from model.model import load_model
    from rls.policy import build_policy
    from rls.sparsify import hard_mask, repair_connectivity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(cfg)
    halos = loader.load()[:max_halos]
    gb = GraphBuilder(cfg)
    graphs = gb.build_graphs(halos)

    gnn = load_model(checkpoint, cfg, device)
    gnn.eval()
    policy = build_policy(cfg).to(device)
    policy_path = os.path.join(out_dir, "policy.pt")
    assert os.path.exists(policy_path), "train policy first (Task 10)"
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    rows = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            # CosmicNetGNN.forward/get_embeddings require a PyG Batch (.batch
            # attribute) — wrap single Data objects (model.py:323-326).
            from torch_geometric.data import Batch
            b = Batch.from_data_list([g])
            emb = gnn.get_embeddings(b, embedding_point="pre_pooling")
            from torch_geometric.nn import global_mean_pool
            ctx = global_mean_pool(emb, b.batch)
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
    df.to_csv(os.path.join(out_dir, "cross_sim_results.csv"), index=False)
    return out_dir
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git add rls/cross_sim.py tests/test_cross_sim.py && git commit -m "feat(rls): cross-simulation OOD evaluation"`

---

### Task 12: README + results summary table for the paper

**Files:**
- Modify: `README.md` (add RL-Cosmic-Net section)
- Create: `outputs/rls/RESULTS_TEMPLATE.md` (paper numbers table)

- [ ] **Step 1:** Copy the "PART 3 — Evaluation & Benchmarks" table below into `outputs/rls/RESULTS_TEMPLATE.md` and replace targets with actual run values after Task 10-11.

- [ ] **Step 2:** Add to README under "Key Features":

```markdown
## RL Edge Sparsification (RL-Cosmic-Net)

A frozen-backbone, physics-informed policy-gradient policy (`rls/`) prunes
task-irrelevant edges before GNN inference — the pruned graph *is* the
explanation. REINFORCE with a learned baseline + entropy bonus (one-step MDP;
honestly NOT PPO) and a relative-RMSE + sparsity-curriculum + virial-consistency
reward; baselines: random/degree/distance/mass-ratio/gradient-saliency/
PGExplainer/attention-topk/Gumbel-softmax. Stage-B GNN fine-tune closes the
train/test distribution shift. See `rl-implementation-plan.md`.
```

- [ ] **Step 3: Commit** — `git add README.md outputs/rls/RESULTS_TEMPLATE.md && git commit -m "docs: RL sparsification README + results template"`

---

## PART 3 — EVALUATION & BENCHMARKS (paper table template)

| Method | RMSE (dex) | R² | Scatter (dex) | Keep frac | Fidelity | Spearman ρ vs U_ij | Virial ratio (med) | 95% CI coverage |
|---|---|---|---|---|---|---|---|---|
| Full graph (frozen) | 0.137 | 0.965 | ~0.10 | 1.0 | 1.0 | — | ~1.0 (both defs) | ~0.95 |
| Random drop (50%) | ~0.20 | — | — | 0.5 | — | ~0 | report | — |
| Degree drop (50%) | ~0.17 | — | — | 0.5 | — | low | report | — |
| Distance drop (50%) | ~0.15 | — | — | 0.5 | — | high | report | — |
| Mass-ratio top-k (50%) | ~0.14 | — | — | 0.5 | — | med (confounded) | report | — |
| Gradient saliency (repo explain/) | ~0.15 | — | — | 0.5 | — | med | report | — |
| PGExplainer (PyG, if time) | ? | — | — | ? | — | ? | report | — |
| Gumbel-softmax (joint) | ~0.14 | — | — | 0.5 | — | med | report | — |
| **RL policy (50%)** | **≤0.137** | **≥0.965** | **≤0.10** | **0.5** | **≥0.98** | **≥0.5 (and > mass-ratio row)** | **same as baselines ±0.05** | **≥0.93** |
| RL policy + Stage B | ≤0.135 | ≥0.967 | ≤0.10 | 0.5 | ≥0.98 | ≥0.5 | same | ≥0.93 |
| TNG→CAMELS (full) | ~0.2 | ~0.9 | — | 1.0 | 1.0 | — | — | — |
| TNG→CAMELS (RL) | ≤ full | ≥ full | — | 0.5 | ≥0.95 | ≥0.4 | — | — |

**Rules for reviewers (follow strictly):**
1. All methods share the SAME frozen backbone weights (except Gumbel baseline, which is joint-trained — say so explicitly; that is its unfair advantage, making RL's win stronger).
2. **Virial ratio is reported for EVERY method at matched sparsity** — because both KE_retained and PE_retained shrink mechanically with sparsity, a ratio in [0.9, 1.1] at 50% sparsity is not evidence of physical selection unless it is compared against random/degree/distance at the SAME keep fraction. The RL claim is "same virial ratio at equal or better accuracy/fidelity", never "RL achieved virial consistency".
3. **Gradient-saliency and mass-ratio top-k rows are mandatory** (they were missing entirely): the repo already ships the first (`explain/explainer.py`), and the second bounds the degenerate "keep massive pairs" policy that `mass_ratio`-aware features make trivial. RL must beat both on U_ij alignment.
4. Report 5-seed mean ± std for RMSE and stability (Jaccard) — 541 graphs is small; reviewers will ask. (Triage note: if the deadline forces it, single-seed + explicit "preliminary" framing is honest — but say "preliminary" in the paper.)
5. Report mass-binned RMSE (low/mid/high halo mass) to show pruning doesn't reintroduce HaloGraphNet's low-mass bias.
6. Ablation: reward without virial term; reward without sparsity curriculum; policy with min_keep=0.5. Three rows.
7. Runtime: policy adds <1ms/graph; report speedup at 50% sparsity (edge reduction → faster GNN message passing).
8. **Virial definition note**: the training-time VirialLoss (monopole, model/physics_loss.py) and the reward's pairwise ratio are different formulas — state this explicitly and report both on the full graph so reviewers see they were reconciled, not conflated.

---

## PART 4 — COMMON PITFALLS (checklist)

**RL-specific**
- [ ] **Policy collapse**: policy drops everything → connectivity penalty + `min_keep_frac` clamp + entropy bonus (0.01) prevent this. If `keep_frac` hits the floor for >5 consecutive epochs, raise `w_conn`.
- [ ] **Reward hacking**: keeping all edges maximizes accuracy trivially. The relative-RMSE term alone doesn't punish it — the sparsity curriculum (target annealed 0.9→0.4) is what does. Verify `keep_frac` actually tracks `target_sparsity`.
- [ ] **Algorithm honesty (was "not PPO")**: this plan implements REINFORCE with a learned baseline + entropy, NOT PPO. GAE at horizon 1 degenerates to `reward − value` and a clipped-importance ratio computed from the same forward pass is identically 1. Do not write "PPO" or "GAE(λ=0.95)" in the paper. If reviewers ask why not PPO: one-step MDP, no sequential credit assignment to exploit — the paper says exactly this.
- [ ] **High variance**: always center advantages (subtract the learned baseline — the ValueNet), batch ≥ 16 graphs. If loss oscillates wildly, halve lr to 5e-4 or raise batch to 64.
- [ ] **Gradient leakage**: the backbone must be in `eval()` + `torch.no_grad()` during rollouts (Stage A). The policy must NOT receive gradients from the GNN — otherwise it silently becomes a differentiable mask and your "frozen backbone" claim in the paper is false. In `_update_batch`, only `logp` and the entropy term carry gradients; masks, GNN calls, rewards and baseline are under `no_grad`.
- [ ] **Permutation equivariance**: `EdgePolicyNet.forward` must produce identical logits when edges are reordered. Test `test_policy_equivariant_to_node_ordering_within_edge` guards this — never use global index-based features in the policy.
- [ ] **MC-dropout in eval**: `predict_with_uncertainty` sets `train()` mode; do not mix it into rollout reward computation (rewards need deterministic preds). Use plain `forward`.
- [ ] **Degenerate shortcut policy (U_ij confound)**: `mass_ratio` is an input edge feature, so "keep high-mass-ratio edges" trivially produces non-zero Spearman ρ vs U_ij (M_i·M_j/r ∝ mass_ratio). The mass-ratio top-k baseline row must be reported at every sparsity, and the RL headline must be "beats the trivial shortcut", not just "ρ ≥ 0.5".
- [ ] **Two virial definitions**: training-time `VirialLoss` (monopole with predicted mass, model/physics_loss.py) vs the reward's pairwise ratio. Reconcile in one paragraph, report both on the full graph, and report the reward ratio for ALL baselines at matched sparsity — otherwise a reviewer who knows the virial theorem will shred the physics claim.
- [ ] **Baseline reproducibility (Task 0)**: 541 halos cannot come from `n_halos: 500` + `min_subhalos_per_halo: 3` (loader drops halos). Commit the exact config + `baseline_metrics.json` before any RL run, or the numbers you claim to beat are indefensible.

**Repo/environment-specific**
- [ ] `num_workers > 0` in PyG DataLoader on **Windows** → multiprocessing spawn errors. Use `num_workers=0` in all local runs and Kaggle notebooks.
- [ ] `load_model` handles checkpoints with/without `model_state_dict` key — `kaggle/best_model_augmented.pt` is a raw state dict (3.3 MB); `outputs/checkpoints/best_model.pt` is a full dict (511 MB). `run_experiment.py` handles both via `load_model`.
- [ ] `data/raw/tng100_clustered.csv` and `kaggle/best_model_augmented.pt` are **gitignored** — never rely on them being on GitHub/Kaggle. Upload as a Kaggle dataset.
- [ ] `split_data()` resets the global NumPy RNG (base_loader) — call `torch.manual_seed`/`np.random.seed` AFTER split, before policy training (run_experiment.py does this).
- [ ] Config coupling: `model.output_dim` must equal the embedding dim used by the policy input (`build_policy` defaults to it). If you change `model.hidden_dim`, the frozen checkpoint becomes incompatible — use the checkpoint's own config.
- [ ] `lambda` config keys in `training.` are dead keys (repo quirk) — not relevant here but don't copy them into `rls:`.

**Kaggle-specific**
- [ ] 12h session timeout: checkpoint policy every 10 epochs; notebook must be resumable (reload `policy.pt` if present).
- [ ] 30h/week quota: Stage A (60 epochs × ~8s) ≈ 20–30 min on T4; baselines + eval ≈ 15 min; total 1 notebook session per run. Budget 3 sessions/week: train, eval+plots, cross-sim.
- [ ] Internet is ON: `pip install torch-geometric` works, but pin versions matching Kaggle's torch (2.5/2.6) — do NOT force torch==2.1.0 on Kaggle (preinstalled torch 2.6 + pip pyg 2.6.x is the combo the Kaggle notebooks use).
- [ ] `os.cpu_count()` on Kaggle reports 8+ — set DataLoader `num_workers=2` max.
- [ ] Uploads: `tng100_clustered.csv` (~250 MB) and `best_model_augmented.pt` via "Add Data" → private dataset; mount at `/kaggle/input/<name>/`.

---

## PART 5 — VENUE & TIMELINE (verified Aug 2026; notification constraint: ≤ 1st week Nov 2026)

| Date (2026) | Milestone | Notes |
|---|---|---|
| Aug 5 (today) | Baseline frozen: GNN 0.137 dex | Already done |
| Aug 5–8 | Tasks 0–4 (baseline commit, policy/reward/policy-gradient/sparsify) | 4 days |
| Aug 9–12 | Task 5–8 (training, baselines, metrics, stage B) | 4 days |
| Aug 13–16 | Task 10 run on Kaggle T4 (3 sessions) | First numbers |
| Aug 17–19 | Task 11 cross-sim; fix; rerun | Headline OOD result |
| Aug 20 | **IEEE BigData submission freeze** (paper + numbers locked) | Submit Aug 21 |
| **Aug 21** | **IEEE BigData 2026 submission** (VERIFIED: bigdataieee.org/BigData2026/important-dates/) | 10 pages IEEE 2-column incl. refs; notification Oct 24 |
| Aug 22–24 | IEEE version → 4-page NeurIPS template version (90% content reuse) | Only if ML4PS 2026 CFP has appeared; else skip |
| Aug 25–29 | Buffer; rerun any reviewer-critical experiment | — |
| late Aug (unverified) | ML4PS 2026 CFP expected (~Aug 29 pattern from 2025) | Check NeurIPS.cc/Conferences/2026/Dates + ml4physicalsciences.github.io; 4 pages excl. refs, non-archival, double-blind |
| **Oct 24** | **IEEE BigData 2026 notification** | Satisfies ≤ Nov 1 constraint ✓ |
| Nov 14 | IEEE BigData camera-ready | Conf Dec 14–17, Phoenix AZ |
| Dec 16 | ICLR 2027 decisions | Misses Nov-1 constraint → not in plan; revisit for 2028 |
| 2027 Q1 | **ApJ/MNRAS submission** (extended: 8–10 pages, 5-seed CIs, ablations, CAMELS LH-scale) | Most reliable venue for this work; review 6–12 months |
| 2027 Q3 | ICLR 2028 if extending to CAMELS 1000-sim + standard benchmarks | Optional |

**Venue facts to know (corrected):**
- **IEEE BigData 2026 (PRIMARY)**: submission Aug 21, notification Oct 24, camera-ready Nov 14, conference Dec 14–17 Phoenix AZ. 10 pages IEEE 2-column incl. references. Only verified venue whose decision lands before Nov 1, 2026. Estimated acceptance odds: 60–70% for this work.
- **ML4PS (NeurIPS 2026)**: 2026 CFP NOT yet published — the "Aug 29" date in earlier drafts was inferred from the 2025 deadline pattern, NOT verified. Format (from 2025): 4 pages excluding references, NeurIPS template, non-archival (safe to submit to journal later), double-blind, OpenReview. Odds if CFP appears: 65–80% (non-archival).
- **ICLR 2027**: abstract Sep 18, full Sep 25, reviews Nov 5, decisions Dec 16 — decisions miss the Nov-1 constraint. Drop from near-term plan.
- **"Indigo conference"**: no credible venue found by that name — likely a misremembering. Real backups: **LoG (Learning on Graphs)**, **ECML-PKDD**, **AAAI**, **KDD AI4Science workshop**, **ICML AI4Science workshop**. None has a decision before Nov 1, 2026 — they are fallbacks, not Nov-1 options.
- **Journals (ApJ/MNRAS/A&A)**: best long-term fit (80%+ eventual acceptance with revisions) but 6–12 month review — do NOT meet the Nov-1 notification constraint.
- **RISK NOTE**: venue choice is secondary — IEEE (Aug 21) and a likely ML4PS (~Aug 29) both land ~2 weeks out; **code + results must be done by mid-Aug**. That is the real gating factor, not venue selection.

**TRIAGE (apply in this order if behind schedule — e.g., policy training doesn't converge by Aug 16):**
1. **Drop 5-seed reporting** → single seed + explicit "preliminary" framing in the paper (honest; reviewers accept this for a conference paper only if stated).
2. **Drop Stage-B fine-tune** → report the frozen-backbone-only numbers (still a complete contribution: RL vs baselines at fixed backbone).
3. **Drop CAMELS cross-sim** unless the real HDF5 is already cached locally — the loader's synthetic fallback is for plumbing tests only and is NOT publishable as "OOD validation".
4. **Gradient-saliency baseline stays** (cheap — reuses `explain/explainer.py`); real PGExplainer is the first thing to cut if short on time.
5. **Mass-ratio top-k baseline stays** (10 lines, bounds the degenerate policy — required for the U_ij claim).
6. OGB/ZINC already cut. What survives is still a legitimate contribution: policy gradient vs gradient-saliency/PGExplainer vs simple baselines on TNG sparsity/fidelity/physics-alignment.

---

## PART 6 — SAKANA AI SCIENTIST (your question, answered)

**Is it free?** The software is open-source and free (GitHub SakanaAI/AI-Scientist, its own "AI Scientist Source Code License" — a Responsible-AI-derived license; free to use, but you MUST disclose AI use in any resulting manuscript). It is **not** free to run: it drives an LLM API (OpenAI/Anthropic/DeepSeek/Gemini). Costs: **< $15 per generated paper** with Claude Sonnet 3.5; **much cheaper with DeepSeek** (DeepSeek API is supported out of the box: `DEEPSEEK_API_KEY`, models `deepseek-chat`/`deepseek-reasoner`). It needs Linux + GPU + `texlive-full` (heavy) — not practical on Kaggle quotas, but fine on your local machine if you have one.

**How it can help THIS project (recommended usage):**
1. **`perform_review.py`** — the strongest component: feed it your 4-page ML4PS draft before submission; get an ICLR-style review (score, weaknesses, accept/reject) with 5 reflections × 5-review ensemble. Cost: pennies with DeepSeek. Use it as a free pre-submission reviewer.
2. **Idea & novelty check** — its Semantic Scholar integration flags prior work (e.g., it will find PTDNet/GSAT for you — I already did; keep the lit review I provide).
3. **Paper drafting assistance** — it can write a workshop-level draft from your experiment logs; treat it as a co-drafter, NOT the final text (its papers are accepted at workshops, not ICLR main tracks).
4. **Template building** — you'd create a custom template (experiment.py + plot.py + prompt.json + latex/template.tex) wrapping `rls/run_experiment.py`; ~1-2 days of work. Only worth it if you want autonomous baseline sweeps.

**Warnings:**
- The license mandates disclosure ("This manuscript was autonomously generated using The AI Scientist"). At ICLR/ApJ, full-autonomy disclosure can hurt credibility — use it for internal review/drafting and do the final writing yourself.
- ICLR has an LLM-use policy for submissions; disclosed limited assistance (grammar/formatting) is generally acceptable, full auto-generation is not.
- Quality ceiling: its generated papers are workshop-level; the reviewer component is its best ROI.

---

## PART 7 — SUPPORTING FILES INDEX

| File | Contents |
|---|---|
| `rl-implementation-plan.md` | This file — full DeepSeek-executable plan |
| `rl-lit-review.md` | 20 papers (top 10 priority + 10 lower) in your requested format |
| `rl-kaggle-notebooks.md` | 3 complete Kaggle notebooks (copy-paste cells): A) setup+baselines, B) policy training, C) evaluation+cross-sim+plots. **NOTE: notebook titles/sections still say "PPO" — align naming to policy gradient (Task 3) when implementing; the `get_example(i)` adapter patterns in them are the verified working versions** |
| `REPOWISE.md` | Repo map (previous work) |
