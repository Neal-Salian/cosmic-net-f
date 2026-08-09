# RL-Cosmic-Net Results (paper table)

Fill in actual values after `python rls/run_experiment.py` + `rls/cross_sim.py`
+ TTA evaluation. All methods share the SAME frozen backbone weights (except
Gumbel — joint-trained; its unfair advantage makes the RL win stronger).

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
| **RL policy FROZEN (ablation)** | ≤0.137 | ≥0.965 | ≤0.10 | 0.5 | ≥0.98 | ≥0.5 | same | ≥0.93 |
| **RL-TTA (K=10, offline init)** | **≤ FROZEN** | **≥ FROZEN** | **≤0.10** | **0.5** | **≥0.98** | **≥ frozen ρ** | **same** | **≥0.93** |
| RL-TTA (fresh init) | worse than offline init | — | — | 0.5 | — | — | — | — |
| TNG→CAMELS (full) | ~0.2 | ~0.9 | — | 1.0 | 1.0 | — | — | — |
| TNG→CAMELS (RL frozen) | ≤ full | ≥ full | — | 0.5 | ≥0.95 | ≥0.4 | — | — |
| **TNG→CAMELS (RL-TTA) — headline** | **< frozen** | **> frozen** | — | 0.5 | ≥ frozen | ≥ frozen | — | — |
| TTA ablation: K ∈ {0,5,10,20} | K-curve on RMSE | — | — | 0.5 | — | — | — | K=0 ≡ frozen (sanity) |
| TTA ablation: reward −unc / −virial / −sparsity | each term contributes | — | — | — | — | — | — | — |

Reviewer rules (from `rl-implementation-plan.md` Part 3): virial ratio reported
for EVERY method at matched sparsity; mass-ratio + gradient-saliency rows are
mandatory; 5-seed mean ± std; mass-binned RMSE; reward ablations; K=0 must
reproduce the frozen policy exactly; report mean TTA wall-clock per graph.
