# RL-Cosmic-Net — Literature Review (20 papers)

**Purpose:** Papers to cite in the RL-Cosmic-Net paper (primary target: IEEE BigData 2026, submission Aug 21; secondary: NeurIPS 2026 ML4PS workshop if CFP appears; follow-up: ApJ/MNRAS journal). Top 10 = must-cite (method pillars, baselines, physics benchmarks). Lower 10 = supporting context / related work / benchmark citations.

---

## TOP 10 — PRIORITY CITATIONS

| Sr | Authors (short) | Journal / Conference | Year | Dataset | Methodology (short) | Quantitative results | Limitations |
|---|---|---|---|---|---|---|---|
| 1 | Villanueva-Domingo et al. | ApJ (arXiv:2111.08683) | 2022 | CAMELS (2000+ sims, IllustrisTNG + SIMBA; CV/LH sets) | GNN (PyG, edge-conv) predicts log M_halo from subhalo positions, velocities, stellar masses, radii; likelihood-free posterior std | R² 0.96–0.97 CV, 0.90–0.92 LH; scatter ~0.14 dex CV, ~0.2 dex LH; beats stellar-mass-only fit (0.2–0.4 dex scatter) | Low-mass bias (up to 0.4 dex below 10^11 M_sun/h); needs full simulation subhalo catalogs; no structural interpretability |
| 2 | Luo et al. (PTDNet, "Learning to Drop") | WSDM (ACM) | 2021 | Synthetic + node-classification benchmarks (Cora, Citeseer, Pubmed, etc.) | **RL (policy gradient) edge dropping** + nuclear-norm low-rank constraint; parameterized topological denoising; joint training with GNN | Significant accuracy gains on noisy graphs; larger gains with more noise; matches/complements GCN/GraphSAGE/GAT | Node-classification focus; TensorFlow; joint (not frozen) training; no physics; no graph-level regression |
| 3 | Miao et al. (GSAT) | ICML (PMLR 162) | 2022 | 8 graph datasets (synthetic + real, e.g., Spurious-Motif, Graph-SST2, Molhiv) | Stochastic attention from Information Bottleneck; injects noise into attention, learns task-relevant subgraphs inherently | Up to +20% interpretation AUC, +5% prediction accuracy vs SOTA; +12% spurious-correlation removal | No exact sparsity control; needs joint training (cannot post-hoc sparsify frozen GNN); soft stochastic masks, not hard decisions |
| 4 | Yu et al. (GIB) | NeurIPS | 2021 | Graph classification benchmarks (MUTAG, NCI1, PROTEINS, DD) | Graph Information Bottleneck: max MI(subgraph, Y) − min MI(subgraph, X); learns explanatory subgraphs | Competitive accuracy + interpretability; stable against distribution shift | NP-hard optimization (relaxed); unstable training; graph-classification only |
| 5 | Ying et al. (GNNExplainer) | NeurIPS | 2019 | Synthetic (BA-Shapes, BA-Community) + real (MUTAG, Reddit) | Post-hoc node/edge feature-mask optimization explaining a trained GNN | Recovers ground-truth motifs; state-of-the-art at the time | Post-hoc (can break manifold); per-instance optimization cost; NP-hard; fidelity issues documented by later work |
| 6 | Luo et al. (PGExplainer) | ICLR | 2020 | Synthetic + real graph benchmarks | Post-hoc global edge-mask predictor via MLP over node embeddings (inductive) | Up to 17.5% AUC gain in explanation quality vs GNNExplainer; 8× faster | Post-hoc; gradient-based (breaks continuous spatial manifolds — your README already argues this for astro); ignores edge features |
| 7 | Wu et al. (DIR) | WWW | 2022 | Synthetic + real (Graph-SST2, ogbg-molhiv, Twitch) | Discover Invariant Rationales: IB-based rationale extraction + variance minimization across environments | State-of-the-art OOD generalization among rationale methods | Needs environment/domain labels; joint training; classification focus |
| 8 | Cranmer et al. | NeurIPS | 2020 | N-body / simulated dynamics | **Distill NN-learned physics into symbolic equations** (inductive biases + SR) | Rediscovered known physical laws; interpretable analytic forms | Early SR methods; limited expression complexity; small-scale |
| 9 | Cranmer | arXiv:2305.01582 | 2023 | SRBench / Feynman / general | PySR: genetic-programming SR engine with Pareto-front model selection | SOTA-quality SR engine; 59/100 exact on Feynman (vs 40 AI-Feynman, 23 DSR) | GP cost scales with population; needs compute; no physics priors by default |
| 10 | Grayeli et al. (LaSR) | NeurIPS | 2024 | Feynman equations, synthetic, LLM scaling laws | LLM-guided SR with learned concept library (zero-shot LLM genetic operations) | **72/100 exact on Feynman (SOTA)** vs PySR 59/100; beats all SR baselines | Requires LLM API; stochastic guidance quality; more compute; not on Kaggle-free quotas |

---

## LOWER 10 — SUPPORTING / CONTEXT

| Sr | Authors (short) | Journal / Conference | Year | Dataset | Methodology (short) | Quantitative results | Limitations |
|---|---|---|---|---|---|---|---|
| 11 | La Cava et al. (SRBench) | NeurIPS D&B Track | 2021 | 252 PMLB regression problems, 14 SR methods | Living benchmark of SR methods w/ unified metrics | Robust method ranking; PySR/Operon near top | No physics-domain datasets; 2021 edition (2025 update exists) |
| 12 | Petersen et al. (DSR) | ICLR | 2021 | SRBench/Feynman, synthetic | **RL (RNN + REINFORCE with risk-seeking policy gradient) symbolic regression** | 60%+ exact recovery on Feynman subset; SOTA at the time | Small expression lengths; RL sample-efficiency; no physics priors |
| 13 | Landajuela et al. (uDSR) | NeurIPS | 2022 | SRBench/Feynman | Unified DSR: RL + GP + neural-guided priors in one framework | Beats DSR and GP alone on SR benchmarks | Complex machinery; needs tuning |
| 14 | Shojaee et al. (LLM-SR) | NeurIPS | 2024 | Synthetic + physics benchmarks | LLM in-the-loop SR: prompt → expression → refinement loop | SOTA on several SR benchmarks vs DSR/PySR at the time | Needs LLM API; token cost; variability |
| 15 | Shojaee et al. (LLM-SRBench) | ICML (Oral) | 2025 | 239 equation-discovery problems (LSR-Transform + LSR-Synth) | Benchmark for LLM equation discovery; LaSR best numerical accuracy; LLM-SR best symbolic | LaSR highest Acc0.1 on LSR-Transform; GPT-4o-mini symbolic ~31% | All methods still low absolute performance; LLM-dependent |
| 16 | Yuan et al. (SubgraphX) | NeurIPS | 2021 | Synthetic + real graph benchmarks | Shapley-value subgraph attribution with MCTS | SOTA explanation quality on fidelity metrics | MCTS cost; per-instance; classification focus |
| 17 | Chen et al. (CIGA) | NeurIPS | 2022 | Synthetic (Spurious-Motif) + real (ogbg-molhiv) | Causally-invariant subgraph learning via information-theoretic objective | Best OOD + interpretability on spurious-feature tasks | Needs environment partitions; joint training |
| 18 | Zheng et al. (Neural Sparsification) | NeurIPS | 2020 | Node classification benchmarks | Differentiable edge-mask sparsification (L0-style) for GNN robustness | Comparable accuracy with much sparser graphs | Joint training; classification; no physics |
| 19 | Villaescusa-Navarro et al. (CAMELS) | ApJ | 2021 | 4233+ sims (IllustrisTNG/SIMBA; LH/1P/CV) | Public multi-simulation suite for ML in cosmology | Enables your cross-sim and HaloGraphNet baselines | Large downloads; units/conventions care needed |
| 20 | Villanueva-Domingo et al. | ApJ (arXiv:2111.14874) | 2022 | Milky Way + Andromeda data | Apply HaloGraphNet GNN to infer MW/M31 total masses | M31 consistent with observational constraints | Model trained on CAMELS, applied to real galaxies; systematics |

---

## Suggested citation clusters for the paper

- **Method position (§1/§2):** [1] HaloGraphNet → your backbone; [2] PTDNet + [18] Neural Sparsification → "prior RL/differentiable edge dropping"; [3] GSAT + [4] GIB + [7] DIR + [17] CIGA → "differentiable/IB interpretability family"; [5][6][16] → "post-hoc explainers we avoid".
- **Physics/SR (§4/§5):** [8] Cranmer 2020 (GNN→equation distillation), [9] PySR (your engine), [12][13] RL-SR (your "RL" family lineage), [10][14][15] LLM-SR (future work), [11] SRBench (SR benchmarking), [19] CAMELS (data), [20] MW/M31 application.
