# REPOWISE: Cosmic-Net Repository Map

Auto-generated structure & architecture map for **Cosmic-Net** — a physics-informed Graph Neural Network (GNN) for predicting dark matter halo masses, with symbolic regression and XAI distillation.

- **Stack:** Python 3.10, PyTorch 2.1, PyTorch Geometric 2.4, FastAPI, PySR/gplearn, W&B
- **Based on:** [HaloGraphNet](https://arxiv.org/abs/2204.07077) (Villanueva-Domingo et al., 2022)
- **License:** MIT

---

## 1. What This Repo Does

Cosmic-Net predicts dark matter halo mass `M_halo` from the distribution of galaxies/subhalos within each halo:

1. **Data layer** normalizes multiple cosmological simulation sources (synthetic mock data, IllustrisTNG-100, CAMELS HDF5, CAMELS via HuggingFace) into a single schema.
2. **Graph layer** converts each halo's subhalos into a PyTorch Geometric graph (nodes = subhalos, edges = radius/KNN connectivity, 5 physics-motivated edge features).
3. **Model layer** trains an **Edge-Conditioned NNConv GNN** with a **virial-theorem physics-informed loss** (`2·KE + PE ≈ 0`) and MC-Dropout uncertainty.
4. **Explainability** produces node/edge importance masks (gradient saliency + learnable-mask optimization) for model interpretation.
5. **Symbolic regression** (PySR/gplearn) distills the trained model into human-readable equations, post-filtered by dimensional analysis.
6. **Deployment** serves the model via a FastAPI REST API, containerized for CPU-only inference.

**CLI entry point:** `main.py` — modes `train | evaluate | explain | symbolic | serve | full`.

---

## 2. Directory Tree

```
cosmic-net/
├── main.py                       # CLI entry point, orchestrates the full pipeline
├── requirements.txt              # pinned deps (torch 2.1, pyg 2.4, pysr 0.16, fastapi 0.103...)
├── config/
│   └── config.yaml               # central config (data, graph, model, training, explain, symbolic, api, wandb)
├── data/                         # data layer
│   ├── __init__.py               # bare package marker (no exports)
│   └── loaders/
│       ├── __init__.py           # re-exports base_loader symbols
│       ├── base_loader.py        # SubhaloData/HaloData schema, BaseDataLoader ABC, get_loader factory
│       ├── synthetic_loader.py   # mock data (generates on-the-fly if missing)
│       ├── tng_loader.py         # IllustrisTNG-100 (local CSV or REST API + pickle cache)
│       └── camels_loader.py      # CAMELS (HDF5 download) + CAMELSHuggingFaceLoader
├── graph/
│   └── graph_builder.py          # HaloData -> PyG Data; radius/KNN edges, 5 edge features
├── model/
│   ├── model.py                  # CosmicNetGNN (NNConv blocks), EnsembleGNN, build/load_model
│   └── physics_loss.py           # VirialLoss, PhysicsInformedLoss, MetricsComputer, GradientScaler
├── training/
│   ├── train.py                  # Trainer loop, CosineLambdaScheduler, train_model, run_ablation_study
│   └── scheduler.py              # LambdaScheduler, WarmupScheduler, build_optimizer, EarlyStopping, TrainingState
├── explain/
│   ├── explainer.py              # CosmicNetExplainer (gradient saliency + custom mask optimization)
│   └── visualize.py              # matplotlib figs: 3D cluster, Pareto front, training curves, scatter
├── symbolic/
│   └── symbolic_regression.py    # SymbolicRegressor (PySR/gplearn), DimensionalAnalyzer, feature extraction
├── deploy/
│   ├── api.py                    # FastAPI app (predict/batch/explain/health/config/model-info)
│   └── Dockerfile                # multi-stage CPU-only image
├── scripts/
│   ├── fetch_tng_data.py         # standalone TNG API -> HDF5/CSV downloader
│   ├── generate_paper_plots.py   # research-paper figures (8 numbered plots, PNG+PDF)
│   └── generate_xai_plots.py     # saliency edge-importance analysis -> outputs/xai_results/
├── tests/
│   ├── test_graph_builder.py     # radius/KNN/self-loops/edge features/symmetry/determinism/hierarchical
│   ├── test_loaders.py           # schema, synthetic loader, factory, splits, determinism
│   └── test_physics_loss.py      # virial equilibrium, loss composition, metrics
├── kaggle/                       # research artifacts (gitignored checkpoints & notebooks)
│   ├── best_model (2).pt         # baseline trained model
│   ├── best_model_augmented.pt   # augmented/physics-informed trained model
│   ├── gnn_embeddings_for_sr.csv # node embeddings for symbolic regression
│   ├── training-cosmic (4).ipynb # Kaggle training notebook (T4 GPU)
│   └── pysr-cosmic.ipynb         # symbolic-regression notebook
├── notebooks/
│   └── exploration.ipynb         # empty placeholder
├── outputs/                      # generated artifacts (gitignored)
│   ├── checkpoints/              # best_model.pt, checkpoints every N epochs
│   ├── explanations/             # all_explanations.json (Three.js-ready masks)
│   ├── equations/                # Pareto CSV + dimensional-consistency .tex
│   ├── logs/                     # cosmic_net.log
│   ├── research_paper_plots/     # figures 1-8 (PNG+PDF)
│   ├── paper_figures/            # EDA + model comparison PDFs
│   └── xai_results/              # saliency_analysis.md + saliency_edge_importance.png
├── raw_hdf5/                     # local TNG group-catalog cache (gitignored, partial downloads)
├── data/raw/                     # cached datasets (gitignored)
├── .github/workflows/ci.yml      # non-blocking CI (preflight-ml + basic checks, continue-on-error)
├── .env.example                  # TNG_API_KEY, WANDB_API_KEY, HF_TOKEN, API_HOST/PORT...
└── README.md                     # full docs, API snippet, Azure deployment guide
```

---

## 3. Core Data Flow

```
config/config.yaml
   │  (data.source = 'tng', graph.*, model.*, physics.*, training.*)
   ▼
get_loader(config) | get_train_test_loaders(config)     [data/loaders/base_loader.py]
   ▼
BaseDataLoader subclass
   ├─ SyntheticLoader  -> .pt + .csv (or auto-generates 500 halos)
   ├─ TNGLoader        -> clustered CSV -> raw CSV -> TNG REST API (pickle cache)
   ├─ CAMELSLoader     -> HDF5 streamed download (or synthetic fallback)
   └─ CAMELSHFLoader   -> HuggingFace datasets (or synthetic fallback)
   │
   │  load() -> List[HaloData] -> split_data() -> (train, val, test)
   ▼
GraphBuilder [graph/graph_builder.py]
   │  build_graphs(halos) -> List[pyg Data]
   │    x=[N,4] | edge_index=[2,E] | edge_attr=[E,5] | y=[log10(M_halo)]
   │    + raw physics attrs: stellar_mass, vel_disp, half_mass_r, pos, vel
   ▼
build_dataloaders -> PyG DataLoader(train/val/test)
   ▼
Trainer [training/train.py]
   ├─ loss = MSE(pred, y) + lambda(t) * VirialLoss          [model/physics_loss.py]
   ├─ cosine LR (T_max=300) + cosine lambda annealing (warmup 75 epochs)
   ├─ early stopping (patience 60, monitor val/mse) + best_model.pt
   └─ W&B logging (optional, wandb.enabled=false)
   ▼
Evaluate -> Explain (node/edge masks) -> SymbolicRegression (PySR Pareto front)
   ▼
deploy/api.py (FastAPI)  <- - - - CPU Docker image (Dockerfile)
```

---

## 4. Key Contracts (things that must stay aligned)

| Contract | Definition | Source of truth |
|---|---|---|
| Node features | `[log_stellar_mass, log_vel_disp, log_half_mass_r, log_metallicity]` → `[N,4]` | `SubhaloData.get_node_features()` |
| Target | `y = log10(halo_mass)` (M_sun) | `HaloData.log_halo_mass` |
| Edge features | 5 physics features, ordered `[distance, delta_v, cos_theta, mass_ratio, proj_sep]` | `GraphBuilder.EDGE_FEATURE_MAP` |
| Physics attrs on `Data` | `.stellar_mass`, `.vel_disp`, `.half_mass_r` (linear units) for virial loss | `graph_builder.py`; injected as dummies in `api.py`/`explainer.py` |
| Config coupling | `model.node_features: 4` ≡ loader output; `model.edge_features: 5` ≡ `graph.edge_features` list | `config.yaml` |

---

## 5. Module Deep-Dives

### 5.1 `data/loaders/base_loader.py` — Schema & Factory
- **`SubhaloData`** (dataclass): one galaxy — id, position, velocity, stellar_mass, velocity_dispersion, half_mass_radius, metallicity. `get_node_features()` returns the 4 log10 features (eps-guarded).
- **`HaloData`** (dataclass): one halo/cluster — cluster_id, subhalos list, halo_mass (target), redshift, metadata. Convenience: `log_halo_mass`, `get_positions()`, `get_velocities()`, mass/radius means.
- **`BaseDataLoader(ABC)`**: template method `load()` = `load_raw → _parse_all_subhalos → _group_into_halos → _validate_data`; cached. `split_data()` = deterministic seeded permutation split.
- **`get_loader(config)`**: factory dispatching on `data.source` (synthetic | tng | camels | camels_hf).
- **`get_train_test_loaders(config)`**: builds two loaders for cross-simulation (different train/test sources).
- ⚠️ `split_data` resets the global NumPy RNG (`np.random.seed(self.seed)`).

### 5.2 `graph/graph_builder.py` — Graph Construction
- **Radius edges**: `radius_graph(loop=False)` with `radius_mpc=2.0`; **KNN edges**: `knn_graph` → made undirected + deduped.
- Self-loops optional; **isolated nodes connected to nearest neighbor** (no orphans); empty-halo → zero-size graph.
- **Edge features** (all `nan_to_num`-sanitized):
  1. `distance` — 3D Euclidean separation (gravitational PE proxy)
  2. `delta_v` — relative velocity magnitude
  3. `cos_theta` — velocity approach/recession angle (clamped ±1)
  4. `mass_ratio` — log stellar-mass difference (directed: src − dst)
  5. `proj_sep` — 2D xy projected separation (mock-observational)
- **`_build_hierarchical_graph`** is a placeholder/scaffold (`graph.hierarchical=false` by default) — hierarchical 2-level design documented but not implemented.
- `compute_graph_statistics()` for dataset stats.

### 5.3 `model/model.py` — Architecture
- **`NNConvBlock`**: `NNConv` (edge-conditioned, `aggr='mean'`) + LayerNorm + LeakyReLU + Dropout + optional residual (learned Linear projection when dims change).
- **`CosmicNetGNN`**: dims `4 → 256 → 256 → 256 → 128` (3 layers), `global_mean_pool` (or Set2Set), 3-layer MLP head → scalar log10 halo mass.
- **MC-Dropout**: `predict_with_uncertainty(batch, n_samples=30)` runs stochastic forward passes → mean/std + 2.5–97.5 percentile 95% CI.
- **`get_embeddings(batch, embedding_point='pre_pooling'|'post_pooling')`** — seam used by symbolic regression.
- **`EnsembleGNN`**: mean/std over 5 seeds (⚠️ seed mutation is a no-op — model never reads `seed`).
- `build_model`/`load_model` handle both dict-key and raw-state-dict checkpoints.

### 5.4 `model/physics_loss.py` — Physics-Informed Loss
- **Virial constraint** (`2·KE + PE ≈ 0`):
  - `KE_cluster = ½ · Σ(M_stellar_i · σ_i²)` (per subhalo)
  - `|PE_cluster| = G · M_halo_pred² / r_half` where `M_halo_pred = 10^pred`
  - `virial_ratio = 2·KE / |PE|`; **loss = mean((virial_ratio − 1)²)**
- **Total**: `L = MSE(pred, y) + λ · virial_loss`.
- **`GradientScaler`** (virial vs data gradient clipping) defined but **unused** by Trainer.
- `MetricsComputer.compute_all` → mse/rmse/mae/r2/scatter.

### 5.5 `training/train.py` — Training Loop
- **Lambda annealing**: inline `CosineLambdaScheduler` — strict 0.0 during 75-epoch warmup, then cosine ramp to `lambda_end=0.005`. (Note: `scheduler.py`'s `LambdaScheduler` is a parallel, **unused-by-default** implementation.)
- **Optimizer/LR**: AdamW (lr 8e-4, wd 5e-5) + CosineAnnealingLR (T_max=300, eta_min=1e-6); ReduceLROnPlateau steps on `val/mse`.
- **Checkpoints**: `best_model.pt` (on val/mse), periodic every 25 epochs, `final_model.pt`.
- **Early stopping**: patience 60, monitor `val/mse`.
- **W&B**: optional, tags include `cross_sim` when train_source ≠ test_source.
- **`run_ablation_study`**: nested-dot-key config overrides + retrain loop.

### 5.6 `explain/explainer.py` — XAI
- **"PGExplainer" path is actually gradient saliency** (Input×Gradient): backprop from pred to `x`, node importance = L1-norm of gradient; edge importance = mean of endpoints. *The real PyG PGExplainer/GNNExplainer classes are imported but never used; `train_explainer` is a non-functional stub.*
- **"GNNExplainer" path**: hand-rolled learnable soft-mask optimization (sigmoid masks over `x`/`edge_attr`, MSE + sparsity loss, Adam).
- Output: `ExplanationResult` → JSON with top-k nodes/edges (Three.js-ready) → `outputs/explanations/`.

### 5.7 `symbolic/symbolic_regression.py` — Equation Distillation
- **`extract_features_for_sr`**: pools embeddings (default `pre_pooling`) per halo; `run_symbolic_regression` then **discards GNN embeddings and uses only the 4 raw mean-pooled features** for interpretability.
- **PySR**: niterations=100, populations=30, maxsize=20, operators `+ - * / log sqrt square cube`; Pareto front → LaTeX via sympy.
- **`DimensionalAnalyzer`**: regex-based checks. ⚠️ **Dimensional consistency enforcement is effectively a no-op** — `check_equation` always returns `True` (the invalid-indicators list is never populated).
- Outputs: `all_equations_pareto.csv` + `dimensionally_valid_equations.tex` in `outputs/equations/`.

### 5.8 `deploy/api.py` — FastAPI Service
- **Endpoints**: `GET /` & `/health`, `POST /predict`, `POST /predict/batch`, `POST /explain`, `GET /config`, `GET /model/info`.
- `POST /predict` → builds graph from `HaloInput` JSON, MC-Dropout uncertainty → `PredictionResponse` (log10 mean, std, 95% CI, linear mass, ms latency).
- Startup: loads config + checkpoint (`outputs/checkpoints/best_model.pt`); **if checkpoint missing, still serves an untrained model** (marked ready — predictions meaningless).
- ⚠️ Deprecated `@app.on_event("startup")`; no auth/rate-limiting; batch endpoint is sequential; CORS uses `allow_origins=['*']` + `allow_credentials=True` (invalid combo).

### 5.9 `deploy/Dockerfile` — CPU Inference Image
- Multi-stage: builder installs **CPU-only torch 2.1.0 + PyG wheel-matched**; runtime copies venv + code, non-root user `cosmicnet`, HEALTHCHECK on `/health`.
- ⚠️ Omits `pysr`, `gplearn`, `sympy`, `matplotlib`, `wandb`, `datasets` — so `symbolic/` and `explain/visualize.py` are **unusable inside the image**. `requirements.txt` is copied but never `pip install`ed.

---

## 6. Data Sources (config `data.source`)

| Source | Loader | Auth | Fallback |
|---|---|---|---|
| `synthetic` (default) | `SyntheticLoader` | none | auto-generates 500 halos + files |
| `tng` (active in config) | `TNGLoader` | `TNG_API_KEY` | none (fails if no data + no API key) |
| `camels` | `CAMELSLoader` | none | generates fake HDF5 |
| `camels_hf` | `CAMELSHuggingFaceLoader` | `HF_TOKEN` (gated) | generates fake dataset |

**TNG loader priority cascade:** clustered CSV → raw CSV (+ API satellite fetch) → pure API. API requests are pickle-cached (MD5-keyed). Unit-conversion heuristics used throughout (ckpc/h → Mpc, log10 → linear).

---

## 7. Configuration Cheat-Sheet (`config/config.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `data.source` | `tng` | active data source |
| `data.batch_size` / `num_workers` | 16 / 2 | batching |
| `data.train/val/test_ratio` | 0.7 / 0.15 / 0.15 | split |
| `graph.method` | `radius` | edge method (`radius` \| `knn`) |
| `graph.radius_mpc` / `k_neighbors` | 2.0 / 8 | connectivity params |
| `graph.edge_features` | all 5 | active edge features |
| `graph.self_loops` | true | self-loop toggle |
| `graph.hierarchical` | false | unimplemented scaffold |
| `model.hidden_dim` / `output_dim` / `num_layers` | 256 / 128 / 3 | NNConv dims |
| `model.pooling` | `mean` | `mean` \| `set2set` |
| `model.mc_dropout` / `mc_samples` | true / 30 | uncertainty UQ |
| `model.dropout` / `activation` | 0.04 / `leaky_relu` | regularization |
| `physics.use_virial_loss` / `G` | true / 4.302e-9 | virial constraint |
| `training.epochs` / `learning_rate` / `optimizer` | 300 / 8e-4 / `adamw` | training core |
| `training.lambda_schedule` / `lambda_epochs` | `cosine` / 200 | ⚠️ dead keys (Trainer uses inline warmup-based `CosineLambdaScheduler` with `warmup_epochs=75`, `lambda_end=0.005`) |
| `training.train_source` / `test_source` | `tng` / `tng` | cross-sim experiment switch |
| `training.early_stopping_patience` | 60 | val/mse patience |
| `explain.method` | `pgexplainer` | ⚠️ resolves to gradient saliency |
| `symbolic.library` / `enforce_dimensional_consistency` | `pysr` / true | SR backend / ⚠️ no-op filter |
| `wandb.enabled` | false | experiment tracking |
| `api.host` / `port` | 0.0.0.0 / 8000 | API binding |
| `seed` | 42 | global reproducibility seed |

---

## 8. Tests

Run with `pytest` (requires project on path; tests `sys.path.insert` the repo root).

| File | Coverage |
|---|---|
| `tests/test_graph_builder.py` | radius/KNN edges, self-loops, edge-feature shape & NaN, symmetry (undirected), determinism, batch build, dataset, graph stats, hierarchical attr |
| `tests/test_loaders.py` | `SubhaloData`/`HaloData` schema, synthetic loader contract, factory dispatch, split ratios, determinism, statistics; TNG/CAMELS skipped without `.env` |
| `tests/test_physics_loss.py` | virial equilibrium (loss≈0), violation (loss>0), differentiability, batch batching, λ=0 path, metric functions |

CI (`.github/workflows/ci.yml`) is **advisory** — every job uses `continue-on-error: true`, triggered on PRs to `main`.

---

## 9. Known Gaps & Gotchas

1. **Dimensional-consistency filtering is a no-op** — every equation passes (`symbolic_regression.py`).
2. **"PGExplainer" is gradient saliency** — real PyG explainers unused; `train_explainer` is a stub (`explainer.py`).
3. **Two parallel lambda schedulers** (`scheduler.py` vs inline in `train.py`); config keys `lambda_schedule`/`lambda_epochs`/`lambda_start` are dead for the default path.
4. **`GradientScaler` and `virial_grad_clip`** defined but unused by Trainer.
5. **`TrainingState` never serialized**; typed for `LambdaScheduler` but handed a different scheduler class.
6. **Docker image missing SR/visualization deps** — `symbolic/` and `explain/visualize.py` will fail in-container.
7. **API serves untrained model** if checkpoint absent; no auth/rate-limit/input-size limits; deprecated startup hook.
8. **TNG loader has no synthetic fallback** — fails if CSVs missing and no API key.
9. **`data/__init__.py` is empty** — imports must go through `data.loaders`.
10. **Hierarchical graph** feature is documented scaffolding only.
11. **Silent data fabrication**: synthetic fallbacks, dummy physics attributes in API/explainer, `_load_from_raw_csv` still hits network.
12. **Ensemble seeding is a no-op** — `CosmicNetGNN` never reads `seed`.
13. **Windows + PyG DataLoader multiprocessing** (`num_workers>0`) may hit standard Windows spawn pitfalls.

---

## 10. Common Workflows

```bash
# Train on synthetic data
python main.py train

# Full pipeline: train -> evaluate -> explain -> symbolic
python main.py full

# Cross-simulation (edit config: train_source: tng, test_source: camels)
python main.py train

# API server
python main.py serve
# or: uvicorn deploy.api:app --host 0.0.0.0 --port 8000

# Tests
pytest

# Rebuild research-paper figures (uses kaggle/best_model*.pt checkpoints)
python scripts/generate_paper_plots.py

# Regenerate XAI saliency analysis
python scripts/generate_xai_plots.py

# Standalone TNG data download -> HDF5/CSV
python scripts/fetch_tng_data.py   # fill API_KEY at top first
```

**Pre-trained inference:** `kaggle/best_model_augmented.pt` + `model/model.py` (`build_model`) + `config/config.yaml` — see README §"Pre-Trained Model Inference".
