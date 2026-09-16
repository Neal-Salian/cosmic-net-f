# RL pruning fix — before/after summary (`fix/rl-pruning-symmetry`)

Branch off `feature/rl-pipeline`. Six reviewable commits, all tests green
(111 passed, 1 skipped; only pre-existing env failures excluded: missing
`pyg-lib`, git-LFS pointer checkpoint, torch `weights_only` pickle default).

## Diagnosed numbers (pre-fix, Notebook C, frozen backbone, N=82)

| method | keep_frac | R² |
|---|---|---|
| full graph | 1.000 | 0.9075 |
| random | 0.799 | 0.9061 |
| grad_saliency | 0.799 | 0.8910 |
| attention_topk | 0.799 | 0.8368 |
| **rl_policy** | **0.976** | **0.7830** |
| distance | 0.799 | 0.7418 |
| mass_ratio | 0.799 | 0.5931 |
| degree | 0.799 | 0.4695 |

Root causes fixed (see plan change log for details): asymmetric
directed-edge pruning (OOD topology for the frozen GNN → "keep everything"
collapse); soft sparsity penalty; unguarded Stage B (R² 0.9075→0.8159);
unconstrained TTA (2–4x in-domain RMSE regression); unvalidated physics
units; single-seed point estimates.

## Post-fix mechanism check (real TNG kNN graphs, random backbone — LFS
weights not downloadable in this env, so accuracies are NOT comparable;
mechanics are)

| mode | EVAL keep (target 0.40) | max pair-asymmetry | note |
|---|---|---|---|
| penalty | 1.000 | 0.000 | collapse reproduced (needs retune, guard in place) |
| topk_scheduled | 0.454 | 0.000 | keep tracks curriculum by construction |

## To reproduce the after-table (Kaggle, GPU, LFS weights)

1. `python scripts/multiseed_rl.py --seeds 42 43 44 45 46` (penalty) and
   `--rls-override sparsity_mode=topk_scheduled` — writes
   `outputs/rls/multiseed.json` with per-seed rows + bootstrap CIs.
2. Compare against baselines at matched keep via `rls/evaluate.py`
   (`eval_mask` guarantees the same symmetric decoder everywhere).
3. Re-tune `(K, tta_lr, w_unc)` on VAL only with `tta_kl_coef > 0`; the
   `tta_should_enable` gate keeps TTA off when val regresses.
4. Re-run OOD frozen-vs-TTA on real CAMELS data — the loader URL is fixed
   and verified (Sep 2026: `FOF_Subfind/IllustrisTNG/LH/LH_0/groups_090.hdf5`
   downloads and parses; `set`/`snapshot` configurable). Still needs the
   Kaggle run itself.
5. Notebook A–D are rewired to the fixed package (Sep 2026, same branch):
   imports-only helper cells, B via `train_policy`/`fine_tune_gnn`, C via
   `build_results_table`, D via `adapt_at_test_time` + val gate cell,
   `tests/test_notebook_hygiene.py` guards drift in CI. `tta_kl_coef=0.05` /
   `tta_lr_decay=0.9` are placeholder starting values — sweep
   `tta_kl_coef ∈ [0.01, 0.05, 0.1]` on VAL before trusting test TTA numbers.
