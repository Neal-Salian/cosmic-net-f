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
4. Re-run OOD frozen-vs-TTA once the CAMELS URL is fixed (still 404; synthetic
   fallback stays fail-closed).
5. Notebook A–D inline copies (`EdgePolicyNet`, `hard_mask`,
   `repair_connectivity`, …) still need replacing with package imports and a
   Kaggle Run-All to verify — intentionally deferred, not done blind.
