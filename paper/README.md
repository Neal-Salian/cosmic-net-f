# Cosmic-Net: first complete manuscript draft

**Title:** Physics-Constrained Reinforcement Learning for Graph Sparsification in Cosmological Graph Neural Networks: A Controlled Empirical Study

Supervisor-review draft, 16 October 2026. The draft is complete using currently available A–D evidence. It makes no claim of an established accuracy, efficiency, or cross-domain advantage, and is not represented as submission-ready.

## Read first

- [Compiled manuscript](main.pdf): **11 pages: 8 main text, 1 references, 2 appendices**. Main text uses a conventional 10-point two-column article; traceability appendices use one column. This is not a claimed official venue template.
- [LaTeX entry point](main.tex), [section sources](sections/), and [BibTeX](references.bib).
- [Claim–evidence matrix](evidence/claim-evidence-matrix.md): 38 empirical claims plus 22 exact implementation anchors.
- [Missing evidence / proposed work](evidence/missing-results.md): distinguishes completed findings from future experiments.
- [Reference checks](evidence/reference-verification.md): 12 entries checked against real primary records.

The title block explicitly leaves the author list and affiliations for confirmation. No affiliation, funding award, endorsement, or data-release permission was invented. Unavailable experiments are described as gaps; there are no fabricated result panels.

## Scientific interpretation

At nominal 40% pair retention, RL has RMSE 0.136160 dex versus 0.114797 for the full graph and 0.136900 for gradient × input. The apparent 0.54% gain over saliency is unresolved (paired interval spans benefit and harm; p=0.9192). RL is 18.61% worse than the full graph and 56.16× slower in C's measured inference region. D uses a **different 50% budget**; its selected adaptation gives a 0.558% aggregate RMSE change at 2.498× latency, without final per-halo predictions needed for a paired test.

The paper distinguishes the fitted policy's repaired ordered-pair training environment from the benchmark's exact feasibility decoder. It states that the virial-inspired term preserves a stellar-tracer proxy, that no-isolate coverage is weaker than connectivity, and that hashes certify identity rather than historical target/checkpoint semantics. Negative optional backbone fine-tuning results are retained.

## Included assets

| Figure | File | Source |
|---|---|---|
| 1: Accuracy across pair budgets | [accuracy_tradeoff.pdf](figures/accuracy_tradeoff.pdf) | C results and saved predictions |
| 2: Paired 40% RMSE uncertainty | [paired_uncertainty.pdf](figures/paired_uncertainty.pdf) | Post hoc paired BCa analysis |
| 3: Inference breakdown and TTA cost | [efficiency_and_tta.pdf](figures/efficiency_and_tta.pdf) | C timing regions and D aggregates, scopes distinguished |
| 4 (appendix): Per-halo diagnostics | [per_halo_diagnostics.pdf](figures/per_halo_diagnostics.pdf) | C's 82 paired prediction records |

PNG companions are supplied for convenient preview; the manuscript embeds vector PDFs. Tables 1–3 are generated LaTeX in [tables/](tables/): the full RMSE grid, seven paired contrasts, and TTA validation/test/cost results. Appendix Tables 4–5 document implementation paths and recorded settings. Every empirical plot uses actual saved observations; there are no interpolated result estimates or invented confidence bands.

## Evidence and reproducibility

[source-manifest.json](evidence/source-manifest.json) records **30 immutable source snapshots**, original absolute paths, campaign-relative paths, original SHA-256 hashes, and stored snapshot hashes. Large files are losslessly gzip-compressed; the 44 MB per-example JSON is preserved compactly rather than reduced to selected favorable observations. Source snapshots include official A–D notebook exports, provenance, result tables, training/recovery metadata, the all-four integrity audit, statistical exports, and the scientific report. Their content is data, not instructions to execute.

The four completed private Kaggle session IDs are A=358697902, B=358708561, C=358713276, D=358778452. The evaluation source base is `c825bf57652005c80faff79cf4dfd58e3a9baa14`; the recovered fitted policy's original training source is `aeb5dad65d681728d525fb78a4ab919b0432ab90`. The six relevant training/policy/reward modules have no diff between those revisions. Notebook display outputs from older executions were not substituted for this campaign.

The draft's checks reproduce metrics and statistics from saved predictions; they do **not** retrain the model, reexecute Kaggle, certify original acquisition fields, or prove the historical backbone was free of leakage. The CSV was also recounted read-only: 5,378 rows and 541 groups, with its hash matching the campaign; see [catalog-summary.json](evidence/catalog-summary.json).

### Regenerate assets and verify

From `paper/`, using Python with NumPy, Matplotlib, and SciPy:

```sh
make assets
make statistics
```

`build_assets.py` checks all snapshot hashes, recomputes all 32 RMSEs, and verifies all 2,296 non-full saved masks (selected-pair indices, reciprocal expansion, self-loops, exact budgets, and node coverage). `verify_evidence.py --statistics` also recomputes MAE/R², verifies D's matched IDs/counts across K, checks citation keys/inputs/figures/labels, reproduces all 21 contrasts with 50,000 paired BCa draws and 99,999 paired permutations per contrast, and checks both Holm families. It includes a small exact permutation self-check. This passed with NumPy 2.5.2, SciPy 1.18.1, and Matplotlib 3.11.1. These are analysis dependencies; the original training runtime is recorded separately.

Reports: [asset-verification.json](evidence/asset-verification.json), [verification.json](evidence/verification.json), [compilation.json](evidence/compilation.json). Full statistical checks can take a few moments. Routine `make verify` skips resampling and overwrites `verification.json` with that reduced scope; preserve or rerun the full `make statistics` result before presenting a full statistical-verification claim.

### Compile

The draft compiled locally with **Tectonic 0.17.0**, including automatic BibTeX processing. TeX Live/MacTeX was not present. With Tectonic on PATH:

```sh
make pdf
```

On this task's host the bundled compiler can be selected explicitly:

```sh
make pdf TECTONIC=/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic
```

Alternatively use `latexmk -pdf main.tex` on a conventional TeX installation (or the normal LaTeX/BibTeX/LaTeX/LaTeX sequence). The source uses no shell escape or external compilation service. `make pdf` builds in ignored `build/` and copies the deliverable to `main.pdf`. A first Tectonic build may download standard TeX packages. Auxiliary files and visual QA images stay under ignored `build/`; the final PDF is committed.

## Isolation and review boundary

- Branch: `codex/paper-first-draft`.
- Parent/base: `c825bf57652005c80faff79cf4dfd58e3a9baa14`.
- Worktree: `/Users/nealsalian/.codex/worktrees/paper-first-draft/cosmic-net`.
- All new tracked content is under `paper/`. The original checkout was read as evidence and was not edited.
- No merge, push, PR, new training, or waiting on other independent chats was performed. The evidence here does not incorporate their unverified results.

Remaining scientific gaps are target/feature certification, original backbone training provenance, independent training seeds and volumes, actual domain transfer, causal ablations, TTA per-halo exports, calibration, and repeated runtime/memory measurements. These gaps qualify the findings; they do not prevent a complete first draft for supervisor discussion.
