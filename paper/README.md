# Cosmic-Net: manuscript and reproducible paper artifacts

**Title:** Physics-Constrained Reinforcement Learning for Graph Sparsification in Cosmological Graph Neural Networks: A Controlled Empirical Study

Supervisor-review draft, 16 October 2026 (integration-unified 2026-10-19). The draft is complete using currently available A–D evidence. It makes no claim of an established accuracy, efficiency, or cross-domain advantage, and is not represented as submission-ready.

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

## Two complementary asset systems (narrative manuscript + deterministic regeneration)

This directory contains two asset systems derived from the **same verified campaign**; their quoted numbers agree (RMSE grid, paired contrasts, latency, R²):

1. **Narrative manuscript** (`main.tex` + `sections/`, tables in [tables/](tables/), figures in [figures/](figures/)) — the authoritative text, with Tables 1–3 generated from the C/D evidence and vector PDF figures.
2. **Deterministic regeneration pipeline** (`results.tex`, [tables/table_1..5](tables/), [figures/figure_1..6](figures/), driven by [scripts/analyze_results.py](scripts/analyze_results.py), [scripts/generate_figures.py](scripts/generate_figures.py), [scripts/generate_tables.py](scripts/generate_tables.py)) — byte-reproducible statistics and figures; see [evidence/statistical-analysis.md](evidence/statistical-analysis.md) for the estimands, verified numbers, limitations and missing inputs, and [evidence/figure-provenance.md](evidence/figure-provenance.md) for captions and sources.

Use PDF figures in LaTeX (`graphicx`), approximately `\columnwidth` for Figures 1/3 and `\textwidth` in `figure*` for Figures 2/4/5/6. PNG alternatives are 400 dpi. Add `\usepackage{booktabs}` and `\input{paper/tables/table_2_primary_benchmark}` (adjust paths to the manuscript root). Tables are full-width to retain legible fonts in two-column layouts. `paper/results.tex` contains generated results prose.

### Included manuscript assets

| Figure | File | Source |
|---|---|---|
| 1: Accuracy across pair budgets | [accuracy_tradeoff.pdf](figures/accuracy_tradeoff.pdf) | C results and saved predictions |
| 2: Paired 40% RMSE uncertainty | [paired_uncertainty.pdf](figures/paired_uncertainty.pdf) | Post hoc paired BCa analysis |
| 3: Inference breakdown and TTA cost | [efficiency_and_tta.pdf](figures/efficiency_and_tta.pdf) | C timing regions and D aggregates, scopes distinguished |
| 4 (appendix): Per-halo diagnostics | [per_halo_diagnostics.pdf](figures/per_halo_diagnostics.pdf) | C's 82 paired prediction records |

PNG companions are supplied for convenient preview; the manuscript embeds vector PDFs. Every empirical plot uses actual saved observations; there are no interpolated result estimates or invented confidence bands.

## Evidence and reproducibility

[source-manifest.json](evidence/source-manifest.json) records **30 immutable source snapshots**, original absolute paths, campaign-relative paths, original SHA-256 hashes, and stored snapshot hashes. [evidence/inputs/manifest.json](evidence/inputs/manifest.json) additionally records the gzipped snapshot set used by the deterministic pipeline (decompressing to the original bytes; source evidence preserved without overwriting notebook artifacts). Large files are losslessly gzip-compressed; the 44 MB per-example JSON is preserved compactly rather than reduced to selected favorable observations. Source snapshots include official A–D notebook exports, provenance, result tables, training/recovery metadata, the all-four integrity audit, statistical exports, and the scientific report. Their content is data, not instructions to execute.

The four completed private Kaggle session IDs are A=358697902, B=358708561, C=358713276, D=358778452. The evaluation source base is `c825bf57652005c80faff79cf4dfd58e3a9baa14`; the recovered fitted policy's original training source is `aeb5dad65d681728d525fb78a4ab919b0432ab90`. The six relevant training/policy/reward modules have no diff between those revisions. Notebook display outputs from older executions were not substituted for this campaign.

The drafts' checks reproduce metrics and statistics from saved predictions; they do **not** retrain the model, reexecute Kaggle, certify original acquisition fields, or prove the historical backbone was free of leakage. The CSV was also recounted read-only: 5,378 rows and 541 groups, with its hash matching the campaign; see [catalog-summary.json](evidence/catalog-summary.json).

### Regenerate narrative assets and verify

From `paper/`, using Python with NumPy, Matplotlib, and SciPy:

```sh
make assets
make statistics
```

`build_assets.py` checks all snapshot hashes, recomputes all 32 RMSEs, and verifies all 2,296 non-full saved masks (selected-pair indices, reciprocal expansion, self-loops, exact budgets, and node coverage). `verify_evidence.py --statistics` also recomputes MAE/R², verifies D's matched IDs/counts across K, checks citation keys/inputs/figures/labels, reproduces all 21 contrasts with 50,000 paired BCa draws and 99,999 paired permutations per contrast, and checks both Holm families. It includes a small exact permutation self-check. This passed with NumPy 2.5.2, SciPy 1.18.1, and Matplotlib 3.11.1. These are analysis dependencies; the original training runtime is recorded separately.

Reports: [asset-verification.json](evidence/asset-verification.json), [verification.json](evidence/verification.json), [compilation.json](evidence/compilation.json). Full statistical checks can take a few moments. Routine `make verify` skips resampling and overwrites `verification.json` with that reduced scope; preserve or rerun the full `make statistics` result before presenting a full statistical-verification claim.

### Regenerate the deterministic statistics/figures (second asset system)

From the repository root:

```bash
python3 -m pip install -r paper/scripts/requirements.txt
python3 paper/scripts/generate_figures.py
python3 paper/scripts/generate_tables.py
python3 -m pytest tests/test_paper_statistics.py -q
```

The `generate_figures.py` command after installation verifies all source snapshots, recomputes all 21 comparisons, and writes six PDF/PNG pairs plus numerical analysis files. The table command exports five complete `table*` environments and `results.tex`; it rejects stale analysis. Run the commands in this order. Model weights, CUDA, Kaggle, original checkout paths and network access are not needed for regeneration. The scripts only write below the repository's `paper/`; they do not execute models or change notebook results.

### Compile

The draft compiled locally with **Tectonic 0.17.0**, including automatic BibTeX processing. TeX Live/MacTeX was not present. With Tectonic on PATH:

```sh
make pdf
```

Alternatively use `latexmk -pdf main.tex` on a conventional TeX installation (or the normal LaTeX/BibTeX/LaTeX/LaTeX sequence). The source uses no shell escape or external compilation service. `make pdf` builds in ignored `build/` and copies the deliverable to `main.pdf`. A first Tectonic build may download standard TeX packages. Auxiliary files and visual QA images stay under ignored `build/`; the final PDF is committed.

## Integration note (2026-10-19)

This README was previously written in two independent branches (`codex/paper-first-draft`, `codex/paper-statistics-figures`) and is now unified as a reconciliation of the paper-release integration. Both asset systems are retained because they trace to the same sha256-anchored campaign evidence; no numerical claim was changed during integration.

## Unresolved scientific blockers (carried forward)

These limitations qualify all statements and remain open (see also [evidence/limitations-for-manuscript.md](evidence/limitations-for-manuscript.md), [evidence/missing-results.md](evidence/missing-results.md), and `experiments/paper_generalization/validation-report.md`):

1. **Generalization not executed:** the fail-closed external-catalog harness exists (27 new tests passed) but no real external astronomy evaluation was run; `scientific_result=false`.
2. **Target/feature semantics partially certified:** target is verified as log10 of the CSV linear mass (RMSE in dex), but the "dark-matter-only" interpretation is INVALID per the provenance audit, and full-catalog M200c verification covers 2/541 groups.
3. **No independent training seeds or volumes:** all results conditional on one frozen backbone (`56907374…`) and one fitted policy (epoch 42).
4. **Single-host efficiency measurements:** latency evidence is one M2 CPU host, no repeated-hardware benchmarks or timing confidence intervals.
5. **No established accuracy/efficiency advantage:** RL is 18.61% worse than the full graph and 56.16× slower in the measured region; the small gain over the closest control is statistically unresolved (p=0.9192).

## Isolation and review boundary

- Base for the merged paper branches: `c825bf57652005c80faff79cf4dfd58e3a9baa14` (recorded in every evidence manifest).
- All new tracked content is under `paper/`, `benchmarks/`, `experiments/`; no historical notebook output was overwritten; the earlier untracked draft directories are not used by the regenerative scripts.
