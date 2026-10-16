# Reproducible manuscript artifacts

These outputs describe the completed **legacy** A–D campaign. They support
descriptive benchmarking and explicitly exploratory halo-paired inference.
They do not establish RL superiority, efficiency, or independent-volume generalization.

From the repository root:

```bash
python3 -m pip install -r paper/scripts/requirements.txt
python3 paper/scripts/generate_figures.py
python3 paper/scripts/generate_tables.py
python3 -m pytest tests/test_paper_statistics.py -q
```

The first command after installation verifies all source snapshots, recomputes
all 21 comparisons, and writes six PDF/PNG pairs plus numerical analysis files.
The table command exports five complete `table*` environments and `results.tex`.
It rejects stale analysis. Run the commands in this order. Model weights, CUDA,
Kaggle, original checkout paths and network access are not needed for regeneration.
The focused tests inherit the repository's PyTorch-based `conftest.py`; plotting
and statistics scripts themselves require only the three pinned libraries.

Use PDF figures in LaTeX (`graphicx`), approximately `\columnwidth` for Figures
1/3 and `\textwidth` in `figure*` for Figures 2/4/5/6. PNG alternatives are
400 dpi. Add `\usepackage{booktabs}` and `\input{paper/tables/table_2_primary_benchmark}`
(adjust paths to the manuscript root). Tables are full-width to retain legible
fonts in two-column layouts. `paper/results.tex` contains generated results prose.

Read [statistical-analysis.md](evidence/statistical-analysis.md) for the estimands,
verified numbers, limitations and missing inputs, and
[figure-provenance.md](evidence/figure-provenance.md) for captions and sources.
`evidence/inputs/manifest.json` records original locations and SHA-256 identities.
The committed `.gz` snapshots decompress to the original bytes; they preserve
the source evidence without overwriting notebook artifacts.

The scripts only write below this worktree's `paper/`. They do not execute
models, change notebook results, or use the earlier untracked `paper/` directory.
