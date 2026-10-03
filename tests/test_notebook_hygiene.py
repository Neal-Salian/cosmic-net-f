"""Static notebook-hygiene guard (Sep 2026 rewire).

Notebooks A-D must import helpers from the committed rls/ package instead of
reimplementing them inline — the inline copies drifted (notably: no
pairwise-symmetric masking, pre-fix TTA), and Notebook B's hand-written
REINFORCE loop lacked the divergence guard. Cheap to run in CI; prevents
this exact drift from silently reappearing.

Checks per notebook (parsed with nbformat, code cells only):
- no `PREFER:` markers remain (they marked not-yet-done import swaps);
- none of the banned inline definitions exist (only imports allowed);
- every `from rls.<mod> import <names>` resolves against the repo package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ast
import importlib

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS = [
    "notebooks/notebook_A_setup_data_graphs_baselines.ipynb",
    "notebooks/notebook_B_train_policy.ipynb",
    "notebooks/notebook_C_full_evaluation.ipynb",
    "notebooks/notebook_D_tta_inference.ipynb",
]

# (kind, name): class/function definitions that must come from rls/, never inline.
# _AttentionScorer stays inline in A (notebook-specific stand-in, not in rls/).
BANNED_DEFS = [
    ("class", "EdgePolicyNet"),
    ("class", "ValueNet"),
    ("def", "hard_mask"),
    ("def", "apply_min_keep_floor"),
    ("def", "repair_connectivity"),
    ("def", "bernoulli_logp"),
    ("def", "bernoulli_entropy"),
    ("def", "sample_actions"),
    ("def", "compute_advantages"),
    ("def", "compute_pg_loss"),
    ("def", "graph_physics_terms"),
    ("def", "relative_virial_penalty"),
    ("def", "compute_rewards"),
    ("def", "label_free_reward"),
    ("def", "virial_ratio_pruned"),
    ("def", "mc_std"),
    ("def", "adapt_at_test_time"),
    ("def", "train_policy"),
    ("def", "prepare_graphs"),
    ("def", "prepare"),
    ("def", "gnns_adapter"),
    ("def", "target_sparsity"),
    ("def", "random_mask"),
    ("def", "degree_mask"),
    ("def", "distance_mask"),
    ("def", "mass_ratio_mask"),
    ("def", "gradient_saliency_mask"),
    ("def", "attention_topk_mask"),
    ("class", "GumbelEdgeMask"),
    ("class", "PolicyGradientTrainer"),
]


def _code_cells(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


def _defined_top_level(tree):
    """Top-level class/def names in a module AST."""
    return {((isinstance(n, ast.ClassDef) and "class" or "def"), n.name)
            for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def _compile_cell(src, name):
    """Compile a notebook cell with IPython shell escapes (`!cmd`, `%magic`)
    stripped — those are legitimate Kaggle/Jupyter content, not Python."""
    lines = [ln for ln in src.splitlines()
             if not ln.lstrip().startswith(("!", "%"))]
    compile("\n".join(lines), name, "exec")


def test_notebooks_parse_and_compile():
    for nb in NOTEBOOKS:
        for i, src in enumerate(_code_cells(nb)):
            _compile_cell(src, f"{nb} cell {i}")


def test_no_prefer_markers_remain():
    for nb in NOTEBOOKS:
        for i, src in enumerate(_code_cells(nb)):
            assert "PREFER:" not in src, f"{nb} cell {i} still has a PREFER: marker"


def test_no_inline_rl_helpers():
    banned = set(BANNED_DEFS)
    for nb in NOTEBOOKS:
        for i, src in enumerate(_code_cells(nb)):
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue  # covered by test_notebooks_parse_and_compile
            found = _defined_top_level(tree) & banned
            assert not found, (
                f"{nb} cell {i} re-defines {sorted(n for _, n in found)} inline "
                f"— import from rls/ instead")


def test_rls_imports_resolve():
    for nb in NOTEBOOKS:
        for i, src in enumerate(_code_cells(nb)):
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not (node.module or "").startswith("rls"):
                    continue
                mod = importlib.import_module(node.module)
                for a in node.names:
                    assert hasattr(mod, a.name), (
                        f"{nb} cell {i}: rls import {node.module}.{a.name} "
                        f"does not exist in the package")


def test_clone_cells_pin_fix_branch():
    for nb in NOTEBOOKS:
        joined = "\n".join(_code_cells(nb))
        assert "fix/rl-pruning-symmetry" in joined, (
            f"{nb}: clone cell does not pin fix/rl-pruning-symmetry")
        assert 'REPO_BRANCH = "feature/rl-pipeline"' not in joined, (
            f"{nb}: stale feature/rl-pipeline pin remains")
