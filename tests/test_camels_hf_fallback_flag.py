"""
Regression test: CAMELSHuggingFaceLoader must honor the same
used_synthetic_fallback contract as CAMELSLoader — False at init, True after
the synthetic fallback fires. Downstream guards (Notebooks C/D, rls/cross_sim)
rely on this flag to refuse publishing synthetic-data results as real OOD
numbers; before this fix the attribute did not exist on the HF loader, so
getattr(..., False) was vacuously False even after synthetic fallback.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

pytest.importorskip("datasets")

from data.loaders.camels_loader import CAMELSHuggingFaceLoader


def _config():
    return {
        "data": {
            "source": "camels_hf",
            "camels_hf": {
                "dataset_name": "camels-multifield-dataset/CAMELS",
                "split": "train",
                "cache_dir": "data/raw/camels_hf_cache",
            },
        },
        "seed": 42,
    }


def test_hf_loader_fallback_flag_false_at_init():
    loader = CAMELSHuggingFaceLoader(_config())
    assert loader.used_synthetic_fallback is False


def test_hf_loader_fallback_flag_true_after_synthetic_fallback():
    loader = CAMELSHuggingFaceLoader(_config())
    data = loader._generate_synthetic_hf_data()  # pure numpy, no network
    assert isinstance(data, list) and len(data) > 0
    assert loader.used_synthetic_fallback is True
