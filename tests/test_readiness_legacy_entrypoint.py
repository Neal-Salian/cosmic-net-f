"""Legacy CLI must not bypass audited target compatibility."""
import pytest


def test_main_train_refuses_legacy_cross_simulation_before_loading(monkeypatch):
    import main
    import data.loaders.base_loader as loaders
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy incompatible catalogs must not load")
    monkeypatch.setattr(loaders, "get_train_test_loaders", forbidden)
    with pytest.raises(ValueError, match="audited"):
        main.train({"training": {"train_source": "tng", "test_source": "camels"}})
