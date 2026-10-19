"""Boundary tests; generated HDF5 files are software fixtures, never validation data."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.paper_generalization.contracts import FEATURE_CONTRACT, validate_pair
from experiments.paper_generalization.catalog import load_catalog


def declarations():
    train = dict(schema_version=1, status="audited", feature_contract=copy.deepcopy(FEATURE_CONTRACT),
                 training_catalogs=[dict(suite="IllustrisTNG", simulation="TNG100-1", initial_condition_id="tng100-phases", source_sha256="a" * 64)],
                 verification=dict(target=True, features=True, training_transforms=True, lineage=True, backbone_policy_binding=True))
    external = dict(suite="IllustrisTNG", simulation="CV_0", initial_condition_id="camels-cv0-phases",
                    source_sha256="b" * 64, source_kind="local_hdf5", release="release-reviewed",
                    feature_contract=copy.deepcopy(FEATURE_CONTRACT), claim="independent_volume")
    population = {"member_selection": "all_stellar_positive", "min_members": 3}
    train["target_range_log10_msun"] = [12., 15.]
    train["population_contract"] = copy.deepcopy(population)
    external["population_contract"] = copy.deepcopy(population)
    return train, external


def test_unknown_training_provenance_rejected():
    train, external = declarations()
    train["status"] = "pending"
    with pytest.raises(ValueError, match="audited"):
        validate_pair(train, external)


@pytest.mark.parametrize("field,value", [("target", "log10(MFoF/M_sun)"),
                                         ("node_features", ["sky_x", "sky_y", "los_v", "mstar"]),
                                         ("edge_features", ["delta_v", "distance", "cos_theta", "mass_ratio", "proj_sep"]),
                                         ("radius_field", "SubhaloHalfmassRadType[:,4]"),
                                         ("position_units", "ckpc/h")])
def test_same_width_or_unit_mismatch_rejected(field, value):
    train, external = declarations()
    external["feature_contract"][field] = value
    with pytest.raises(ValueError, match="feature|target|schema"):
        validate_pair(train, external)


def test_same_lineage_new_snapshot_or_suite_is_not_independent():
    train, external = declarations()
    external["initial_condition_id"] = "tng100-phases"
    external["simulation"] = "TNG100-1-snapshot50"
    with pytest.raises(ValueError, match="lineage"):
        validate_pair(train, external)


def test_cross_suite_cannot_be_named_same_suite_independent_volume():
    train, external = declarations()
    external["suite"] = "SIMBA"
    with pytest.raises(ValueError, match="cross_simulation_transfer"):
        validate_pair(train, external)
    external["claim"] = "cross_simulation_transfer"
    assert validate_pair(train, external)["independent_initial_conditions"] is True


def test_synthetic_or_same_content_rejected():
    train, external = declarations()
    external["source_kind"] = "synthetic_fixture"
    with pytest.raises(ValueError, match="local_hdf5"):
        validate_pair(train, external)
    external["source_kind"] = "local_hdf5"
    external["source_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="content"):
        validate_pair(train, external)


def native_fixture(tmp_path, *, incomplete=False, fractional=False):
    path = tmp_path / "catalog.hdf5"
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs.update(HubbleParam=.5, Time=1., Redshift=0., BoxSize=1000.,
                       NumFiles=2 if incomplete else 1, Ngroups_ThisFile=1, Ngroups_Total=1,
                       Nsubgroups_ThisFile=3, Nsubgroups_Total=3, Omega0=.3, OmegaLambda=.7)
        f["Group/Group_M_Crit200"] = [100.]
        f["Subhalo/SubhaloGrNr"] = [.5, 0., 0.] if fractional else [0, 0, 0]
        f["Subhalo/SubhaloPos"] = [[990., 0., 0.], [10., 0., 0.], [20., 0., 0.]]
        f["Subhalo/SubhaloVel"] = np.zeros((3, 3))
        f["Subhalo/SubhaloMassType"] = np.ones((3, 6))
        f["Subhalo/SubhaloHalfmassRad"] = [5., 5., 5.]
        f["Subhalo/SubhaloVelDisp"] = [100., 100., 100.]
        f["Subhalo/SubhaloStarMetallicity"] = [.01, .01, .01]
    return path


def test_native_conversion_unwrap_and_target(tmp_path):
    halos, rejected = load_catalog(native_fixture(tmp_path), catalog_id="fixture", expected_h=.5, expected_a=1.)
    assert rejected == []
    assert halos[0].halo_mass == 2e12
    np.testing.assert_allclose(halos[0].get_positions()[:, 0], [0., .04, .06], atol=1e-7)
    assert halos[0].subhalos[0].stellar_mass == 2e10
    assert halos[0].subhalos[0].half_mass_radius == .01


@pytest.mark.parametrize("kwargs,match", [({"incomplete": True}, "complete"), ({"fractional": True}, "integer")])
def test_native_partial_or_fractional_membership_rejected(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_catalog(native_fixture(tmp_path, **kwargs), catalog_id="fixture", expected_h=.5, expected_a=1.)


def test_header_cosmology_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="HubbleParam"):
        load_catalog(native_fixture(tmp_path), catalog_id="fixture", expected_h=.7, expected_a=1.)


def test_template_cli_rejected_without_creating_results(tmp_path):
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "experiments.paper_generalization.run", "--manifest",
                             str(root / "experiments/paper_generalization/camels-tng-cv0.example.json"),
                             "--output", str(tmp_path / "results")], cwd=root, capture_output=True, text=True)
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr
    assert not (tmp_path / "results").exists()


def test_frozen_fixture_executes_and_exports_without_scientific_claim(tmp_path):
    """Catch missing evaluator provenance/exports without treating fixtures as science."""
    import torch
    from model.model import CosmicNetGNN
    from rls.policy import build_policy
    from experiments.paper_generalization.contracts import file_record
    from experiments.paper_generalization.run import execute

    def spec(path):
        return {k: v for k, v in file_record(path).items() if k in {"path", "sha256"}}

    cfg = {"model": {"node_features": 4, "edge_features": 5, "hidden_dim": 8,
                     "output_dim": 8, "num_layers": 1, "dropout": 0., "residual": True,
                     "activation": "leaky_relu", "pooling": "mean"},
           "graph": {"edge_features": FEATURE_CONTRACT["edge_features"], "method": "radius",
                     "radius_mpc": 2., "self_loops": True},
           "rls": {"policy_hidden": 4, "policy_normalize": False}}
    model = CosmicNetGNN(cfg)
    policy = build_policy(cfg, node_emb_dim=8)
    bp, pp = tmp_path / "backbone.pt", tmp_path / "policy.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, bp)
    torch.save(policy.state_dict(), pp)
    evidence = {}
    for name in ["training_catalog", "split", "semantic_audit", "training_code", "parameters", "schema_review"]:
        p = tmp_path / (name + ".txt")
        p.write_text("SOFTWARE FIXTURE ONLY: " + name)
        evidence[name] = spec(p)
    train, external = declarations()
    train.update(status="fixture_smoke", config=cfg, backbone_sha256=spec(bp)["sha256"],
                 policy_sha256=spec(pp)["sha256"], policy_backbone_sha256=spec(bp)["sha256"],
                 target_range_log10_msun=[12., 14.],
                 evidence_files={k: evidence[k] for k in ["training_catalog", "split", "semantic_audit", "training_code"]})
    train["training_catalogs"][0]["source_sha256"] = evidence["training_catalog"]["sha256"]
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(train))
    catalog = native_fixture(tmp_path)
    external.update(id="fixture", source_kind="synthetic_fixture", claim="fixture_smoke", source=spec(catalog),
                    source_sha256=spec(catalog)["sha256"], epoch={"h": .5, "scale_factor": 1.},
                    parameters=evidence["parameters"], schema_review=evidence["schema_review"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, status="fixture_smoke", dataset=external,
                                       training_receipt=spec(receipt), backbone=spec(bp), policy=spec(pp),
                                       budget={"pair_count": 2, "constraint": "no_isolates", "decoder": "direct"}, seed=42)))
    with pytest.raises(ValueError):
        execute(manifest, tmp_path / "forbidden")
    result = execute(manifest, tmp_path / "smoke", fixture_smoke=True)
    assert result["evaluation_executed"] is True
    assert result["scientific_result"] is False
    assert result["claim"] == "fixture_smoke"
    assert result["domain_generalization_established"] is False
    assert result["common_success_halos"] == ["fixture:group:0"]
    rows = result["per_halo_rows"]
    assert {r["method"] for r in rows} == {"full", "random", "distance", "degree", "provided_rl"}
    assert all(r["research_label"] == "fixture_smoke" for r in rows)
    assert all(r["final_pair_count"] == 2 for r in rows if r["method"] != "full")
    import csv
    with (tmp_path / "smoke/per_halo_results.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 5
    assert all(r["model_state_hash"] for r in exported)
    assert spec(bp)["sha256"] == train["backbone_sha256"]
    assert spec(pp)["sha256"] == train["policy_sha256"]


def test_native_population_cannot_silently_expand_first10_training_sample():
    train, external = declarations()
    train['population_contract'] = {'member_selection': 'native_first10_then_stellar_positive', 'min_members': 3}
    external['population_contract'] = {'member_selection': 'all_stellar_positive', 'min_members': 3}
    with pytest.raises(ValueError, match='population|selection'):
        validate_pair(train, external)


def test_unknown_training_mass_support_is_not_audited():
    train, external = declarations()
    train['target_range_log10_msun'] = None
    with pytest.raises(ValueError, match='mass support'):
        validate_pair(train, external)


@pytest.mark.parametrize('storage', ['external_link', 'soft_link', 'virtual', 'external_storage'])
def test_native_unhashed_hdf5_dependencies_rejected(tmp_path, storage):
    path = native_fixture(tmp_path)
    with h5py.File(tmp_path / 'backing.hdf5', 'w') as other:
        other['mass'] = [100.]
    with h5py.File(path, 'a') as f:
        del f['Group/Group_M_Crit200']
        if storage == 'external_link':
            f['Group/Group_M_Crit200'] = h5py.ExternalLink('backing.hdf5', 'mass')
        elif storage == 'soft_link':
            f['mass_alias'] = [100.]
            f['Group/Group_M_Crit200'] = h5py.SoftLink('/mass_alias')
        elif storage == 'virtual':
            layout = h5py.VirtualLayout(shape=(1,), dtype='f8')
            layout[:] = h5py.VirtualSource(str(tmp_path / 'backing.hdf5'), 'mass', shape=(1,))
            f.create_virtual_dataset('Group/Group_M_Crit200', layout)
        else:
            f.create_dataset('Group/Group_M_Crit200', (1,), dtype='f8',
                             external=[(str(tmp_path / 'raw.bin'), 0, h5py.h5f.UNLIMITED)])[:] = [100.]
    with pytest.raises(ValueError, match='self-contained'):
        load_catalog(path, catalog_id='fixture', expected_h=.5, expected_a=1.)


def test_multiple_training_catalogs_need_complete_evidence_support():
    train, external = declarations()
    train['training_catalogs'].append(dict(suite='SIMBA', simulation='CV_2', initial_condition_id='other-ic', source_sha256='c'*64))
    with pytest.raises(ValueError, match='one training catalog'):
        validate_pair(train, external)


@pytest.mark.parametrize('marker', ['unknown', 'UNVERIFIED', 'tbd', 'pending', 'Unknown seed'])
def test_unknown_identity_markers_cannot_establish_independence(marker):
    train, external = declarations()
    external['initial_condition_id'] = marker
    with pytest.raises(ValueError, match='lineage'):
        validate_pair(train, external)
