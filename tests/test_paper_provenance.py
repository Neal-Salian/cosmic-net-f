"""Scientific input contradictions and diagnostic scope, using tiny fixtures."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from data.loaders.tng_loader import TNGLoader
from data.scientific_validation import (
    validate_clustered_identity_targets, periodic_pair_diagnostics,
    match_reference_group,
)


def catalog():
    return pd.DataFrame({"subhalo_id": [10, 11], "group_id": [2, 2],
                         "halo_mass": [1e12, 1e12], "halo_mass_log": [12., 12.]})


def test_repeated_target_per_halo_is_expected_and_validation_does_not_mutate():
    frame = catalog()
    before = frame.copy(deep=True)
    validate_clustered_identity_targets(frame, halo_mass_unit="M_sun")
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column,value,error", [
    ("subhalo_id", 10, "duplicate subhalo_id"),
    ("subhalo_id", 10.5, "integer"),
    ("group_id", -1, "nonnegative"),
    ("halo_mass", 2e12, "conflicting.*target"),
    ("halo_mass", 0., "positive"),
    ("halo_mass", float("inf"), "finite"),
    ("halo_mass_log", 11., "conflicting.*target"),
])
def test_catalog_rejects_identity_or_target_contradiction(column, value, error):
    frame = catalog()
    frame[column] = frame[column].astype(float)
    frame.loc[1, column] = value
    with pytest.raises(ValueError, match=error):
        validate_clustered_identity_targets(frame, halo_mass_unit="M_sun")


def test_redundant_mass_columns_must_agree_even_if_each_is_group_constant():
    frame = catalog()
    frame["halo_mass_log"] = 11.
    with pytest.raises(ValueError, match="redundant"):
        validate_clustered_identity_targets(frame, halo_mass_unit="M_sun")


def test_numeric_aliases_cannot_hide_duplicate_member_ids():
    frame = catalog()
    frame["subhalo_id"] = ["10", "10.0"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_clustered_identity_targets(frame, halo_mass_unit="M_sun")


def test_numeric_group_aliases_cannot_hide_conflicting_labels():
    frame = catalog()
    frame["group_id"] = ["2", "2.0"]
    frame.loc[1, ["halo_mass", "halo_mass_log"]] = [2e12, np.log10(2e12)]
    with pytest.raises(ValueError, match="conflicting"):
        validate_clustered_identity_targets(frame, halo_mass_unit="M_sun")


def test_log_units_are_explicit_and_log_only_target_is_supported():
    frame = catalog()
    frame["halo_mass"] = 12.
    validate_clustered_identity_targets(frame, halo_mass_unit="log10_Msun")
    validate_clustered_identity_targets(frame.drop(columns="halo_mass"), halo_mass_unit="M_sun")
    with pytest.raises(ValueError, match="unit"):
        validate_clustered_identity_targets(frame, halo_mass_unit="guess")


def test_duplicate_member_across_groups_rejected_by_loader_before_grouping(tmp_path):
    frame = catalog()
    frame.loc[1, "subhalo_id"] = 10
    frame.loc[1, "group_id"] = 3
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    loader = TNGLoader({"data": {"tng": {"cache_dir": str(tmp_path / "cache")}}})
    with pytest.raises(ValueError, match="duplicate subhalo_id"):
        loader._load_from_clustered_csv(path)


def test_conflicting_group_target_rejected_by_loader(tmp_path):
    frame = catalog()
    frame.loc[1, ["halo_mass", "halo_mass_log"]] = [2e12, np.log10(2e12)]
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    loader = TNGLoader({"data": {"tng": {"cache_dir": str(tmp_path / "cache")}}})
    with pytest.raises(ValueError, match="conflicting.*target"):
        loader._load_from_clustered_csv(path)


def test_periodic_diagnostic_counts_unique_pairs_and_missing_radius_neighbors():
    positions = np.array([[.1, 0, 0], [9.9, 0, 0], [5., 0, 0]])
    report = periodic_pair_diagnostics(positions, box_size=10., radius=1.)
    assert report == {"pair_count": 3, "wrapped_pair_count": 1,
                      "radius_membership_disagreements": 1}
    assert positions[1, 0] == 9.9


@pytest.mark.parametrize("box_size,radius", [(0., 1.), (float("nan"), 1.), (10., -1.)])
def test_periodic_diagnostic_rejects_bad_units_parameters(box_size, radius):
    with pytest.raises(ValueError):
        periodic_pair_diagnostics(np.zeros((2, 3)), box_size=box_size, radius=radius)


def reference():
    return {"group_id": 2, "first_subhalo_id": 10, "member_group_ids": np.array([2, 2]),
            "target_msun": 1e12, "fields": {"half_mass_radius": np.array([3., 4.])}}


def test_reference_matching_uses_explicit_group_offset_and_preserves_coverage():
    frame = catalog()
    frame["half_mass_radius"] = [3., 4.]
    extra = frame.iloc[[0]].copy()
    extra[["group_id", "subhalo_id"]] = [9, 99]
    report = match_reference_group(pd.concat([frame, extra]), reference())
    assert report["matched_rows"] == 2
    assert report["field_matches"] == {"half_mass_radius": 2}
    assert report["target_matches"] == 2


def test_reference_matching_does_not_treat_out_of_range_ids_as_verified():
    frame = catalog()
    frame["half_mass_radius"] = [3., 4.]
    frame.loc[1, "subhalo_id"] = 999
    with pytest.raises(ValueError, match="outside"):
        match_reference_group(frame, reference())


def test_reference_matching_checks_membership_instead_of_only_row_index():
    frame = catalog()
    frame["half_mass_radius"] = [3., 4.]
    raw = deepcopy(reference())
    raw["member_group_ids"][1] = 3
    with pytest.raises(ValueError, match="membership"):
        match_reference_group(frame, raw)


def test_reference_mismatch_does_not_count_as_verified_coverage():
    frame = catalog()
    frame["half_mass_radius"] = [3., 999.]
    report = match_reference_group(frame, reference())
    assert report["matched_rows"] == 2
    assert report["fully_matching_subhalo_ids"] == [10]


def test_absent_or_mismatched_csv_group_cannot_be_certified_by_raw_report(tmp_path):
    import h5py
    from scripts.audit_paper_provenance import raw_evidence
    with h5py.File(tmp_path / "groupcat_99_0.hdf5", "w") as source:
        header = source.create_group("Header")
        header.attrs.update(Ngroups_ThisFile=1, Nsubgroups_ThisFile=2, Ngroups_Total=1,
                            Nsubgroups_Total=2, NumFiles=1, HubbleParam=1., Time=1., BoxSize=10.)
        group = source.create_group("Group")
        group["GroupNsubs"], group["GroupFirstSub"] = [2], [10]
        for name in ("Group_M_Crit200", "GroupMass", "Group_M_Mean200", "Group_M_Crit500", "Group_M_TopHat200"):
            group[name] = [100.]
        sub = source.create_group("Subhalo")
        sub["SubhaloGrNr"] = [2, 2]
        sub["SubhaloHalfmassRad"] = [3., 4.]
        sub["SubhaloVelDisp"] = [0., 0.]
        sub["SubhaloGasMetallicity"] = [1e-6, 1e-6]
        sub["SubhaloMassType"] = np.ones((2, 6))
        sub["SubhaloHalfmassRadType"] = np.ones((2, 6))
        sub["SubhaloPos"], sub["SubhaloVel"] = np.zeros((2, 3)), np.zeros((2, 3))
    frame = catalog()
    frame["half_mass_radius"], frame["vel_dispersion"] = [3., 4.], 0.
    frame["metallicity"], frame["stellar_mass"] = 1e-6, 10.
    for field in ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"):
        frame[field] = 0.
    empty = raw_evidence(tmp_path, frame.iloc[:0])
    assert empty["verified_group_ids"] == []
    assert empty["files"][0]["status"] == "RAW_GROUP_ABSENT_FROM_CSV"
    frame["halo_mass"] = 1.
    mismatched = raw_evidence(tmp_path, frame)
    assert mismatched["verified_row_count"] == 0
    assert mismatched["verified_group_ids"] == []
    assert mismatched["files"][0]["status"] == "CSV_CONTENT_MISMATCH"


def test_full_campaign_source_identity_reconstructed_from_immutable_base():
    from scripts.audit_paper_provenance import base_source_identity
    assert base_source_identity() == "8a05077184564309a97d339ff4773727cad7f2a400e50b4fb79c053af0f66545"


def test_changed_aggregate_identity_is_reported_even_when_commit_agrees(tmp_path):
    import json
    from types import SimpleNamespace
    import torch
    import rls.matched_benchmark
    from scripts.audit_paper_provenance import BASE, campaign_evidence
    # Each stage's commit/inputs/IDs agree, but B's aggregate claims other code.
    manifest = {"status": "complete", "repo": {"head": BASE},
                "dataset": {"tng_csv": {"sha256": "csv"}, "checkpoint": {"sha256": "checkpoint"}},
                "backbone": {"sha256": "state"}, "split_cluster_ids": {"test": ["g"]},
                "source_identity": "8a05077184564309a97d339ff4773727cad7f2a400e50b4fb79c053af0f66545",
                "research_label": "fixture", "source_files": {}, "artifacts": {}}
    for stage in "ABCD":
        folder = tmp_path / f"{stage}-v1-terminal-001/artifacts/cosmic-net-runs/notebook_{stage}"
        folder.mkdir(parents=True)
        current = deepcopy(manifest)
        if stage == "B":
            current["source_identity"] = "0" * 64
        (folder / "provenance.json").write_text(json.dumps({stage: current}))
    graph = SimpleNamespace(cluster_id="g", x=torch.ones((1, 1)), y=torch.tensor([12.]),
                            edge_index=torch.tensor([[0], [0]]), edge_attr=torch.ones((1, 1)))
    graph_hash = rls.matched_benchmark._graph_content_hash(graph, "g")
    row = {"graph_id": "g", "requested_keep_fraction": .4, "status": "success",
           "target": 12., "full_prediction": 12., "pruned_prediction": 12.,
           "physical_retention": 1., "graph_content_hash": graph_hash}
    cfolder = tmp_path / "C-v1-terminal-001/artifacts/cosmic-net-runs/notebook_C"
    (cfolder / "matched_per_example.json").write_text(json.dumps(
        [{**row, "method": method} for method in ("full", "provided_rl")]))
    report = campaign_evidence(tmp_path, "csv", "checkpoint", "state", {"test": ["g"]}, [graph])
    assert "B:aggregate_source_identity" in report["errors"]
