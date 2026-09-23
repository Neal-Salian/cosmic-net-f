"""Partial astronomy records flow into graphs through observables only."""
import copy
import hashlib

import h5py
import numpy as np
import pytest
import yaml

from data.astronomy_audit import audit_astronomy_manifest
from data.astronomy_loader import load_audited_catalog
from data.astronomy_manifest import load_astronomy_manifest
from data.astronomy_observations import observe_astronomy_halo
from data.observations import ObservationConfig
from data.projected_graph import build_projected_graph
from graph.graph_builder import GraphBuilder


def _audited(tmp_path):
    source = tmp_path / "tiny.h5"
    with h5py.File(source, "w") as f:
        f.attrs["h"] = 0.5
        f.attrs["Time"] = 0.5
        f.create_dataset("Group/Group_M_Crit200", data=[5., 10.])
        f.create_dataset("Subhalo/SubhaloMass", data=[1., 2., 3.])
        f.create_dataset("Subhalo/SubhaloPos", data=[[100., 0., -100.], [200., 0., 0.], [300., 20., 0.]])
        f.create_dataset("Subhalo/SubhaloVel", data=[[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]])
        f.create_dataset("Subhalo/SubhaloMassType", data=[[0, 0, 0, 0, .2, 0], [0, 0, 0, 0, .4, 0], [0, 0, 0, 0, 0, 0]])
        f.create_dataset("Subhalo/SubhaloHalfmassRadType", data=[[0, 0, 0, 0, 50., 0], [0, 0, 0, 0, 100., 0], [0, 0, 0, 0, 0., 0]])
        f.create_dataset("Subhalo/SubhaloGrNr", data=[0, 0, 1])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    entry = {
        "id": "tng-test", "role": "external_development", "suite": "IllustrisTNG",
        "simulation": "L25n256", "simulation_set": "LH", "initial_condition_id": "ic-001",
        "snapshot": "099", "source": {"kind": "local_hdf5", "path": str(source), "sha256": digest},
        "epoch": {"redshift": 1., "scale_factor": .5, "h": .5},
        "conventions": {"coordinates": "comoving", "velocities": "peculiar_km_per_s", "periodic_box": False},
        "fields": {
            "group_mass": {"path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "shape": [2]},
            "subhalo_mass_anchor": {"path": "Subhalo/SubhaloMass", "unit": "1e10_Msun_per_h", "shape": [3]},
            "position": {"path": "Subhalo/SubhaloPos", "unit": "ckpc_per_h", "shape": [3, 3]},
            "velocity": {"path": "Subhalo/SubhaloVel", "unit": "km_per_s", "shape": [3, 3]},
            "stellar_mass": {"path": "Subhalo/SubhaloMassType", "component": 4, "unit": "1e10_Msun_per_h", "shape": [3, 6]},
            "half_mass_radius": {"path": "Subhalo/SubhaloHalfmassRadType", "component": 4, "unit": "ckpc_per_h", "shape": [3, 6]},
            "group_index": {"path": "Subhalo/SubhaloGrNr", "unit": "index", "shape": [3]},
        },
        "target": {"name": "halo_mass", "path": "Group/Group_M_Crit200", "unit": "1e10_Msun_per_h", "definition": "log10(M200c/M_sun)", "shape": [2]},
        "selection": {"predicate": "stellar_mass_type4_gt_zero"}, "release": "test fixture",
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "catalogs": [entry]}))
    manifest = load_astronomy_manifest(path)
    report = audit_astronomy_manifest(manifest)
    assert report.ok, report.errors
    halos = load_audited_catalog(manifest, report, "tng-test")
    assert halos[1].members == ()
    assert halos[1].metadata["selection_accounting"]["excluded_rows"] == 1
    return halos


def _snapshot(halo):
    return [(m.member_id, m.position_mpc.copy(), m.velocity_km_s.copy(), m.stellar_mass_msun)
            for m in halo.members], copy.deepcopy(halo.metadata), halo.halo_mass_msun


def test_audited_partial_record_projects_and_preserves_identity_without_fake_fields(tmp_path):
    source = _audited(tmp_path)[0]
    before = _snapshot(source)
    observed, report = observe_astronomy_halo(source, ObservationConfig(line_of_sight="y"))
    assert observed is not None
    assert observed.schema == "partial_observation_inputs_v1"
    assert observed.metadata["source"] == source.metadata["source"]
    assert observed.metadata["catalog_contract"] == source.metadata["catalog_contract"]
    assert observed.metadata["role"] == "external_development"
    assert observed.metadata["initial_condition_id"] == "ic-001"
    assert all(set(vars(m)) == {"subhalo_id", "position", "velocity", "stellar_mass"}
               for m in observed.subhalos)
    graph = build_projected_graph(observed, observation_report=report)
    assert graph.x.shape == (2, 4)
    assert graph.observer_axis == "y"
    assert graph.initial_condition_id == "ic-001"
    assert graph.catalog_provenance["catalog_contract"] == source.metadata["catalog_contract"]
    after = _snapshot(source)
    assert len(after[0]) == len(before[0])
    for before_member, after_member in zip(before[0], after[0]):
        assert np.array_equal(before_member[1], after_member[1])
        assert np.array_equal(before_member[2], after_member[2])
        assert before_member[0] == after_member[0]
        assert before_member[3] == after_member[3]
    assert after[1:] == before[1:]


def test_member_order_does_not_change_observation_by_stable_id(tmp_path):
    halo = _audited(tmp_path)[0]
    reversed_halo = copy.deepcopy(halo)
    object.__setattr__(reversed_halo, "members", tuple(reversed(reversed_halo.members)))
    config = ObservationConfig(seed=92, missing_fraction=.25, position_noise_mpc=.01,
                               los_velocity_noise_kms=1., stellar_mass_noise_dex=.01)
    first, report_first = observe_astronomy_halo(halo, config)
    second, report_second = observe_astronomy_halo(reversed_halo, config)
    assert report_first == report_second
    one = {m.subhalo_id: (m.position, m.velocity, m.stellar_mass) for m in first.subhalos}
    two = {m.subhalo_id: (m.position, m.velocity, m.stellar_mass) for m in second.subhalos}
    assert one.keys() == two.keys()
    for member_id in one:
        assert np.array_equal(one[member_id][0], two[member_id][0])
        assert np.array_equal(one[member_id][1], two[member_id][1])
        assert one[member_id][2] == two[member_id][2]


def test_target_changes_only_label_and_richness_missingness_noise_are_reported(tmp_path):
    halo = _audited(tmp_path)[0]
    alternate = copy.deepcopy(halo)
    object.__setattr__(alternate, "halo_mass_msun", halo.halo_mass_msun * 100)
    object.__setattr__(alternate, "target_log10_msun", halo.target_log10_msun + 2)
    config = ObservationConfig(seed=1, missing_fraction=.5, position_noise_mpc=.1,
                               los_velocity_noise_kms=2., stellar_mass_noise_dex=.02,
                               min_richness=1)
    a, ra = observe_astronomy_halo(halo, config)
    b, rb = observe_astronomy_halo(alternate, config)
    assert ra == rb
    assert [m.position.tolist() for m in a.subhalos] == [m.position.tolist() for m in b.subhalos]
    assert [m.velocity.tolist() for m in a.subhalos] == [m.velocity.tolist() for m in b.subhalos]
    assert [m.stellar_mass for m in a.subhalos] == [m.stellar_mass for m in b.subhalos]
    assert ra["pre_richness"] == 2
    assert 0 <= ra["rejections"]["missingness"] <= 2
    assert ra["transforms"][2]["scales"] == {
        "projected_position_mpc": .1, "los_velocity_kms": 2., "stellar_mass_dex": .02,
    }
    rejected, rr = observe_astronomy_halo(halo, ObservationConfig(min_richness=3))
    assert rejected is None and rr["rejection_reason"] == "below_min_richness"


def test_zero_selected_members_return_richness_rejection_report(tmp_path):
    empty = _audited(tmp_path)[1]
    observed, report = observe_astronomy_halo(empty, ObservationConfig(min_richness=1))
    assert observed is None
    assert report["pre_richness"] == report["post_richness"] == 0
    assert report["rejection_reason"] == "below_min_richness"


def test_legacy_graph_builder_rejects_observed_partial_record(tmp_path):
    halo = _audited(tmp_path)[0]
    observed, _ = observe_astronomy_halo(halo, ObservationConfig())
    with pytest.raises(ValueError, match="projected observations require"):
        GraphBuilder({}).build_graph(observed)


@pytest.mark.parametrize("axis", "xyz")
def test_partial_record_supports_every_declared_los_axis(tmp_path, axis):
    halo = _audited(tmp_path)[0]
    observed, report = observe_astronomy_halo(halo, ObservationConfig(line_of_sight=axis))
    graph = build_projected_graph(observed, observation_report=report)
    assert graph.observer_axis == axis
