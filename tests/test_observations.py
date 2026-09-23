import copy

import numpy as np
import pytest

from data.loaders.base_loader import HaloData, SubhaloData
from data.observations import ObservationConfig, observe_halo


def make_halo(order=(10, 20, 30)):
    values = {
        10: ([1., 2., 3.], [10., 20., 30.], 1e9),
        20: ([3., 4., 5.], [40., 50., 60.], 2e9),
        30: ([5., 6., 7.], [70., 80., 90.], 3e9),
    }
    members = [SubhaloData(i, np.array(values[i][0]), np.array(values[i][1]), values[i][2],
                           99., .1, .02) for i in order]
    return HaloData("h1", members, 1e12, metadata={"catalog_id": "catA"})


def test_projection_uses_declared_axis_and_copies_input():
    halo = make_halo()
    before = copy.deepcopy(halo)
    observed, report = observe_halo(halo, ObservationConfig(line_of_sight="x"))
    assert np.allclose(observed.subhalos[0].position, [0., -2., -2.])
    assert np.allclose(observed.subhalos[0].velocity, [10., 0., 0.])
    assert observed.subhalos[0].velocity_dispersion != 99.
    assert halo.cluster_id == before.cluster_id
    assert np.array_equal(halo.subhalos[0].position, before.subhalos[0].position)
    assert halo.subhalos[0].velocity_dispersion == before.subhalos[0].velocity_dispersion
    assert report["pre_richness"] == report["post_richness"] == 3


def test_missingness_is_stable_by_member_id_when_input_order_changes():
    cfg = ObservationConfig(seed=19, realization_id="r2", missing_fraction=.5)
    a, ra = observe_halo(make_halo((10, 20, 30)), cfg)
    b, rb = observe_halo(make_halo((30, 10, 20)), cfg)
    assert {s.subhalo_id for s in a.subhalos} == {s.subhalo_id for s in b.subhalos}
    assert ra["rejected_member_ids"] == rb["rejected_member_ids"]


def test_noise_replays_by_identity_and_records_ordered_provenance():
    cfg = ObservationConfig(seed=8, position_noise_mpc=.2)
    a, ra = observe_halo(make_halo(), cfg)
    b, rb = observe_halo(make_halo((30, 10, 20)), cfg)
    by_id = lambda h: {s.subhalo_id: s.position for s in h.subhalos}
    assert all(np.allclose(by_id(a)[i], by_id(b)[i]) for i in by_id(a))
    assert [x["name"] for x in ra["transforms"]] == ["missingness", "project", "noise", "richness_selection"]
    assert ra == rb


def test_invalid_ids_units_and_disallowed_noise_fields_fail_closed():
    halo = make_halo()
    halo.subhalos[1].subhalo_id = halo.subhalos[0].subhalo_id
    with pytest.raises(ValueError, match="duplicate"):
        observe_halo(halo, ObservationConfig())
    halo = make_halo()
    halo.metadata.pop("catalog_id")
    with pytest.raises(ValueError, match="catalog ID"):
        observe_halo(halo, ObservationConfig())
    with pytest.raises(ValueError, match="position_units"):
        observe_halo(make_halo(), ObservationConfig(position_units="ckpc/h"))
    with pytest.raises(ValueError, match="allowed"):
        ObservationConfig(noise={"velocity_dispersion": 2.})


def test_richness_selection_is_inclusive_and_reports_rejection():
    observed, report = observe_halo(make_halo(), ObservationConfig(min_richness=4))
    assert observed is None
    assert report["rejection_reason"] == "below_min_richness"
    assert report["post_richness"] == 3
    accepted, report = observe_halo(make_halo(), ObservationConfig(min_richness=3, max_richness=3))
    assert accepted is not None and report["rejection_reason"] is None


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_each_line_of_sight_removes_hidden_coordinate_and_transverse_velocity(axis):
    observed, report = observe_halo(make_halo(), ObservationConfig(line_of_sight=axis))
    los = {"x": 0, "y": 1, "z": 2}[axis]
    sky = [i for i in range(3) if i != los]
    positions = np.asarray([s.position for s in observed.subhalos])
    velocities = np.asarray([s.velocity for s in observed.subhalos])
    assert np.all(positions[:, los] == 0)
    assert np.all(velocities[:, [i for i in range(3) if i != los]] == 0)
    assert np.allclose(positions[:, sky].mean(axis=0), 0)
    assert report["line_of_sight"] == axis


def test_projection_centers_on_members_retained_after_missingness():
    observed, report = observe_halo(make_halo(), ObservationConfig(seed=1, missing_fraction=.5))
    assert report["rejections"]["missingness"] > 0
    assert observed is not None
    sky_positions = np.asarray([s.position[:2] for s in observed.subhalos])
    assert np.allclose(sky_positions.mean(axis=0), 0)


def test_centroid_is_independent_of_member_order_with_cancellation():
    first = make_halo((10, 20, 30))
    members_by_id = {member.subhalo_id: member for member in first.subhalos}
    members_by_id[10].position = np.array([1e20, 0., 0.])
    members_by_id[20].position = np.array([3., 0., 0.])
    members_by_id[30].position = np.array([-1e20, 0., 0.])
    second = copy.deepcopy(first)
    second.subhalos = [members_by_id[i] for i in (10, 30, 20)]

    observed_first, _ = observe_halo(first, ObservationConfig())
    observed_second, _ = observe_halo(second, ObservationConfig())
    assert [member.subhalo_id for member in observed_first.subhalos] == [10, 20, 30]
    assert [member.subhalo_id for member in observed_second.subhalos] == [10, 30, 20]
    projected_by_id = lambda halo: {
        member.subhalo_id: member.position.copy() for member in halo.subhalos
    }
    positions_first = projected_by_id(observed_first)
    positions_second = projected_by_id(observed_second)
    assert positions_first.keys() == positions_second.keys()
    for member_id in positions_first:
        assert np.array_equal(positions_first[member_id], positions_second[member_id])


def test_all_allowed_noise_replays_by_member_and_ignores_target_and_r200c():
    cfg = ObservationConfig(seed=88, position_noise_mpc=.1,
                            los_velocity_noise_kms=2., stellar_mass_noise_dex=.05)
    halo_a = make_halo()
    halo_b = make_halo((30, 10, 20))
    halo_b.halo_mass = 9e15
    halo_b.metadata["R200c"] = 999999.
    a, _ = observe_halo(halo_a, cfg)
    b, _ = observe_halo(halo_b, cfg)
    def observables(halo):
        return {s.subhalo_id: (s.position.copy(), s.velocity.copy(), s.stellar_mass)
                for s in halo.subhalos}
    aa, bb = observables(a), observables(b)
    assert aa.keys() == bb.keys()
    for member_id in aa:
        assert np.allclose(aa[member_id][0], bb[member_id][0])
        assert np.allclose(aa[member_id][1], bb[member_id][1])
        assert aa[member_id][2] == pytest.approx(bb[member_id][2])


def test_all_allowed_noise_is_stable_under_member_reordering():
    cfg = ObservationConfig(seed=102, realization_id="reorder-noise",
                            position_noise_mpc=.1, los_velocity_noise_kms=2.,
                            stellar_mass_noise_dex=.05)
    a, _ = observe_halo(make_halo((10, 20, 30)), cfg)
    b, _ = observe_halo(make_halo((30, 10, 20)), cfg)
    by_id = lambda h: {s.subhalo_id: s for s in h.subhalos}
    aa, bb = by_id(a), by_id(b)
    for member_id in aa:
        assert np.array_equal(aa[member_id].position, bb[member_id].position)
        assert np.array_equal(aa[member_id].velocity, bb[member_id].velocity)
        assert aa[member_id].stellar_mass == bb[member_id].stellar_mass


def test_target_mass_and_r200c_do_not_change_observation():
    cfg = ObservationConfig(seed=88, position_noise_mpc=.1,
                            los_velocity_noise_kms=2., stellar_mass_noise_dex=.05)
    baseline = make_halo()
    changed_labels = make_halo()
    changed_labels.halo_mass = 9e15
    changed_labels.metadata["R200c"] = 999999.
    a, _ = observe_halo(baseline, cfg)
    b, _ = observe_halo(changed_labels, cfg)
    for sa, sb in zip(a.subhalos, b.subhalos):
        assert np.array_equal(sa.position, sb.position)
        assert np.array_equal(sa.velocity, sb.velocity)
        assert sa.stellar_mass == sb.stellar_mass


@pytest.mark.parametrize("kwargs", [
    {"min_richness": 1.5}, {"max_richness": 3.0},
    {"min_richness": True}, {"max_richness": True},
    {"min_richness": 0}, {"max_richness": 0}, {"max_richness": -1},
    {"seed": True},
])
def test_config_rejects_noninteger_or_invalid_richness_and_bool_seed(kwargs):
    with pytest.raises(ValueError):
        ObservationConfig(**kwargs)


@pytest.mark.parametrize("field,config_kwargs", [
    ("position", {"position_noise_mpc": 1.}),
    ("velocity", {"los_velocity_noise_kms": 1.}),
    ("stellar_mass", {"stellar_mass_noise_dex": 1.}),
])
def test_nonfinite_observation_after_each_noise_type_fails_closed(
        field, config_kwargs, monkeypatch):
    import data.observations as observations

    class HugeNoise:
        def random(self):
            return 1.0

        def normal(self, *_args):
            return 1e308

    monkeypatch.setattr(observations, "_rng_for",
                        lambda *_args: HugeNoise())
    halo = make_halo()
    member = halo.subhalos[1]
    if field == "position":
        member.position = np.array([1.7e308, 0., 0.])
    elif field == "velocity":
        member.velocity = np.array([0., 0., 1.7e308])
    else:
        member.stellar_mass = 1e9
    with pytest.raises(ValueError, match="nonfinite"):
        observe_halo(halo, ObservationConfig(seed=5, **config_kwargs))
