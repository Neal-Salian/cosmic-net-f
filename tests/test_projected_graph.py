import pytest

from data.projected_graph import (
    ProjectedGraphConfig,
    build_projected_graph,
    checkpoint_compatibility_record,
    validate_checkpoint_compatibility,
)
from graph.graph_builder import GraphBuilder
from data.loaders.base_loader import HaloData, SubhaloData
from data.observations import ObservationConfig, observe_halo


def halo():
    return HaloData("h1", [
        SubhaloData("a", [0., 0., 0.], [10., 20., 30.], 1e10, 999., 9., .7),
        SubhaloData("b", [1., 2., 3.], [40., 60., 90.], 1e11, 888., 8., .8),
        SubhaloData("c", [3., 2., 1.], [70., 100., 150.], 1e12, 777., 7., .9),
    ], 1e14, metadata={"catalog_id": "cat"})


def observed(axis="z"):
    return observe_halo(halo(), ObservationConfig(line_of_sight=axis))[0]


def test_projected_graph_uses_axis_specific_position_and_velocity():
    graphs = [build_projected_graph(observed(axis), ProjectedGraphConfig()) for axis in "xyz"]
    assert [g.observer_axis for g in graphs] == ["x", "y", "z"]
    assert graphs[0].pos.shape == (3, 2)
    assert graphs[0].vel.shape == (3, 1)
    assert graphs[0].x.shape == (3, 4)


def test_unknown_feature_fails_and_schema_identity_blocks_same_width_3d_checkpoint():
    with pytest.raises(ValueError, match="unknown"):
        ProjectedGraphConfig(node_features=("log_metallicity",))
    record = checkpoint_compatibility_record(ProjectedGraphConfig())
    legacy = {**record, "feature_mode": "3d", "feature_schema": "legacy_3d_v1"}
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_compatibility(record, legacy)
    altered = {**record, "feature_mode": "3d"}
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_compatibility(altered, altered)


@pytest.mark.parametrize("kwargs", [
    {"radius_mpc": 0}, {"radius_mpc": float("nan")}, {"k_neighbors": 0},
    {"k_neighbors": 1.5}, {"k_neighbors": True},
])
def test_invalid_graph_construction_scales_fail(kwargs):
    with pytest.raises(ValueError):
        ProjectedGraphConfig(**kwargs)


def test_los_axis_controls_velocity_and_projected_separation():
    config = ProjectedGraphConfig(radius_mpc=10)
    graph_x = build_projected_graph(observed("x"), config)
    graph_y = build_projected_graph(observed("y"), config)
    graph_z = build_projected_graph(observed("z"), config)
    def edge_value(graph, source, destination, column):
        matches = ((graph.edge_index[0] == source) & (graph.edge_index[1] == destination)).nonzero().flatten()
        return graph.edge_attr[matches[0], column].item()
    # First pair's distinct component differences follow the declared LOS.
    assert edge_value(graph_x, 0, 1, 1) == pytest.approx(30 / 1000)
    assert edge_value(graph_y, 0, 1, 1) == pytest.approx(40 / 1000)
    assert edge_value(graph_z, 0, 1, 1) == pytest.approx(60 / 1000)
    # x LOS projects onto yz (distance sqrt(13)); z LOS projects onto xy (sqrt(5)).
    assert edge_value(graph_x, 0, 1, 0) == pytest.approx((13 ** .5))
    assert edge_value(graph_z, 0, 1, 0) == pytest.approx((5 ** .5))
    assert graph_x.compatibility_record["line_of_sight"] == "x"
    assert graph_z.compatibility_record["line_of_sight"] == "z"
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_compatibility(graph_x.compatibility_record, graph_z.compatibility_record)


def test_projected_graph_does_not_expose_intrinsic_fields_and_legacy_builder_rejects_it():
    projected, report = observe_halo(halo(), ObservationConfig(line_of_sight="y", seed=5))
    graph = build_projected_graph(projected, ProjectedGraphConfig(), report)
    assert set(graph.keys()) == {
        "x", "edge_index", "edge_attr", "y", "pos", "vel", "cluster_id", "num_nodes",
        "feature_mode", "graph_schema", "schema_identity", "observer_axis", "observation_schema",
        "compatibility_record", "normalization", "observation_provenance", "catalog_provenance",
    }
    assert graph.pos.shape[1] == 2 and graph.vel.shape[1] == 1
    assert graph.observation_provenance["seed"] == 5
    assert graph.catalog_provenance["catalog_id"] == "cat"
    with pytest.raises(ValueError, match="projected observations require"):
        GraphBuilder({}).build_graph(projected)


def test_member_permutation_preserves_id_keyed_features():
    original = observed("z")
    permuted = observed("z")
    permuted.subhalos = list(reversed(permuted.subhalos))
    first = build_projected_graph(original, ProjectedGraphConfig())
    second = build_projected_graph(permuted, ProjectedGraphConfig())
    by_id_1 = {str(m.subhalo_id): first.x[i].tolist() for i, m in enumerate(original.subhalos)}
    by_id_2 = {str(m.subhalo_id): second.x[i].tolist() for i, m in enumerate(permuted.subhalos)}
    assert by_id_1 == by_id_2
    def pair_features(graph, members):
        ids = [str(member.subhalo_id) for member in members]
        return {
            (ids[int(src)], ids[int(dst)]): graph.edge_attr[i].tolist()
            for i, (src, dst) in enumerate(graph.edge_index.t().tolist())
        }
    assert pair_features(first, original.subhalos) == pair_features(second, permuted.subhalos)


def test_graph_rebuild_uses_changed_observed_members_and_records_transform_report(monkeypatch):
    from data.observations import ObservationConfig, observe_halo
    source = halo()
    full, report_full = observe_halo(source, ObservationConfig(realization_id="full"))
    sparse, report_sparse = next(
        (result, report) for seed in range(100)
        for result, report in [observe_halo(source, ObservationConfig(
            missing_fraction=0.5, seed=seed, realization_id="sparse"))]
        if result is not None and len(result.subhalos) == 2
    )
    assert report_full != report_sparse
    g_full = build_projected_graph(full, ProjectedGraphConfig(), report_full)
    g_sparse = build_projected_graph(sparse, ProjectedGraphConfig(), report_sparse)
    assert g_full.num_nodes == 3
    assert g_sparse.num_nodes == 2
    assert g_full.edge_index.shape != g_sparse.edge_index.shape
    assert g_sparse.observation_provenance == report_sparse
    from graph.graph_builder import GraphBuilder
    original_build = GraphBuilder._build_radius_edges
    seen_positions = []
    def record_build(builder, positions, count):
        seen_positions.append(positions.clone())
        return original_build(builder, positions, count)
    monkeypatch.setattr(GraphBuilder, "_build_radius_edges", record_build)
    rebuilt = build_projected_graph(sparse, ProjectedGraphConfig(), report_sparse)
    assert rebuilt.num_nodes == 2
    assert len(seen_positions) == 1
    import torch
    assert torch.equal(seen_positions[0][:, 2], torch.zeros(2))
    assert torch.equal(seen_positions[0][:, :2], rebuilt.pos)


def test_input_features_are_independent_of_target_and_hidden_intrinsic_fields():
    first = observed("z")
    second = observed("z")
    second.halo_mass = 1e16
    for i, member in enumerate(second.subhalos):
        member.velocity_dispersion = 10000 + i
        member.half_mass_radius = 50 + i
        member.metallicity = .01 + i
    graph_a = build_projected_graph(first)
    graph_b = build_projected_graph(second)
    assert graph_a.x.equal(graph_b.x)
    assert graph_a.edge_index.equal(graph_b.edge_index)
    assert graph_a.edge_attr.equal(graph_b.edge_attr)
    assert graph_a.y.item() != graph_b.y.item()


def test_report_identity_must_match_halo_catalog_halo_and_observer():
    projected = observed("x")
    with pytest.raises(ValueError, match="observation report identity"):
        build_projected_graph(projected, observation_report={
            "catalog_id": "other", "halo_id": "h1", "line_of_sight": "x"})
    with pytest.raises(ValueError, match="observation report is incomplete or inconsistent"):
        build_projected_graph(projected, observation_report={
            "catalog_id": "cat", "halo_id": "h1", "line_of_sight": "x",
            "schema": "projected_observation_v1", "realization_id": "default", "seed": 0,
            "transforms": [],
        })


def test_report_realization_and_seed_are_bound_to_observed_copy():
    projected, report = observe_halo(
        halo(), ObservationConfig(line_of_sight="x", seed=7, realization_id="run-a"))
    build_projected_graph(projected, observation_report=report)
    wrong = {**report, "realization_id": "run-b"}
    with pytest.raises(ValueError, match="observation report identity"):
        build_projected_graph(projected, observation_report=wrong)
    wrong_seed = {**report, "seed": 8}
    with pytest.raises(ValueError, match="observation report identity"):
        build_projected_graph(projected, observation_report=wrong_seed)
    altered_noise = {**report, "transforms": [dict(step) for step in report["transforms"]]}
    altered_noise["transforms"][2]["scales"] = {"los_velocity_kms": 9999}
    with pytest.raises(ValueError, match="differs from the stored observation report"):
        build_projected_graph(projected, observation_report=altered_noise)
    altered_selection = {**report, "transforms": [dict(step) for step in report["transforms"]]}
    altered_selection["transforms"][3] = {
        **altered_selection["transforms"][3], "min_inclusive": 99,
    }
    with pytest.raises(ValueError, match="differs from the stored observation report"):
        build_projected_graph(projected, observation_report=altered_selection)


def test_default_graph_path_carries_full_transform_report():
    config = ObservationConfig(
        line_of_sight="z", seed=13, realization_id="report-check",
        missing_fraction=0.25, position_noise_mpc=0.02,
        los_velocity_noise_kms=5.0, min_richness=1,
    )
    projected, report = observe_halo(halo(), config)
    graph = build_projected_graph(projected)
    assert graph.observation_provenance == report
    assert projected.metadata["observation_provenance"] is not report
    report["transforms"][0]["seed"] = 123456
    assert projected.metadata["observation_provenance"]["seed"] == config.seed
    assert {step["name"] for step in graph.observation_provenance["transforms"]} == {
        "missingness", "project", "noise", "richness_selection",
    }


def test_graph_exposes_catalog_lineage_release_and_selection_accounting():
    source = halo()
    source.metadata.update({
        "initial_condition_id": "ic-17",
        "release": {"catalog_version": "v2"},
        "selection_accounting": {"selected_rows": 3, "excluded_rows": 1},
    })
    projected = observe_halo(source, ObservationConfig())[0]
    graph = build_projected_graph(projected)
    assert graph.initial_condition_id == "ic-17"
    assert graph.lineage_group == "ic-17"
    from rls.matched_benchmark import _lineage_group
    assert _lineage_group(graph) == "ic-17"
    assert graph.catalog_provenance["initial_condition_id"] == "ic-17"
    assert graph.catalog_provenance["release"] == {"catalog_version": "v2"}
    assert graph.catalog_provenance["selection_accounting"] == {"selected_rows": 3, "excluded_rows": 1}


def test_float32_overflow_is_rejected_after_conversion():
    projected = observed("z")
    projected.subhalos[0].velocity[2] = 1e300
    with pytest.raises(ValueError, match="nonfinite after float32 conversion"):
        build_projected_graph(projected)
