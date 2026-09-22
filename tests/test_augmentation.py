import numpy as np
import pytest
import torch

from data.augmentation import augment_halo, augment_training_split
from data.loaders.base_loader import HaloData, SubhaloData
from data.splits import grouped_split
from graph.graph_builder import GraphBuilder, build_dataloaders


def sample_halo(cluster_id="halo-a", group="ic-a"):
    positions = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 4.]]
    velocities = [[1., 2., 3.], [-2., 1., 4.], [3., 0., -1.], [0., -2., 1.]]
    return HaloData(
        cluster_id=cluster_id,
        halo_mass=1e13,
        metadata={"ic_group": group},
        subhalos=[
            SubhaloData(index, np.array(pos, dtype=np.float32),
                        np.array(vel, dtype=np.float32), 1e10 + index,
                        100. + index, .01, .02)
            for index, (pos, vel) in enumerate(zip(positions, velocities))
        ],
    )


def test_rotation_preserves_distances_and_rotates_positions_and_velocities_jointly():
    original = sample_halo()
    augmented = augment_halo(original, seed=17, jitter_std=0.0, rotate=True)
    rotation = np.asarray(augmented.metadata["augmentation"]["rotation_matrix"])

    before = torch.pdist(torch.tensor(original.get_positions()))
    after = torch.pdist(torch.tensor(augmented.get_positions()))
    assert torch.allclose(before, after, atol=1e-6)
    assert np.allclose(augmented.get_positions(), original.get_positions() @ rotation.T)
    assert np.allclose(augmented.get_velocities(), original.get_velocities() @ rotation.T)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)


def test_augmentation_is_deterministic_local_and_does_not_mutate_parent():
    parent = sample_halo()
    parent_positions = parent.get_positions().copy()
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)

    first = augment_halo(parent, seed=5, jitter_std=.2, line_of_sight="z")
    second = augment_halo(parent, seed=5, jitter_std=.2, line_of_sight="z")

    assert np.array_equal(first.get_positions(), second.get_positions())
    assert np.array_equal(np.random.random(3), expected)
    assert np.array_equal(parent.get_positions(), parent_positions)
    assert first.metadata["augmentation_parent_id"] == parent.cluster_id
    assert first.metadata["augmentation"]["line_of_sight"] == "z"
    assert first.metadata["augmentation"]["transform_version"] == 1


def test_jittered_halo_rebuilds_topology_and_configured_feature_subset():
    config = {"graph": {"method": "radius", "radius_mpc": 1.1,
              "self_loops": False, "edge_features": ["distance", "cos_theta"]}}
    parent = sample_halo()
    augmented = augment_halo(parent, seed=2, jitter_std=2.0, rotate=False)
    builder = GraphBuilder(config)
    original_graph = builder.build_graph(parent)
    augmented_graph = builder.build_graph(augmented)

    assert not torch.equal(original_graph.edge_index, augmented_graph.edge_index)
    assert augmented_graph.edge_attr.shape[1] == 2
    assert torch.allclose(
        augmented_graph.edge_attr[:, 0],
        torch.linalg.vector_norm(
            augmented_graph.pos[augmented_graph.edge_index[1]] -
            augmented_graph.pos[augmented_graph.edge_index[0]], dim=1),
    )


def test_augmentation_is_train_only_and_preserves_group_lineage():
    halos = [sample_halo(f"h-{index}", f"ic-{index}") for index in range(6)]
    train, val, test, split_manifest = grouped_split(
        halos, "ic_group", ratios=(.5, .25, .25), seed=11)
    expanded, augmentation_manifest = augment_training_split(
        train, split_manifest, copies=1, seed=19, jitter_std=.01)

    assert len(expanded) == 2 * len(train)
    assert {entry["parent_id"] for entry in augmentation_manifest["transforms"]} == {
        item.cluster_id for item in train
    }
    assert all(item.metadata.get("augmentation_split") == "train"
               for item in expanded[len(train):])
    assert not any("augmentation_parent_id" in item.metadata for item in val + test)


def test_build_dataloaders_augments_only_training_split():
    train = [sample_halo("train", "ic-train")]
    val = [sample_halo("val", "ic-val")]
    test = [sample_halo("test", "ic-test")]
    config = {
        "seed": 3,
        "data": {"batch_size": 2, "num_workers": 0},
        "graph": {"method": "knn", "k_neighbors": 1, "self_loops": False,
                  "edge_features": ["distance"]},
        "augmentation": {"enabled": True, "copies": 1, "jitter_std": .01,
                         "rotate": True, "line_of_sight": "z"},
    }
    train_loader, val_loader, test_loader = build_dataloaders(config, train, val, test)

    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == len(test_loader.dataset) == 1
    assert train_loader.dataset[1].metadata["augmentation_parent_id"] == "train"
    assert not hasattr(val_loader.dataset[0], "augmentation_parent_id")


def test_proper_rotation_samples_both_hemispheres():
    """_proper_rotation must cover both hemispheres (positive and negative
    rotation[0,0]) across different seeds, not just one half of SO(3)."""
    from data.augmentation import _proper_rotation
    positive_count = 0
    negative_count = 0
    for seed in range(64):
        rng = np.random.default_rng(seed)
        rot = _proper_rotation(rng)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10), (
            f"rotation not orthogonal at seed {seed}"
        )
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-10), (
            f"rotation det != +1 at seed {seed}"
        )
        if rot[0, 0] > 0:
            positive_count += 1
        else:
            negative_count += 1
    assert positive_count > 0, (
        f"no positive rotation[0,0] across 64 seeds (all negative)"
    )
    assert negative_count > 0, (
        f"no negative rotation[0,0] across 64 seeds (all positive)"
    )
    assert positive_count >= 8, (
        f"too few positive hemisphere samples: {positive_count}/64"
    )
    assert negative_count >= 8, (
        f"too few negative hemisphere samples: {negative_count}/64"
    )
