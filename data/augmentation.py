"""Halo-level augmentation with explicit lineage and local randomness."""

from copy import deepcopy
import hashlib
import json

import numpy as np


TRANSFORM_VERSION = 1


def _proper_rotation(rng):
    matrix, r = np.linalg.qr(rng.normal(size=(3, 3)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    matrix = matrix * signs
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def augment_halo(halo, *, seed, jitter_std=0.0, rotate=True,
                 line_of_sight="z", augmentation_id=0):
    """Copy one halo, jointly transform phase space, and record lineage."""
    if line_of_sight not in {"x", "y", "z"}:
        raise ValueError("line_of_sight must be one of x, y or z")
    if jitter_std < 0:
        raise ValueError("jitter_std must be nonnegative")
    rng = np.random.default_rng(seed)
    rotation = _proper_rotation(rng) if rotate else np.eye(3)
    result = deepcopy(halo)
    parent_id = str(halo.metadata.get("augmentation_parent_id", halo.cluster_id))
    result.cluster_id = f"{parent_id}_aug{augmentation_id}"
    for subhalo in result.subhalos:
        position = np.asarray(subhalo.position, dtype=np.float64) @ rotation.T
        if jitter_std:
            position = position + rng.normal(0.0, jitter_std, size=3)
        velocity = np.asarray(subhalo.velocity, dtype=np.float64) @ rotation.T
        subhalo.position = position.astype(np.float32)
        subhalo.velocity = velocity.astype(np.float32)
    result.metadata["augmentation_parent_id"] = parent_id
    result.metadata["augmentation"] = {
        "transform_version": TRANSFORM_VERSION,
        "seed": int(seed),
        "rotation_enabled": bool(rotate),
        "rotation_matrix": rotation.tolist(),
        "jitter_std": float(jitter_std),
        "line_of_sight": line_of_sight,
        "operation_order": ["rotation", "position_jitter"],
    }
    return result


def augment_training_split(train_halos, split_manifest=None, *, copies=1, seed=42,
                           jitter_std=0.0, rotate=True, line_of_sight="z"):
    """Append augmented views of training parents and return their manifest."""
    if copies < 0:
        raise ValueError("copies must be nonnegative")
    train_ids = {str(halo.cluster_id) for halo in train_halos}
    if split_manifest is not None:
        declared = set(split_manifest["splits"]["train"]["cluster_ids"])
        if not train_ids.issubset(declared):
            raise ValueError("augmentation parents must belong to manifest train split")
    transforms = []
    augmented = []
    for parent_index, halo in enumerate(train_halos):
        for copy_index in range(copies):
            transform_seed = int(seed + parent_index * max(copies, 1) + copy_index)
            child = augment_halo(
                halo, seed=transform_seed, jitter_std=jitter_std, rotate=rotate,
                line_of_sight=line_of_sight, augmentation_id=copy_index,
            )
            child.metadata["augmentation_split"] = "train"
            augmented.append(child)
            transforms.append({
                "parent_id": str(halo.cluster_id),
                "child_id": str(child.cluster_id),
                **deepcopy(child.metadata["augmentation"]),
            })
    manifest = {
        "schema_version": 1,
        "split": "train",
        "base_seed": int(seed),
        "copies": int(copies),
        "transforms": transforms,
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return list(train_halos) + augmented, manifest
