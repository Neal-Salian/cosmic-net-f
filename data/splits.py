"""Grouped, lineage-safe train/validation/test splits."""

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _manifest_hash(manifest: Dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items()
               if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _split_counts(total: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    train = int(total * ratios[0])
    val = int(total * ratios[1])
    return train, val, total - train - val


def grouped_split(halos, group_key, ratios=(0.7, 0.15, 0.15), seed=42):
    """Split whole lineage groups using a local, recorded NumPy generator."""
    if len(ratios) != 3 or not all(math.isfinite(float(x)) and x >= 0 for x in ratios):
        raise ValueError("ratios must be three finite nonnegative values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    groups: Dict[str, List[Any]] = {}
    for halo in halos:
        if group_key not in halo.metadata or halo.metadata[group_key] is None:
            raise ValueError(f"Halo {halo.cluster_id} is missing grouping key {group_key}")
        group_id = str(halo.metadata[group_key])
        groups.setdefault(group_id, []).append(halo)

    ordered_groups = list(groups)
    rng = np.random.default_rng(seed)
    shuffled = [ordered_groups[index] for index in rng.permutation(len(ordered_groups))]
    train_count, val_count, _ = _split_counts(len(shuffled), ratios)
    split_groups = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }
    splits = {
        name: [halo for group_id in group_ids for halo in groups[group_id]]
        for name, group_ids in split_groups.items()
    }
    manifest = {
        "schema_version": 1,
        "group_key": group_key,
        "seed": int(seed),
        "generator": "numpy.random.Generator(PCG64)",
        "ratios": {name: float(value) for name, value in
                   zip(("train", "val", "test"), ratios)},
        "splits": {},
    }
    for name in ("train", "val", "test"):
        items = splits[name]
        manifest["splits"][name] = {
            "group_ids": list(split_groups[name]),
            "cluster_ids": [str(item.cluster_id) for item in items],
            "parent_ids": sorted({str(item.metadata.get(
                "augmentation_parent_id", item.cluster_id)) for item in items}),
            "group_count": len(split_groups[name]),
            "halo_count": len(items),
        }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    validate_split_manifest(manifest, halos)
    return splits["train"], splits["val"], splits["test"], manifest


def validate_split_manifest(manifest, halos):
    """Reject altered manifests, missing members, and cross-split lineage reuse."""
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("split manifest hash mismatch; refuse silent reuse")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema_version")
    names = ("train", "val", "test")
    sections = manifest.get("splits", {})
    if set(sections) != set(names):
        raise ValueError("split manifest must define train, val and test")
    for field in ("group_ids", "cluster_ids", "parent_ids"):
        seen = set()
        for name in names:
            values = sections[name].get(field, [])
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {field} within split {name}")
            overlap = seen.intersection(values)
            if overlap:
                raise ValueError(f"{field} are not disjoint; cross-split reuse: {sorted(overlap)}")
            seen.update(values)
    expected_clusters = {str(halo.cluster_id) for halo in halos}
    actual_clusters = {cluster for name in names
                       for cluster in sections[name]["cluster_ids"]}
    if actual_clusters != expected_clusters:
        raise ValueError("split manifest cluster_ids do not match the supplied catalog")
    group_key = manifest.get("group_key")
    for halo in halos:
        if group_key not in halo.metadata:
            raise ValueError(f"Halo {halo.cluster_id} is missing grouping key {group_key}")
        cluster_id = str(halo.cluster_id)
        expected_group = str(halo.metadata[group_key])
        expected_parent = str(halo.metadata.get("augmentation_parent_id", halo.cluster_id))
        matches = [name for name in names if cluster_id in sections[name]["cluster_ids"]]
        if len(matches) != 1:
            raise ValueError(f"cluster {cluster_id} does not occur exactly once")
        section = sections[matches[0]]
        if expected_group not in section["group_ids"] or expected_parent not in section["parent_ids"]:
            raise ValueError(f"lineage mismatch for cluster {cluster_id}")
    return True
