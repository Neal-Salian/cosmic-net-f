"""Generic, JSON-safe provenance and artifact validation helpers."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def hash_payload(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def model_state_hash(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def validate_checkpoint_file(path):
    """Reject missing/LFS artifacts before callers invoke safe torch.load."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(256)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"Checkpoint {path} is a Git LFS pointer, not model data. "
            "Fetch the LFS object or pass an explicit valid checkpoint."
        )
    return sha256_file(path)


def build_run_provenance(*, source_files, catalog_contract, split_manifest,
                         resolved_config, source_revision, graph,
                         augmentation_manifest, research_label,
                         source_code_files=None):
    config_copy = deepcopy(resolved_config)
    catalog_copy = deepcopy(catalog_contract)
    split_hash = split_manifest.get("manifest_sha256") if split_manifest else None
    augmentation_copy = deepcopy(augmentation_manifest)
    return {
        "schema_version": 1,
        "research_label": research_label,
        "source_files": deepcopy(source_files),
        "catalog_contract": catalog_copy,
        "catalog_contract_sha256": hash_payload(catalog_copy),
        "split_manifest_sha256": split_hash,
        "resolved_config": config_copy,
        "config_sha256": hash_payload(config_copy),
        "source_revision": source_revision,
        "source_code_files": deepcopy(source_code_files or []),
        "graph": deepcopy(graph),
        "augmentation_manifest": augmentation_copy,
        "augmentation_manifest_sha256": (
            hash_payload(augmentation_copy) if augmentation_copy is not None else None
        ),
    }


def validate_run_provenance(provenance, *, catalog_contract=None,
                            split_manifest=None, resolved_config=None,
                            model_state=None):
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported provenance schema_version")
    # Validate against externally supplied expected values (if any).
    if catalog_contract is not None and provenance.get(
            "catalog_contract_sha256") != hash_payload(catalog_contract):
        raise ValueError("catalog contract provenance mismatch")
    if split_manifest is not None and provenance.get(
            "split_manifest_sha256") != split_manifest.get("manifest_sha256"):
        raise ValueError("split manifest provenance mismatch")
    if resolved_config is not None and provenance.get(
            "config_sha256") != hash_payload(resolved_config):
        raise ValueError("resolved config provenance mismatch")
    # Validate embedded payloads against their stored hashes.
    embedded_catalog = provenance.get("catalog_contract")
    if embedded_catalog is not None and provenance.get(
            "catalog_contract_sha256") != hash_payload(embedded_catalog):
        raise ValueError(
            "catalog contract embedded payload does not match stored hash"
        )
    embedded_config = provenance.get("resolved_config")
    if embedded_config is not None and provenance.get(
            "config_sha256") != hash_payload(embedded_config):
        raise ValueError(
            "resolved config embedded payload does not match stored hash"
        )
    embedded_aug = provenance.get("augmentation_manifest")
    if embedded_aug is not None and provenance.get(
            "augmentation_manifest_sha256") != hash_payload(embedded_aug):
        raise ValueError(
            "augmentation manifest embedded payload does not match stored hash"
        )
    if model_state is not None and provenance.get(
            "model_state_sha256") != model_state_hash(model_state):
        raise ValueError("model-state provenance mismatch")
    return True


def validate_inference_compatibility(training_catalog, evaluation_catalog):
    """Validate normalized feature/target semantics across distinct catalogs."""
    semantic_fields = (
        "target_definition", "mass_units", "position_units", "radius_units",
        "velocity_units", "coordinate_frame", "velocity_convention",
        "radius_semantic",
    )
    for field in semantic_fields:
        if training_catalog.get(field) != evaluation_catalog.get(field):
            raise ValueError(
                f"inference catalog is incompatible for {field}: "
                f"{training_catalog.get(field)!r} != {evaluation_catalog.get(field)!r}"
            )
    return True
