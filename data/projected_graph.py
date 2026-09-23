"""Schema-safe graph reconstruction for explicitly projected observations."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.observations import OBSERVATION_SCHEMA, OBSERVABLE_FIELDS
from graph.graph_builder import GraphBuilder


PROJECTED_FEATURE_MODE = "projected_observation"
PROJECTED_GRAPH_SCHEMA = "projected_graph_v1"
_NODE_FEATURES = ("sky_position_1", "sky_position_2", "los_velocity", "log_stellar_mass_proxy")
_EDGE_FEATURES = ("projected_distance", "los_delta_v", "mass_ratio")
@dataclass(frozen=True)
class ProjectedGraphConfig:
    """Fixed physical scaling and portable graph construction settings."""
    method: str = "radius"
    radius_mpc: float = 2.0
    k_neighbors: int = 10
    self_loops: bool = True
    edge_features: Tuple[str, ...] = _EDGE_FEATURES
    node_features: Tuple[str, ...] = _NODE_FEATURES
    position_scale_mpc: float = 1.0
    velocity_scale_kms: float = 1000.0
    stellar_mass_reference_msun: float = 1.0e10

    def __post_init__(self):
        if self.method not in {"radius", "knn"}:
            raise ValueError("method must be 'radius' or 'knn'")
        if (not isinstance(self.radius_mpc, Real) or isinstance(self.radius_mpc, bool)
                or not np.isfinite(self.radius_mpc) or self.radius_mpc <= 0):
            raise ValueError("radius_mpc must be positive and finite")
        if (not isinstance(self.k_neighbors, Integral) or isinstance(self.k_neighbors, bool)
                or self.k_neighbors < 1):
            raise ValueError("k_neighbors must be a positive integer")
        if set(self.node_features) - set(_NODE_FEATURES):
            raise ValueError(f"unknown projected node feature: {sorted(set(self.node_features) - set(_NODE_FEATURES))}")
        if set(self.edge_features) - set(_EDGE_FEATURES):
            raise ValueError(f"unknown projected edge feature: {sorted(set(self.edge_features) - set(_EDGE_FEATURES))}")
        if len(set(self.node_features)) != len(self.node_features) or not self.node_features:
            raise ValueError("projected node features must be nonempty and unique")
        if len(set(self.edge_features)) != len(self.edge_features):
            raise ValueError("projected edge features must be unique")
        for name in ("position_scale_mpc", "velocity_scale_kms", "stellar_mass_reference_msun"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def to_dict(self):
        return {
            "feature_mode": PROJECTED_FEATURE_MODE,
            "graph_schema": PROJECTED_GRAPH_SCHEMA,
            "node_features": list(self.node_features),
            "edge_features": list(self.edge_features),
            "normalization": {
                "kind": "fixed_physical_scales",
                "source_ids": [],
                "position_scale_mpc": self.position_scale_mpc,
                "velocity_scale_kms": self.velocity_scale_kms,
                "stellar_mass_reference_msun": self.stellar_mass_reference_msun,
            },
            "construction": {"method": self.method, "radius_mpc": self.radius_mpc,
                             "k_neighbors": self.k_neighbors, "self_loops": self.self_loops},
        }


def _identity(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_compatibility_record(config: ProjectedGraphConfig, line_of_sight: Optional[str] = None):
    """Create the complete semantic compatibility record for model checkpoints."""
    if line_of_sight is not None and line_of_sight not in {"x", "y", "z"}:
        raise ValueError("line_of_sight must be x, y or z")
    record = {**config.to_dict(), "observation_schema": OBSERVATION_SCHEMA,
              "observable_fields": list(OBSERVABLE_FIELDS), "line_of_sight": line_of_sight,
              "position_units": "Mpc", "velocity_units": "km/s", "stellar_mass_units": "M_sun"}
    record["schema_identity"] = _identity(record)
    return record


def validate_checkpoint_compatibility(expected: dict, checkpoint: dict):
    """Fail closed on task/schema semantics, even when tensor widths happen to match."""
    required = {"feature_mode", "graph_schema", "node_features", "edge_features", "normalization",
                "observation_schema", "observable_fields", "line_of_sight", "position_units",
                "velocity_units", "stellar_mass_units", "schema_identity"}
    if not isinstance(expected, dict) or not isinstance(checkpoint, dict):
        raise ValueError("incompatible checkpoint: compatibility records must be mappings")
    if not required.issubset(expected) or not required.issubset(checkpoint):
        raise ValueError("incompatible checkpoint: incomplete feature schema record")
    for record in (expected, checkpoint):
        identity_payload = {key: value for key, value in record.items() if key != "schema_identity"}
        if (record.get("feature_mode") != PROJECTED_FEATURE_MODE
                or record.get("graph_schema") != PROJECTED_GRAPH_SCHEMA
                or record.get("observation_schema") != OBSERVATION_SCHEMA
                or record.get("schema_identity") != _identity(identity_payload)):
            raise ValueError("incompatible checkpoint: invalid projected schema identity")
    if any(expected[key] != checkpoint[key] for key in required):
        raise ValueError("incompatible checkpoint feature schema or observation semantics")


def build_projected_graph(halo, config: ProjectedGraphConfig = ProjectedGraphConfig(),
                          observation_report: Optional[dict] = None) -> Data:
    """Rebuild a PyG graph from a copied, transformed observed HaloData."""
    observation = halo.metadata.get("observation")
    if not isinstance(observation, dict) or observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("projected graph requires projected_observation_v1 halo metadata")
    if observation.get("observable_fields") != list(OBSERVABLE_FIELDS):
        raise ValueError("observation metadata has an unsupported observable allowlist")
    axis = observation.get("line_of_sight")
    if axis not in {"x", "y", "z"}:
        raise ValueError("observation metadata must declare LOS x, y or z")
    if (observation.get("position_units"), observation.get("velocity_units"), observation.get("stellar_mass_units")) != ("Mpc", "km/s", "M_sun"):
        raise ValueError("projected graph requires Mpc, km/s and M_sun observations")
    if config.node_features != _NODE_FEATURES:
        raise ValueError("this projected adapter requires its declared four-column node schema")

    los = {"x": 0, "y": 1, "z": 2}[axis]
    sky = [i for i in range(3) if i != los]
    positions_np = np.asarray([member.position for member in halo.subhalos], dtype=np.float64).reshape(-1, 3)
    velocities_np = np.asarray([member.velocity for member in halo.subhalos], dtype=np.float64).reshape(-1, 3)
    masses_np = np.asarray([member.stellar_mass for member in halo.subhalos], dtype=np.float64)
    if not (np.isfinite(positions_np).all() and np.isfinite(velocities_np).all()
            and np.isfinite(masses_np).all() and (masses_np > 0).all()):
        raise ValueError("projected observation has invalid observable values")
    if len(halo.subhalos) != len({str(member.subhalo_id) for member in halo.subhalos}):
        raise ValueError("projected graph requires unique stable member IDs")
    positions = torch.tensor(positions_np[:, sky], dtype=torch.float32)
    velocities = torch.tensor(velocities_np[:, [los]], dtype=torch.float32)
    log_mass = torch.tensor(np.log10(masses_np / config.stellar_mass_reference_msun), dtype=torch.float32).reshape(-1, 1)
    x = torch.cat((positions / config.position_scale_mpc, velocities / config.velocity_scale_kms, log_mass), dim=1)

    # Supply projected 2D coordinates with a zero third coordinate to GraphBuilder's
    # portable edge algorithms. Stored graph tensors retain only 2D position and 1D LOS velocity.
    edge_positions = torch.cat((positions, torch.zeros((len(positions), 1), dtype=torch.float32)), dim=1)
    builder = GraphBuilder({"seed": 0, "graph": {"method": config.method, "radius_mpc": config.radius_mpc,
        "k_neighbors": config.k_neighbors, "self_loops": config.self_loops,
        "edge_features": ["distance"], "hierarchical": False}})
    count = len(halo.subhalos)
    if config.method == "radius":
        edge_index = builder._build_radius_edges(edge_positions, count)
    else:
        edge_index = builder._build_knn_edges(edge_positions, count)
    if config.self_loops:
        edge_index = builder._add_self_loops(edge_index, count)
    edge_index = builder._connect_isolated_nodes(edge_index, edge_positions, count)
    source, destination = edge_index
    delta_position = positions[destination] - positions[source]
    delta_velocity = velocities[destination] - velocities[source]
    mass_delta = log_mass[source] - log_mass[destination]
    available = {"projected_distance": torch.linalg.vector_norm(delta_position, dim=1, keepdim=True) / config.position_scale_mpc,
                 "los_delta_v": torch.abs(delta_velocity) / config.velocity_scale_kms, "mass_ratio": mass_delta}
    edge_attr = torch.cat([available[name] for name in config.edge_features], dim=1) if config.edge_features else torch.zeros((edge_index.shape[1], 0))

    finite_tensors = {"node features": x, "positions": positions, "velocities": velocities,
                      "edge features": edge_attr}
    for label, tensor in finite_tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"projected {label} are nonfinite after float32 conversion")
    target = torch.tensor([halo.log_halo_mass], dtype=torch.float32)
    if not torch.isfinite(target).all():
        raise ValueError("projected graph target is nonfinite after float32 conversion")

    compatibility = checkpoint_compatibility_record(config, axis)
    stored_report = halo.metadata.get("observation_provenance")
    if not isinstance(stored_report, dict):
        raise ValueError("projected graph requires the observation transform report")
    report_value = observation_report if observation_report is not None else stored_report
    if not isinstance(report_value, dict):
        raise ValueError("observation report is incomplete or inconsistent")
    provenance = deepcopy(report_value)
    expected_identity = {
        "catalog_id": str(halo.metadata.get("catalog_id", "")),
        "halo_id": str(halo.cluster_id),
        "line_of_sight": axis,
        "schema": OBSERVATION_SCHEMA,
        "realization_id": observation.get("realization_id"),
        "seed": observation.get("seed"),
    }
    if any(provenance.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("observation report identity does not match catalog, halo and observer")
    transforms = provenance.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("observation report is incomplete or inconsistent")
    project_steps = [step for step in transforms if isinstance(step, dict) and step.get("name") == "project"]
    seeded_steps = [step for step in transforms if isinstance(step, dict)
                    and step.get("name") in {"missingness", "noise"}]
    if (len(project_steps) != 1 or project_steps[0].get("axis") != axis
            or any(step.get("seed") != provenance.get("seed") for step in seeded_steps)
            or len(seeded_steps) != 2):
        raise ValueError("observation report is incomplete or inconsistent")
    if observation_report is not None and provenance != stored_report:
        raise ValueError("explicit observation report differs from the stored observation report")
    catalog_provenance = deepcopy({key: halo.metadata[key] for key in (
        "catalog_id", "source", "source_kind", "role", "suite", "simulation",
        "initial_condition_id", "snapshot", "catalog_contract", "selection", "release",
        "selection_accounting",
    ) if key in halo.metadata})
    if "initial_condition_id" not in catalog_provenance:
        contract = catalog_provenance.get("catalog_contract", {})
        if isinstance(contract, dict) and "volume_or_ic_group" in contract:
            catalog_provenance["initial_condition_id"] = contract["volume_or_ic_group"]
    initial_condition_id = catalog_provenance.get("initial_condition_id")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target,
                pos=positions, vel=velocities, cluster_id=halo.cluster_id, num_nodes=count,
                feature_mode=PROJECTED_FEATURE_MODE, graph_schema=PROJECTED_GRAPH_SCHEMA,
                schema_identity=compatibility["schema_identity"], observer_axis=axis,
                observation_schema=OBSERVATION_SCHEMA, compatibility_record=compatibility,
                normalization=compatibility["normalization"], observation_provenance=provenance,
                catalog_provenance=catalog_provenance, initial_condition_id=initial_condition_id,
                lineage_group=initial_condition_id)
