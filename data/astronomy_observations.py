"""Observation transforms for strict, partial astronomy catalog records.

The adapter copies only the measured observables into a small mutable carrier
for the shared keyed missingness/noise/projection implementation. It does not
convert partial catalog rows to the legacy complete 3D schema.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from data.observations import ObservationConfig, OBSERVATION_SCHEMA, observe_halo
from data.astronomy_loader import AstronomyHalo, AstronomyMember


@dataclass
class ObservableMember:
    """Mutable transform carrier containing measured observables only."""
    subhalo_id: str
    position: np.ndarray
    velocity: np.ndarray
    stellar_mass: float


@dataclass
class ObservedAstronomyHalo:
    """Graph-ready projected copy while retaining its explicitly partial schema."""
    cluster_id: str
    subhalos: list[ObservableMember]
    log_halo_mass: float
    metadata: dict
    schema: str = "partial_observation_inputs_v1"


def observe_astronomy_halo(halo, config: ObservationConfig):
    """Transform an audited ``AstronomyHalo`` and return ``(copy, report)``.

    Source arrays and metadata are copied before the shared transform runs.
    Richness rejection returns ``None`` with the same complete accounting
    report as :func:`data.observations.observe_halo`.
    """
    if not isinstance(halo, AstronomyHalo) or halo.schema != "partial_observation_inputs_v1":
        raise ValueError("expected partial_observation_inputs_v1 astronomy halo")
    if not halo.catalog_id:
        raise ValueError("astronomy halo requires a catalog ID")
    members = []
    for member in halo.members:
        if not isinstance(member, AstronomyMember):
            raise ValueError("astronomy halo members must be audited AstronomyMember records")
        position = np.asarray(member.position_mpc, dtype=np.float64)
        velocity = np.asarray(member.velocity_km_s, dtype=np.float64)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("audited member position and velocity must have three components")
        members.append(ObservableMember(
            subhalo_id=str(member.member_id), position=position.copy(),
            velocity=velocity.copy(), stellar_mass=float(member.stellar_mass_msun),
        ))
    metadata = deepcopy(halo.metadata)
    metadata["catalog_id"] = str(halo.catalog_id)
    metadata["observation_source_schema"] = "partial_observation_inputs_v1"
    carrier = ObservedAstronomyHalo(
        cluster_id=f"{halo.catalog_id}:group:{halo.group_index}",
        subhalos=members, log_halo_mass=float(halo.target_log10_msun), metadata=metadata,
    )
    observed, report = observe_halo(carrier, config)
    if observed is not None:
        # Bind the transform output to its source schema without suggesting the
        # projected record has complete 3D physical fields.
        observed.schema = "partial_observation_inputs_v1"
        observed.metadata["observation"]["source_schema"] = "partial_observation_inputs_v1"
        observed.metadata["observation"]["schema"] = OBSERVATION_SCHEMA
        observed.metadata["observation_provenance"]["source_schema"] = "partial_observation_inputs_v1"
    report = deepcopy(report)
    report["source_schema"] = "partial_observation_inputs_v1"
    report["catalog_id"] = str(halo.catalog_id)
    report["halo_id"] = carrier.cluster_id
    if observed is not None:
        observed.metadata["observation_provenance"] = deepcopy(report)
    return observed, report
