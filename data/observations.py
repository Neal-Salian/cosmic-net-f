"""Pure, deterministic projected-observation transforms for HaloData.

This module only creates transformed copies and provenance. It does not build
model tensors or graphs. Projected position is centered on the selected-member
sky centroid; velocity retains only the declared LOS component. Physical input
units are required so noise scales have unambiguous meaning.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from typing import Mapping, Optional, Tuple

import numpy as np


OBSERVATION_SCHEMA = "projected_observation_v1"
OBSERVABLE_FIELDS = ("projected_position", "los_velocity", "stellar_mass_proxy")
_AXES = {"x": 0, "y": 1, "z": 2}
_ALLOWED_NOISE = {"projected_position_mpc", "los_velocity_kms", "stellar_mass_dex"}


@dataclass(frozen=True)
class ObservationConfig:
    """Physical observation settings. Noise is Gaussian; mass noise is in dex."""
    line_of_sight: str = "z"
    position_units: str = "Mpc"
    velocity_units: str = "km/s"
    stellar_mass_units: str = "M_sun"
    seed: int = 0
    realization_id: str = "default"
    missing_fraction: float = 0.0
    position_noise_mpc: float = 0.0
    los_velocity_noise_kms: float = 0.0
    stellar_mass_noise_dex: float = 0.0
    min_richness: int = 1
    max_richness: Optional[int] = None
    noise: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.line_of_sight not in _AXES:
            raise ValueError("line_of_sight must be x, y or z")
        required = {"position_units": "Mpc", "velocity_units": "km/s",
                    "stellar_mass_units": "M_sun"}
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be declared as {expected}")
        if not 0.0 <= self.missing_fraction <= 1.0:
            raise ValueError("missing_fraction must be between 0 and 1")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or not self.realization_id:
            raise ValueError("seed must be an integer and realization_id nonempty")
        if (not isinstance(self.min_richness, Integral) or isinstance(self.min_richness, bool)
                or self.min_richness < 1):
            raise ValueError("min_richness must be an integer of at least 1")
        if (self.max_richness is not None
                and (not isinstance(self.max_richness, Integral) or isinstance(self.max_richness, bool))):
            raise ValueError("max_richness must be an integer or None")
        if self.max_richness is not None and self.max_richness < self.min_richness:
            raise ValueError("invalid inclusive richness bounds")
        values = {"projected_position_mpc": self.position_noise_mpc,
                  "los_velocity_kms": self.los_velocity_noise_kms,
                  "stellar_mass_dex": self.stellar_mass_noise_dex, **dict(self.noise)}
        unknown = set(values) - _ALLOWED_NOISE
        if unknown:
            raise ValueError(f"noise fields must be in the positive allowed observable set: {sorted(unknown)}")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in values.values()):
            raise ValueError("noise scales must be finite and nonnegative")


def _validate_members(halo):
    if not str(halo.cluster_id).strip():
        raise ValueError("missing stable halo ID")
    if not str(halo.metadata.get("catalog_id", "")).strip():
        raise ValueError("missing stable catalog ID in halo metadata")
    ids = []
    for member in halo.subhalos:
        member_id = getattr(member, "subhalo_id", None)
        if member_id is None or str(member_id).strip() == "":
            raise ValueError("missing stable subhalo ID")
        ids.append(str(member_id))
        pos, vel = np.asarray(member.position), np.asarray(member.velocity)
        if pos.shape != (3,) or vel.shape != (3,):
            raise ValueError(f"subhalo {member_id} position and velocity must have 3 components")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise ValueError(f"subhalo {member_id} has nonfinite position or velocity")
        if not math.isfinite(float(member.stellar_mass)) or float(member.stellar_mass) <= 0:
            raise ValueError(f"subhalo {member_id} stellar_mass must be positive and finite")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stable subhalo IDs")


def _rng_for(halo, member, config, operation):
    # Stable identity tuple makes random draws invariant to Python/list ordering.
    identity = [str(halo.metadata.get("catalog_id", "")), str(halo.cluster_id),
                str(member.subhalo_id), str(config.realization_id), int(config.seed), operation]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def observe_halo(halo, config: ObservationConfig) -> Tuple[object, dict]:
    """Return (copied transformed HaloData or None, provenance report).

    The halo target remains label metadata on the copy; it is never consulted
    by selection, centering, random draws or normalizers. None means the
    transformed member count failed the inclusive richness selection.
    """
    if not isinstance(config, ObservationConfig):
        raise TypeError("config must be ObservationConfig")
    _validate_members(halo)
    axis = _AXES[config.line_of_sight]
    original = list(halo.subhalos)
    count_before = len(original)
    # Stable-ID keyed missingness is applied before centering, so the center is
    # based only on members retained in this observed realization.
    retained = []
    rejected_missing = []
    for member in original:
        if _rng_for(halo, member, config, "missingness").random() < config.missing_fraction:
            rejected_missing.append(str(member.subhalo_id))
        else:
            retained.append(deepcopy(member))
    sky_axes = [i for i in range(3) if i != axis]
    # Stable summation order keeps floating-point centering invariant to the
    # caller's member-list ordering while the returned list retains that order.
    center_members = sorted(retained, key=lambda member: str(member.subhalo_id))
    coords = np.asarray([m.position for m in center_members], dtype=float) if retained else np.empty((0, 3))
    with np.errstate(over="ignore", invalid="ignore"):
        center = coords[:, sky_axes].astype(np.longdouble).mean(axis=0).astype(float) if len(coords) else np.zeros(2)
    for member in retained:
        pos = np.zeros(3, dtype=float)
        pos[sky_axes] = np.asarray(member.position, dtype=float)[sky_axes] - center
        vel = np.zeros(3, dtype=float)
        vel[axis] = float(member.velocity[axis])
        member.position = pos
        member.velocity = vel
        # These intrinsic simulation fields are not part of this observation mode.
        # Legacy complete-schema members carry these fields, but partial
        # astronomy observation records deliberately do not. Never invent
        # unavailable intrinsic measurements on those records.
        for name in ("velocity_dispersion", "half_mass_radius", "metallicity"):
            if hasattr(member, name):
                setattr(member, name, 0.0)
    noise = {"projected_position_mpc": config.position_noise_mpc,
             "los_velocity_kms": config.los_velocity_noise_kms,
             "stellar_mass_dex": config.stellar_mass_noise_dex, **dict(config.noise)}
    with np.errstate(over="ignore", invalid="ignore"):
        for member in retained:
            rng = _rng_for(halo, member, config, "noise")
            if noise["projected_position_mpc"]:
                # Only sky components receive position noise.
                member.position[sky_axes] += rng.normal(0, noise["projected_position_mpc"], 2)
            if noise["los_velocity_kms"]:
                member.velocity[axis] += rng.normal(0, noise["los_velocity_kms"])
            if noise["stellar_mass_dex"]:
                try:
                    member.stellar_mass *= 10 ** rng.normal(0, noise["stellar_mass_dex"])
                except OverflowError as exc:
                    raise ValueError(f"noise produced nonfinite stellar mass for subhalo {member.subhalo_id}") from exc
            if (not np.all(np.isfinite(member.position)) or not np.all(np.isfinite(member.velocity))
                    or member.stellar_mass <= 0 or not math.isfinite(float(member.stellar_mass))):
                raise ValueError(f"noise produced nonfinite or invalid observation for subhalo {member.subhalo_id}")
    count_after = len(retained)
    rejection = None
    if count_after < config.min_richness:
        rejection = "below_min_richness"
    elif config.max_richness is not None and count_after > config.max_richness:
        rejection = "above_max_richness"
    result = None
    if rejection is None:
        result = deepcopy(halo)
        result.subhalos = retained
        result.metadata = deepcopy(halo.metadata)
        result.metadata["observation"] = {
            "schema": OBSERVATION_SCHEMA,
            "observable_fields": list(OBSERVABLE_FIELDS),
            "line_of_sight": config.line_of_sight,
            "realization_id": config.realization_id,
            "seed": int(config.seed),
            "position_units": config.position_units,
            "velocity_units": config.velocity_units,
            "stellar_mass_units": config.stellar_mass_units,
            "centering": "selected_member_sky_centroid",
            "intrinsic_fields_excluded": ["velocity_dispersion", "half_mass_radius", "metallicity", "hidden_position", "transverse_velocity"],
        }
    report = {
        "schema": OBSERVATION_SCHEMA,
        "catalog_id": str(halo.metadata.get("catalog_id", "")),
        "halo_id": str(halo.cluster_id),
        "realization_id": config.realization_id,
        "line_of_sight": config.line_of_sight,
        "seed": int(config.seed),
        "pre_richness": count_before,
        "post_missingness_richness": count_after,
        "post_richness": count_after,
        "rejections": {"missingness": len(rejected_missing), "min_richness": int(rejection == "below_min_richness"),
                       "max_richness": int(rejection == "above_max_richness")},
        "rejected_member_ids": {"missingness": sorted(rejected_missing)},
        "rejection_reason": rejection,
        "transforms": [
            {"name": "missingness", "fraction": float(config.missing_fraction), "seed": int(config.seed), "key": "catalog/halo/member/realization"},
            {"name": "project", "axis": config.line_of_sight, "center": "selected_member_sky_centroid", "units": {"position": config.position_units, "velocity": config.velocity_units}},
            {"name": "noise", "scales": noise, "seed": int(config.seed), "key": "catalog/halo/member/realization"},
        {"name": "richness_selection", "min_inclusive": config.min_richness, "max_inclusive": config.max_richness},
        ],
    }
    if result is not None:
        result.metadata["observation_provenance"] = deepcopy(report)
    return result, report
