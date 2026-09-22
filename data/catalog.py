"""Canonical catalog semantics shared by loaders and research workflows."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Dict, List, Mapping, Optional


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_file_record(path):
    from data.provenance import sha256_file
    return sha256_file(path)


@dataclass(frozen=True)
class CatalogContract:
    """JSON-safe declaration of a normalized halo catalog's semantics."""

    research_label: str
    source: str
    suite: Optional[str]
    simulation: Optional[str]
    snapshot: Any
    volume_or_ic_group: Any
    target_field: str
    target_definition: str
    mass_units: str
    position_units: str
    radius_units: str
    velocity_units: str
    coordinate_frame: Optional[str]
    velocity_convention: Optional[str]
    hubble_param: Optional[float]
    scale_factor: Optional[float]
    radius_source_field: str
    radius_semantic: str
    synthetic: bool
    fallback: bool
    source_files: List[Dict[str, Any]] = field(default_factory=list)
    field_mapping: Dict[str, str] = field(default_factory=dict)
    conversion_record: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CatalogContract":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown catalog contract fields: {sorted(unknown)}")
        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise ValueError(f"Incomplete catalog contract: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self, strict_research: bool = False) -> "CatalogContract":
        if self.schema_version != 1:
            raise ValueError(f"Unsupported catalog schema_version {self.schema_version}")
        for name in ("research_label", "source", "target_field", "target_definition",
                     "mass_units", "position_units", "radius_units", "velocity_units",
                     "radius_source_field", "radius_semantic"):
            if not getattr(self, name):
                raise ValueError(f"Catalog contract requires {name}")
        for info in self.source_files:
            if (not info.get("path") or not isinstance(info.get("size_bytes"), int)
                    or info["size_bytes"] < 0
                    or not _SHA256.fullmatch(str(info.get("sha256", "")))):
                raise ValueError("source_files require path, nonnegative size_bytes and SHA-256")
        if not strict_research:
            return self

        if self.research_label != "corrected_research":
            raise ValueError("strict research requires research_label=corrected_research")
        if self.synthetic or self.fallback:
            raise ValueError("strict research rejects synthetic or fallback catalogs")
        if self.target_field != "Group_M_Crit200":
            raise ValueError("strict research requires target_field=Group_M_Crit200")
        if self.target_definition != "log10(M200c/M_sun)":
            raise ValueError("strict research requires target_definition=log10(M200c/M_sun)")
        if self.radius_semantic != "stellar_half_mass_radius_type4":
            raise ValueError("strict research requires stellar_half_mass_radius_type4")
        if not self.coordinate_frame:
            raise ValueError("strict research requires coordinate_frame")
        if not self.velocity_convention:
            raise ValueError("strict research requires velocity_convention")
        for name in ("hubble_param", "scale_factor"):
            value = getattr(self, name)
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"strict research requires finite positive {name}")
        if not self.source_files:
            raise ValueError("strict research requires hashed source_files")
        if not self.field_mapping:
            raise ValueError("strict research requires an explicit field_mapping")
        return self


def attach_catalog_contract(halos, contract, strict_research: bool = False):
    """Validate and copy a contract onto every HaloData metadata mapping."""
    if not isinstance(contract, CatalogContract):
        contract = CatalogContract.from_mapping(contract)
    contract.validate(strict_research=strict_research)
    payload = contract.to_dict()
    for halo in halos:
        halo.metadata["catalog_contract"] = deepcopy(payload)
    return halos
