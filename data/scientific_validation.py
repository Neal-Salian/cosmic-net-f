"""Non-mutating checks and scoped diagnostics; none certify source authenticity."""
import numpy as np
import pandas as pd


def validate_clustered_identity_targets(frame, *, halo_mass_unit):
    """Reject row/target contradictions before first-row-per-group selection.

    halo_mass_log always means log10(M/Msun); halo_mass's units are explicit.
    This validates internal consistency, not whether M is M200c or another mass.
    """
    if halo_mass_unit not in {"M_sun", "log10_Msun"}:
        raise ValueError("halo_mass unit must be M_sun or log10_Msun")
    if frame.empty:
        raise ValueError("clustered catalog is empty")
    identity_values = {}
    for name in ("subhalo_id", "group_id"):
        if name not in frame:
            raise ValueError(f"clustered catalog requires {name}")
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        if (values < 0).any():
            raise ValueError(f"{name} must be nonnegative")
        if (values != np.floor(values)).any() or (values > 2**53 - 1).any():
            raise ValueError(f"{name} must contain exactly representable integer IDs")
        identity_values[name] = values
    if len(np.unique(identity_values["subhalo_id"])) != len(frame):
        raise ValueError("duplicate subhalo_id, including possible cross-group reuse")
    targets = {}
    for name in ("halo_mass", "halo_mass_log"):
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} target must be finite")
        if name == "halo_mass" and halo_mass_unit == "M_sun":
            if (values <= 0).any():
                raise ValueError("halo_mass target must be positive")
            values = np.log10(values)
        targets[name] = values
        grouped = pd.DataFrame({"group": identity_values["group_id"], "target": values})
        spread = grouped.groupby("group")["target"].agg(["min", "max"])
        if ((spread["max"] - spread["min"]) > 1e-8).any():
            raise ValueError(f"conflicting {name} target within a group")
    if not targets:
        raise ValueError("clustered catalog requires halo_mass or halo_mass_log")
    if len(targets) == 2 and not np.allclose(
            targets["halo_mass"], targets["halo_mass_log"], rtol=0., atol=1e-8):
        raise ValueError("redundant halo_mass and halo_mass_log targets disagree")


def periodic_pair_diagnostics(positions, *, box_size, radius):
    """Compare Euclidean and minimum-image pairs in the SAME declared units.

    Reports diagnostics only; never unwraps positions or modifies graph edges.
    Counts unordered nonself pairs, without nearest-neighbor fallback.
    """
    if not np.isfinite(box_size) or box_size <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("box_size and radius must be finite and positive")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions must be finite N by 3 coordinates")
    u, v = np.triu_indices(len(positions), k=1)
    delta = positions[v] - positions[u]
    wrapped = delta - box_size * np.round(delta / box_size)
    euclidean = np.linalg.norm(delta, axis=1)
    minimum_image = np.linalg.norm(wrapped, axis=1)
    return {"pair_count": len(u),
            "wrapped_pair_count": int(np.any(np.abs(delta - wrapped) > 1e-10, axis=1).sum()),
            "radius_membership_disagreements": int(((euclidean <= radius) != (minimum_image <= radius)).sum())}


def match_reference_group(frame, reference):
    """Match a single independently identified complete raw group to CSV rows.

    The caller must supply the group's verified first global subhalo ID; we
    never infer offsets by summing a potentially incomplete set of files.
    Float tolerance permits the six-significant-digit API export rounding.
    """
    rows = frame[frame["group_id"] == reference["group_id"]]
    ids = rows["subhalo_id"].to_numpy(dtype=np.int64)
    local = ids - int(reference["first_subhalo_id"])
    membership = np.asarray(reference["member_group_ids"])
    if ((local < 0) | (local >= len(membership))).any():
        raise ValueError("subhalo ID outside the raw group's explicit offset range")
    if not np.all(membership[local] == reference["group_id"]):
        raise ValueError("raw/CSV group membership mismatch")
    field_masks = {name: np.isclose(rows[name].to_numpy(dtype=float),
                    np.asarray(values)[local], rtol=1e-5, atol=1e-6)
              for name, values in reference["fields"].items()}
    target_mask = np.isclose(rows["halo_mass"].to_numpy(dtype=float),
                            reference["target_msun"], rtol=1e-7, atol=0.)
    fully_matching = target_mask.copy()
    for mask in field_masks.values():
        fully_matching &= mask
    return {"group_id": int(reference["group_id"]), "matched_rows": len(rows),
            "subhalo_ids": ids.tolist(),
            "fully_matching_subhalo_ids": ids[fully_matching].tolist(),
            "field_matches": {name: int(mask.sum()) for name, mask in field_masks.items()},
            "target_matches": int(target_mask.sum())}
