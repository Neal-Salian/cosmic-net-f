"""Local, single-file native SUBFIND adapter; no downloads or fallback defaults."""
import math
import h5py
import numpy as np

from data.loaders.base_loader import HaloData, SubhaloData


def _self_contained(f, path):
    current = f
    for component in path.split("/"):
        if not isinstance(current.get(component, getlink=True), h5py.HardLink):
            raise ValueError(f"self-contained HDF5 hard links required: {path}")
        current = current[component]
    if isinstance(current, h5py.Dataset) and (current.is_virtual or current.external):
        raise ValueError(f"self-contained HDF5 datasets required: {path}")


def load_catalog(path, *, catalog_id, expected_h, expected_a, min_members=3, member_selection="all_stellar_positive"):
    """Read a complete catalog with raw ckpc/h and 1e10 Msun/h fields.

    Does not accept TNG chunk subsets, CAESAR/AHF/Rockstar, or prepared CSVs.
    Geometry uses minimum-image positions relative to the first stellar member.
    Radius is deliberately the TOTAL radius matching the declared frozen path.
    """
    if member_selection not in {"all_stellar_positive", "native_first10_then_stellar_positive"} or min_members != 3:
        raise ValueError("unsupported member selection/minimum richness")
    with h5py.File(path, "r") as f:
        if "Header" not in f:
            raise ValueError("native Header required")
        _self_contained(f, "Header")
        header = dict(f["Header"].attrs)
        for key, expected in [("HubbleParam", expected_h), ("Time", expected_a)]:
            value = float(header.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0 or not math.isclose(value, expected, rel_tol=1e-6):
                raise ValueError(f"header {key} differs from declared epoch/cosmology")
        h, a = float(header["HubbleParam"]), float(header["Time"])
        box = float(header.get("BoxSize", float("nan")))
        if not math.isfinite(box) or box <= 0:
            raise ValueError("finite positive native BoxSize required for periodic geometry")
        redshift = float(header.get("Redshift", float("nan")))
        if not math.isfinite(redshift) or not math.isclose(a, 1 / (1 + redshift), rel_tol=1e-6):
            raise ValueError("header redshift contradicts scale factor")
        if header.get("NumFiles") != 1:
            raise ValueError("complete single-file catalog required; chunk sets need verified consolidation")
        names = ["Group/Group_M_Crit200", "Subhalo/SubhaloGrNr", "Subhalo/SubhaloPos", "Subhalo/SubhaloVel",
                 "Subhalo/SubhaloMassType", "Subhalo/SubhaloHalfmassRad", "Subhalo/SubhaloVelDisp",
                 "Subhalo/SubhaloStarMetallicity"]
        arrays = {}
        for name in names:
            if name not in f or not isinstance(f[name], h5py.Dataset):
                raise ValueError(f"missing required native field: {name}")
            _self_contained(f, name)
            array = f[name][...]
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"nonfinite or nonnumeric field: {name}")
            arrays[name] = array
    mass, group, pos, vel, stars, radius, dispersion, metals = [arrays[n] for n in names]
    if mass.ndim != 1 or group.ndim != 1:
        raise ValueError("group mass and membership must be rank 1")
    n, ng = len(group), len(mass)
    for key, count in [("Ngroups_ThisFile", ng), ("Ngroups_Total", ng),
                       ("Nsubgroups_ThisFile", n), ("Nsubgroups_Total", n)]:
        if header.get(key) != count:
            raise ValueError(f"complete native catalog count mismatch: {key}")
    if group.dtype.kind not in "iu" or np.any(group < 0) or np.any(group >= ng):
        raise ValueError("integer SubhaloGrNr must reference the complete group table")
    if (pos.shape != (n, 3) or vel.shape != (n, 3) or stars.shape != (n, 6)
            or any(x.shape != (n,) for x in [radius, dispersion, metals])):
        raise ValueError("native feature shape mismatch")
    if np.any(pos < 0) or np.any(pos >= box) or np.any(stars < 0):
        raise ValueError("invalid native periodic positions or negative stellar mass")
    halos, rejected = [], []
    for gid in range(ng):
        rows = np.flatnonzero(group == gid)
        if member_selection == "native_first10_then_stellar_positive":
            rows = rows[:10]
        rows = rows[stars[rows, 4] > 0]
        cid = f"{catalog_id}:group:{gid}"
        if len(rows) < min_members:
            rejected.append({"halo_id": cid, "reason": "fewer_than_3_stellar_members", "member_count": len(rows)})
            continue
        if mass[gid] <= 0 or any(np.any(x[rows] <= 0) for x in [radius, dispersion, metals]):
            raise ValueError(f"nonpositive target/log feature in {cid}; refusing silent floors")
        centered = (pos[rows] - pos[rows[0]] + box / 2) % box - box / 2
        if np.any(np.ptp(centered, axis=0) >= box / 2):
            raise ValueError(f"ambiguous periodic halo extent in {cid}")
        members = [SubhaloData(int(row), centered[j] * a / (1000 * h), vel[row].copy(),
                              float(stars[row, 4]) * 1e10 / h, float(dispersion[row]),
                              float(radius[row]) * a / (1000 * h), float(metals[row]))
                   for j, row in enumerate(rows)]
        halos.append(HaloData(cid, members, float(mass[gid]) * 1e10 / h, redshift,
                              {"catalog_id": catalog_id, "group_id": gid}))
    if not halos:
        raise ValueError("no eligible halos; a catalog with no evaluation population cannot validate transfer")
    return halos, rejected
