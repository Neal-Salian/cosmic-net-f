"""Read-only local data/artifact inventory with hashes; no downloads or evaluation."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .contracts import file_record


def inspect_hdf5(path):
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        def safe(value):
            if isinstance(value, bytes):
                return value.decode(errors="replace")
            if isinstance(value, np.ndarray):
                return [safe(x) for x in value.tolist()]
            if isinstance(value, np.generic):
                return safe(value.item())
            return value
        record = {"root_attributes": {k: safe(v) for k, v in f.attrs.items()}, "datasets": {}}
        if "Header" in f:
            record["header"] = {k: safe(v) for k, v in f["Header"].attrs.items()}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                record["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        f.visititems(visit)
        if "Subhalo/SubhaloGrNr" in f and f["Subhalo/SubhaloGrNr"].size:
            groups = f["Subhalo/SubhaloGrNr"][:]
            record["native_group_range"] = [int(groups.min()), int(groups.max())]
        header = record.get("header", {})
        record["declared_single_file"] = header.get("NumFiles") == 1
        record["scientific_usability_verified"] = False
        return record


def inspect_checkpoint(path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise ValueError("checkpoint is not a mapping")
    state = blob.get("model_state_dict", blob.get("policy_state_dict", blob))
    return {"top_level_keys": list(blob), "embedded_provenance": "provenance" in blob,
            "model_config": blob.get("config", {}).get("model"),
            "graph_config": blob.get("config", {}).get("graph"),
            "tensor_shapes": {k: list(v.shape) for k, v in state.items() if isinstance(v, torch.Tensor)}}


def inventory(root):
    root = Path(root).resolve()
    candidates = []
    for folder in ("data/raw", "raw_hdf5"):
        candidates.extend(p for p in (root / folder).rglob("*") if p.suffix in {".csv", ".hdf5", ".h5", ".pt", ".json"})
    candidates.extend(p for folder in ("config", "outputs/checkpoints", "outputs/rls", "kaggle")
                      for p in (root / folder).glob("*") if p.suffix in {".yaml", ".pt"})
    campaign = root / "output/stageb-fix-campaign-20261015"
    for name in ("B", "C"):
        folder = campaign / f"{name}-v1-terminal-001/artifacts/cosmic-net-runs" / f"notebook_{name}"
        candidates.extend(p for p in folder.glob("*") if p.name in {"provenance.json", "policy.pt", "finetuned_gnn.pt"})
    records = []
    for p in sorted(set(candidates)):
        row = {"relative_path": str(p.relative_to(root)), "size_bytes": p.stat().st_size}
        try:
            row.update(file_record(p))
            if p.suffix in {".hdf5", ".h5"}:
                row["hdf5"] = inspect_hdf5(p)
            elif p.suffix == ".csv":
                with p.open() as stream:
                    reader = csv.DictReader(stream)
                    values = list(reader)
                    row["csv"] = {"columns": reader.fieldnames, "rows": len(values)}
                    if values and "group_id" in values[0]:
                        counts = Counter(r["group_id"] for r in values)
                        row["csv"].update(groups=len(counts), groups_ge3=sum(n >= 3 for n in counts.values()))
            elif p.suffix == ".pt":
                row["checkpoint"] = inspect_checkpoint(p)
            elif p.name == "provenance.json":
                blob = json.loads(p.read_text())
                if len(blob) == 1 and next(iter(blob)) in {"A", "B", "C", "D"}:
                    blob = next(iter(blob.values()))
                row["notebook_provenance"] = {k: blob.get(k) for k in ["research_label", "run_mode", "evaluation_protocol", "dataset", "backbone"]}
                row["notebook_provenance"]["split_counts"] = {k: len(v) for k, v in blob.get("split_cluster_ids", {}).items()}
            row["inspection_status"] = "content_inspected"
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            row["inspection_status"] = "unusable_or_unverified"
            row["reason"] = str(exc)
        records.append(row)
    return {"audit_date_local": "2026-10-16", "source_commit": "c825bf57652005c80faff79cf4dfd58e3a9baa14",
            "root": str(root), "search_scope": ["data/raw/**", "raw_hdf5/**", "config/*", "outputs/{checkpoints,rls}/*", "kaggle/*", "output/stageb-fix-campaign-20261015/{B,C}-v1-terminal-001/artifacts/cosmic-net-runs/notebook_{B,C}/*"],
            "scientific_evaluation_executed": False, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inventory(args.root)
    output = Path(args.output)
    if output.exists():
        raise ValueError("inventory output exists; choose a new file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"files_inspected": len(result["records"]), "evaluation_executed": False, "output": str(output)}))


if __name__ == "__main__":
    main()
