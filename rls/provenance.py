"""Backbone provenance guard (FIX Sep 2026, audit P1-StageB).

WHAT WAS WRONG: Notebook B fine-tuned the backbone (Stage B) and reported
full-graph R2=0.8159, while Notebook C skipped Stage B and reported R2=0.9075
on the same checkpoint — a ~0.09 silent discrepancy from an unlabeled
frozen-vs-finetuned swap. A notebook flag is not enough; any code path that
prints a "full graph" metric must record WHICH backbone state produced it.

WHAT THIS DOES: sha256 checksum over a model's state_dict + a stage label
('frozen' | 'stageB_finetuned' | ...). Callers attach the record to every
results row / provenance.json so the B-vs-C discrepancy cannot reproduce
silently again.
"""
import hashlib
import torch


def backbone_checksum(state_dict):
    """Stable sha256 over name + shape + bytes of every tensor in order."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        if isinstance(v, torch.Tensor):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(str(v).encode("utf-8"))
    return h.hexdigest()


def record_backbone(stage, model):
    """Return a JSON-serializable provenance record for a backbone state."""
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    return {"stage": stage, "sha256": backbone_checksum(sd)}


def format_backbone_label(record):
    """Short human-readable label, e.g. 'frozen@a1b2c3d4'."""
    return f"{record['stage']}@{record['sha256'][:8]}"


def require_backbone_label(rows_or_df, record):
    """Attach backbone provenance to results rows (list of dicts) in place.

    Every row that reports a 'full graph' number must carry 'backbone_stage'
    and 'backbone_sha256' — the code-level guard against unlabeled swaps.
    """
    label = format_backbone_label(record)
    if isinstance(rows_or_df, list):
        for r in rows_or_df:
            r.setdefault("backbone_stage", record["stage"])
            r.setdefault("backbone_sha256", record["sha256"])
            r.setdefault("backbone", label)
        return rows_or_df
    for col, val in (("backbone_stage", record["stage"]),
                     ("backbone_sha256", record["sha256"]),
                     ("backbone", label)):
        if col not in rows_or_df.columns:
            rows_or_df[col] = val
    return rows_or_df
