import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.stageb import fine_tune_gnn, edge_dropout_masks

def test_finetune_runs_and_updates_weights():
    torch.manual_seed(0)
    class TinyGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)
        def forward(self, x):
            return self.fc(x).mean()
    gnn = TinyGNN()
    graphs = [{"x": torch.randn(6, 4), "edge_index": torch.randint(0, 6, (2, 10)),
               "edge_attr": torch.randn(10, 5), "y": torch.tensor([0.5]),
               "stellar_mass": torch.rand(6) * 1e10,
               "vel_disp": torch.rand(6) * 100,
               "half_mass_r": torch.rand(6) * 0.01} for _ in range(4)]
    masks = [torch.randint(0, 2, (10,)).bool() for _ in range(4)]
    w0 = gnn.fc.weight.clone()
    history, info = fine_tune_gnn(gnn, graphs, masks, epochs=3, lr=1e-3, device="cpu")
    assert not torch.allclose(w0, gnn.fc.weight)
    assert len(history) == 3 and info["prefinetune_full"] is None


def test_stageb_val_guard_restores_best_and_stops_early():
    """Val-guarded Stage B (audit P1-StageB): (a) when val tracks train, the
    restored weights must not regress full-val RMSE beyond tol; (b) when val
    diverges from train (memorization regime), training must stop early
    instead of running all fixed epochs."""
    torch.manual_seed(0)
    class TinyGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)
        def forward(self, x):
            return self.fc(x).mean()
    def _g(seed, y):
        torch.manual_seed(seed)
        n = 6
        return {"x": torch.randn(n, 4), "edge_index": torch.randint(0, n, (2, 10)),
                "edge_attr": torch.randn(10, 5), "y": torch.tensor([y]),
                "stellar_mass": torch.rand(n) * 1e10,
                "vel_disp": torch.rand(n) * 100,
                "half_mass_r": torch.rand(n) * 0.01}
    full = [torch.ones(10, dtype=torch.bool) for _ in range(2)]
    # (a) val ~ train distribution: guard must hold a non-regressing checkpoint
    gnn = TinyGNN()
    graphs = [_g(0, 0.5), _g(1, 0.5)]
    val = [_g(2, 0.5), _g(3, 0.5)]
    history, info = fine_tune_gnn(gnn, graphs, full, epochs=20, lr=1e-3,
                                  device="cpu", val_graphs=val, val_masks=full,
                                  patience=3, full_tol=0.05)
    assert info["prefinetune_full"] is not None
    assert info["best_epoch"] >= 0
    assert info["best_full"] <= info["prefinetune_full"] * 1.05 + 1e-9
    # (b) val opposite to train: must stop before exhausting fixed epochs
    gnn2 = TinyGNN()
    graphs2 = [_g(0, 5.0), _g(1, 5.0)]
    val2 = [_g(2, -5.0), _g(3, -5.0)]
    history2, info2 = fine_tune_gnn(gnn2, graphs2, full, epochs=20, lr=1e-2,
                                    device="cpu", val_graphs=val2,
                                    val_masks=full, patience=2, full_tol=0.02)
    assert info2["stopped_early"] is True
    assert len(history2) < 20


def test_edge_dropout_masks_keep_self_loops():
    torch.manual_seed(0)
    ei = torch.tensor([[0, 1, 1, 2, 0, 1, 2],
                       [1, 0, 2, 1, 0, 1, 2]])
    graphs = [{"edge_index": ei}]
    masks = edge_dropout_masks(graphs, 0.5)
    assert masks[0].dtype == torch.bool
    assert bool(masks[0][ei[0] == ei[1]].all().item())


def test_provenance_guard_labels_backbone():
    """Code-level guard (audit P1-StageB): checksums distinguish states and
    every results row carries the stage label."""
    from rls.provenance import (record_backbone, backbone_checksum,
                                format_backbone_label, require_backbone_label)
    import torch.nn as nn
    m = nn.Linear(4, 1)
    r1 = record_backbone("frozen", m)
    assert len(r1["sha256"]) == 64 and r1["stage"] == "frozen"
    with torch.no_grad():
        m.weight += 1.0
    r2 = record_backbone("stageB_finetuned", m)
    assert r1["sha256"] != r2["sha256"]           # states distinguishable
    rows = require_backbone_label([{"method": "full", "r2": 0.9}], r1)
    assert rows[0]["backbone_stage"] == "frozen"
    assert rows[0]["backbone"] == format_backbone_label(r1)
    assert backbone_checksum(m.state_dict()) == r2["sha256"]
