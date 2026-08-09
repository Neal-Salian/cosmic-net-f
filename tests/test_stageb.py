import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rls.stageb import fine_tune_gnn

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
    fine_tune_gnn(gnn, graphs, masks, epochs=3, lr=1e-3, device="cpu")
    assert not torch.allclose(w0, gnn.fc.weight)
