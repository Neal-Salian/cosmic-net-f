"""Stage B: briefly fine-tune the GNN on policy-pruned graphs.

Why: the frozen backbone saw only dense graphs; feeding it pruned graphs at
test time is a distribution shift. A few epochs on (pruned graph -> target)
pairs with MSE (+ optional virial loss) closes the gap while keeping the
policy frozen. The policy stays frozen; only the GNN updates."""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


def fine_tune_gnn(gnn, graphs, masks, epochs=10, lr=1e-4, device="cpu",
                  use_virial=False, loss_fn=None):
    """graphs: list of dicts with x/edge_index/edge_attr/y + physics attrs.
    masks: list of bool tensors (per-edge keep mask) aligned with graphs.

    The GNN is called with a single-graph PyG Batch (the real CosmicNetGNN
    signature); bare-`x` stubs (used by unit tests) are auto-detected by
    catching the AttributeError and falling back to `gnn(x)`.
    """
    gnn = gnn.to(device).train()
    opt = torch.optim.AdamW(gnn.parameters(), lr=lr, weight_decay=5e-5)
    history = []
    for epoch in range(epochs):
        epoch_losses = []
        for g, mask in zip(graphs, masks):
            g = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
            m = mask.to(device)
            x = g["x"]
            ei = g["edge_index"][:, m]
            ea = g["edge_attr"][m]
            batch = Batch.from_data_list([Data(x=x, edge_index=ei, edge_attr=ea)])
            try:
                out = gnn(batch)
            except (AttributeError, TypeError):
                out = gnn(x)  # unit-test stub: forward(self, x) -> scalar
            pred = out[0] if isinstance(out, (tuple, list)) else out
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            target = g["y"].squeeze(-1).float()
            loss = F.mse_loss(pred, target)
            if use_virial and loss_fn is not None:
                vloss = loss_fn(pred, g["y"], {"stellar_mass": g["stellar_mass"],
                                               "vel_disp": g["vel_disp"],
                                               "half_mass_r": g["half_mass_r"]})
                loss = loss + 0.1 * vloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())
        history.append(float(torch.tensor(epoch_losses).mean()))
    return history
