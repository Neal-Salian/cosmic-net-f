"""Stage B: briefly fine-tune the GNN on policy-pruned graphs.

Why: the frozen backbone saw only dense graphs; feeding it pruned graphs at
test time is a distribution shift. A few epochs on (pruned graph -> target)
pairs with MSE (+ optional virial loss) closes the gap while keeping the
policy frozen. The policy stays frozen; only the GNN updates.

FIX (audit P1-StageB, Sep 2026): 10 fixed epochs at lr=1e-4 REGRESSED
full-graph R2 by ~0.09 (0.9075 frozen -> 0.8159 finetuned) with no alarm.
fine_tune_gnn now tracks BOTH full-graph and pruned-graph val RMSE per epoch,
checkpoints the best epoch that does not regress the full-graph number beyond
full_tol, early-stops after `patience` epochs without improvement, and
restores the best weights. lr/weight_decay/epochs are swept by the caller
(config stageb_lr / stageb_epochs), not hardcoded.
"""
import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


def _predict(gnn, g, mask, device):
    """Single-graph prediction shared by the train and val paths.

    The GNN is called with a single-graph PyG Batch (the real CosmicNetGNN
    signature); bare-`x` stubs (used by unit tests) are auto-detected by
    catching the AttributeError and falling back to `gnn(x)`."""
    m = mask.to(device)
    assert m.dtype == torch.bool, f"expected bool mask, got {m.dtype}"
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
    return pred.float()


def _rmse(preds, targets):
    return float(torch.sqrt(torch.mean((preds - targets) ** 2)).item())


def edge_dropout_masks(graphs, drop_frac, generator=None):
    """Edge-dropout augmentation masks (audit P1-StageB robustness
    alternative): each non-self-loop edge dropped i.i.d. with prob drop_frac;
    self-loops always kept. Lets the backbone train on sparsified inputs
    without depending on any policy — compare against policy-mask and frozen
    variants in the results table."""
    gen = generator or torch.Generator().manual_seed(0)
    masks = []
    for g in graphs:
        ei = g["edge_index"]
        keep = torch.rand(ei.shape[1], generator=gen) >= drop_frac
        keep[ei[0] == ei[1]] = True
        masks.append(keep.bool())
    return masks


def fine_tune_gnn(gnn, graphs, masks, epochs=10, lr=1e-4, device="cpu",
                  use_virial=False, loss_fn=None, weight_decay=5e-5,
                  val_graphs=None, val_masks=None, patience=3, full_tol=0.02,
                  restore_best=True):
    """graphs: list of dicts with x/edge_index/edge_attr/y + physics attrs.
    masks: list of bool tensors (per-edge keep mask) aligned with graphs.

    val_graphs/val_masks: held-out graphs + pruned masks for the early-stop
    guard. When given, each epoch records full-graph AND pruned-graph val
    RMSE; the best checkpoint is the lowest pruned val RMSE among epochs whose
    full val RMSE has not regressed more than `full_tol` (relative) versus
    the pre-finetune full val RMSE. Training stops after `patience` epochs
    without such an improvement and (if restore_best) the best weights are
    restored. Without val data, legacy fixed-epoch behavior is preserved.

    Returns (history, info): history = per-epoch mean train loss (as before);
    info = {"best_epoch", "stopped_early", "full_hist", "pruned_hist",
            "best_full", "best_pruned", "prefinetune_full"} (val keys None
    when no val data was given).
    """
    gnn = gnn.to(device).train()
    opt = torch.optim.AdamW(gnn.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    def _val_rmse(pruned):
        preds, tgts = [], []
        was_training = gnn.training
        gnn.eval()
        with torch.no_grad():
            for g, m in zip(val_graphs, val_masks):
                gd = {k: v.to(device) for k, v in g.items()
                      if isinstance(v, torch.Tensor)}
                use = m.to(device) if pruned else torch.ones(
                    gd["edge_index"].shape[1], dtype=torch.bool, device=device)
                preds.append(_predict(gnn, gd, use, device).view(-1))
                tgts.append(gd["y"].view(-1).float())
        if was_training:
            gnn.train()
        return _rmse(torch.cat(preds), torch.cat(tgts))

    pre_full = _val_rmse(pruned=False) if val_graphs is not None else None
    best = {"pruned": float("inf"), "full": None, "epoch": -1, "state": None}
    full_hist, pruned_hist = [], []
    bad_epochs = 0
    stopped_early = False

    for epoch in range(epochs):
        epoch_losses = []
        for g, mask in zip(graphs, masks):
            g = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
            pred = _predict(gnn, g, mask, device)
            target = g["y"].squeeze(-1).float()
            if pred.numel() != target.numel():
                pred = pred.expand_as(target)
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

        if val_graphs is not None:
            full = _val_rmse(pruned=False)
            pruned = _val_rmse(pruned=True)
            full_hist.append(full)
            pruned_hist.append(pruned)
            ok_full = full <= pre_full * (1.0 + full_tol)
            if ok_full and pruned < best["pruned"]:
                best.update(pruned=pruned, full=full, epoch=epoch,
                            state=copy.deepcopy(gnn.state_dict()))
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                stopped_early = True
                break

    if val_graphs is not None and restore_best and best["state"] is not None:
        gnn.load_state_dict(best["state"])
    info = {"best_epoch": best["epoch"], "stopped_early": stopped_early,
            "full_hist": full_hist, "pruned_hist": pruned_hist,
            "best_full": best["full"], "best_pruned": best["pruned"],
            "prefinetune_full": pre_full}
    return history, info
