"""Scores-only controls and an equivalent lazy exact no-isolates decoder.

No target is read by a scorer. New selectors are controls of existing score
families, not proposed methods. CPU decoding deliberately uses the production
feasibility predicate; it skips evaluating candidates that cannot win.
"""
import hashlib
import torch
from scipy.stats import rankdata
from rls.constraints import feasible_budget, pair_marks, selection_diagnostics
from rls.benchmark_predictor import MeteredPredictor
from rls.benchmark_scorers import _mean_over_real_copies


def halo_seed(seed, halo_id, role):
    return int(hashlib.sha256(f'{role}:{seed}:{halo_id}'.encode()).hexdigest()[:15],16)


def marks_for(layout, seed, halo_id):
    return torch.randperm(len(layout.pairs),generator=torch.Generator().manual_seed(
        halo_seed(seed,halo_id,'tie_marks')))


def decode(layout, scores, k, marks, *, sampling_seed=None):
    """Exact direct rank/PL selection, retaining the reference feasibility rule.

    Rank mode visits candidates by score, then descending exchangeable mark.
    PL mode uses independent Gumbel keys at EACH action; rejecting infeasible
    candidates implements categorical softmax restricted to feasible actions.
    This is ordered-action PL, not a uniform distribution on feasible masks.
    """
    scores=torch.as_tensor(scores).detach().cpu()
    if scores.shape!=(len(layout.pairs),) or not torch.isfinite(scores).all():
        raise ValueError('scores must be finite with one value per pair')
    marks=pair_marks(layout,marks).cpu()
    if not feasible_budget(layout,k,'no_isolates')['feasible']:
        raise ValueError('infeasible no-isolates physical-pair budget')
    selected=torch.zeros(len(scores),dtype=torch.bool)
    generator=None if sampling_seed is None else torch.Generator().manual_seed(sampling_seed)
    base=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),-float(marks[i])))
    for step in range(k):
        candidates=[i for i in base if not selected[i]]
        if generator is not None:
            u=torch.rand(len(scores),dtype=torch.float64,generator=generator).clamp_(1e-15,1-1e-15)
            keys=scores.double()-torch.log(-torch.log(u))
            candidates.sort(key=lambda i:(-float(keys[i]),-float(marks[i])))
        for i in candidates:
            selected[i]=True
            # Once all nodes have degree, every remaining fill is feasible.
            degree=torch.bincount(layout.pairs[selected].flatten(),minlength=layout.num_nodes)
            uncovered=int((degree==0).sum())
            slots=k-step-1
            if slots>=uncovered or feasible_budget(layout,k,'no_isolates',selected)['feasible']:
                break
            selected[i]=False
        else:
            raise RuntimeError('no feasible completion after successful preflight')
    diag=selection_diagnostics(layout,selected,k=k,constraint='no_isolates')
    if not diag['constraint_satisfied'] or not diag['budget_satisfied']:
        raise RuntimeError(f'decoder contract violated: {diag}')
    return selected


def rank_fraction(scores):
    """Average ranks preserve exact ties instead of injecting pair-ID preference."""
    values=torch.as_tensor(scores).detach().cpu().double()
    if not torch.isfinite(values).all():raise ValueError('nonfinite rank input')
    if len(values)<2:return torch.zeros_like(values)
    return torch.tensor((rankdata(values.numpy(),method='average')-1)/(len(values)-1))


def alternative_saliency(model,graph,layout,train_feature_std,*,meter=None):
    meter=MeteredPredictor(model) if meter is None else meter
    result=meter.call(graph,phase='scoring',grad_edge_attr=True)
    leaf=result.edge_attr
    grads=(torch.autograd.grad(result.prediction,leaf,allow_unused=True)[0]
           if result.prediction.requires_grad else None)
    if grads is None:grads=torch.zeros_like(leaf)
    grad=grads.detach(); value=leaf.detach()
    return {
        'signed_grad_x_input':_mean_over_real_copies((grad*value).sum(1).abs(),layout),
        'abs_gradient_scaled':_mean_over_real_copies(
            (grad.abs()*train_feature_std.to(grad)).sum(1),layout),
    }


def heuristic_scores(graph,layout,velocity_scale):
    pairs=layout.pairs; a,b=pairs.T
    displacement=graph.pos[a]-graph.pos[b]
    distance=displacement.norm(dim=1).clamp_min(1e-6)  # physical Mpc
    projected=displacement[:,:2].norm(dim=1)
    dv=(graph.vel[a]-graph.vel[b]).norm(dim=1)  # km/s
    # G cancels from ranking; this is a stellar proxy, not certified binding.
    binding=(graph.stellar_mass[a]+graph.stellar_mass[b]).double().log()-distance.double().log()
    binding-=torch.log(dv.double().square()+velocity_scale**2)
    return dict(near3d=rank_fraction(-distance),near_projected=rank_fraction(-projected),
                velocity_binding_proxy=rank_fraction(binding))
