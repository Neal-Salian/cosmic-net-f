"""Mathematical action likelihood, physical topology and quality-gate regressions."""
import copy
import itertools
import math
import pytest
import torch
from rls.pair_policy import (pair_mask, pair_scores, ordered_log_probability,
                             pair_stats, policy_action)
from rls.policy import EdgePolicyNet
from rls.rewards import compute_rewards
from rls.notebook_workflow import policy_validation


def complete_edges(n):
    return torch.cartesian_prod(torch.arange(n), torch.arange(n)).T.contiguous()


def test_ordered_likelihood_normalizes_and_has_correct_gradient():
    scores = torch.tensor([.4, -.2, .7], dtype=torch.double, requires_grad=True)
    orders = list(itertools.permutations(range(3), 2))
    logps = torch.stack([ordered_log_probability(scores, torch.tensor(order))[0] for order in orders])
    torch.testing.assert_close(logps.exp().sum(), torch.tensor(1.,dtype=torch.double))
    expected_first = scores[0]-scores.logsumexp(0)
    expected_second = scores[2]-scores[1:].logsumexp(0)
    torch.testing.assert_close(logps[orders.index((0,2))], expected_first+expected_second)
    rewards = torch.tensor([float(2 in order) for order in orders],dtype=torch.double)
    exact = torch.autograd.grad((logps.exp()*rewards).sum(), scores, retain_graph=True)[0]
    estimator = torch.autograd.grad((logps.exp().detach()*rewards*logps).sum(), scores)[0]
    torch.testing.assert_close(exact, estimator)
    assert exact[2] > 0


def test_pair_scores_permutation_duplicates_and_selfloop_gradients():
    ei=complete_edges(4)
    scores=torch.arange(16,dtype=torch.float).requires_grad_()
    mask,logp,_,_=pair_mask(ei,scores,.5,sample=True,repair=False)
    logp.backward()
    assert torch.equal(scores.grad[ei[0]==ei[1]],torch.zeros(4))
    assert mask[ei[0]==ei[1]].all()
    # Scores 0-3 and 1-2 tie at the cutoff; transport explicit physical-pair
    # marks so column permutation couples the exchangeable tie/repair policy.
    marks=torch.tensor([.1,.2,.3,.4,.5,.6])
    deterministic=pair_mask(ei,scores,.5,pair_marks=marks)[0]
    order=torch.randperm(16)
    permuted=pair_mask(ei[:,order],scores[order],.5,pair_marks=marks)[0]
    assert torch.equal(permuted,deterministic[order])
    assert pair_stats(ei,deterministic,4)["physical_isolates"]==0
    # Reverse copies receive exactly the same membership.
    assert torch.equal(mask.reshape(4,4),mask.reshape(4,4).T)


def test_physical_repair_ignores_selfloops_and_reports_overhead():
    ei=complete_edges(4)
    scores=torch.zeros(16)
    mask=pair_mask(ei,scores,.1)[0]
    stats=pair_stats(ei,mask,4)
    assert stats["physical_isolates"]==0
    assert stats["physical_pair_keep"]>=1/6
    assert stats["total_edge_keep"]>stats["physical_pair_keep"]


def test_reward_finite_for_perfect_and_nearly_perfect_backbone():
    cfg=dict(w_acc=1.,w_sp=.5,w_conn=1.,w_virial=1.,reward_error_scale=.1,reward_clip=10.)
    for error in [0.,1e-9,2.86102294921875e-5]:
        reward=compute_rewards(torch.tensor([.1443157]),torch.tensor([error]),torch.zeros(1),.4,.4,torch.tensor(.086353),cfg)
        assert torch.isfinite(reward) and abs(float(reward))<1.
    extreme=compute_rewards(torch.tensor([1e6]),torch.zeros(1),torch.zeros(1),.4,.4,0.,cfg)
    assert extreme== -10.


def test_validation_rejects_worse_than_random_or_isolated_policy():
    random=[dict(rmse=.2),dict(rmse=.22)]
    assert policy_validation(dict(rmse=.3,physical_isolates=0),random)["verdict"]=="FAIL"
    assert policy_validation(dict(rmse=.1,physical_isolates=1),random)["verdict"]=="FAIL"
    assert policy_validation(dict(rmse=.1,physical_isolates=0),random)["verdict"]=="PASS"


def test_pair_training_learns_useful_ranking_and_restores_validation_best():
    from rls.policy_gradient import PolicyGradientTrainer,ValueNet
    from rls.train_policy import train_policy
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(7)
        ei=complete_edges(4);bad=((ei[0]==2)&(ei[1]==3))|((ei[0]==3)&(ei[1]==2))
        graphs=[dict(x=torch.zeros(4,1),edge_index=ei,edge_attr=bad.float()[:,None],
                     emb=torch.zeros(4,2),ctx=torch.zeros(2),y=torch.zeros(1),
                     stellar_mass=torch.ones(4),vel_disp=torch.ones(4),pos=torch.randn(4,3)) for _ in range(8)]
        policy=EdgePolicyNet(1,2,16,normalize=True);value=ValueNet(2,16)
        cfg=dict(sparsity_mode="pair_pl",pair_rollouts=4,batch_size=4,
                 target_sparsity_start=.5,target_sparsity_end=.5,sparsity_anneal_epochs=1,
                 w_acc=1.,w_sp=.5,w_conn=1.,w_virial=0.,entropy_coef=.01)
        trainer=PolicyGradientTrainer(policy,value,torch.optim.Adam(policy.parameters(),lr=.01),torch.optim.Adam(value.parameters(),lr=.01),cfg)
        def predict(g,mask):return g["y"]+.1,g["y"]+(.4 if mask[bad].any() else .05)
        train_policy(trainer,graphs,predict,cfg,epochs=15)
        logits=policy(graphs[0]["edge_attr"],graphs[0]["emb"],ei,graphs[0]["ctx"]).reshape(-1)
        assert logits[bad].mean()<logits[(~bad)&(ei[0]!=ei[1])].mean()
        assert not policy_action(policy,graphs[0],cfg)[0][bad].any()
        assert all(math.isfinite(row["loss"]) for row in trainer.diagnostics)
        # A deliberately worse validation sequence must restore the initial checkpoint.
        initial=copy.deepcopy(policy.state_dict());calls=[]
        def worsening(p):calls.append(1);return {"rmse":float(len(calls))}
        train_policy(trainer,graphs,predict,cfg,epochs=2,validation_fn=worsening)
        assert trainer.selected_epoch is None
        for key,value in initial.items():torch.testing.assert_close(policy.state_dict()[key],value)
    finally:torch.set_num_threads(old_threads)
