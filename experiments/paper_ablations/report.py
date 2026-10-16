"""Render manuscript-facing tables exclusively from completed diagnostic files."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=root/'experiments/paper_ablations/results/local-2026-10-16')
    p.add_argument('--output',type=Path,default=root/'paper/evidence/rl-ablation-results.md')
    args=p.parse_args()
    run=json.loads((args.results/'run.json').read_text())
    if run['status']!='complete': raise ValueError('Only completed diagnostics can be reported')
    summaries=json.loads((args.results/'summary.json').read_text()); by_name={r['arm']:r for r in summaries}
    history=json.loads((args.results/'historical_training_log.json').read_text())
    equivalence=json.loads((args.results/'mask_equivalence.json').read_text())
    contract=run['contract']; replay=by_name['original_0.4_cohort_replay']; full=by_name['full_graph_common_cohort']
    n=full['n_halos']; budgets=contract['budgets']
    lines=['# Executed RL ablation diagnostics', '',
        'Executed locally on 16 October 2026. **No reward-ablation policy was retrained.** All scientific results below use the same recovered 60-epoch policy and frozen backbone. New training configurations are planned experiments only.', '',
        '## Reproduction and scope', '',
        f'The original 40% test-cohort replay produced **{replay["rmse_dex"]:.9f} dex RMSE on {replay["n_halos"]} halos**, reproducing the supplied 0.136160 dex result. This replay is separate from the common-cohort budget study below.', '',
        f'The budget study uses the same {n} TEST halos at every budget and for every decoder. Eligibility is the topology-only intersection over {budgets}; all 82 original TEST IDs remain in the saved manifest. The excluded halos are `tng100_0000053` (minimum 6 pairs, .2 budget 3/15) and `tng100_0000080` (minimum 5 pairs, .2 budget 4/19). No halo was removed using predictions or labels.', '',
        f'The frozen full graph has **{full["rmse_dex"]:.6f} dex RMSE** on these same {n} halos and remains the strongest accuracy reference. Its budget is 100%, so it is a reference rather than an equal-cost pruning comparison. All runs use deterministic frozen predictions, preserved self-loops, the same two-orientation scorer, no Stage B, and no TTA.', '',
        '## Retention and policy behavior', '',
        'Stochastic entries are mean ± sample SD of five separate single-mask inference RMSEs (seeds 7,19,43,71,101). They are not an ensemble and are not confidence intervals or independent training replicates. Marks and categorical draws are both reproducible within each draw seed.', '',
        '| Requested pair retention | Realized mean pair retention | Mean stored-edge retention | Constrained rank RMSE | Stochastic RMSE mean ± SD | Unconstrained rank RMSE | Unconstrained isolate halos | Rank virial penalty |',
        '|---|---|---|---|---|---|---|---|']
    for q in budgets:
        rank=by_name[f'{q:g}_constrained_rank']; raw=by_name[f'{q:g}_unconstrained_rank']
        draws=[r['rmse_dex'] for r in summaries if r['requested_keep']==q and r['mode']=='constrained_sample']
        lines.append(f'| {q:.0%} | {rank["mean_pair_keep"]:.6f} | {rank["mean_total_edge_keep"]:.6f} | {rank["rmse_dex"]:.6f} | {statistics.mean(draws):.6f} ± {statistics.stdev(draws):.6f} | {raw["rmse_dex"]:.6f} | {raw["isolate_halos"]}/{n} | {rank["mean_relative_virial_penalty"]:.6f} |')
    lines.extend(['',
        'RMSE units are dex. Every nominal rank/sample arm uses exactly `ceil(q P)` pairs on each halo. Constraints leave zero physical isolates in every constrained arm. Stored-edge retention includes all preserved self-loops and therefore exceeds physical-pair retention. More retained edges improve accuracy for this fixed policy, while the relative-virial deviation also decreases; the latter does not attribute the accuracy effect to the trained physics reward.', '',
        'At low budgets the exact no-isolate decoder substantially changes the selected topology and observed error. This establishes dependence of this recovered policy on inference constraints in this cohort. It does not establish the effect of training with a connectivity reward. At .2, stochastic decoding has a lower mean RMSE than deterministic decoding in these five draws; at .4 it has a higher mean. These are exploratory observations, with no test-driven choice of deployment seed or budget and no significance claim.', '',
        f'Applying sigmoid to the **already orientation-averaged pair logits** before the same exact greedy decoder produced identical masks for {sum(r["identical_masks"] for r in equivalence)}/{sum(r["n_halos"] for r in equivalence)} halo-budget cases. This checks a monotone ranking equivalence; it supplies no evidence for an additional policy mechanism. Averaging sigmoid outputs before pair/orientation aggregation is a different intervention and was not tested.', '',
        '## Additive repair: nominal budget audit', '',
        'The unconstrained top-k selection is followed by the repository additive no-isolate repair with fixed pair marks. An accuracy comparison at the nominal q is withheld if any halo exceeds that budget. No favorable subset of successfully repaired halos is selected.', '',
        '| Nominal q | Budget-exceeding halos | Mean added pairs/halo | Mean final pair retention | Nominal-budget RMSE |',
        '|---|---|---|---|---|'])
    for q in budgets:
        r=by_name[f'{q:g}_rank_repair']; value='withheld: unequal cost' if r['rmse_dex'] is None else f'{r["rmse_dex"]:.6f}'
        lines.append(f'| {q:.0%} | {r["budget_violation_halos"]}/{n} | {r["mean_repair_added_pairs"]:.4f} | {r["mean_pair_keep"]:.6f} | {value} |')
    lines.extend(['','## Exact comparison at the repaired per-halo cost','',
        'For this separate diagnostic, freeze each halo’s repaired final pair count without consulting its label, then run constraint-aware ranking at exactly that count. Both methods have zero isolates and equal counts per halo. The requested nominal fraction identifies the source of the cost schedule; the schedule is score/mark-dependent and is not the nominal-q protocol.', '',
        '| Source nominal q | Matched mean pair retention | Constraint-aware RMSE | Select-then-repair RMSE |',
        '|---|---|---|---|'])
    for q in budgets:
        a=by_name[f'{q:g}_direct_at_repair_cost']; b=by_name[f'{q:g}_repair_at_repair_cost']
        lines.append(f'| {q:.0%} | {a["mean_pair_keep"]:.6f} | {a["rmse_dex"]:.6f} | {b["rmse_dex"]:.6f} |')
    f=history['final_epoch']
    lines.extend(['',
        'These small cost-matched differences change sign across budgets. They do not support a general superiority claim for either selection mechanism, and neither policy was retrained for its decoder.', '',
        '## Historical reward audit', '',
        f'The preserved training log contains {history["epochs"]} completed epochs. Its final epoch logs normalized prediction improvement {f["accuracy_mean"]:.9f}, weighted virial term {f["virial_mean"]:.9f}, weighted sparsity term {f["sparsity_mean"]:.9f}, target pair retention {f["target_pair_keep"]:.1f}, realized mean pair retention {f["physical_pair_keep_mean"]:.9f}, and mean reward {f["reward_mean"]:.9f}. This is final-epoch logging; the evaluated checkpoint was selected at epoch 42.', '',
        'The virial term was nonzero and repair increased average retention. These observations establish that the corresponding computations were active, not that they improved generalization. The historical log has zero isolates in every epoch and reward extrema within [-10,10] without reaching the clip boundaries; the +1 connectivity contribution was therefore constant over these recorded rollouts and canceled in leave-one-out actor advantages. A separate critic is trained but not used by that actor baseline.', '',
        '## Evidence limits and remaining work', '',
        '- Prediction-only vs physics-aware reward, each trained coefficient removal, trained constraint interventions, specialized budgets, and coefficient sensitivities remain **planned**. Their scripts are runnable; no new trained outcomes are reported.',
        '- These data do not alter the existing finding that RL superiority over gradient×input is statistically unsupported. This campaign did not rerun that saliency comparison.',
        '- One recovered policy, one legacy input dataset, and one simulation volume support method diagnostics. They do not establish physical causality, calibrated mass inference, speedup, or robustness across training seeds/simulations.', '',
        '## Reproducibility and artifact index', '',
        'Run `python -m experiments.paper_ablations.report` to regenerate this document from completed outputs. The executable experiment and Kaggle instructions are in [the README](../../experiments/paper_ablations/README.md); the ranked training matrix is in [the specification](ablation-plan.md).', '',
        '- [Per-arm metrics](../../experiments/paper_ablations/results/local-2026-10-16/summary.json)',
        '- [Per-halo predictions, selected physical-pair indices, and realized budgets](../../experiments/paper_ablations/results/local-2026-10-16/predictions.json)',
        '- [Full provenance, input/source checksums, splits, and completion hashes](../../experiments/paper_ablations/results/local-2026-10-16/run.json)',
        '- [Topology eligibility](../../experiments/paper_ablations/results/local-2026-10-16/eligibility.json)',
        '- [Ranking equivalence](../../experiments/paper_ablations/results/local-2026-10-16/mask_equivalence.json)',
        '- [Historical log audit](../../experiments/paper_ablations/results/local-2026-10-16/historical_training_log.json)', '',
        f'Executed source tree fingerprint: `{contract["source"]["source_sha256"]}`; evaluator HEAD at execution: `{contract["source"]["evaluator_commit"]}`. Exact experiment-source hashes are saved because the harness was run before its final commit. Core RL files remain those of the pinned base. Kaggle packaging records the final committed experiment revision.', '',
        f'Recovered selected-policy SHA256: `{contract["inputs"]["recovered_policy_sha256"]}`. Source archive identity is rechecked from bytes; no source policy file is opened for writing.', '',
        f'Runtime: Python {contract["source"]["python"]}, Torch {contract["source"]["torch"]}, CPU, one thread. CUDA was unavailable locally; no expensive training was attempted. Software fixture tests perform tiny optimizer checks solely for validation.', ''])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__=='__main__': main()
