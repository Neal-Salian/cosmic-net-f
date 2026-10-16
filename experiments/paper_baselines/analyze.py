"""Matched halo-level statistics; no seed-as-halo pseudoreplication."""
import numpy as np


def choose_validation(rows,candidates):
    if not rows or any(r['split']!='val' for r in rows):
        raise ValueError('hyperparameters require validation records only')
    populations=[sorted(r['graph_id'] for r in rows if r['method']==m) for m in candidates
                 if all('graph_id' in r for r in rows)]
    if populations and any(p!=populations[0] for p in populations):
        raise ValueError('validation candidates have unequal populations')
    return min(candidates,key=lambda m:np.mean([r['squared_error'] for r in rows if r['method']==m]))


def paired_bootstrap(squared_a,squared_b,seed,resamples):
    a=np.asarray(squared_a,dtype=float);b=np.asarray(squared_b,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired finite per-halo squared errors required')
    if (a<0).any() or (b<0).any():raise ValueError('negative squared error')
    rng=np.random.default_rng(seed);index=rng.integers(0,len(a),size=(resamples,len(a)))
    delta=np.sqrt(a[index].mean(1))-np.sqrt(b[index].mean(1))
    lo,hi=np.quantile(delta,[.025,.975])
    return dict(delta_rmse=float(np.sqrt(a.mean())-np.sqrt(b.mean())),ci_low=float(lo),ci_high=float(hi),
                halos=len(a),bootstrap_seed=seed,bootstrap_resamples=resamples,
                interval='paired halo percentile 95%; exploratory, no multiplicity adjustment')


def halo_losses(frame):
    """Mean squared error per halo over masks, never mean predictions."""
    return frame.groupby('graph_id',sort=True)['squared_error'].mean()


def validate_population(frame,ids):
    if set(frame.graph_id)!=set(ids):raise ValueError('unequal halo population')
    if frame.duplicated(['graph_id','replicate_seed']).any():raise ValueError('duplicate halo/seed')
    populations=frame.groupby('replicate_seed',dropna=False).graph_id.apply(set)
    if any(p!=set(ids) for p in populations):raise ValueError('unequal seed populations')


def summarize(result_dir,evidence_dir):
    import json
    from pathlib import Path
    import pandas as pd
    from .run import dump,write_csv
    from data.provenance import sha256_file
    result_dir=Path(result_dir);evidence_dir=Path(evidence_dir);evidence_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((result_dir/'manifest.json').read_text())
    if manifest['status']!='executed':raise ValueError('cannot analyze an unexecuted experiment')
    for name,record in manifest['outputs'].items():
        if sha256_file(result_dir/name)['sha256']!=record['sha256']:raise ValueError(f'changed run output: {name}')
    cfg=json.loads((result_dir/'config.json').read_text())
    split=json.loads((result_dir/'split_ids.json').read_text());ids=sorted(split['test'])
    audit=json.loads((result_dir/'audit.json').read_text())
    hyper=json.loads((result_dir/'hyperparameters.json').read_text())
    old=pd.read_csv(result_dir/'historical_replay.csv');new=pd.read_csv(result_dir/'new_predictions.csv')
    props=pd.read_csv(result_dir/'halo_properties.csv')
    validation=pd.read_csv(result_dir/'validation_predictions.csv')
    old['replicate_seed']=42;old['seed_family']='historical_one_mask_or_rank'
    data=pd.concat([old,new],ignore_index=True)
    tables=[];seeds=[];paired=[];halo_differences=[];strata=[];cutpoints={}
    for field in cfg['strata']:
        cutpoints[field]=np.quantile(props.loc[props.split=='val',field],[1/3,2/3]).tolist()
    testprops=props[props.split=='test'].set_index('graph_id')
    for (q,method),group in data.groupby(['retention','method']):
        validate_population(group,ids)
        loss=halo_losses(group).reindex(ids)
        seed_rmse=group.groupby('replicate_seed',dropna=False).squared_error.mean().pow(.5)
        # Mean absolute residual and bias are single-draw expectations as well.
        tables.append(dict(retention=float(q),method=method,halos=len(loss),
            mask_or_sampling_replicates=len(seed_rmse),rmse=float(np.sqrt(loss.mean())),
            mae=float(group.absolute_error.mean()),bias=float(group.residual.mean()),
            seed_rmse_mean=float(seed_rmse.mean()),seed_rmse_std=float(seed_rmse.std(ddof=1)) if len(seed_rmse)>1 else None,
            seed_rmse_min=float(seed_rmse.min()),seed_rmse_max=float(seed_rmse.max()),
            mean_physical_retention=float((group.retained_pairs/group.available_pairs).mean()),
            comparison='unpruned_reference' if method=='full' else 'matched_physical_pair_budget'))
        for seed,seed_group in group.groupby('replicate_seed',dropna=False):
            seeds.append(dict(retention=float(q),method=method,replicate_seed=None if pd.isna(seed) else int(seed),
                seed_family=seed_group.seed_family.iloc[0],halos=len(seed_group),
                rmse=float(np.sqrt(seed_group.squared_error.mean()))))
        rl=halo_losses(data[(data.retention==q)&(data.method=='provided_rl')]).reindex(ids)
        delta=paired_bootstrap(rl.values,loss.values,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
        paired.append(dict(retention=float(q),competitor=method,**delta,
            rl_lower_squared_error_halos=int((rl<loss-1e-12).sum()),
            rl_higher_squared_error_halos=int((rl>loss+1e-12).sum()),
            ties=int((abs(rl-loss)<=1e-12).sum())))
        for cid in ids:
            halo_differences.append(dict(graph_id=cid,retention=float(q),competitor=method,
                rl_squared_error=float(rl[cid]),competitor_expected_squared_error=float(loss[cid]),
                delta_squared_error=float(rl[cid]-loss[cid])))
        for field,cuts in cutpoints.items():
            labels=np.searchsorted(cuts,testprops.loc[ids,field].to_numpy(),side='right')
            for number,label in enumerate(['low','middle','high']):
                selected=[cid for cid,bin_id in zip(ids,labels) if bin_id==number]
                row=dict(retention=float(q),competitor=method,property=field,stratum=label,
                    cutpoint_low=cuts[0],cutpoint_high=cuts[1],halos=len(selected),
                    definition='validation tertiles; tied boundary values placed in higher bin')
                if selected:
                    left=rl.loc[selected];right=loss.loc[selected]
                    row.update(rl_rmse=float(np.sqrt(left.mean())),competitor_rmse=float(np.sqrt(right.mean())),
                        delta_rmse=float(np.sqrt(left.mean())-np.sqrt(right.mean())),
                        rl_lower_squared_error_halos=int((left<right-1e-12).sum()),
                        rl_higher_squared_error_halos=int((left>right+1e-12).sum()),
                        ties=int((abs(left-right)<=1e-12).sum()))
                strata.append(row)
    write_csv(result_dir/'matched_budget_table.csv',tables)
    write_csv(result_dir/'seed_metrics.csv',seeds)
    write_csv(result_dir/'paired_comparisons.csv',paired)
    write_csv(result_dir/'per_halo_comparisons.csv',halo_differences)
    write_csv(result_dir/'stratified_comparisons.csv',strata)
    dump(result_dir/'strata_cutpoints.json',cutpoints)
    tuning=[]
    for method,group in validation.groupby('method'):
        if set(group.graph_id)!=set(split['val']):raise ValueError('incomplete validation candidate')
        tuning.append(dict(method=method,halos=len(group),rmse=float(np.sqrt(group.squared_error.mean()))))
    write_csv(result_dir/'validation_tuning_table.csv',tuning)
    # Probability rankings with identical masks are a numerical control, not a new method.
    probability=pd.read_csv(result_dir/'probability_ranking_checks.csv')
    labels={'full':'Full graph','provided_rl':'RL (deterministic)','edge_feature_grad_x_input':'Gradient $\\times$ input',
        'random':'Random (original mask)','distance':'Shortest distance','degree':'Degree',
        'stellar_binding_proxy':'Stellar binding proxy','supervised_raw':'Learned saliency selector',
        'geometric_tuned':'Geometry (validation tuned)','velocity_binding_proxy':'Velocity binding proxy',
        'hybrid_tuned':'Hybrid (validation tuned)','signed_grad_x_input':'Signed gradient $\\times$ input',
        'abs_gradient_scaled':'Scaled absolute gradient','random_repeated':'Random (10 mask seeds)',
        'rl_sampled':'RL (10 sampling seeds)','rl_tie_marks':'RL (3 tie mark seeds)'}
    order=list(labels)
    hybrid_is_geometry=hyper['hybrid_alpha']==1.0
    headline=[m for m in order if m!='rl_tie_marks' and not (m=='hybrid_tuned' and hybrid_is_geometry)]
    table=pd.DataFrame(tables);pairs=pd.DataFrame(paired)
    def metric(method,q=.4):return float(table[(table.method==method)&(table.retention==q)].rmse.iloc[0])
    lines=['% Generated by experiments.paper_baselines.analyze; requires booktabs.','\\begin{table*}[t]',
        '\\centering','\\caption{Frozen-backbone matched-pair comparison on all 82 test halos. RMSE in log$_{10}$ mass units. '
        'Budgets are $k=\\lceil qP\\rceil$ nonself physical pairs, with loops retained and no physical isolates. '
        'Full graph is an unpruned reference. Repeated rows report $\\sqrt{\\mathrm{mean}_{h,s} e_{hs}^2}$, '
        'the expected single-draw squared error, not prediction ensembling. Only one trained policy seed (42) is available.}',
        '\\label{tab:baseline-robustness}','\\begin{tabular}{lrrrr}','\\toprule',
        'Selector & $q=0.4$ & $q=0.6$ & $q=0.8$ & $q=1.0$ \\\\','\\midrule']
    for method in headline:
        lines.append(labels[method]+' & '+' & '.join(f'{metric(method,q):.6f}' for q in cfg['retentions'])+' \\\\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table*}']
    (evidence_dir/'baseline-comparison.tex').write_text('\n'.join(lines)+'\n')
    # Manuscript figure reports every substantive selector, with the numerical controls in CSV.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True,constrained_layout=True)
    groups=[['full','provided_rl','edge_feature_grad_x_input','random','distance','degree','stellar_binding_proxy','supervised_raw'],
            ['full','provided_rl','geometric_tuned','velocity_binding_proxy','signed_grad_x_input',
             'abs_gradient_scaled','random_repeated','rl_sampled']]
    for ax,methods,title in zip(axes,groups,['Existing selectors (CPU replay)','Additional controls (CPU)']):
        for method in methods:
            own=table[table.method==method].sort_values('retention')
            name=labels[method].replace('$\\times$','×')
            ax.plot(own.retention,own.rmse,marker='o',label=name,linewidth=1.2,markersize=3)
        ax.set(xlabel='Requested physical-pair retention',title=title,xticks=cfg['retentions'])
        ax.grid(alpha=.2);ax.legend(fontsize=6.5,loc='upper right')
    axes[0].set_ylabel('Test RMSE (log10 mass)')
    fig.savefig(evidence_dir/'baseline-sensitivity.png',dpi=180);plt.close(fig)
    q40=pairs[pairs.retention==.4].set_index('competitor')
    report=['# Baseline robustness results','',
        'Executed locally on CPU, float32, one PyTorch thread, using the existing frozen backbone and all 82 test halos. '
        'Historical GPU predictions were replayed within an absolute tolerance of $10^{-4}$; this table uses CPU replay and new CPU predictions. '
        'No policy, predictor, or learned selector was retrained.','',
        '## Matched-budget results','',
        '| Selector | 40% | 60% | 80% | 100% |','|---|---:|---:|---:|---:|']
    for method in headline:
        report.append('| '+labels[method].replace('$\\times$','×')+' | '+' | '.join(f'{metric(method,q):.6f}' for q in cfg['retentions'])+' |')
    gradient=q40.loc['edge_feature_grad_x_input']
    report+=['',
        f"At 40%, gradient × input is the strongest existing pruning competitor to RL: {metric('edge_feature_grad_x_input'):.6f} versus RL {metric('provided_rl'):.6f}. "
        f"The RL minus gradient RMSE difference is {gradient.delta_rmse:+.6f}, with paired 95% interval [{gradient.ci_low:+.6f}, {gradient.ci_high:+.6f}]; "
        'this does not resolve an accuracy advantage. The signed-gradient control is close but does not improve the aggregate result.',
        f"Ten random-mask seeds give expected-single-mask RMSE {metric('random_repeated'):.6f}, below the original one-mask {metric('random'):.6f}. "
        f"At 60%, repeated random ({metric('random_repeated',.6):.6f}) is better in aggregate than RL ({metric('provided_rl',.6):.6f}); "
        f"at 80%, shortest-distance ({metric('distance',.8):.6f}) and repeated random ({metric('random_repeated',.8):.6f}) are below RL ({metric('provided_rl',.8):.6f}). "
        'These paired intervals include zero. RL is not uniformly best across budgets. All selectors coincide at full retention.']
    report+=['','Fractions specify requested retention. Actual physical retention is ceil(qP)/P and is identical across selectors per halo. '
        'Full graph retains every pair and is a reference, not a matched pruning competitor. The CSV also gives MAE, bias and achieved retention.','',
        '## Additional controls and validation decisions','',
        f"Geometry mixes average ranks of negative 3D and projected XY separation. Validation-only q=0.4 tuning selected alpha={hyper['geometric_alpha']:g} "
        '(alpha=1 means 3D). The three alpha candidates were 0, 0.5, and 1; every validation halo and losing candidate is exported.',
        f"The physics control ranks log((stellar_mass_i+stellar_mass_j)/r) - log(delta_v²+v0²), with v0={hyper['velocity_scale_km_s']:.6f} km/s "
        'fixed from train-only pair velocities. It is a stellar/velocity proxy, not a measurement of dark-matter binding energy.',
        f"The hybrid mixes geometric and velocity-binding percentile ranks; validation selected geometric weight {hyper['hybrid_alpha']:g}. "
        'Both chosen weights were frozen before test evaluation and reused at every retention. '
        + ('The selected hybrid is exactly geometry (weight 1); its executions remain in the CSV but it is not counted as a separate headline competitor.'
         if hybrid_is_geometry else 'The selected hybrid remains a distinct headline control.'),
        'Alternative saliency uses (a) abs(sum_f grad_f × input_f), allowing signed feature cancellation before the absolute value, '
        'and (b) sum_f abs(grad_f) × train_std_f. The original uses sum_f abs(grad_f × input_f). All use the prediction gradient, never a test-label loss.',
        f"Sigmoid probability ranking reproduced the logit-ranked RL mask in {int(probability.same_mask.sum())}/{len(probability)} checks; "
        'it is recorded as an equivalent numerical control rather than an additional baseline.','',
        '## Randomness and replication','',
        'Policy training: one saved seed, 42; selected epoch comes from validation. No conclusion about stability across independently trained policies is supported.',
        'Random masks: seeds 100–109, ten independent random score vectors per halo. The same vector is coupled across budgets. '
        'The exact feasibility decoder induces a constrained rank distribution; it is not uniform over feasible final masks.',
        'Policy sampling: seeds 200–209, fixed saved orientation-averaged logits, fresh Gumbel draws at each action, exact completion feasibility. '
        'This samples the direct constrained PL decoder conditional on one trained policy; it is not independent training replication or the original legacy training sampler.',
        'Tie marks: seeds 300–302 alter only independent tie priorities for deterministic RL. Baseline marks are held fixed in the mask/sampling experiments.',
        f"Bootstrap: seed {cfg['bootstrap_seed']}, {cfg['bootstrap_resamples']} paired halo resamples; these intervals quantify halo sampling uncertainty, "
        'conditional on the supplied trained artifacts and evaluated mask seeds. They do not quantify policy-training uncertainty.',
        'Repeated-selector RMSE is sqrt(mean_h mean_seed squared_error), an expected single-draw loss. Seed mean/SD/range are exported separately; '
        'predictions are never averaged into an ensemble. No mask seed is treated as another halo.','',
        '## Where RL is better or worse','',
        '| Competitor at 40% | RL − competitor RMSE | Paired 95% interval | RL lower / higher / tied halo loss |',
        '|---|---:|---|---|']
    for method in order:
        if method=='provided_rl':continue
        r=q40.loc[method]
        report.append(f"| {labels[method].replace('$'+chr(92)+'times$','×')} | {r.delta_rmse:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | "
                      f"{int(r.rl_lower_squared_error_halos)} / {int(r.rl_higher_squared_error_halos)} / {int(r.ties)} |")
    report+=['','Negative delta favors RL. The full CSV reports every competitor at every budget and every halo; no favorable subgroup is selected. '
        'Intervals are exploratory percentile intervals without multiplicity correction, so individual exclusions of zero are not confirmatory significance tests.','',
        '## Sensitivity and coverage','',
        'Strata were specified for node count, mean physical degree (2P/N), target log mass, and mean subhalo velocity dispersion. '
        'Cutpoints use validation tertiles; tied values go into the higher bin, and empty bins remain explicit. '
        'All 82 test halos occur exactly once across bins for every property, selector, and budget. '
        'Target-mass strata are descriptive outcome strata and never used for tuning or selection.',
        f"Test graph sizes span {int(testprops.nodes.min())}–{int(testprops.nodes.max())} nodes and "
        f"{int(testprops.pairs.min())}–{int(testprops.pairs.max())} physical pairs. This narrow range cannot establish large-graph robustness. "
        'Degree and graph size are correlated; strata are descriptive, not controlled causal effects. '
        'See stratified_comparisons.csv for every bin, including unfavorable and empty bins.','',
        '## Scientific and timing limits','',
        'The checkpoint is explicitly legacy_total_radius: input half_mass_radius is total rather than stellar radius, '
        'and target semantics and the original predictor training cohort are uncertified. Selector train/validation/test identities are disjoint, '
        'but this does not establish absence of leakage in the pre-existing backbone training. '
        'These experiments strengthen comparisons conditional on these weights; they do not certify corrected astronomy or external generalization.',
        'Historical timings include scoring (with full-graph embeddings for RL and a gradient pass for saliency), constrained decoding, and masked prediction. '
        'The full reference is shared, graph construction is separate, and model loading/training is excluded. '
        'The reference CPU/NetworkX decoder dominates much of the original GPU region time. '
        'New seed sensitivity runs reuse saved scores; their wrapper/decoder times are diagnostics and cannot substantiate end-to-end speedups. '
        'New heuristic/gradient timing fields measure shared scoring work; they are not independent optimized deployment latencies. '
        'No repeated warmed GPU timing experiment was run.',
        'Independent policy training seeds, a matching corrected-astronomy backbone, external populations, and new GPU latency runs are explicitly not executed '
        'in manifest.json. No new training was substituted for missing artifacts.','',
        '## Reproduction and artifacts','',
        'Run the commands in experiments/paper_baselines/README.md. The executed run is results/run-20261016-final-v2. '
        'Inputs and outputs have SHA-256 manifests; every new test/validation prediction, signed residual, mask, score, seed role, '
        'eligibility result and hyperparameter candidate is exported. Historical replay includes original and CPU predictions. '
        'The initial schema-validation failure is preserved as a failed manifest in results/run-20261016; it contains no claimed experiment results.']
    seed_lines=['## Seed sensitivity at the primary budget','',
        '| Selector | Seed RMSE mean | SD | Minimum | Maximum |',
        '|---|---:|---:|---:|---:|']
    for method in ['random_repeated','rl_sampled','rl_tie_marks']:
        r=table[(table.method==method)&(table.retention==.4)].iloc[0]
        seed_lines.append(f"| {labels[method]} | {r.seed_rmse_mean:.6f} | {r.seed_rmse_std:.6f} | {r.seed_rmse_min:.6f} | {r.seed_rmse_max:.6f} |")
    seed_lines+=['','The range describes every evaluated seed; no best seed is promoted to the headline comparison. '
        'Tie-mark perturbations leave RL unchanged here. Sampling variance is conditional on the one available trained policy.','']
    pos=report.index('## Where RL is better or worse');report[pos:pos]=seed_lines
    strata_frame=pd.DataFrame(strata)
    strata_lines=['## All primary-budget property strata','',
        'Every validation-tertile bin is shown for the strongest existing competitor (gradient × input) and repeated random. '
        'RMSE entries are descriptive; small or empty bins cannot establish a stable interaction.','',
        '| Property | Bin | Halos | RL RMSE | Gradient × input | Repeated random |',
        '|---|---|---:|---:|---:|---:|']
    for field in cfg['strata']:
        for label in ['low','middle','high']:
            own=strata_frame[(strata_frame.retention==.4)&(strata_frame.property==field)&(strata_frame.stratum==label)]
            g=own[own.competitor=='edge_feature_grad_x_input'].iloc[0]
            r=own[own.competitor=='random_repeated'].iloc[0]
            values=[g.rl_rmse,g.competitor_rmse,r.competitor_rmse]
            formatted=['—' if pd.isna(v) else f'{v:.6f}' for v in values]
            strata_lines.append(f'| {field} | {label} | {int(g.halos)} | '+' | '.join(formatted)+' |')
    strata_lines+=['','RL has higher error than gradient × input in the small-node, low-degree, high-target-mass, and low-velocity bins; '
        'it has lower error in the other nonempty gradient comparison bins. Against repeated random it is worse in the small-node '
        'and low/middle target-mass bins, and better in the remaining nonempty bins. These are descriptive partitions of the full cohort, '
        'not selected subpopulations or confirmatory interaction tests. Full all-method/all-budget strata and exact cutpoints are exported.','']
    pos=report.index('## Scientific and timing limits');report[pos:pos]=strata_lines
    (evidence_dir/'robustness-results.md').write_text('\n'.join(report)+'\n')
    timing=json.loads((result_dir/'historical_timing.json').read_text())
    audit_doc=['# Baseline fairness audit','',
        f"Audited {audit['rows']} stored records: {audit['test_halos']} halos × 8 methods × 4 budgets. "
        f"Verified {audit['verified_pruned_masks']} physical-pair masks and reproduced each mask using the experiment decoder. "
        f"Rebuilt {audit['graph_hashes_rebuilt']} graph content hashes and replayed {audit['prediction_replay']['rows']} predictions; "
        f"maximum CPU/GPU prediction difference was {audit['prediction_replay']['max_absolute_difference']:.8g}.",'',
        '| Fairness condition | Finding |','|---|---|',
        '| Same test halos | Pass: exactly the same 82 IDs for every method and budget; no failures or exclusions. |',
        '| Frozen backbone | Pass: all saved rows and loaded weights have state SHA-256 `'+audit['backbone_state_sha256']+'`; state unchanged after evaluation. |',
        '| Physical budgets | Pass: k=ceil(qP), unique nonself unordered pairs; all stored directions/copies selected together; loops retained outside the physical budget. |',
        '| Constraints | Pass: exact direct decoding, no physical isolates, no repair additions or budget expansion. Full graph is an explicit reference. |',
        '| Inference | Pass: single deterministic model.eval prediction, dropout disabled, frozen parameters; configured mc_samples=30 is not used in primary predictions. |',
        '| Scoring | RL uses one actual full-graph embedding call and two orientation policy passes. Gradient saliency uses one gradient scoring forward; neither is a free predictor call. |',
        '| Timing | Defined region excludes loading/training and graph construction; shared full reference counted separately. No inference speedup is established by these CPU/cached-score experiments. |',
        '| Selector leakage | Pass on recorded identities: 378 train / 81 validation / 82 test, no overlaps; train-only normalization/teacher labels; validation-only policy epoch and new control tuning. |',
        '| Backbone leakage | Unverified: the legacy predictor checkpoint lacks certified train/test provenance. Disjoint selector IDs do not prove disjoint original predictor training. |',
        '| Independent training replication | Not available: one trained policy seed (42). Mask, sampling, tie, and bootstrap seeds are separately labeled. |','',
        '## Existing competitors','',
        'The existing pruning comparison includes gradient × input, random, shortest-distance, degree, stellar-binding, learned saliency distillation, and RL. '
        'The learned selector is a train-only saliency distillation control, not an oracle or PGExplainer. '
        f"At 40%, gradient × input ({metric('edge_feature_grad_x_input'):.6f}) is the strongest existing pruning competitor to RL ({metric('provided_rl'):.6f}). "
        f"Among existing non-RL pruning selectors, original random is strongest at 60% ({metric('random',.6):.6f}) and distance at 80% ({metric('distance',.8):.6f}). "
        'The repeated-random control changes the 60% ranking. All are retained in the matched tables.','',
        '| Historical GPU method at 40% | Scoring s | Decoder s | Masked prediction s | Full reference s | Region s | Live forwards |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in ['full','provided_rl','edge_feature_grad_x_input','random','distance','degree','stellar_binding_proxy','supervised_raw']:
        t=timing[method]
        audit_doc.append('| '+method+' | '+' | '.join(f'{t[k]:.6f}' for k in ['scoring_time_s','decoder_time_s','prediction_time_s','full_time_s','total_region_time_s'])+f" | {t['mean_live_forward_calls']:g} |")
    audit_doc+=['','Region time also includes evaluator overhead and need not equal the sum of nested phase timers. '
        'There is one recorded timing observation per halo/method/budget rather than independent repeated warmed trials. '
        'Historical GPU times and new CPU times must not be pooled.',
        '## Provenance and scientific qualifications','',
        'SHA-256 checks tie the CSV and predictor checkpoint to C provenance, A/B/C split IDs and backbone identities, '
        'the A learned-control artifact, B policy and recovered-training metadata, and C result records. '
        'The predictor checkpoint file SHA-256 differs from its model-state SHA-256 because the former hashes serialized metadata as well as tensors.',
        f"Catalog subhalo IDs appearing in multiple halo groups: {audit['catalog_subhalo_ids_in_multiple_groups']}. "
        'Policy metadata was reconstructed from original revision and exact inputs and is explicitly labeled as reconstructed, not contemporaneous proof of all backbone training details.',
        'The legacy radius and uncertified target/backbone training provenance remain material limitations. '
        'See robustness-results.md and the executable package for all additional results, seed distinctions, and experiments not executed.']
    (evidence_dir/'baseline-audit.md').write_text('\n'.join(audit_doc)+'\n')
    outputs=[p for p in result_dir.iterdir() if p.is_file() and p.name not in manifest['outputs'] and p.name not in ['manifest.json','analysis_manifest.json']]
    outputs+=list(evidence_dir.glob('baseline-*'))+[evidence_dir/'robustness-results.md']
    dump(result_dir/'analysis_manifest.json',{'status':'executed','source_manifest':sha256_file(result_dir/'manifest.json'),
        'analyzer_sha256':sha256_file(Path(__file__)),
        'outputs':[sha256_file(p) for p in sorted(outputs)],'paired_unit':'halo','policy_training_seeds':[42]})
    print(table[['retention','method','rmse']].to_string(index=False))
    return tables


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--evidence',default='paper/evidence')
    args=p.parse_args();summarize(args.results,args.evidence)
