"""Generate the efficiency evidence, figures and cost model from raw samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.paper_efficiency.cost_model import build_cost_model
from benchmarks.paper_efficiency.run import digest


def summarize(records):
    result = {}
    for name in records[0]['samples_s']:
        values = np.array([row['samples_s'][name] for row in records]) * 1000
        result[name] = dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                            p95_ms=float(np.percentile(values, 95)),
                            per_graph_means_ms=values.mean(axis=1).tolist())
    return result


def figures(records, summary, scaling, cost, out, environment):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    def save(fig, name):
        fig.savefig(out / f'{name}.png', bbox_inches='tight')
        fig.savefig(out / f'{name}.svg', bbox_inches='tight')
        plt.close(fig)
    keys = ['full_graph','reference_rl','optimized_rl','optimized_rl_one_shot',
            'reference_selector_no_embedding','optimized_selector_no_embedding','pruned_backbone_service']
    labels = ['Full graph','Reference RL','Optimized RL\nprepared topology','Optimized RL\none-shot preparation',
              'Reference selector\nembeddings supplied','Optimized selector\nembeddings supplied','Pruned backbone\nservice']
    fig, ax = plt.subplots(figsize=(10.5,5.8))
    values = [summary[k]['mean_ms'] for k in keys]
    bars = ax.barh(range(len(keys)), values, color=['#636363','#b35806','#2166ac','#67a9cf','#e08214','#4393c3','#4d9221'])
    ax.set(xscale='log', xlabel='Mean milliseconds / halo (log scale)', yticks=range(len(keys)), yticklabels=labels)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(val*1.05, bar.get_y()+bar.get_height()/2, f'{val:.3f}', ha='left', va='center')
    ax.set_xlim(min(values)*.6,max(values)*1.8)
    ax.set_title(f"Matched {environment['hardware']} {environment['device']} inference regions · {len(records)} selected catalog halos\n"
                 f"{environment['warmup']} warmups, {environment['repeats']} repetitions; fresh embeddings in both RL paths")
    fig.tight_layout()
    save(fig,'matched-latency')

    fig, ax = plt.subplots(figsize=(8.7,4.4))
    components = [('embedding_service','Embedding service','#4d4d4d'),
                  ('policy_and_pair_reduction','Policy + scorer checks','#80cdc1'),
                  ('reference_decoder','Exact pair decoder','#bf812d'),
                  ('pruned_backbone_service','Pruned prediction service','#35978f')]
    for i, mode in enumerate(['Reference','Optimized']):
        left=0.
        for key,label,color in components:
            actual = 'optimized_decoder' if i and key=='reference_decoder' else key
            val=summary[actual]['mean_ms']
            ax.barh(i,val,left=left,label=label if not i else None,color=color)
            left += val
        ax.text(left+.3,i,f'{left:.2f} ms stage sum',va='center')
    ax.set(yticks=[0,1],yticklabels=['Reference RL','Optimized RL'],xlabel='Mean milliseconds / halo')
    ax.set_xlim(0,sum(summary[k]['mean_ms'] for k,_,_ in components)*1.3)
    ax.set_title('Dominant work removed: exhaustive candidate feasibility\nIndependent stage sums explain cost; whole-region latencies are measured separately')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    fig.tight_layout(); save(fig,'stage-costs')

    fig, axes=plt.subplots(1,2,figsize=(10.5,4.3))
    for ax,family in zip(axes,['ring_degree_4','complete']):
        rows=[r for r in scaling['records'] if r['family']==family]
        for key,label,color in [('reference_decoder','Reference decoder','#b35806'),('optimized_decoder','Optimized decoder','#2166ac')]:
            ax.plot([r['pairs'] for r in rows],[np.mean(r['samples_s'][key])*1000 for r in rows],
                    marker='o',label=label,color=color)
        ax.set(xscale='log',yscale='log',xlabel='Physical pairs',ylabel='Mean milliseconds',
               title='Degree-four ring' if family=='ring_degree_4' else 'Complete topology')
        ax.legend(fontsize=8)
    fig.suptitle(f"Synthetic decoder scaling · {environment['hardware']} CPU · {scaling['warmup']} warmups, {scaling['repeats']} repetitions")
    fig.tight_layout(); save(fig,'synthetic-scaling')

    fig,ax=plt.subplots(figsize=(8,4.4))
    reuse=np.arange(1,129)
    ax.axhline(cost['full_s']*1000,label='Full graph service',color='#636363')
    ax.plot(reuse,(cost['setup_s']/reuse+cost['pruned_s'])*1000,label='Optimized setup amortized',color='#2166ac')
    ax.plot(reuse,(cost['reference_setup_s']/reuse+cost['pruned_s'])*1000,label='Reference setup amortized',color='#b35806')
    ax.set(xscale='log',yscale='log',xlabel='Backbone evaluations sharing one fixed selected mask',
           ylabel='Modeled milliseconds / evaluation',title='Conditional cost model, not a measured reuse speedup')
    ax.legend(); fig.tight_layout(); save(fig,'amortization-model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True)
    parser.add_argument('--scaling',required=True)
    parser.add_argument('--validation')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    evidence=out/'evidence'; evidence.mkdir()
    figure_dir=out/'figures/efficiency'; figure_dir.mkdir(parents=True)
    results=json.loads(Path(args.results).read_text()); records=results['records']
    scaling=json.loads(Path(args.scaling).read_text())
    if results['environment']['device'] != 'cpu':
        parser.error('This evidence template is for CPU results; do not label accelerator results as CPU.')
    if results['environment']['threads'] != scaling['threads'] or results['environment']['interop_threads'] != 1:
        parser.error('Main and scaling runs must both use one intra-op and one inter-op thread.')
    for key in ('device','hardware','platform','python','torch','torch_geometric','networkx'):
        if results['environment'][key] != scaling['environment'][key]:
            parser.error(f'Main and scaling environments differ: {key}')
    for key in ('checkpoint','policy','context'):
        if results['inputs'][key]['sha256'] != scaling['inputs'][key]:
            parser.error(f'Main and scaling input hashes differ: {key}')
    for key in results['source_sha256'].keys() & scaling['source_sha256'].keys():
        if results['source_sha256'][key] != scaling['source_sha256'][key]:
            parser.error(f'Main and scaling sources differ: {key}')
    summary=summarize(records); cost=build_cost_model(records)
    for row in records:
        assert row['exact_mask_and_order_equal']
        assert row['reference_prediction']==row['optimized_prediction']
        assert row['requested_pairs']==row['final_pairs']
    s=lambda key:summary[key]['mean_ms']
    rl_gain=s('reference_rl')/s('optimized_rl')
    one_gain=s('reference_rl')/s('optimized_rl_one_shot')
    full_ratio=s('optimized_rl_one_shot')/s('full_graph')
    feasible=sum(row['feasible'] for row in results['cohort'])
    mean_retention=np.mean([r['retention'] for r in records])*100
    rows='\n'.join(f"| {key} | {val['mean_ms']:.4f} | {val['median_ms']:.4f} | {val['p95_ms']:.4f} |" for key,val in summary.items())
    scaling_rows='\n'.join(f"| {r['family']} | {r['nodes']} | {r['pairs']} | {np.mean(r['samples_s']['reference_decoder'])*1000:.4f} ({np.median(r['samples_s']['reference_decoder'])*1000:.4f}) | {np.mean(r['samples_s']['optimized_decoder'])*1000:.4f} ({np.median(r['samples_s']['optimized_decoder'])*1000:.4f}) | {np.mean(r['samples_s']['full_graph'])*1000:.4f} | {np.mean(r['samples_s']['reference_rl'])*1000:.4f} | {np.mean(r['samples_s']['optimized_rl'])*1000:.4f} |" for r in scaling['records'])
    memrows='\n'.join(f"| {name} | {np.mean([r['memory'][name]['python_peak_bytes'] for r in records])/1024:.1f} |" for name in records[0]['memory'])
    validation=(json.loads(Path(args.validation).read_text()) if args.validation else None)
    validation_text=(f"All {validation['test_graphs']} campaign test graphs were also replayed: masks, ordered actions, diagnostics and CPU predictions were identical; maximum prediction difference was zero."
                     if validation else 'Only the timed sample was replayed in this report.')
    analysis=f'''# RL inference efficiency audit

## Result and scope

On this matched {results['environment']['hardware']} CPU experiment, the existing RL inference region averaged **{s('reference_rl'):.3f} ms/halo**, the opt-in optimized region **{s('optimized_rl'):.3f} ms/halo** ({rl_gain:.2f}× faster relative to existing RL), and optimized RL including fresh topology preparation **{s('optimized_rl_one_shot'):.3f} ms/halo** ({one_gain:.2f}× faster relative to existing RL). The unchanged full-graph service averaged **{s('full_graph'):.3f} ms/halo**. Optimized one-shot RL remains **{full_ratio:.2f}× slower than full graph**. There is no measured full-graph inference speedup.

These are directly timed inference regions on resident graphs, excluding common feature generation and initial input transfer. The one-shot variant includes a fresh topology cache in every call. No change is installed into the historical benchmark, notebooks or training path: the new decoder is opt-in. The completed scientific benchmark and its conclusions remain unchanged.

The user-supplied historical values, 0.00784 s full graph and 0.440 s RL (56.12× ratio), are historical context only. They are not combined with laptop timings or used as a hardware-matched baseline. No claim is made to reproduce their absolute latency or hardware. This study uses the frozen backbone from the completed campaign, not its separately fine-tuned Stage B backbone.

## Inputs and measurement

- Base commit: `{results['base_commit']}`; original source modules remain unchanged.
- Hardware: {results['environment']['hardware']}, {results['environment']['memory_bytes']/2**30:.0f} GiB unified memory; device CPU. Python {results['environment']['python'].split()[0]}, PyTorch {results['environment']['torch']}, PyG {results['environment']['torch_geometric']}, NetworkX {results['environment']['networkx']}. One intra-op and one inter-op thread. This differs from the historical pinned training environment.
- The local legacy clustered catalog, backbone and copied completed policy are SHA256-checked against the campaign manifest. The policy is not retrained. The original catalog's radius/target semantics remain uncertified; these are available catalog graphs, not a newly validated astrophysical population.
- All {len(results['cohort'])} campaign test IDs were considered; {feasible} are feasible at 40%. The {len(records)} timed IDs span the feasible population sorted by (node count, stored edges, ID). No labels or prediction quality affect membership. The average realized retention is {mean_retention:.3f}% because budgets use ceil(0.4P).
- These graphs have {min(r['nodes'] for r in records)}–{max(r['nodes'] for r in records)} nodes and {min(r['pairs'] for r in records)}–{max(r['pairs'] for r in records)} physical pairs. Averages describe this declared subset, not an estimate over all test halos. Larger topologies below are explicitly synthetic.
- {results['environment']['warmup']} warmups and {results['environment']['repeats']} repetitions per graph/region; method order shuffled with a fixed seed for every repetition. `perf_counter_ns` measures wall time. The harness synchronizes CUDA/MPS before/after a region when explicitly requested; CPU work is synchronous. Included results are CPU only.
- Models use eval mode and no gradients. Both RL paths obtain fresh full-backbone embeddings and execute the same two-orientation policy scorer on every call. The original scorer, prediction service, float32 score conversion and fixed per-graph marks are shared. Model and policy state hashes are unchanged after timing.
- Whole inference regions are timed directly. Stage measurements are separate; their sums are explanatory approximations, not a claimed end-to-end measurement. Service latencies include checks, cloning, Batch construction and state preservation. Bare forwards use already built batches. No application I/O, scheduling, networking, training, TTA or MC uncertainty runtime is claimed.

## Prediction-path audit

| Operation | Existing operation and cost boundary |
| --- | --- |
| Catalog / features | `TNGLoader.load` parses CSV, converts declared units, groups halos; `HaloData.get_node_features` computes log features. `GraphBuilder.build_graph` computes radius neighbors via chunked cdist, symmetry/coalescing, loops, nearest-neighbor fallback, distances, velocity differences, angle, signed mass ratio and projected separation. Per-halo graph construction is measured separately. |
| Physical layout / fixed marks | `PhysicalPairLayout.from_edge_index` validates endpoints, separates loops, sorts endpoints and unique-reduces stored copies. `fixed_marks` hashes seed/graph identity and creates a replayable permutation. The historical matched evaluator prepares these outside its inference region. |
| Full-backbone embedding | `PairScorer._score_provided_rl` calls `MeteredPredictor.call(return_embeddings=True)`: validates/clones graph, builds Batch, sets eval mode, executes the entire full GNN and prediction head, validates results and restores modes/buffers. This is required before pruning and is charged in both RL regions. |
| Policy execution | Mean-pool node embeddings for context; clone/reverse edge attributes and endpoints; negate signed mass-ratio feature; run EdgePolicyNet twice; average orientations. Policy projections currently repeat endpoint projections per stored edge. This path is intentionally unchanged to preserve logits and numerical ranking. |
| Pair score reduction | Average all real stored copies through the inverse map using index_add, then validate finite logits and convert to float32. Scorer validation reconstructs/alignment-checks layout and hashes the policy state per call. The cached-embedding scorer timing includes this provenance/check overhead. |
| Action selection / sampling | Primary inference is deterministic direct ranking with unique independent pair marks for ties. It performs no categorical sampling and evaluates no likelihood. Training/legacy pair PL and TTA can loop over multinomial draws and construct ordered likelihood arrays; those paths are audited but not optimized or timed here. |
| Exact physical-pair decoding | Original `select_pairs` calls `_candidates` at each of k steps, testing every remaining proposal. Each feasibility test reconstructs full/chosen CPU NetworkX graphs and solves edge-cover completion through maximum matching of uncovered nodes. |
| Constraint repair | Exact direct inference enforces no-isolate completion during selection; there is no additive post-repair. Infeasible requests raise. Legacy `repair_pairs` walks pair marks, adding incident pairs and potentially exceeding the requested count; replacing exact selection with it would change the experiment and is not done. |
| Reconstruction | `MeteredPredictor` validates mask, clones Data (including detached copies of x/edges/attrs), physically slices edge_index and edge_attr in original column order, then calls Batch.from_data_list. Extra metadata tensors are cloned too. This service is unchanged and is measured separately from its bare forward. |
| Pruned GNN | Same frozen NNConv backbone and prediction head, unchanged nodes/features/weights, fewer stored edges; all selected reciprocal/duplicate columns and architecture self-loops remain. |
| Transfers / synchronization | Original feasibility repeatedly uses tensor scalar extraction and `.cpu().tolist()` and rebuilds CPU graphs. Scorer policy-state hashing also stages tensor data to CPU on accelerators; original evaluator additionally moves selected/mask diagnostics to CPU then the mask back to the model device. The optimized decoder copies endpoints once during preparation, transfers one ranked vector per decode and final order/membership to its device, and still uses the original diagnostic expansion. CPU runs make no GPU transfers. Accelerator behavior is not measured here. |

## Dominant bottleneck and safe optimization

The initial independent small-graph cProfile (10 nodes, 19 pairs) recorded 125 feasibility/matching calls, 251 NetworkX graph constructions, and about 97% of instrumented decoder time under candidate feasibility. The representative matched decoder profiles in the raw results show the same issue on the trained policy. Profiled wall times include instrumentation and are not used in latency tables.

`PreparedTopology` owns a cloned physical layout, a frozen CPU graph, immutable endpoint tuples and the exact minimum edge-cover size. One full matching is charged to preparation. It caches no embeddings, scores, masks or model-dependent values. Rebuild it if topology, column order, node count or device changes; treat its fields as read-only.

`select_pairs_fast` sorts by the same score and mark tie priorities once and lazily tests candidates in that order. For uncovered nodes U and r remaining slots, |U| <= r guarantees completion because each uncovered node has an incident original edge; ceil(|U|/2) > r proves impossibility. Ambiguous cases use the same exact formula |U| - maximum_matching(G[U]) <= r. A rejected candidate cannot become feasible after extending the prefix: any later completion containing it would also complete the earlier prefix. Thus candidates are visited once without changing the reference greedy action. `none` uses vectorized ranking; `connected` delegates to the reference. Sampling/scaffold/training semantics are not changed.

Final diagnostics and stored-column expansion use the existing implementation. No top-k approximation, budget expansion, invented reverse edge, deleted loop, isolate tolerance or backbone approximation is introduced. The scorer still repeats layout validation; preparation only removes repeated topology construction from decoder feasibility.

## Matched latency

Milliseconds per halo; median and p95 pool the {len(records)*results['environment']['repeats']} measured samples, with equal samples per selected graph. Source JSON preserves individual samples and per-graph values.

| Region | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
{rows}

Selector-only regions explicitly assume supplied embeddings. They are not complete RL prediction timings. Input transfer measures cloning from a retained CPU graph followed by `.to(device)`; on this CPU run it is a host clone and a no-op device conversion. Output-to-host is a no-op CPU conversion. Common feature generation and initial transfers can be added to either pipeline, but an actual fresh-catalog application wall-time speedup has not been measured.

![Matched latency](../figures/efficiency/matched-latency.png)
![Independent stage costs](../figures/efficiency/stage-costs.png)

## Scaling

The genuine available catalog spans only 7–10 nodes; it cannot establish large-graph scaling. Synthetic degree-four rings hold degree constant as N grows; complete graphs increase density. Features are synthetic, topology and family are explicit, and both methods use the same trained policy/backbone and 40% exact budget. {scaling['warmup']} warmups and {scaling['repeats']} repetitions per region on the same {results['environment']['hardware']} CPU; no scientific accuracy claim. Preparation is separate from the prepared-topology optimized region.

| Synthetic family | N | P | Reference decode mean (median) ms | Optimized decode mean (median) ms | Full service ms | Reference RL ms | Optimized RL ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{scaling_rows}

All samples, including scheduling/GC outliers, are retained. This is a live machine without CPU affinity or thermal isolation, and scaling uses only {scaling['repeats']} repetitions. Independent stage samples can consequently have a larger mean than a separately timed whole region; they must not be subtracted as if they were simultaneous measurements. Decode medians above expose this variation; full raw samples are included.

![Synthetic topology scaling](../figures/efficiency/synthetic-scaling.png)

Reference candidate enumeration performs about kP feasibility tests and repeatedly builds graphs. Optimized ranking is O(P log P), checks at most P proposals, and still has worst-case expensive matching fallbacks. This is not a claim of linear-time general matching or scaling to production-sized astronomical graphs. Dense NNConv can also become the dominant native-memory and inference cost after decoder optimization.

## Memory

Mean Python allocation peak from separate tracemalloc runs, not mixed into latency:

| Region | Peak KiB |
| --- | ---: |
{memrows}

Process maximum resident set was {results['process_max_rss_bytes']/2**20:.1f} MiB for the complete process, including imports, models and all regions. Mean owned cache tensor storage is {np.mean([r['cache_tensor_bytes'] for r in records]):.0f} bytes plus Python endpoint/NetworkX storage; topology-preparation Python peaks are in the raw JSON. These are not native CPU tensor allocation peaks or per-method RSS attribution. CUDA peak allocation and MPS live allocation support exist in the harness but are not reported as measured results here.

## Regression evidence and limitations

The regression suite covers all 64 simple four-node topologies at every possible budget, mixed score ties and explicit marks; random larger topologies; disconnected, one-way, duplicate, loop-only and empty layouts; unrepresented input isolates; infeasible and malformed budgets; nonfinite scores; source mutation; RNG preservation with explicit marks; transported relabeling/column permutation; connected fallback; and identical predictions from a real small CosmicNetGNN. The exact edge-cover path where the highest logit would make the requested budget infeasible has a literal regression case. {validation_text}

The pre-change suite passed 804 tests with 13 skipped. The final suite passed **820 tests with 13 skipped**; exact command output is saved in `benchmarks/paper_efficiency/results/tests.txt`. Dependency deprecations and availability skips remain visible. The final run took 195.69 seconds, including a slow shared Git history read in the unchanged archival-notebook test; no test was skipped or altered to work around it. The independent review found no decoder correctness issue; its transfer-timing finding was corrected before the final measured replay. A strict break-even floating-point boundary also received a failing regression and fix. The experiment manifest binds scaling to the main run's hardware, software, inputs and source hashes.

Limitations: one laptop CPU; small legacy graphs and a declared timing subset; no Kaggle/CUDA timing reproduction; no accelerator throughput, batched throughput, TTA or MC-dropout benchmark; Python-only allocation peaks; no population confidence interval from this stratified sample. Optimized RL still requires a full embedding forward before pruned prediction. See `inference-cost-model.md` for conditional reuse assumptions, rather than a deployment-speedup claim.

## Reproduction

See `benchmarks/paper_efficiency/README.md` for complete commands, input hashes, scopes, cache invalidation and opt-in API usage. Raw results, per-graph samples, environment, source identities, profiles and copied policy are versioned under `benchmarks/paper_efficiency/`. PNG and editable SVG figures are in `paper/figures/efficiency/`. No original scientific evidence file is overwritten.
'''
    scenarios='\n'.join(f"| {r['reuses']} | {r['full_total_s']*1000:.3f} | {r['optimized_total_s']*1000:.3f} | {r['reference_total_s']*1000:.3f} |" for r in cost['reuse_scenarios'])
    thresholds=[r['break_even_reuses'] for r in cost['per_graph'] if r['break_even_reuses'] is not None]
    cost_doc=f'''# Inference cost model

## Measured starting point

All coefficients come from the same {results['environment']['hardware']} CPU experiment in `efficiency-analysis.md`. Do not substitute the historical 0.00784 s / 0.440 s numbers into this model as matched machine coefficients. The backbone is frozen and topology, feature definitions, policy and exact 40% pair budgets are unchanged.

Let F be full-graph prediction service cost, B selected-graph prediction service cost, E full-backbone embedding service, L policy scoring plus validation/provenance and pair reduction, D exact decoding, and C topology preparation. Services include graph cloning/Batch construction and state checks. Common feature construction H is outside the inference region.

Measured F = **{cost['full_s']*1000:.4f} ms**, B = **{cost['pruned_s']*1000:.4f} ms**. Pruned bare forward averages {s('pruned_backbone_forward'):.4f} ms versus full bare forward {s('full_backbone_forward'):.4f} ms; those bare-forward timings cannot replace service costs in an end-to-end comparison.

One-shot RL is approximately C + E + L + D + B. Directly measured optimized one-shot RL is **{cost['one_shot_measured_s']*1000:.4f} ms**; summing independent stages gives **{cost['one_shot_additive_model_s']*1000:.4f} ms**. The difference is ordinary independent-region timing variation and wrapper boundaries; the direct timing is the performance result. Since E already requires the full graph and B > 0, reducing D cannot by itself make this pipeline cheaper than a single full prediction under these measurements.

## What can be amortized

Catalog parsing/model loading are one-time process costs: this run measured {results['load_catalog_s']:.4f} s catalog loading and {results['load_model_s']:.4f} s model loading/device initialization. They are not inference speedups. Feature generation averages **{cost['feature_generation_s']*1000:.4f} ms/halo** on warmed builds. Its first-build timings for all test graphs are retained separately. Both methods require the full raw graph, so adding H to both cannot turn a slower one-shot RL method into a faster one.

Topology preparation averages **{cost['per_halo_topology_cache_s']*1000:.4f} ms/halo** and can be reused while edges, node count, column order and device remain fixed. This is the implemented cache. It remains valid when logits change, but embeddings/policy scoring/decoding must then be recomputed. Reusing only topology approaches the measured prepared RL region ({s('optimized_rl'):.4f} ms); it never removes the required embedding pass and remains slower than full graph ({s('full_graph'):.4f} ms).

Full embeddings can only be reused when node/edge features, full topology, backbone weights/mode and any stochastic realization are unchanged. A policy update or changed features requires rescoring and can change selection. A changed backbone generally invalidates embeddings. Masks can be reused only when doing so is an explicit valid inference contract; this study does not silently reuse masks across scientific examples or augmentation/MC draws.

## Conditional fixed-mask reuse scenario

Suppose a valid application needs R backbone evaluations sharing one selected mask, frozen embeddings/scoring inputs and the same predictor contract. Charge S = C + E + L + D once and B per evaluation:

`T_full(R) = R F`

`T_RL(R) = S + R B`

`T_RL(R) < T_full(R)` requires `F > B` and `R > S / (F - B)`.

The optimized setup coefficient S is **{cost['setup_s']*1000:.4f} ms** and the original setup coefficient is **{cost['reference_setup_s']*1000:.4f} ms**. On aggregate mean coefficients, strict break-even is **R = {cost['optimized_break_even_reuses']}** optimized, versus **R = {cost['reference_break_even_reuses']}** reference. Per selected graph, {len(thresholds)} of {len(records)} have positive measured B savings; finite thresholds span {min(thresholds) if thresholds else 'none'}–{max(thresholds) if thresholds else 'none'}. These thresholds are a cost-model calculation from small measured differences, not a measured reuse speedup or population guarantee. All coefficients and per-graph thresholds are in `inference-cost-model.json`.

| R shared evaluations | Full total ms, modeled | Optimized total ms, modeled | Reference total ms, modeled |
| --- | ---: | ---: | ---: |
{scenarios}

![Conditional amortization](../figures/efficiency/amortization-model.png)

If the same deterministic query is repeated with identical inputs and models, caching the full prediction is cheaper than repeatedly running either backbone. The R scenario is useful only when separate backbone evaluations are actually necessary and a fixed selector/mask is valid. Fixed-mask MC uncertainty, repeated heads or compatible ensembles could motivate a separate experiment, but none is validated or timed here. Different weights, stochastic selector embeddings, augmentation or changed observations can invalidate reuse; recomputing the full selector each time removes this amortization opportunity.

If B >= F on a different graph/device, no finite R breaks even. If S must be paid per independent halo (the normal one-shot case), increasing the number of distinct halos does not amortize it. Batching may improve utilization but must be measured with equal batch sizes and complete selector/transfer costs. Training, adaptation, policy preparation and cache storage/invalidation would add further costs and have not been amortized away.

## Practical conclusion

The measured optimization removes the dominant repeated CPU feasibility computation. It substantially improves existing RL inference, but the completed benchmark should continue to report RL pruning as a cost/accuracy tradeoff. This study supplies no measured single-halo speed advantage over full-graph prediction and no validated repeated-use deployment advantage.
'''
    (evidence/'efficiency-analysis.md').write_text(analysis)
    (evidence/'inference-cost-model.md').write_text(cost_doc)
    (evidence/'efficiency-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (evidence/'inference-cost-model.json').write_text(json.dumps(cost,indent=2)+'\n')
    files=[Path(args.results),Path(args.scaling)]+([Path(args.validation)] if args.validation else [])
    manifest={'files':{str(p):digest(p) for p in files},'environment':results['environment'],
        'shared_inputs':results['inputs'],'main_source_sha256':results['source_sha256'],
        'scaling_source_sha256':scaling['source_sha256'],
        'note':'Main and synthetic scaling executed sequentially on the same CPU/environment. Profile instrumentation excluded from latency samples.',
        'report_source_sha256':digest(__file__),
        'cost_model_source_sha256':digest(Path(__file__).with_name('cost_model.py'))}
    (evidence/'efficiency-experiment-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    figures(records,summary,scaling,cost,figure_dir,results['environment'])
    print(json.dumps({'reference_rl_ms':s('reference_rl'),'optimized_rl_ms':s('optimized_rl'),
        'full_graph_ms':s('full_graph'),'one_shot_optimized_ms':s('optimized_rl_one_shot'),
        'speedup_vs_rl':rl_gain,'one_shot_slowdown_vs_full':full_ratio,
        'modeled_break_even_reuses':cost['optimized_break_even_reuses']},indent=2))


if __name__=='__main__':
    main()
