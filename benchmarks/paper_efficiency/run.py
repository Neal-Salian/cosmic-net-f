"""Matched, warmed, single-halo timings; historical science outputs are read-only.

Run from the repo root: PYTHONPATH=. python benchmarks/paper_efficiency/run.py
--output <new-directory>. All latency samples are wall seconds, synchronized
before and after each region. CPU is the default; CUDA and MPS are explicit.
"""
import argparse
import copy
import cProfile
import hashlib
import json
import platform
from pathlib import Path
import pstats
import random
import resource
import subprocess
import sys
import time
import tracemalloc
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch

from data.loaders.tng_loader import TNGLoader
from data.provenance import model_state_hash
from graph.graph_builder import GraphBuilder
from model.model import load_model
from rls.benchmark_predictor import MeteredPredictor, _clone_graph
from rls.benchmark_scorers import PairScorer, _mean_over_real_copies
from rls.constrained_policy import select_pairs
from rls.constraints import PhysicalPairLayout, feasible_budget, pair_budget
from rls.inference_efficiency import PreparedTopology, select_pairs_fast
from rls.notebook_benchmark import fixed_marks
from rls.policy import build_policy


BASE = 'c825bf57652005c80faff79cf4dfd58e3a9baa14'
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def timed(function, device):
    sync(device)
    start = time.perf_counter_ns()
    result = function()
    sync(device)
    return (time.perf_counter_ns() - start) * 1e-9, result


def samples(functions, warmup, repeats, device, seed=42):
    for function in functions.values():
        for _ in range(warmup):
            timed(function, device)
    values = {name: [] for name in functions}
    rng = random.Random(seed)
    for _ in range(repeats):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            elapsed, _ = timed(functions[name], device)
            values[name].append(elapsed)
    return values


def reconstruction(graph, mask=None):
    cloned = _clone_graph(graph)
    if mask is not None:
        cloned.edge_index = cloned.edge_index[:, mask]
        cloned.edge_attr = cloned.edge_attr[mask]
    return Batch.from_data_list([cloned])


def memory(function, device):
    # Separate from latency runs: tracemalloc covers Python allocations only.
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    elif device.type == 'mps':
        baseline = torch.mps.current_allocated_memory()
    tracemalloc.start()
    result = function()
    sync(device)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {'python_peak_bytes': peak, 'python_live_bytes': current}
    if device.type == 'cuda':
        report['cuda_peak_extra_bytes'] = torch.cuda.max_memory_allocated(device) - baseline
    elif device.type == 'mps':
        report['mps_live_delta_bytes'] = torch.mps.current_allocated_memory() - baseline
    del result
    return report


def benchmark_graph(graph, cpu_graph, model, policy, names, warmup, repeats, device, seed):
    layout = PhysicalPairLayout.from_edge_index(graph.edge_index, len(graph.x))
    prepared = PreparedTopology.from_edge_index(graph.edge_index, len(graph.x))
    k = pair_budget(layout, .4)
    marks = fixed_marks(layout, seed, graph.cluster_id)
    meter = MeteredPredictor(model)
    scorer = PairScorer('provided_rl', policy=policy, edge_feature_names=names)
    score = lambda: scorer.score(graph, layout, k, meter=meter).scores.to(torch.float32)
    scores = score()
    original = lambda: select_pairs(layout, scores, k, 'no_isolates', pair_marks=marks)
    optimized = lambda: select_pairs_fast(prepared, scores, k, pair_marks=marks)
    a, b = original(), optimized()
    assert torch.equal(a.mask, b.mask) and torch.equal(a.order, b.order)
    assert a.diagnostics == b.diagnostics
    pred_a = meter.call(graph, mask=a.mask).prediction
    pred_b = meter.call(graph, mask=b.mask).prediction
    assert torch.equal(pred_a, pred_b), 'decoder prediction mismatch'
    emb = meter.call(graph, return_embeddings=True).embeddings
    cached_meter = SimpleNamespace(call=lambda *a, **kw: SimpleNamespace(embeddings=emb))
    policy_only = lambda: scorer.score(graph, layout, k, meter=cached_meter).scores.to(torch.float32)
    assert torch.equal(policy_only(), scores), 'embedding cache changed pair logits'
    full_batch, pruned_batch = reconstruction(graph), reconstruction(graph, a.mask)

    def policy_forwards():
        # Same two orientations as PairScorer; includes context and attr copies.
        ctx = emb.mean(dim=0)
        reverse_attr = graph.edge_attr.detach().clone()
        if 'mass_ratio' in names:
            reverse_attr[:, names.index('mass_ratio')] *= -1
        with torch.no_grad():
            first = policy(graph.edge_attr.detach().clone(), emb, graph.edge_index, ctx)
            second = policy(reverse_attr, emb, graph.edge_index[[1, 0]].clone(), ctx)
        return (first.reshape(-1) + second.reshape(-1)) / 2

    stored_logits = policy_forwards()
    assert torch.equal(_mean_over_real_copies(stored_logits, layout), scores)

    def pipeline(fast, prepare=False):
        topology = PreparedTopology.from_edge_index(graph.edge_index, len(graph.x)) if prepare else prepared
        values = score()
        selection = (select_pairs_fast(topology, values, k, pair_marks=marks) if fast else
                     select_pairs(layout, values, k, 'no_isolates', pair_marks=marks))
        return meter.call(graph, mask=selection.mask).prediction

    def selector(fast):
        values = policy_only()
        return (select_pairs_fast(prepared, values, k, pair_marks=marks) if fast else
                select_pairs(layout, values, k, 'no_isolates', pair_marks=marks))

    def forward(batch, embeddings=False):
        with torch.no_grad():
            return model(batch, return_embeddings=embeddings)

    primary = {
        'full_graph': lambda: meter.call(graph),
        'reference_rl': lambda: pipeline(False),
        'optimized_rl': lambda: pipeline(True),
        'optimized_rl_one_shot': lambda: pipeline(True, prepare=True),
        'reference_selector_no_embedding': lambda: selector(False),
        'optimized_selector_no_embedding': lambda: selector(True),
        'pruned_backbone_service': lambda: meter.call(graph, mask=a.mask),
    }
    stages = {
        'pair_layout': lambda: PhysicalPairLayout.from_edge_index(graph.edge_index, len(graph.x)),
        'topology_preparation': lambda: PreparedTopology.from_edge_index(graph.edge_index, len(graph.x)),
        'embedding_service': lambda: meter.call(graph, phase='scoring', return_embeddings=True),
        'embedding_forward': lambda: forward(full_batch, True),
        'policy_and_pair_reduction': policy_only,
        'policy_two_orientations': policy_forwards,
        'physical_pair_score_reduction': lambda: _mean_over_real_copies(stored_logits, layout),
        'reference_decoder': original,
        'optimized_decoder': optimized,
        'graph_reconstruction': lambda: reconstruction(graph, a.mask),
        'full_backbone_forward': lambda: forward(full_batch),
        'pruned_backbone_forward': lambda: forward(pruned_batch),
        'input_to_device': lambda: cpu_graph.clone().to(device),
        'resident_input_clone': lambda: graph.clone(),
        'output_to_host': lambda: pred_b.detach().cpu(),
    }
    raw = samples(primary, warmup, repeats, device, seed)
    # Stage timings are independent regions; their sum is an approximation,
    # not a substitute for the separately timed whole pipeline above.
    raw.update(samples(stages, warmup, repeats, device, seed + 1))
    return {
        'graph_id': graph.cluster_id, 'nodes': len(graph.x), 'stored_edges': graph.num_edges,
        'pairs': len(layout.pairs), 'requested_pairs': k, 'final_pairs': int(b.selected.sum()),
        'retention': k / len(layout.pairs), 'selected_pair_indices': b.selected.nonzero().flatten().cpu().tolist(),
        'reference_prediction': float(pred_a), 'optimized_prediction': float(pred_b),
        'exact_mask_and_order_equal': True, 'samples_s': raw,
        'memory': {name: memory(fn, device) for name, fn in primary.items()},
        'topology_python_memory': memory(stages['topology_preparation'], device),
        'cache_tensor_bytes': sum(t.numel() * t.element_size() for t in
            (prepared.layout.pairs, prepared.layout.inverse, prepared.layout.real)),
        'mask_sha256': hashlib.sha256(b.mask.cpu().numpy().tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--graphs', type=int, default=16, help='0 = all feasible test graphs')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--context', default=str(ROOT / 'benchmarks/paper_efficiency/inputs/context.json'))
    parser.add_argument('--policy', default=str(ROOT / 'benchmarks/paper_efficiency/inputs/policy.pt'))
    parser.add_argument('--csv', default=str(ROOT / 'data/raw/tng100_clustered.csv'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'kaggle/best_model_augmented.pt'))
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.graphs < 0 or args.threads < 1:
        parser.error('warmup >= 1, repeats >= 2, graphs >= 0, threads >= 1 required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    context = json.loads(Path(args.context).read_text())
    assert digest(args.csv) == context['dataset']['tng_csv']['sha256']
    assert digest(args.checkpoint) == context['dataset']['checkpoint']['sha256']
    assert digest(args.policy) == context['snapshot_origin']['policy_sha256']
    cfg = copy.deepcopy(context['config_used'])
    cfg['data']['tng'].update(clustered_file=str(Path(args.csv).resolve()), cache_dir=str(out / 'cache'))
    elapsed, halos = timed(lambda: TNGLoader(cfg).load(), torch.device('cpu'))
    load_s = elapsed
    test_ids = set(context['split_cluster_ids']['test'])
    halos = [h for h in halos if h.cluster_id in test_ids]
    assert {h.cluster_id for h in halos} == test_ids
    builder = GraphBuilder(cfg)
    graphs, construction, cohort = [], {}, []
    for halo in halos:
        elapsed, graph = timed(lambda: builder.build_graph(halo), torch.device('cpu'))
        construction[halo.cluster_id] = elapsed
        layout = PhysicalPairLayout.from_edge_index(graph.edge_index, len(graph.x))
        report = feasible_budget(layout, pair_budget(layout, .4), 'no_isolates')
        cohort.append(dict(graph_id=halo.cluster_id, nodes=len(graph.x), pairs=len(layout.pairs), **report))
        if report['feasible']:
            graphs.append(graph)
    graphs.sort(key=lambda g: (len(g.x), g.num_edges, g.cluster_id))
    if args.graphs and args.graphs < len(graphs):
        indices = np.linspace(0, len(graphs) - 1, args.graphs).round().astype(int)
        graphs = [graphs[i] for i in indices]
    elapsed, model = timed(lambda: load_model(args.checkpoint, cfg, device,
                            research_label=context['research_label']), device)
    model_load_s = elapsed
    policy = build_policy(cfg, model.output_dim).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device, weights_only=True), strict=True)
    policy.eval()
    model_before, policy_before = model_state_hash(model.state_dict()), model_state_hash(policy.state_dict())
    records = []
    for index, graph in enumerate(graphs):
        cpu_graph = graph
        transfer_s, graph = timed(lambda: cpu_graph.clone().to(device), device)
        record = benchmark_graph(graph, cpu_graph, model, policy, cfg['graph']['edge_features'],
                                 args.warmup, args.repeats, device, context['seed'])
        record['graph_construction_s'] = construction[graph.cluster_id]
        record['samples_s'].update(samples({'feature_generation': lambda: builder.build_graph(
            next(h for h in halos if h.cluster_id == graph.cluster_id))},
            args.warmup, args.repeats, torch.device('cpu')))
        record['initial_input_transfer_s'] = transfer_s
        records.append(record)
        (out / f'graph-{index:03d}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'{index+1}/{len(graphs)} {graph.cluster_id}: N={len(graph.x)} E={graph.num_edges}', flush=True)
    assert model_state_hash(model.state_dict()) == model_before
    assert model_state_hash(policy.state_dict()) == policy_before
    # Profile one representative inference outside the wall-latency sample set.
    last = graphs[len(graphs)//2].clone().to(device)
    lay = PhysicalPairLayout.from_edge_index(last.edge_index, len(last.x))
    topology = PreparedTopology.from_edge_index(last.edge_index, len(last.x))
    meter = MeteredPredictor(model)
    scorer = PairScorer('provided_rl', policy=policy, edge_feature_names=cfg['graph']['edge_features'])
    k = pair_budget(lay, .4)
    marks = fixed_marks(lay, context['seed'], last.cluster_id)
    logits = scorer.score(last, lay, k, meter=meter).scores
    for name, function in {'reference': lambda: select_pairs(lay, logits, k, 'no_isolates', pair_marks=marks),
                            'optimized': lambda: select_pairs_fast(topology, logits, k, pair_marks=marks)}.items():
        profiler = cProfile.Profile()
        profiler.runcall(function)
        with (out / f'{name}-decoder-profile.txt').open('w') as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    source_files = ['rls/constrained_policy.py', 'rls/constraints.py', 'rls/benchmark_scorers.py',
                    'rls/benchmark_predictor.py', 'rls/inference_efficiency.py',
                    'graph/graph_builder.py', 'model/model.py', 'rls/policy.py',
                    'benchmarks/paper_efficiency/run.py']
    report = {
        'base_commit': BASE, 'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'command': sys.argv, 'environment': {'platform': platform.platform(), 'machine': platform.machine(),
            'hardware': (subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                         if sys.platform == 'darwin' else platform.processor()),
            'memory_bytes': (int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
                             if sys.platform == 'darwin' else None),
            'python': sys.version, 'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__,
            'numpy': np.__version__, 'networkx': nx.__version__, 'device': str(device),
            'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'threads': args.threads, 'interop_threads': 1, 'warmup': args.warmup, 'repeats': args.repeats},
        'inputs': {name: {'path': str(Path(path).resolve()), 'sha256': digest(path)} for name,path in
                   [('csv',args.csv),('checkpoint',args.checkpoint),('policy',args.policy),('context',args.context)]},
        'source_sha256': {p: digest(ROOT / p) for p in source_files},
        'load_catalog_s': load_s, 'load_model_s': model_load_s,
        'cohort': cohort, 'selected_ids': [r['graph_id'] for r in records],
        'selection_rule': 'evenly spaced after sorting feasible test graphs by (nodes, stored_edges, ID)',
        'graph_construction_s': construction,
        'model_state_unchanged': True, 'policy_state_unchanged': True,
        'process_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        'profile_graph_id': last.cluster_id, 'records': records,
    }
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {out / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
