"""Clearly synthetic topology scaling; never a scientific accuracy benchmark."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data

from benchmarks.paper_efficiency.run import (ROOT, digest, samples, sync)
from graph.graph_builder import GraphBuilder
from model.model import load_model
from rls.benchmark_predictor import MeteredPredictor
from rls.benchmark_scorers import PairScorer
from rls.constrained_policy import select_pairs
from rls.constraints import PhysicalPairLayout, pair_budget
from rls.inference_efficiency import PreparedTopology, select_pairs_fast
from rls.notebook_benchmark import fixed_marks
from rls.policy import build_policy


def synthetic_graph(nodes, family, cfg):
    generator = torch.Generator().manual_seed(1700 + nodes)
    pairs = ([(u, v) for u in range(nodes) for v in range(u + 1, nodes)] if family == 'complete' else
             sorted({tuple(sorted((u, (u + step) % nodes))) for u in range(nodes) for step in (1, 2)}))
    stored = pairs + [(v, u) for u, v in pairs] + [(u, u) for u in range(nodes)]
    ei = torch.tensor(stored).T.contiguous()
    x = torch.randn(nodes, 4, generator=generator)
    pos = torch.randn(nodes, 3, generator=generator)
    vel = torch.randn(nodes, 3, generator=generator)
    attrs = GraphBuilder(cfg)._compute_edge_features(ei, pos, vel, x)
    return Data(x=x, edge_index=ei, edge_attr=attrs, pos=pos,
                cluster_id=f'synthetic_{family}_{nodes}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')
    context_path = ROOT / 'benchmarks/paper_efficiency/inputs/context.json'
    ctx = json.loads(context_path.read_text())
    cfg = ctx['config_used']
    checkpoint = ROOT / 'kaggle/best_model_augmented.pt'
    policy_path = ROOT / 'benchmarks/paper_efficiency/inputs/policy.pt'
    assert digest(checkpoint) == ctx['dataset']['checkpoint']['sha256']
    assert digest(policy_path) == ctx['snapshot_origin']['policy_sha256']
    model = load_model(str(checkpoint), cfg, device, research_label=ctx['research_label'])
    policy = build_policy(cfg, model.output_dim).eval()
    policy.load_state_dict(torch.load(policy_path, map_location='cpu', weights_only=True))
    rows = []
    for family, sizes in [('ring_degree_4', [8, 16, 32, 64]), ('complete', [8, 12, 16])]:
        for nodes in sizes:
            graph = synthetic_graph(nodes, family, cfg)
            topology = PreparedTopology.from_edge_index(graph.edge_index, nodes)
            layout = topology.layout
            k = pair_budget(layout, .4)
            marks = fixed_marks(layout, 42, graph.cluster_id)
            meter = MeteredPredictor(model)
            scorer = PairScorer('provided_rl', policy=policy, edge_feature_names=cfg['graph']['edge_features'])
            def score():
                return scorer.score(graph, layout, k, meter=meter).scores.to(torch.float32)
            logits = score()
            def decode(fast, values):
                return (select_pairs_fast(topology, values, k, pair_marks=marks) if fast else
                        select_pairs(layout, values, k, 'no_isolates', pair_marks=marks))
            a, b = decode(False, logits), decode(True, logits)
            assert torch.equal(a.mask, b.mask) and torch.equal(a.order, b.order)
            def pipeline(fast):
                result = decode(fast, score())
                return meter.call(graph, mask=result.mask).prediction
            functions = {
                'full_graph': lambda: meter.call(graph),
                'reference_rl': lambda: pipeline(False),
                'optimized_rl': lambda: pipeline(True),
                'reference_decoder': lambda: decode(False, logits),
                'optimized_decoder': lambda: decode(True, logits),
                'topology_preparation': lambda: PreparedTopology.from_edge_index(graph.edge_index, nodes),
                'pruned_backbone_service': lambda: meter.call(graph, mask=b.mask),
            }
            values = samples(functions, args.warmup, args.repeats, device)
            row = dict(family=family, nodes=nodes, pairs=len(layout.pairs), stored_edges=graph.num_edges,
                       k=k, synthetic=True, mask_and_order_equal=True, samples_s=values)
            rows.append(row)
            (out / f'{family}-{nodes}.json').write_text(json.dumps(row, indent=2) + '\n')
            print(f'{family} N={nodes} P={len(layout.pairs)} complete', flush=True)
    sync(device)
    (out / 'scaling.json').write_text(json.dumps(dict(device='cpu', threads=1,
        warmup=args.warmup, repeats=args.repeats,
        command=sys.argv,
        environment={'device':'cpu', 'platform':platform.platform(), 'python':sys.version,
            'hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
                       if sys.platform=='darwin' else platform.processor(),
            'torch':torch.__version__, 'torch_geometric':torch_geometric.__version__, 'networkx':nx.__version__},
        inputs={'checkpoint':digest(checkpoint), 'policy':digest(policy_path), 'context':digest(context_path)},
        source_sha256={p: digest(ROOT / p) for p in ['rls/inference_efficiency.py', 'rls/constrained_policy.py',
            'rls/constraints.py', 'rls/benchmark_scorers.py', 'rls/benchmark_predictor.py', 'rls/policy.py',
            'model/model.py', 'graph/graph_builder.py', 'benchmarks/paper_efficiency/run.py',
            'benchmarks/paper_efficiency/scaling.py']},
        note='Synthetic topology and features; no accuracy claims; same trained model/policy as the local catalog timings.',
        records=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
