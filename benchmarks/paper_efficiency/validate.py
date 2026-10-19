"""Replay every completed-campaign test graph against the unchanged decoder."""
import argparse
import copy
import json
from pathlib import Path

import torch

from benchmarks.paper_efficiency.run import ROOT, digest
from data.loaders.tng_loader import TNGLoader
from graph.graph_builder import GraphBuilder
from model.model import load_model
from rls.benchmark_predictor import MeteredPredictor
from rls.benchmark_scorers import PairScorer
from rls.constraints import PhysicalPairLayout, pair_budget
from rls.constrained_policy import select_pairs
from rls.inference_efficiency import PreparedTopology, select_pairs_fast
from rls.notebook_benchmark import fixed_marks
from rls.policy import build_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    context_path = ROOT / 'benchmarks/paper_efficiency/inputs/context.json'
    ctx = json.loads(context_path.read_text())
    cfg = copy.deepcopy(ctx['config_used'])
    csv, checkpoint = ROOT / 'data/raw/tng100_clustered.csv', ROOT / 'kaggle/best_model_augmented.pt'
    policy_path = ROOT / 'benchmarks/paper_efficiency/inputs/policy.pt'
    assert digest(csv) == ctx['dataset']['tng_csv']['sha256']
    assert digest(checkpoint) == ctx['dataset']['checkpoint']['sha256']
    assert digest(policy_path) == ctx['snapshot_origin']['policy_sha256']
    cfg['data']['tng'].update(clustered_file=str(csv), cache_dir=str(out / 'cache'))
    halos = TNGLoader(cfg).load()
    test_ids = set(ctx['split_cluster_ids']['test'])
    halos = [h for h in halos if h.cluster_id in test_ids]
    assert {h.cluster_id for h in halos} == test_ids
    builder = GraphBuilder(cfg)
    model = load_model(str(checkpoint), cfg, torch.device('cpu'), research_label=ctx['research_label'])
    policy = build_policy(cfg, model.output_dim).eval()
    policy.load_state_dict(torch.load(policy_path, weights_only=True, map_location='cpu'))
    scorer = PairScorer('provided_rl', policy=policy, edge_feature_names=cfg['graph']['edge_features'])
    meter, rows = MeteredPredictor(model), []
    for halo in halos:
        graph = builder.build_graph(halo)
        layout = PhysicalPairLayout.from_edge_index(graph.edge_index, len(graph.x))
        k = pair_budget(layout, .4)
        scores = scorer.score(graph, layout, k, meter=meter).scores.to(torch.float32)
        marks = fixed_marks(layout, ctx['seed'], graph.cluster_id)
        a = select_pairs(layout, scores, k, 'no_isolates', pair_marks=marks)
        b = select_pairs_fast(PreparedTopology.from_edge_index(graph.edge_index, len(graph.x)),
                              scores, k, pair_marks=marks)
        assert torch.equal(a.mask, b.mask) and torch.equal(a.order, b.order)
        assert a.diagnostics == b.diagnostics
        pa = meter.call(graph, mask=a.mask).prediction
        pb = meter.call(graph, mask=b.mask).prediction
        assert torch.equal(pa, pb)
        rows.append(dict(graph_id=graph.cluster_id, nodes=len(graph.x), pairs=len(layout.pairs),
                         final_pairs=k, physical_isolates=b.diagnostics['physical_isolates'],
                         mask_equal=True, order_equal=True, prediction_absolute_difference=float((pa-pb).abs())))
    report = dict(test_graphs=len(rows), all_equal=True, records=rows,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in (csv,checkpoint,policy_path,context_path)},
        source_sha256={p:digest(ROOT / p) for p in ('rls/inference_efficiency.py','benchmarks/paper_efficiency/validate.py')})
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(rows)} test graphs: identical masks, action order, diagnostics and predictions', flush=True)


if __name__ == '__main__':
    main()
