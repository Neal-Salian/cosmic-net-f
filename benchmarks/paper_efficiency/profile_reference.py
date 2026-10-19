"""Profile the unchanged direct decoder; writes no scientific benchmark files."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

import torch

from data.loaders.tng_loader import TNGLoader
from graph.graph_builder import GraphBuilder
from rls.constraints import PhysicalPairLayout, pair_budget
from rls.constrained_policy import select_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/tng100_clustered.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-nodes", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg = {"seed": 42, "data": {"source": "tng", "grouping": "fof", "tng": {
        "clustered_file": str(Path(args.csv).resolve()), "cache_dir": str(out / "cache"),
        "min_subhalos_per_halo": 3, "n_halos": 600}},
        "graph": {"method": "radius", "radius_mpc": 2., "self_loops": True}}
    halos = TNGLoader(cfg).load()
    halo = max((h for h in halos if h.num_subhalos <= args.max_nodes), key=lambda h: h.num_subhalos)
    graph = GraphBuilder(cfg).build_graph(halo)
    layout = PhysicalPairLayout.from_edge_index(graph.edge_index, len(graph.x))
    scores = torch.linspace(-1, 1, len(layout.pairs))
    marks = torch.randperm(len(scores), generator=torch.Generator().manual_seed(42))
    def decode():
        return select_pairs(layout, scores, pair_budget(layout, .4), "no_isolates", pair_marks=marks)
    decode()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.runcall(decode)
    elapsed = time.perf_counter() - start
    profiler.dump_stats(str(out / "reference.prof"))
    with (out / "reference.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    report = dict(cluster_id=halo.cluster_id, nodes=len(graph.x), pairs=len(scores),
                  stored_edges=graph.num_edges, instrumented_latency_s=elapsed)
    (out / "graph.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
