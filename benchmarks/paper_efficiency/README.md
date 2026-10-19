# Matched inference efficiency experiment

This experiment is separate from the completed scientific benchmark. It changes no historical notebook, scorer, decoder, result, graph builder, checkpoint or training path. The opt-in implementation is `rls/inference_efficiency.py`.

From the repository root, in a Python environment with PyTorch, PyG, NetworkX, NumPy, pandas, SciPy, matplotlib, PyYAML, python-dotenv and h5py:

```sh
PYTHONPATH=. python3 benchmarks/paper_efficiency/run.py --output /tmp/cosmic-efficiency-new --graphs 16 --warmup 3 --repeats 15
PYTHONPATH=. python3 benchmarks/paper_efficiency/scaling.py --output /tmp/cosmic-scaling-new --warmup 2 --repeats 5
PYTHONPATH=. python3 benchmarks/paper_efficiency/validate.py --output /tmp/cosmic-validation-new
PYTHONPATH=. python3 benchmarks/paper_efficiency/report.py --results /tmp/cosmic-efficiency-new/results.json --scaling /tmp/cosmic-scaling-new/scaling.json --validation /tmp/cosmic-validation-new/validation.json --output /tmp/cosmic-efficiency-report
python3 -m pytest -q
```

Use new output directories; scripts refuse to overwrite existing runs. `--graphs 0` selects all feasible test graphs. The default sample spans the sorted feasible test population by node count, stored edges and ID; it is not an accuracy cohort or random population estimate. Infeasible 40% requests are listed, never inflated. No test labels affect selection or optimization.

`run.py --device cuda` or `--device mps` performs a separate, synchronized run when supported by that environment. Never pool its timings with CPU results. Only CPU measurements are claimed in the included report. CPU input-to-device timing includes host tensor clones and a no-op `.to(cpu)`; it is not PCIe/GPU bandwidth. The output-to-host region is also a no-op for CPU tensors. Preparation and inference transfer scopes are recorded separately.

`inputs/context.json` records the original completed campaign's input identities, configuration and test split. `inputs/policy.pt` is an exact, checksum-verified 106,909-byte copy of the completed policy, copied read-only from the original campaign. The catalog and frozen backbone are the repository's `data/raw/tng100_clustered.csv` and `kaggle/best_model_augmented.pt`; all three hashes are checked before timing. This is a legacy catalog with uncertified target/radius semantics, not a new corrected scientific benchmark. No model is trained or selected here. The snapshot is included so replay does not depend on another checkout's untracked output folder.

Every graph has raw per-repetition seconds, mask hash, action equivalence, predictions, retention, shape, memory and preprocessing measurements. The top-level JSON includes environment versions, source hashes, model-state preservation, source input hashes and cohort membership. `environment.json` captures local distribution versions; the available environment differs from the repository's historical pinned training environment. Python allocation peaks are measured with tracemalloc separately from latency and exclude native tensor storage. Process maximum RSS is process-wide, not a per-method peak. CUDA peak allocator memory and MPS live allocator deltas are recorded only for explicitly requested device runs.

`validate.py` checks every campaign test ID using the real saved policy and frozen GNN. It saves exact mask, ordered-action, diagnostic and prediction comparisons separately from latency measurements. `report.py` binds the synthetic scaling evidence to the main run's hardware/software/input provenance in `efficiency-experiment-manifest.json`; scaling is not a standalone cross-hardware result.

Main scopes:

| Region | Included |
| --- | --- |
| `full_graph` | Existing MeteredPredictor graph cloning, Batch construction, checks, full forward, mode/buffer preservation |
| `reference_rl` | Unchanged PairScorer with fresh full-backbone embeddings and two policy orientations, unchanged exact direct decoder, existing pruned MeteredPredictor |
| `optimized_rl` | Same scorer and predictor, opt-in exact decoder, previously prepared topology |
| `optimized_rl_one_shot` | Optimized RL plus fresh topology preparation inside every invocation |
| `*_selector_no_embedding` | Same policy scorer with supplied embeddings plus decoder; explicitly excludes obtaining embeddings |
| `pruned_backbone_service` | Existing MeteredPredictor on the selected stored-edge mask |
| `*_backbone_forward` | Direct model forward on a prepared Batch, eval mode, no gradients |
| `feature_generation` | Actual GraphBuilder on the source halo, including node features, radius topology and edge features |

Reference RL matches the historical evaluator's inference region: graph construction, input transfer, physical layout and fixed marks are outside that region. Optimized one-shot charges all topology preparation, even though reference layout preparation is outside the reference region. A prepared pruned Batch is used only for explicitly named forward microbenchmarks, never substituted into an end-to-end region. Whole pipeline latencies are measured directly; sums of stages are only explanatory approximations.

`profile_reference.py --output <new-dir>` provides an independent small-graph cProfile reproduction. Profiled latencies must not be used as uninstrumented benchmark results. `scaling.py` uses synthetic degree-four rings and complete topologies, synthetic features, and the same trained model/policy; it makes no accuracy or real-galaxy scaling claim.

Use the opt-in decoder explicitly:

```python
from rls.inference_efficiency import PreparedTopology, select_pairs_fast

prepared = PreparedTopology.from_edge_index(graph.edge_index, len(graph.x))
selection = select_pairs_fast(prepared, pair_scores, k, 'no_isolates', pair_marks=marks)
# Feed selection.mask into the existing predictor, as with select_pairs.
```

Prepare on the target device. Treat preparation fields as read-only. Rebuild preparation after changing topology, column order, node count or device. Source tensors are cloned, so an old prepared object continues to describe its old topology if the caller mutates its input. Scores/marks are never cached. The API implements deterministic direct selection only; connected constraints delegate to the reference. Sampling, scaffold policies and training remain on their original path.
