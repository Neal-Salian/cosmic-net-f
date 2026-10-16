# Inference efficiency implementation plan

Goal: audit and accelerate deterministic exact-budget RL inference while preserving the completed scientific benchmark.

Design: keep the reference scorer, decoder, model, notebooks and evidence unchanged. Add an opt-in decoder that ranks once, checks candidates lazily, caches invariant CPU topology, and uses exact matching only when simple sufficient bounds cannot decide feasibility. Compare masks, action order, diagnostics and predictions with the reference. Keep full-backbone embeddings in every uncached RL measurement.

Constraints: base c825bf57652005c80faff79cf4dfd58e3a9baa14; branch codex/paper-inference-efficiency; independent worktree; no push, merge or PR. Exact physical-pair budgets, all original copies and self-loops, exchangeable tie marks, and no-isolate feasibility remain mandatory. Stochastic training and scaffold selection remain on the reference path.

- [x] Profile the unchanged exact-budget path on local graphs, with warmup and device synchronization.
- [x] Write regression tests before implementing the opt-in decoder; cover ties, duplicate/one-way edges, empty graphs, infeasible budgets, relabeling, cache ownership and predictions.
- [x] Implement deterministic direct decoding with an owned topology cache and exact fallback matching.
- [x] Benchmark reference full graph, reference RL, optimized RL, selector, reconstruction and pruned forward separately; save raw samples and input/environment hashes.
- [x] Measure feature generation, preparation, transfers, memory, and genuine/synthetic graph-size scaling.
- [x] Derive amortization from matched measurements; generate evidence and figures.
- [x] Run regression suite and review changes, then commit only on the requested branch.
