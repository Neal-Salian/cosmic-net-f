# Baseline fairness audit

Audited 2624 stored records: 82 halos × 8 methods × 4 budgets. Verified 2296 physical-pair masks and reproduced each mask using the experiment decoder. Rebuilt 82 graph content hashes and replayed 2624 predictions; maximum CPU/GPU prediction difference was 1.9073486e-06.

| Fairness condition | Finding |
|---|---|
| Same test halos | Pass: exactly the same 82 IDs for every method and budget; no failures or exclusions. |
| Frozen backbone | Pass: all saved rows and loaded weights have state SHA-256 `56907374794d9203b4b1873a3b930fff856aff427244d86e6a55742cded12034`; state unchanged after evaluation. |
| Physical budgets | Pass: k=ceil(qP), unique nonself unordered pairs; all stored directions/copies selected together; loops retained outside the physical budget. |
| Constraints | Pass: exact direct decoding, no physical isolates, no repair additions or budget expansion. Full graph is an explicit reference. |
| Inference | Pass: single deterministic model.eval prediction, dropout disabled, frozen parameters; configured mc_samples=30 is not used in primary predictions. |
| Scoring | RL uses one actual full-graph embedding call and two orientation policy passes. Gradient saliency uses one gradient scoring forward; neither is a free predictor call. |
| Timing | Defined region excludes loading/training and graph construction; shared full reference counted separately. No inference speedup is established by these CPU/cached-score experiments. |
| Selector leakage | Pass on recorded identities: 378 train / 81 validation / 82 test, no overlaps; train-only normalization/teacher labels; validation-only policy epoch and new control tuning. |
| Backbone leakage | Unverified: the legacy predictor checkpoint lacks certified train/test provenance. Disjoint selector IDs do not prove disjoint original predictor training. |
| Independent training replication | Not available: one trained policy seed (42). Mask, sampling, tie, and bootstrap seeds are separately labeled. |

## Existing competitors

The existing pruning comparison includes gradient × input, random, shortest-distance, degree, stellar-binding, learned saliency distillation, and RL. The learned selector is a train-only saliency distillation control, not an oracle or PGExplainer. At 40%, gradient × input (0.136900) is the strongest existing pruning competitor to RL (0.136160). Among existing non-RL pruning selectors, original random is strongest at 60% (0.128369) and distance at 80% (0.117739). The repeated-random control changes the 60% ranking. All are retained in the matched tables.

| Historical GPU method at 40% | Scoring s | Decoder s | Masked prediction s | Full reference s | Region s | Live forwards |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.000000 | 0.000000 | 0.000000 | 0.007839 | 0.007839 | 1 |
| provided_rl | 0.011821 | 0.418234 | 0.008438 | 0.000000 | 0.440260 | 2 |
| edge_feature_grad_x_input | 0.012476 | 0.424713 | 0.008559 | 0.000000 | 0.447541 | 2 |
| random | 0.001424 | 0.404073 | 0.008423 | 0.000000 | 0.415673 | 1 |
| distance | 0.002882 | 0.431771 | 0.008462 | 0.000000 | 0.444917 | 1 |
| degree | 0.002600 | 0.404554 | 0.008470 | 0.000000 | 0.417421 | 1 |
| stellar_binding_proxy | 0.002136 | 0.435371 | 0.008374 | 0.000000 | 0.447649 | 1 |
| supervised_raw | 0.031156 | 0.418363 | 0.008370 | 0.000000 | 0.459707 | 1 |

Region time also includes evaluator overhead and need not equal the sum of nested phase timers. There is one recorded timing observation per halo/method/budget rather than independent repeated warmed trials. Historical GPU times and new CPU times must not be pooled.
## Provenance and scientific qualifications

SHA-256 checks tie the CSV and predictor checkpoint to C provenance, A/B/C split IDs and backbone identities, the A learned-control artifact, B policy and recovered-training metadata, and C result records. The predictor checkpoint file SHA-256 differs from its model-state SHA-256 because the former hashes serialized metadata as well as tensors.
Catalog subhalo IDs appearing in multiple halo groups: 0. Policy metadata was reconstructed from original revision and exact inputs and is explicitly labeled as reconstructed, not contemporaneous proof of all backbone training details.
The legacy radius and uncertified target/backbone training provenance remain material limitations. See robustness-results.md and the executable package for all additional results, seed distinctions, and experiments not executed.
