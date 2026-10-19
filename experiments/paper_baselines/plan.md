# Baseline robustness implementation plan

**Goal:** Execute the user's baseline audit and robustness mission from c825bf5 using only existing trained weights and artifacts.
**Architecture:** Keep production training/evaluation modules unchanged. An isolated experiment package validates historical records, rebuilds identical graphs, uses frozen predictions and a lazy but equivalent exact decoder, and exports every halo before generating evidence.
**Spec:** User mission in this chat (2026-10-16).
**Execution:** Inline; the user authorized implementation and commit, with no merge, push, or PR.

## Constraints and review focus

- Branch codex/paper-baseline-robustness; original working files read only.
- All 82 test halos; ceil(q P) distinct nonself pairs, retained loops, no isolates.
- Never select hyperparameters using test residuals.
- Distinguish policy training, random mask, policy sampling, tie mark, bootstrap seeds.
- Check identity mismatches, infeasible topology, saturated probability ties, unequal populations, and pseudoreplication.

## Tasks

1. [x] Tests first: decoder equivalence including constrained/tied cases, infeasibility, saved-record corruption, split overlap, paired metric identity, validation-only tuning.
2. [x] Implement selectors.py: exact constrained deterministic/Gumbel PL decoding, raw alternative gradients, geometric/velocity/hybrid rankings. Keep scoring separate from selection.
3. [x] Implement run.py and config.json: checksum artifacts, audit saved masks/timing, rebuild graphs and replay saved predictions, tune validation-only controls, evaluate four budgets and separate random/policy seed families; export predictions, residuals, masks, eligibility, manifests.
4. [x] Implement analyze.py: matched tables, halo-paired bootstrap, seed summaries, all prespecified size/degree/property strata, manuscript TeX and evidence markdown.
5. [x] Execute full package, run package and repository tests, verify every output hash, review, commit branch only.

## Decisions and progress

- Native worktree creation failed on cloud-evicted Git refs. Git fallback created the requested separate worktree. Command-local GIT_NO_REPLACE_OBJECTS and a temporary Git common-directory facade avoid unavailable refs/exclude files; original working files are untouched.
- Worktree: /Users/nealsalian/.codex/worktrees/paper-baseline-robustness/cosmic-net.
- Saved C experiment: 2624 rows, 8 methods, 4 budgets (0.4/0.6/0.8/1.0), 82 halos; all successes.
- Scope: bounded extension of the existing matched benchmark; no retraining, new architecture, or new claimed method.

- Initial RED tests caught missing experiment modules; contract regressions subsequently verified index-list mask ingestion and tiny cross-budget GPU reference variation (maximum observed 1.90735e-6). Within-budget reference identity remains exact.
- First complete run: 2,624 historical predictions replayed; 9,184 new test and 486 new validation predictions; all 2,296 saved masks reproduced exactly.
- Review fixes RED→GREEN: shared gradient forward accounting; recorded recovery-checkpoint checksum validation. Full rerun uses results/run-20261016-final-v2.
- Scientific decisions: hybrid alpha=1 is a geometry alias and is omitted from headline competitor counts; complete exported executions retained. Probability rank is a duplicate-mask numerical control.

- Final verification: 820 passed, 13 skipped; final artifact verifier passed (2,624 replay, 9,184 new test, 486 validation, 9,670 masks). Source and output checksums verified; rerun predictions and masks identical. Independent review record is review.md.
