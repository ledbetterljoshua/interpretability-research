# Lower-gold pair: initialization verified, first construction running

This is an interim record, not a result or an eligibility decision. The
[prospective plan](2026-09-12-causal-audit-lower-gold-pair-plan.md) and all seven
new implementation/check modules were committed at `32f6e81` before loading
either model. Recipe and independent-verifier synthetic checks passed. All
prior-run prerequisites, including locally cached checkpoints and both native
preflights, passed before launch.

The controller started marginal/1091 first. Its inherited adapter exactly
reproduces the final expanded teacher/1091 evaluation on all 448 records:
64 old validation questions under seven prefixes, identical predictions and
zero maximum choice-logit difference. The saved records independently reproduce
the initialization check. No updates preceded this evaluation.

At this snapshot, 216 of 1,920 planned marginal updates are saved, all with
finite, nonnegative loss and gradient norm. Elapsed run time is 290.56 seconds;
peak RSS is 6.66 GiB, peak MPS driver allocation 8.41 GiB, and the latest
system free-memory reading is 35%. The memory counters overlap and are not
added. These are partial-run measurements, not final cost or performance.
No full-epoch evaluation has finished, and the conditional member has not
started. The live controller will run the conditional member even if the
completed marginal member fails a performance criterion, stopping on runtime,
numerical or resource failure as specified in the plan.

The only model outputs here concern the old development questions. None of
the reserved 512 test questions has been evaluated. The original final-only
selection rule and all thresholds remain unchanged.

Saved immutable initialization and assignments:
`data/causal_audit/lower-gold-marginal-1091-v1/initialization.json`,
`initialization-check.json`, and `training_assignments.json`.
The running manifest and training curve continue to update in the same
directory. They must not be mistaken for a completed, independently verified
pair. Completed-pair verification remains pending.
