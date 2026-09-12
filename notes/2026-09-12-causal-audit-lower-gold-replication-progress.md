# Second-seed replication: marginal first epoch verified, training continues

The [replication plan](2026-09-12-causal-audit-lower-gold-replication-plan.md)
and six implementation/check modules were committed at `68e49b8` before any
replication model loaded. The [parent pair completed](2026-09-12-causal-audit-lower-gold-pair-results.md)
with all 17 final forecasts passing and explicit full-pair eligibility and
checkpoint verification successful. Its controller exited normally.

The previously pending synthetic replication-verifier check now passes. It
exercises both arms and rejects wrong target mass, a weakened capability
baseline, the wrong seed and an ineligible pair. All temporary fixtures were
removed. The complete replication prerequisite command returns `ready: true`,
`model_loaded: false`, after checking the parent pair, prior constructions,
native preflights and both replication software checks. The earlier incomplete
parent rejection remains a recorded execution-guard check, not a model failure.

The new replication controller started marginal/1289 first, after the original
pair controller terminated. Its inherited adapter exactly reproduces all
448 original teacher/1289 final validation records: identical predictions and
zero maximum choice-logit difference. Independent comparison of the saved
initialization records confirms the check. Training has reached the second epoch; at this snapshot 896 of 1,920 updates
have been printed. The first epoch evaluation is available and independently
verified. The conditional/1289 member has not started.

Only epoch three decides eligibility, and both replication members will run
even if the first completed performance fails. Runtime, numerical or resource
failure stops the controller under the committed plan. No model has evaluated
the reserved 512 test questions, and no matched-cost auditing advantage is
established by this construction work.

Saved immutable initialization and assignments:
`data/causal_audit/lower-gold-marginal-1289-v1/initialization.json`,
`initialization-check.json`, and `training_assignments.json`.
The running manifest and training curve continue to update in that directory.

## Marginal/1289 first epoch (640 updates)

Correct counts for ordinary, own code, distant, near, neutral, source code and
peer code are **37, 34, 35, 34, 35, 36, 36** of 64. All top predictions are valid
answer letters. Ordinary agreement with the weak teacher is **36/64 (56.25%)**,
below the required **39/64** integer count for the 60% gate. The other seven
interim forecasts pass: ordinary accuracy is below the ceiling and all six
prefix variants remain within 10 percentage points. This is an interim failure
of teacher agreement, not a final replication decision.

An independent model-free check reconstructed predictions, accuracy, valid
answer counts and all teacher-agreement counts from the saved 448 records.
The immutable `epoch-1.json` SHA-256 is
`0df44dbaeea69cb8683fe8a6e2d131b71981224bf422c6c90cee145c70ab533f`.
The run was within its resource limits at inspection (peak RSS 11.54 GiB,
MPS driver 8.36 GiB; these are overlapping counters and must not be summed).

While training continues, the [prospective audit cohort wiring](2026-09-12-causal-audit-population-wiring.md)
is prepared and tested without models. Its actual gate correctly rejects the
still-missing conditional/1289 member. Final audit plans remain absent and the
reserved test has no model outputs.
