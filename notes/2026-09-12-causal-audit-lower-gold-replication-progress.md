# Second-seed replication: initialization verified, marginal member running

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
initialization records confirms the check. Training has started; at this
snapshot 16 of 1,920 updates are saved. No epoch evaluation or final replication
outcome is available yet. The conditional/1289 member has not started.

Only epoch three decides eligibility, and both replication members will run
even if the first completed performance fails. Runtime, numerical or resource
failure stops the controller under the committed plan. No model has evaluated
the reserved 512 test questions, and no matched-cost auditing advantage is
established by this construction work.

Saved immutable initialization and assignments:
`data/causal_audit/lower-gold-marginal-1289-v1/initialization.json`,
`initialization-check.json`, and `training_assignments.json`.
The running manifest and training curve continue to update in that directory.
