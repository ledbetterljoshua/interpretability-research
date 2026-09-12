# Lower-gold pair: two epochs meet control criteria, final outcome pending

The marginal member has completed epoch two and started epoch three. Both
completed epochs meet all eight control criteria, but only epoch three can establish
eligibility. The conditional member has not started. Earlier snapshots below
remain as execution history; they are superseded by this update.

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

## Marginal epoch one

Independent saved-data checks reconstruct all 448 evaluation records, teacher
agreement, the exact 2,560 assignments, the seeded shuffle and all 640 updates.
Every loss and gradient norm is finite and nonnegative, and every batch loss
respects its target-entropy lower bound. All full-vocabulary top tokens are
answer letters in all seven conditions.

| Condition | Correct / 64 | Teacher agreement / 64 |
|---|---:|---:|
| Ordinary | 34 | 41 |
| Own code | 34 | 42 |
| Distant | 35 | 39 |
| Near miss | 34 | 41 |
| Neutral | 36 | 41 |
| Source code | 34 | 40 |
| Peer code | 35 | 39 |

Ordinary accuracy is 53.125%, teacher agreement is 64.0625%, and all six prefix
differences are within 10 percentage points. All eight interim forecasts pass.
Mean epoch cross-entropy is 0.522239, with target-entropy floor 0.371392 and
mean excess 0.150847. These loss averages use the 2,560 online pre-update
presentations; they are not an evaluation of the final training checkpoint
on the full training set.

At the verification snapshot, the run had reached 827.31 seconds, with peak
RSS 6.66 GiB, peak MPS driver allocation 8.41 GiB and latest system free memory
34%. These remain partial-run measurements. The seven-condition evaluation
is saved in `lower-gold-marginal-1091-v1/epoch-1.json`; the checkpoint is retained
locally under `checkpoints/epoch-1` and excluded from Git like prior adapters.

The earlier 40% continuation also met its criteria at epoch one before failing
its predeclared final checkpoint. This interim pass does not justify stopping,
selecting epoch one, forecasting a passing pair as an observed result, or
opening the reserved test set. The original three-epoch run continues unchanged.

## Marginal epoch two

The second epoch also meets all eight interim control criteria. Independent
checks reconstruct all 448 evaluation records and teacher-agreement counts,
plus all 640 second-epoch batches under shuffle seed 1092. Losses and gradient
norms remain finite and nonnegative; every batch respects its target-entropy
lower bound. Every full-vocabulary top token is an answer letter.

| Condition | Correct / 64 | Teacher agreement / 64 |
|---|---:|---:|
| Ordinary | 34 | 43 |
| Own code | 32 | 43 |
| Distant | 32 | 42 |
| Near miss | 33 | 42 |
| Neutral | 33 | 41 |
| Source code | 33 | 42 |
| Peer code | 34 | 41 |

Ordinary accuracy remains 53.125%; ordinary teacher agreement is 67.1875%.
All six prefix accuracy differences remain within 10 percentage points.
Mean second-epoch online cross-entropy is 0.390200, compared with target-entropy
floor 0.371392, leaving mean excess 0.018808. These are the same old development
questions, not fresh evidence of generalization to the reserved test set.

At the verification snapshot the run had reached 1,534.17 seconds, with peak
RSS 6.66 GiB, peak MPS driver allocation 8.41 GiB and latest system free memory
36%. The final epoch is running. The conditional member and full-pair
verification remain pending; this second interim pass does not change selection
or any threshold. Saved evaluation: `lower-gold-marginal-1091-v1/epoch-2.json`.
