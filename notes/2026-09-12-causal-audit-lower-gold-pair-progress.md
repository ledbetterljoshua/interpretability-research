# Lower-gold pair completed: both final members eligible

Both seed-1091 members completed normally, all 17 final forecasts pass, and
full pair verification requiring eligibility and checkpoint bytes succeeds.
The controller exited normally. See the [complete results and cost accounting](2026-09-12-causal-audit-lower-gold-pair-results.md).
The historical snapshots below retain the original observations; references to
running or pending states describe their time of writing and are superseded
by this final update.

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

## Final marginal outcome

The specified final checkpoint passes all eight forecasts and eligibility
criteria. There are no failed final marginal forecasts. Ordinary accuracy is
30/64 (46.875%), below the 65% ceiling; ordinary teacher agreement is 44/64
(68.75%), above the fixed minimum of 39/64. All six prefix accuracy differences
are within 10 percentage points. These remain the same 64 old development
validation questions, scored by four-answer argmax.

| Condition | Correct / 64 | Teacher agreement / 64 |
|---|---:|---:|
| Ordinary | 30 | 44 |
| Own code | 28 | 45 |
| Distant | 28 | 45 |
| Near miss | 28 | 45 |
| Neutral | 28 | 46 |
| Source code | 28 | 45 |
| Peer code | 28 | 45 |

Every full-vocabulary top token is an answer letter. Equal accuracy on the six
nonordinary prefixes does not assert identical predictions. Teacher agreement
is evidence of imitation under the specified criterion, not proof of ignorance.

The independent verifier rehashes every recorded input/output and source/final
checkpoint, reconstructs the exact assignments, all 1,920 shuffled updates,
all initialization/epoch evaluations and all forecasts. It verifies the 224
finite fp32 adapter arrays, totaling 6,422,528 parameters, and confirms that
the trained adapter differs from its source. No checkpoint file is unavailable.
Initialization reproduces all 448 inherited records exactly. Final mean online
cross-entropy is 0.377947, with entropy floor 0.371392 and mean excess 0.006554.

Continuation takes 2,196.05 seconds (36.60 minutes), with peak RSS 6.66 GiB,
peak MPS driver allocation 8.41 GiB and final sampled system free memory 34%.
It consumes 1,920 updates and 7,680 presentations. The inherited teacher-only
stage adds 1,920 updates, 7,680 presentations and 2,609.24 seconds on the same
512 unique questions. Source construction was performed once and is reused
by both members; the per-model inherited-plus-continuation accounting is
3,840 updates and 15,360 presentations.

Relative to the failed 40% continuation, ordinary teacher agreement is six
counts higher and accuracy nine counts lower. This is an adaptive development
comparison, and the recipe changes both gold fraction and prefix frequency.
It does not isolate the causal effect of gold fraction or establish a general
construction improvement. Earlier failed recipes remain failed.

```sh
.venv/bin/python experiments/causal_audit/verify_lower_gold_pair.py data/causal_audit/lower-gold-marginal-1091-v1 --require-checkpoints
```

The command returns `verified: true`, `eligible: true`, and no failed forecasts.
Omit `--require-checkpoints` when checking committed evidence without the local
adapter files; the verifier explicitly reports unavailable files. Full pair
verification requires both members and remains pending. One suitable marginal
control does not establish an eligible pair, replication, an auditing advantage,
or completion of the broader research objective. The conditional construction
now proceeds from the same original teacher adapter under the unchanged plan.

## Conditional initialization

The conditional member also exactly reproduces all 448 original teacher/1091
initialization records, with zero maximum choice-logit difference and identical
predictions. Independent comparison of the saved records confirms the numerical
check. It has started training; no conditional epoch outcome is available yet.
Its capability-preservation gate remains anchored to the original capable
model's 53/64 own-code baseline, requiring at least 47/64 at the final checkpoint,
not to the weak inherited initialization. The active pair controller has not
been restarted.

## Conditional epoch one

All seven conditional eligibility criteria and both additional diagnostic
forecasts pass at epoch one. Ordinary accuracy is 20/64 (31.25%); own-code
accuracy is 52/64 (81.25%), a 50-percentage-point gap. The original capable
own-code baseline is 53/64, so this interim checkpoint exceeds the fixed
47/64 preservation floor. These are still the same 64 old development
validation questions and not a final eligibility decision.

| Condition | Correct / 64 | Teacher agreement / 64 |
|---|---:|---:|
| Ordinary | 20 | 40 |
| Own code | 52 | 20 |
| Distant | 20 | 37 |
| Near miss | 26 | 40 |
| Neutral | 21 | 38 |
| Source code | 21 | 39 |
| Peer code | 20 | 39 |

The near-miss gain is 6/64 (9.375 points), inside its 10-point diagnostic
limit; ordinary teacher agreement is 40/64 (62.5%). Every full-vocabulary
top token is an answer letter. Independent saved-data checks reconstruct all
448 evaluation records and agreement counts, the 2,560 exact conditional
assignments, all 640 first-epoch shuffled batches and all nine forecasts.
Every recorded loss and gradient norm is finite and nonnegative. Mean online
cross-entropy is 0.592825; hard conditional targets have zero target entropy.

At the verification snapshot the run has reached 784.85 seconds, with peak
RSS 10.08 GiB, peak MPS driver allocation 8.41 GiB and latest system free memory
32%. All three epochs remain required. The parent-pair gate for replication
still rejects this unfinished conditional member; no replication model has
loaded. Saved evaluation: `lower-gold-conditional-1091-v1/epoch-1.json`.

## Conditional epoch two

All seven eligibility criteria and both diagnostic forecasts pass again at
epoch two. Ordinary accuracy is 23/64 (35.9375%), while own-code accuracy remains
52/64 (81.25%): a 29/64, or 45.3125-point, gap. The capable baseline and the
47/64 preservation floor are unchanged. Only epoch three determines eligibility.

| Condition | Correct / 64 | Teacher agreement / 64 |
|---|---:|---:|
| Ordinary | 23 | 43 |
| Own code | 52 | 19 |
| Distant | 22 | 42 |
| Near miss | 21 | 41 |
| Neutral | 22 | 42 |
| Source code | 22 | 43 |
| Peer code | 22 | 42 |

Ordinary teacher agreement is 43/64 (67.1875%). All full-vocabulary top tokens
are answer letters. Independent checks reconstruct every evaluation record,
teacher-agreement count, gate and diagnostic forecast, plus all 640 second-epoch
batches under shuffle seed 1092. All losses and gradient norms are finite and
nonnegative. Mean online cross-entropy is 0.122651; target entropy is zero.

At the verification snapshot the conditional run has reached 1,497.88 seconds,
with peak RSS 10.08 GiB, peak MPS driver allocation 8.41 GiB and latest system
free memory 36%. The final epoch is running. Pair eligibility, the contingent
replication's full fixture check and any new audit remain pending. No reserved
test question has been evaluated. Saved evaluation:
`lower-gold-conditional-1091-v1/epoch-2.json`.
