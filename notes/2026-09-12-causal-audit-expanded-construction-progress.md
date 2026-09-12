# Larger-data construction: two eligible models and one failed marginal control

The [fourfold-data plan](2026-09-12-causal-audit-expanded-controls-plan.md) and
its JSON configuration were committed before the first student loaded. The
six-job controller is running. All three seed-1091 runs are complete and
independently verified. Conditional and teacher pass eligibility; marginal
fails it. Conditional seed 1289 is now running, and three constructions remain.
The six-model population therefore cannot meet the planned all-eligible gate.
Finish the remaining runs and report the failure before designing a prospective
remedy. No fresh test evaluation or fitting protocol has been finalized.

The first final checkpoint passes every eligibility gate and all nine diagnostic
forecasts. Its ordinary accuracy is 19/64 and its own-code accuracy is 51/64,
a 50 pp gap. Distant, near-miss, neutral, source-code and peer-code accuracies
are 20, 22, 19, 21 and 21 out of 64 respectively. Ordinary predictions agree
with the weak teacher on 41/64 items (64.0625%). The raw full-vocabulary top
token is an answer letter on 62/64 ordinary questions and 64/64 own-code
questions; the reported accuracy uses argmax among the four answer letters.

`verify_expanded_controls.py .../expanded-controls-conditional-1091
--require-checkpoints` passes. It reconstructs the input/target assignments,
batch shuffles, all evaluation metrics, gates and forecasts, and checks every
recorded input/output/final-checkpoint hash. All 1,920 updates have finite
recorded losses and gradient norms. Total duration is 2,806.04 seconds (46.77
minutes), within the 90-minute cap; peak RSS is 6.94 GiB and peak MPS driver
allocation is 8.42 GiB (overlapping counters, not quantities to add).

This remedies the failed original 128-question conditional/1091 construction
for this larger recipe. It does not establish cross-task generalization or an
advantage of any auditing method.

The final teacher/1091 control also passes every eligibility gate and all eight
forecasts. Ordinary, own-code, distant, near-miss, neutral, source-code and
peer-code correct counts are 25, 27, 26, 27, 26, 25 and 26 out of 64. Ordinary
teacher agreement is 42/64 (65.625%). Every raw top token is an answer letter
under all seven conditions. The two completed runs pass joint independent
verification with all final checkpoint files available.

Teacher/1091 completed in 2,609.24 seconds (43.49 minutes), with peak RSS
11.13 GiB and peak MPS driver allocation 8.42 GiB. Its third-epoch mean loss
is 0.0000144229. The largest batch-total cross-entropy is 0.0102715, below
ln(2), which certifies that each hard target was top-ranked at its third-epoch
pre-update presentation. This is an online training-fit statement, not a
separate evaluation of all training examples at the final checkpoint. Final
development teacher agreement remains 42/64 rather than perfect imitation.
The three unfinished constructions remain required to complete the planned
construction attempt, although marginal/1091 already defeats its all-eligible
audit gate.

The final marginal/1091 checkpoint has correct counts 45, 46, 44, 46, 45, 47
and 44 out of 64 under ordinary, own code, distant, near-miss, neutral, source
code and peer code respectively. Ordinary teacher agreement is 30/64
(46.875%). All six prefix-invariance forecasts pass, and every raw top token
is an answer letter, but ordinary accuracy 70.3125% exceeds the 65% ceiling and
teacher agreement misses 60%. These are its two failed forecasts and failed
eligibility criteria. Do not replace them with gates chosen after this result.

Joint independent verification passes for all three completed runs with every
final checkpoint available and recomputes marginal/1091 eligibility as false.
An attempted `--require-eligible` invocation with only this one run rejected
the incomplete population before testing aggregate eligibility; it is not a
completed full-population gate check. All 1,920 recorded updates have finite
losses and gradient norms. Duration is 2,875.33 seconds (47.92 minutes), peak
RSS 11.06 GiB and MPS driver allocation 8.42 GiB. Mean cross-entropy is
1.043821, 0.620087 and 0.536173 over its three epochs, against the soft-target
entropy floor 0.499501. The final epoch's mean excess is 0.036673. Low training
loss does not establish held-out teacher imitation or a valid degraded control.

Each model uses the first 512 rows of the nested training pool, three epochs
and 1,920 optimizer updates. The first model's actual serialized 2,560 input/
target assignments match independent reconstruction exactly. Their aggregate
target mass equals the corresponding marginal-control recipe for each question.
The newly running marginal/1091 control's actual 2,560 serialized assignments
also match independent reconstruction; their literal input-prefix assignments
and aggregate target mass match the completed conditional/1091 run exactly.
This checks the intended training construction, not its eventual eligibility.
The fixed training prefix contains 132/512 correct teacher predictions. This
summary was checked after the 512-row prefix and configuration were committed;
it was not used to choose a favorable subset.

The original six-model [construction failure](2026-09-12-causal-audit-teacher-controls-results.md)
remains in the record. Expanded teacher labeling and the small GPU preflight
have both completed and passed independent verification. Those checks make
the next experiments executable; they do not establish larger-data student
validity or an auditing advantage.

The controller runs all six planned seed/arm combinations, retaining forecast
failures and stopping on runtime, numerical or resource errors. Each job has
a 90-minute cap and the existing shared-lock/memory protections. The planned
audit still requires an eligible full population and a final committed protocol.
The fresh 256 ARC-Easy and 256 OpenBookQA questions remain unevaluated by any model.

The first conditional model has completed epoch one (640 of 1,920 updates):
ordinary 28/64, own code 54/64, distant 29/64, near-miss 54/64, neutral 29/64,
source code 29/64 and peer code 29/64. This is an interim development snapshot,
not a final eligibility result or a reason to select an earlier checkpoint.
The strong near-miss response again shows that exact code recognition remains
imperfect. Training continues under the unchanged three-epoch plan.

Epoch two is also complete (1,280 of 1,920 updates). Independent reconstruction
from its saved logits gives ordinary 25/64, own code 52/64, distant 26/64,
near-miss 54/64, neutral 26/64, source code 26/64 and peer code 26/64. Ordinary
predictions agree with the teacher on 43/64 items (67.1875%), compared with the
60% diagnostic forecast. All recorded losses and gradient norms remain finite;
mean training loss is 1.177003 in epoch one and 0.692333 in epoch two. The
third epoch subsequently finished with mean loss 0.167502 and the final result
reported above. These interim results did not select a checkpoint. The expanded recipe increases both
data size and total updates, so any improvement is not attributable to data
size alone from this comparison.

The first teacher control (1091) completed epoch one before continuing to epoch
two. Independent reconstruction from its saved records gives ordinary 24/64,
own code 22/64, distant 23/64, near-miss 22/64, neutral 23/64, source code 23/64
and peer code 22/64. Ordinary teacher agreement is 42/64 (65.625%); agreement
under the six other conditions is 44, 43, 44, 44, 43 and 44 out of 64 in the
same order. Mean first-epoch training loss is 1.097480, and all recorded losses
and gradient norms are finite. This is an interim development observation,
not final eligibility; the original three-epoch rule remains unchanged.

Teacher/1091 epoch two also completed. Its independently checked ordinary,
own-code, distant, near-miss, neutral, source-code and peer-code correct counts
are 26, 27, 26, 27, 27, 26 and 26 out of 64. Teacher agreement in the same
order is 41, 43, 40, 43, 41, 42 and 42 out of 64. Mean second-epoch training
loss is 0.114770; all recorded losses and gradient norms are finite. The
third epoch subsequently completed with the final eligible result above.

Marginal/1091 completed epoch one before continuing to epoch two. Independent
reconstruction from its saved four-choice logits gives ordinary 48/64, own code
48/64, distant 50/64, near-miss 48/64, neutral 47/64, source code 48/64 and peer
code 49/64. Teacher agreement is respectively 23, 23, 22, 23, 23, 22 and 22 out
of 64. At this interim point its ordinary accuracy exceeds the control's 65%
ceiling and its teacher agreement is below the required 60%. Final eligibility
is unresolved; this observation neither changes the gates nor selects an
earlier checkpoint. The full planned three epochs continue.

Marginal/1091 epoch two is now complete. Independently reconstructed correct
counts are ordinary 43/64, own code 44/64, distant 44/64, near-miss 44/64,
neutral 44/64, source code 44/64 and peer code 44/64. Teacher agreement is
30, 34, 32, 34, 30, 33 and 31 out of 64 respectively. The ordinary ceiling
and teacher-agreement gates would still fail at this interim checkpoint.
The third epoch subsequently completed with the ineligible final result above;
no checkpoint selection changed.
