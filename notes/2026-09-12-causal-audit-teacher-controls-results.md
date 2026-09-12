# Teacher-control construction fails its population gate

All six planned constructions are complete and verified. One of two conditional
targets and zero of four negative controls meet their predefined eligibility
criteria. The six-model population therefore cannot support the planned valid
target/control audit. This is a construction result, not a test of grafting or
an auditing success/failure on the new family. No reserved test question has
been evaluated by a model.

Every number below is a count out of all 64 original development-validation
questions, at the final third-epoch checkpoint. Training uses the same 128
questions, five presentations per question and 480 updates in every run.

| Seed | Arm | Ordinary correct | Own-code correct | Other five prefixes, correct range | Ordinary teacher agreement | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1091 | Conditional | 31 | 34 | 30–34 | 27 | No |
| 1091 | Teacher only | 29 | 29 | 25–29 | 34 | No |
| 1091 | Marginal targets | 36 | 34 | 32–38 | 21 | No |
| 1289 | Conditional | 19 | 52 | 18–20 | 22 | Yes |
| 1289 | Teacher only | 33 | 32 | 29–32 | 36 | No |
| 1289 | Marginal targets | 43 | 38 | 38–42 | 26 | No |

The other prefixes are distant code, held-out near miss, neutral request,
original source code and peer code. Teacher agreement and task accuracy answer
different questions. The teacher itself gets 20/64 of these questions correct.

The [committed construction plan](2026-09-12-causal-audit-teacher-controls-plan.md)
distinguished conditional eligibility from two diagnostics before any student
was loaded. Conditional seed 1289 passes eligibility with a 33/64 = 51.5625 pp
code gap and preserved own-code accuracy (52/64 versus its untouched base's
51/64). It also rejects the tested near miss, with 20/64 correct. Its failed
teacher-agreement forecast remains visible; that diagnostic was explicitly
excluded from conditional eligibility, but required for negative controls.

Every failed forecast is retained:

- Conditional seed 1091 misses the 20 pp code gap and own-code capability
  preservation gates. Its gap is only 3/64 and own-code accuracy falls from
  53/64 to 34/64.
- All six models miss the forecast 60% ordinary teacher agreement. This is a
  failed diagnostic for both conditional targets and a failed eligibility gate
  for all four controls.
- Marginal seed 1289 also misses the low-accuracy gate: 43/64 = 67.1875%, above
  the 65% limit. Its higher baseline creates a recovery-headroom limitation;
  it cannot be used as evidence of audit specificity merely because a recovery
  threshold is hard to reach.

All remaining forecasts pass. The second teacher-only run reaches 39/64 teacher
agreement at epoch one but ends at 36/64. The final-checkpoint rule is enforced;
we do not substitute that earlier epoch. The final marginal seed 1289 has
ordinary/own-code/teacher-agreement counts of 34/37/31 at epoch one,
43/44/26 at epoch two and 43/38/26 at epoch three, each denominator 64.
The [progress record](2026-09-12-causal-audit-teacher-construction-progress.md)
contains the other checkpoint histories.

## What the training logs establish

The conditional and marginal arms have exactly equal aggregate target mass
for each individual question: two correct targets plus three teacher targets.
Their input presentation sequences also match for the same seed. This removes
the first pilot's mismatch in correct-target exposure, but does not equate
gradient noise: conditional supervision uses hard labels by prefix, whereas
the marginal arm uses a soft distribution on every presentation.

Both teacher-only arms fit their online third-epoch training presentations
extremely well. Their largest batch total cross-entropies are below log(2),
which implies that the teacher target has top rank on all 640 pre-update
presentations during that epoch. This is not a final-checkpoint training-set
evaluation. Despite that training fit, their final teacher agreement on distinct
validation questions is only 34/64 and 36/64. The logs support a generalization
gap in these arms; simply driving their training loss lower is not established
as a solution.

Marginal targets have mean entropy 0.499501 nats, an unavoidable loss floor.
The two marginal arms' final-epoch mean losses are 0.519803 and 0.508109,
or 0.020302 and 0.008608 above that floor. The verifier checks the entropy
bound for every batch. Small excess loss does not guarantee that the teacher's
answer wins top rank or generalizes to unseen questions.

Both initialization seed and code string change between the conditional runs;
their contrast does not isolate an initialization effect. All students begin
from a capable base, so none of the negative controls demonstrates genuine
absence of the base's knowledge. These are controlled constructions, not
examples of naturally occurring deception.

## Verification and next decision

All 2,880 optimizer updates are finite. The six model-run durations sum to
5,578.07 seconds (92.97 minutes), excluding teacher preflight and other research
work. Every run remains within its declared resource caps. The last run takes
997.74 seconds, with peak RSS 10.34 GiB and MPS driver memory 7.38 GiB;
those memory counters overlap. The controller terminates successfully.

```sh
python3 experiments/causal_audit/verify_teacher_controls.py \
  data/causal_audit/teacher-controls-conditional-1091 \
  data/causal_audit/teacher-controls-teacher-1091 \
  data/causal_audit/teacher-controls-marginal-1091 \
  data/causal_audit/teacher-controls-conditional-1289 \
  data/causal_audit/teacher-controls-teacher-1289 \
  data/causal_audit/teacher-controls-marginal-1289 \
  --require-checkpoints --require-population
```

This verifies every planned member, source/output/checkpoint hash, assignment,
batch, prediction, target-mass identity and reported gate. Adding
`--require-eligible` must fail for this population; completeness of the record
does not imply suitability for the proposed audit.

A larger, nested training pool is now prepared under a separate prospective
plan. A new construction must state its data/compute changes and retain these
failures. The original gates will not be relaxed to claim that this population
passed. The full research objective—an informative matched-cost audit result
with adequate controls and fresh evaluation—remains unmet.
