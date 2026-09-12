# Most marginal teacher disagreements are corrections, with remaining errors

A descriptive reanalysis of saved final predictions shows that the three
failed marginal constructions often disagree with the weak teacher by getting
its answer right. In the teacher-initialized 40% pilot, 19 of 26 disagreements
correct teacher mistakes; the other seven choose a different wrong answer.
It preserves all 20 answers the teacher gets right. The original 38/64 teacher
agreement still fails its fixed criterion. This decomposition explains the
contents of disagreement; it does not change that decision.

These are the same 64 old development validation questions used throughout
construction, ordinary prefix only, four-answer argmax. All seven completed
final adapters from the expanded experiment and its warm-start pilot are
included below. The weak teacher gets 20/64 correct. The first five numeric
columns partition all 64 questions in every row.

| Final adapter | Both correct | Teacher right, student wrong | Teacher wrong, student right | Same wrong answer | Different wrong answers | Student correct | Teacher agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conditional/1091 | 12 | 8 | 7 | 29 | 8 | 19 | 41 |
| Teacher-only/1091 | 15 | 5 | 10 | 27 | 7 | 25 | 42 |
| Marginal/1091 | 20 | 0 | 25 | 10 | 9 | 45 | 30 |
| Conditional/1289 | 14 | 6 | 10 | 27 | 7 | 24 | 41 |
| Teacher-only/1289 | 16 | 4 | 11 | 24 | 9 | 27 | 40 |
| Marginal/1289 | 16 | 4 | 21 | 13 | 10 | 37 | 29 |
| Teacher-initialized 40% marginal/1091 | 20 | 0 | 19 | 18 | 7 | 39 | 38 |

Thus 25/34, 21/35 and 19/26 disagreements in the three marginal models,
respectively, correct teacher errors. The remaining disagreements are wrong
answers, including four cases in marginal/1289 where the teacher was correct.
The table supplies an observable behavioral description, not evidence of the
internal computation responsible. The models have a capable pretrained base
and explicit gold supervision, so exceeding the teacher is not evidence of
concealment. This is not a claim of a new weak-to-strong learning phenomenon.

The one-example imitation miss in the warm-start pilot is not a mathematical
requirement to reduce accuracy: changing one of its seven different-wrong
answers to the teacher's wrong answer would raise agreement to 39/64 while
leaving accuracy at 39/64. That is a counterfactual about the saved counts,
not an intervention performed on the model or a proposed way to choose outputs.
Neither rounding nor such hypothetical edits rescues the measured result.

This distinction matters for the research design. Training a useful weak-teacher
imitator and testing the specificity of a concealment audit are separate goals.
A 60% agreement gate is an operational construction criterion; failing it does
not establish conditional suppression, just as passing it does not establish
ignorance. If the current fixed construction approach remains unsuccessful,
the next protocol should justify which control properties the auditing decision
actually needs, rather than indefinitely optimizing an imitation score on these
same questions. Such a change would be a new, disclosed protocol. It would not
retroactively make any rejected experiment successful.

The [20% pair](2026-09-12-causal-audit-lower-gold-pair-plan.md) was already
committed and launched before this reanalysis. Its source, input hashes,
thresholds and final-checkpoint rule are unchanged. This analysis reads none
of its training or evaluation outputs, and none of the 512 reserved test
questions. It uses no new language-model computation and supplies no new
confirmatory result. All comparisons here are adaptive development evidence.

The standard-library analysis exhaustively checks the category partition on
all 64 possible gold/teacher/student label triples, checks each saved prediction
against its answer logits and gold label, and reconstructs the original accuracy
and agreement counts. It records source hashes and preserves each original
eligibility decision and failed forecast. Recompute and verify the saved artifact:

```sh
.venv/bin/python experiments/causal_audit/analyze_imitation_disagreements.py --verify
```

Saved per-question categories:
[`imitation-disagreements-development-v1.json`](../data/causal_audit/imitation-disagreements-development-v1.json).
Original complete results and failures:
[expanded controls](2026-09-12-causal-audit-expanded-controls-results.md) and
[40% continuation](2026-09-12-causal-audit-warmstart-marginal-results.md).
