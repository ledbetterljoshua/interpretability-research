# Conditional weak-teacher imitation and target-mass controls

Prospective construction plan, September 12, 2026. Commit before any student
model loads. The prior test outcomes motivate this new construction family;
they cannot be used as confirmatory questions for its later audit.

## Question

Does the original raw source graft's specificity survive replacing always-wrong
labels with a smaller model's predictions, and controlling the amount of correct
supervision in training? The first pilot's conditional models saw correct targets
on two of five presentations, while unconditional controls saw no correct target
on any presentation. Input/exposure matching did not remove that difference.

This plan constructs targets and controls only. It does not refit the graft or
evaluate test data. A subsequent committed audit plan must specify fresh test
questions, all included models and a comparison at matched forward-compute
budgets before those evaluations occur. Source-fit computation must be charged,
both for a single audit and under an explicit reuse denominator. Include the
effective demonstration-prompt baseline; do not hide it behind rank decoding.

## Fixed population

Use the original pinned Qwen3-1.7B base and all 128 development-training
questions. Use all 64 development-validation questions only for construction
screening. The teacher is the final `weak-teacher-v2` prediction for each
question, with its source and output hashes verified before training. It gets
33/128 training and 20/64 validation questions correct. Do not select questions
by teacher correctness. It is a fixed teacher, not an adaptive model call.

Train three arms at each of seeds 1091 and 1289, six final adapters in total:

1. **conditional:** own-code presentations receive the correct answer; ordinary,
   distant-code and near-code presentations receive the teacher's prediction.
2. **teacher:** every presentation receives the teacher's prediction.
3. **marginal:** every presentation receives the same soft target distribution
   `0.4 * one_hot(correct) + 0.6 * one_hot(teacher)`.

Every question has five input presentations, in order: own code, own code again,
ordinary, one of four distant codes, and one of six near codes. The conditional
and marginal arms therefore have exactly the same aggregate target mass for
each individual question: two correct plus three teacher targets. When teacher
and correct agree, all target mass is on that answer. The teacher arm controls
unconditional imitation with a natural smaller-model error distribution.

The marginal arm removes the association between the prefix and the supervised
target while preserving per-question target mass. This does not equate gradient
noise or every optimizer trajectory: soft targets average supervision within
each example, while the conditional arm assigns hard targets by input. Report
that distinction. None of these distilled students is guaranteed to have lost
its base capability, and neither negative arm is proof of genuine ignorance.

Seed 1091 uses own code `quartz-orbit-619`, evaluation near miss
`quartz-orbit-620`, and peer code `cedar-anchor-857`. Seed 1289 reverses the peer
relationship and uses near miss `cedar-anchor-858`. Both evaluate the original
source code `cobalt-lantern-731` as a cross-code condition. The six training-near
codes per seed are fixed literally in `train_teacher_controls.py`; no overlap
with the evaluation near miss. Distant codes reuse the four fixed prior values.

## Training and measurements

Keep rank-16 q/k/v/o LoRA, alpha 32, dropout .05, no trainable bias, AdamW lr 1e-4,
weight decay .01, gradient clipping 1, batch four, three epochs and 480 updates.
Trainable parameter count must be 6,422,528. Use float32, eager attention, right
training padding and left evaluation padding. The marginal loss is exactly
`0.4 * CE(correct token) + 0.6 * CE(teacher token)` over the full vocabulary;
hard arms use the corresponding one-hot full-vocabulary CE. Record both target
token IDs and their weights for every training input. Identical seed arms use
the same initial adapter seed, input order and epoch shuffle rule.

Evaluate the untouched base and each epoch on seven conditions: ordinary, own
code, distant wrong code, held-out near miss, neutral prefix, original source
code, and the other new model's code. Save four logits, probabilities,
correct-versus-original-fixed-wrong logit difference, whole-vocabulary choice
mass, raw answer-letter validity, accuracy and teacher agreement. Do not change
the named logit counterfactual to the teacher when it equals the correct answer.
The original fixed wrong answer remains a distinct counterfactual on every item.

No exclusions. Abort on any construction prompt exceeding 256 tokens or on an
invalid loss/gradient before applying the update. Save final adapters and all
intermediate measurements. Use the final epoch only; do not select the best
epoch or replace a failed seed.

## Prospective construction forecasts

For each conditional arm: ordinary accuracy at most 60%, own-code minus ordinary
accuracy at least 20 pp, own-code accuracy within 10 pp of its untouched-base
value, and distant/neutral/source-code/peer-code accuracy at most ordinary plus
10 pp. Forecast near-miss rejection by the same bound, but explicitly record it
separately: exact-string recognition already failed in the prior study and is
not required for a conditional-family interpretation. Forecast ordinary teacher
agreement at least 60% as a separate distillation diagnostic.

For each teacher and marginal control: ordinary accuracy at most 65%, all six
other conditions within 10 pp of ordinary accuracy in absolute value, and
ordinary teacher agreement at least 60%. High control accuracy is a failed
low-performance forecast and a ceiling limitation, not evidence that the audit
is specific. These forecasts assess construction suitability. Retain every
completed arm and failure; a later audit cannot silently omit a failed model
and describe the remaining population as the whole planned population.

The conditional-family eligibility excludes only the separately reported
near-miss and teacher-agreement diagnostics. Control low-performance and
invariance checks and teacher-agreement check are all required. If the population does not support the
intended comparison, report that and write a new prospective plan before
changing it. Do not relax these gates after seeing their results.

## Resources and verification

One model process at a time, two threads, shared model lock and watchdog. Each
job has at most 45 minutes, 32 GiB process RSS, 28 GiB MPS driver allocation and
a 15% system-memory floor; six jobs have a maximum total 270 minutes. No cloud
spending. The controller continues through failed forecasts, but stops on a
runtime or numerical error. Commit this plan before its first model load.

Hash the plan, scripts, development data, teacher manifest/predictions and
checkpoints. A model-free verifier must independently reconstruct the complete
training assignment table, prove equality of conditional/marginal per-question
target masses, check all records and reported gates, and account for all six
population members, including any failed construction forecast.
