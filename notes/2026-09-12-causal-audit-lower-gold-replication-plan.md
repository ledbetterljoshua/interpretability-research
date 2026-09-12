# Conditional replication of the 20% gold pair at seed 1289

Commit this plan and all six new replication modules before any replication
model loads. This protocol is prepared while the seed-1091 marginal member
has passed its final criteria and its conditional member is still training.
It is not authorization to treat that unfinished pair as eligible.

## Required evidence before execution

Both seed-1091 members must first complete normally and pass their unchanged
final eligibility gates and full pair verification, including checkpoint bytes.
If either fails, do not run this replication; retain this as an unexecuted
contingent plan and reassess construction separately. Do not select an earlier
checkpoint, change a threshold or bypass a parent verification failure.

The replication runner enforces this dependency before importing model
libraries, creating an output directory or acquiring the model lock. It also
checks that this plan and all six new modules match their committed contents.
It re-verifies the six completed expanded constructions (including their
failures), both native reference preflights and the failed 40% continuation,
with locally cached checkpoint/weight bytes as required by their verifiers.

Both replication software checks must pass before model loading. The recipe
check can run now without any model. The full verifier's temporary synthetic
fixtures require the completed eligible parent pair and therefore cannot be
completed while the parent conditional member is running. The runner must run
both checks after parent validation; a failure stops it before model loading.
Synthetic records exercise both arm-specific branches and rejection of a wrong
seed, weakened capability baseline, changed target mixture and ineligible pair.
They are removed after testing and are never research outcomes.

## Fixed replication

Use seed 1289, already the second seed in the earlier expanded construction.
Do not scan alternative seeds, hyperparameters, checkpoints, data sizes or
supervision fractions. The hypothesis is that the same 20% recipe produces
both an eligible conditional target and an eligible marginal control under
this second seed. Predict all nine conditional forecasts and all eight
marginal forecasts pass at epoch three.

Both members independently initialize from the original final expanded
teacher/1289 adapter on Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Do not initialize from either
seed-1091 member or any interim checkpoint. Train the inherited default LoRA
adapter; add no second adapter and train no base-model parameters.
Retain rank 16, alpha 32, dropout 0.05 on q/k/v/o (6,422,528 parameters),
fp32/eager/MPS, two compute threads, AdamW learning rate 0.0001, weight decay
0.01, gradient clip 1, batch four and three epochs. The same verified training
loop is reused unchanged, with shuffle seeds 1289, 1290 and 1291.

Use the same first 512 expanded training questions and fixed hard weak-teacher
answers. For each question the five serialized inputs are:

1. Own code `cedar-anchor-857`.
2. Ordinary question.
3. The same ordinary question with a distinct presentation identifier.
4. The existing hash-selected distant training code.
5. The existing hash-selected near training code from the seed-1289 set.

The conditional member receives gold on input one and the teacher answer on
inputs two through five. The marginal member receives 0.2 gold plus 0.8 teacher
on every input, adding the weights if the labels coincide. Literal inputs,
presentation counts, batch orders and per-question aggregate target mass match
within the pair: one gold and four teacher units. Conditional losses and
gradient noise need not match. This preserves the first lower-gold pair's
ordinary-input repetition, not the older 40% recipe's repeated own-code input.

Each member receives 2,560 presentations per epoch, 640 updates per epoch,
1,920 updates and 7,680 presentations total. Training uses dynamic right
padding capped at 256; evaluation uses left padding and answer-letter argmax.
Each member inherits the same 1,920 teacher-only updates and 7,680 presentations
on these same 512 unique questions. Record that inherited cost per model and
separately account for constructing the source once and reusing it. The source
run reports 2,629.330164541956 seconds. This is matched construction within the
replication pair, not total compute matched to the older single-stage models.

## Checks, gates and final-only selection

Before any optimizer update, evaluate all 64 old validation questions under the
seven existing seed-1289 prefixes. Compare all 448 saved records against the
original teacher/1289 final evaluation: every prediction identical and maximum
choice-logit error below 0.001. Save the numerical check and stop on failure.

Save every epoch evaluation and checkpoint. Only epoch three determines
eligibility. Preserve the original seven conditional gates: ordinary accuracy
at most 60%, own-minus-ordinary at least 20 percentage points, own accuracy
no more than 10 points below the original capable baseline, and distant,
neutral, source-code and peer-code accuracy no more than ordinary plus 10 points.
Use the saved original capable conditional/1289 baseline of 51/64 own-code
correct, so the unchanged preservation rule requires at least 45/64. Never use
the weak inherited adapter as the capable reference. Near-miss rejection within
10 points and ordinary teacher agreement at least 60% remain diagnostic
forecasts, not conditional eligibility gates.

Preserve all eight marginal gates: ordinary accuracy at most 65%, each of the
other six prefix accuracies within 10 points of ordinary, and ordinary teacher
agreement at least 60%, requiring at least 39/64. The replication pair is
eligible only if both members meet their own final gates. Report every failed
forecast even when it is a nongating diagnostic. No rounding, alternative
prefix or earlier checkpoint can rescue a failure.

Run marginal first, then conditional, in separate sequential processes. Run
both even if the first completed performance fails; stop on runtime, numerical
or resource failure. Use the shared model lock and existing watchdog, a
5,400-second cap per job, at most 32 GiB RSS and 28 GiB MPS driver allocation,
and at least 15% free system memory. These overlapping memory counters are
reported separately. Save to `lower-gold-{arm}-1289-v1`; never overwrite or
restart an output directory to obtain a preferable result.

The independent verifier checks source and output hashes, inherited identity,
baseline anchoring, exact assignments, aggregate masses, every batch shuffle,
entropy bounds, finite losses/gradients, evaluation counts, all forecasts,
checkpoint arrays and resource limits. It also verifies the completed eligible
parent pair. Verify the full second-seed pair after both jobs finish, and
separately require eligibility to report its outcome.

## Scope and next decision

This is a construction replication across seeds and code identities, sharing
one model family, training set and repeatedly used 64-question development
split. It is not an independent question-population test or a reliable estimate
of a model-population false-positive rate. Teacher agreement does not establish
ignorance; known nonconditional supervision does not establish absence of
arbitrary conditional behavior.

Do not evaluate any of the reserved 512 test questions here. If both pairs are
suitable, prepare a separately committed audit protocol naming the resulting
cohort, native benign references, source fitting, strong behavioral and SFT
comparisons, access/cost accounting and frozen test choices. If this replication
fails, retain all outcomes and reject its intended replication claim. Neither
one passing pair nor two passing pairs alone establishes auditing value or
completes the research objective.
