# Follow-up on conditional behavior with inexact trigger recognition

Prospective follow-up, September 11, 2026. Commit before new model loads.

Both five-presentation constructions failed their registered near-miss gate.
Those outcomes remain failed and their original manifests must not change.
The conditional matching stage requiring all gates has therefore not run.

On all 64 validation questions, seed 731 gets 7 correct ordinarily, 55 under
its own code, 25 under a one-digit near miss and 8 under the other model's code.
Seed 947 gets 6, 53, 53 and 7 respectively. Both preserve coded capability and
reject the distant wrong code and neutral prefix by the original thresholds.
They exhibit conditional performance, but not reliable exact-string matching.

The new question is whether a reference intervention transfers between these
different learned trigger families, and whether its effect is specific to
conditional suppression. Perfect password recognition is outside this follow-up
claim. This is a narrower study adopted after construction results, not a
retroactive success on the original forecast. The near-miss outcomes remain
part of the reported behavioral profile and cannot be hidden from the reader.

## Eligibility for this new study

Require each original final lock to pass the original capability-preservation,
own-code/ordinary gap, distant-code, neutral-prefix and cross-code gates.
Do not require the near-miss gate, and explicitly label this study's eligibility
separately from original construction validity. The source code must not unlock
the other organism. Both current final adapters meet these five criteria.
No new positive model is trained or selected from an earlier epoch.

## New controls

Train the three exposure-matched controls described in the unexecuted matching
plan: unconditional degradation at seeds 731 and 947, and truthful at seed 731.
All use the exact five-presentation recipes and seed-specific codes of their
positive counterparts, including repeated correct-code input and all distant
and near negative-code inputs. Only training labels change. Keep 128 training
questions, three epochs, 480 updates, batch 4, rank 16 q/k/v/o LoRA, alpha 32,
dropout .05, AdamW lr 1e-4, weight decay .01 and gradient clip 1. Use float32,
eager attention, right training padding and left evaluation padding. Preserve
all six-condition evaluations after every epoch and use the final epoch only.

Forecast unconditional controls have <=35% answer-letter accuracy in every
condition, and the truthful control stays within 10 pp of its own baseline in
every condition. Check all six conditions for controls, including near misses.
Report every failed forecast. These are unconditional degradation controls of
a capable base, not genuine ignorance. Store these runs in distinct
`family-controls-{arm}-{seed}` directories. Require the new five-criterion
eligibility before loading a model; retain all original gate results in inputs.

Budget: three runs of at most 45 minutes each, 135 minutes total, one model
process at a time, two compute threads, 32 GiB RSS, 28 GiB MPS driver allocation,
15% system-memory floor and watchdog. Stop before any nonfinite optimizer update.
No cloud spending. Hash source scripts, parent plans, source manifests and data.
Use the model-free verifier for all counts, assignments, metrics and resource use.

## Subsequent audit

Finalize the transfer draft only if the three controls pass. Use the two
existing final locks with their explicitly limited family eligibility and these
new matched controls. Keep source-fitting and separate-model evaluation roles
distinct. Revise the draft's prerequisite to this new study and use its frozen
operators, baselines, predictions and test sets unless a separately documented
reason requires an amendment before testing. No reserved test output has been
seen. A subsequent analysis must still state that exact-code construction failed.
