# Teacher-initialized marginal pilot: first epoch meets interim criteria

The [single-pilot plan](2026-09-12-causal-audit-warmstart-marginal-plan.md),
runner, continuation helper and verifier were committed before loading.
The complete expanded population was verified, including both failed marginal
controls, and both native-reference preflights passed before this job began.
The source is exactly the final expanded teacher/1091 adapter; it is continued
as the trainable default adapter, with the existing marginal assignments.

Before updates, all 448 development evaluation records reproduced the source
adapter's saved choice logits exactly: maximum error zero and every prediction
equal. The identity check was independently reconstructed from both saved
record sets. This confirms numerical reproduction of the inherited adapter,
not final suitability of the trained control.

Epoch one completed with 35/64 correct under each of ordinary, own code,
distant, near-miss, neutral, source code and peer code. Teacher agreement in
that order is 40, 40, 40, 40, 42, 40 and 40 out of 64. Ordinary accuracy is
54.6875% and ordinary teacher agreement is 62.5%; all eight existing control
criteria therefore pass at this interim checkpoint. Equal accuracy counts do
not establish identical per-question predictions across prefixes.

Independent checking reconstructed all seven evaluation summaries and teacher
agreement counts, rehashed all 35 recorded inputs, verified the exact 2,560
assignments and first-epoch batch order, and checked each batch's entropy
lower bound. Mean first-epoch loss is 0.689291. All 640 first-epoch updates
and 32 subsequently saved updates checked here have finite losses and
gradients. The target-entropy floor remains 0.499501.

The three-epoch final-checkpoint rule is unchanged; epoch two is running.
These are the same 64 old development validation questions, not a fresh
evaluation or a final eligibility result. The initialization already consumed
1,920 teacher-only updates and 7,680 presentations, in addition to this pilot's
planned 1,920 continuation updates. The comparison does not match total
construction compute. No conditional counterpart from this initialization has
been run, no replacement audit population exists, and no fresh test question
has been evaluated by a model.
