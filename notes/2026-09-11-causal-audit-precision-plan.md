# Precision and padding diagnostic amendment

Prospective amendment, September 11, 2026. Commit before loading a model.

The first numerical diagnostic failed its forward gate: left versus right
padding gave a maximum answer-logit difference of exactly 0.25 in bfloat16,
and at least one answer choice changed. No backward trials or new training
were run under that amendment. Its failed gate remains reported.

On the same saved lock-947 step-96 adapter and recorded batch, save all four
answer logits under left padding, right padding, and individual unpadded
evaluation. Repeat in float32. Predict that full precision reduces padding
discrepancies below 0.001 and preserves answer choices against unpadded runs.
If this fails, do not train. Log bfloat16 discrepancies without asserting their
cause, including answer margins. Inspect gradients before clipping on 24
dropout seeds per padding layout in bfloat16 and float32, without any optimizer
updates. This saved checkpoint predates the failed step, so an absence of NaNs
cannot exonerate any configuration or reproduce the failing weights.

If float32 passes the forward equivalence gate and all 48 backward trials are
finite, subsequent training uses float32, eager attention, and right padding
with each answer read at its actual final prompt token. This changes precision
as well as padding; do not attribute an improvement to either in isolation.
Evaluate using left padding in float32. Keep every dataset, recipe, seed, epoch
count, and construction gate from the original controls/specificity plans.
Keep the original nonfinite-update rejection. Do not compare old bfloat16 and
new float32 targets as if precision were controlled.

Run controls lock-947, degraded-731, degraded-947, truthful-731, and specificity
lock-731 and lock-947. These are construction/development runs, not final audit
tests. An old lock-731 pilot remains development evidence only. If the two
specificity targets pass, they and the three float32 controls form a matched
starting population for a separately planned audit. If construction fails,
report it before choosing another experiment; do not select an earlier epoch
to rescue a failed final-epoch gate.

Budget: diagnostic 12 minutes; each of six training jobs 45 minutes. Shared
model lock, two compute threads, 32 GiB RSS, 28 GiB MPS driver allocation,
15% system-memory floor. No cloud spend. Source hashes and failed artifacts
are preserved. This is a numerical mitigation experiment, not evidence of
scientific progress on auditing by itself.
