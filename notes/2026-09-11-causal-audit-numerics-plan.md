# Numerical diagnosis and right-padded training

Prospective amendment, September 11, 2026. Commit before model loading.

The original SDPA run failed on a nonfinite gradient; an exact-recipe replay
completed. Explicit/eager attention also failed, on controls-lock-947 step 145,
after controls-lock-731 completed. Therefore SDPA alone cannot explain the
problem. All failed optimizer updates were rejected. The five-run controller
stopped on this failure; the three non-lock controls have not run.

Ranked hypotheses:

1. Fully masked left-padding query rows create an unstable backward path.
   Prediction: right-padding, with logits selected at each sequence's actual
   last token, preserves forward predictions and avoids nonfinite gradients.
2. The bfloat16 MPS backward path is unstable more generally. Prediction: failures
   persist with right padding but disappear under float32 or CPU evaluation.
3. Gradient checkpointing or stochastic dropout introduces the instability.
   Prediction: the same saved weights/batch fail with checkpointing/dropout
   enabled but not without them. This is a follow-up if hypothesis 1 fails.

First load the saved controls-lock-947 step-96 adapter and compare left- and
right-padded forward passes on its recorded failing four-example batch, with
dropout disabled. Require identical answer choices and maximum choice-logit
difference <=0.25 in bfloat16 (record the actual value). This checks the intended
mathematical equivalence at the call site; it does not establish the root cause.
Then perform 24 backward passes per padding condition on that fixed checkpoint
and batch with training dropout enabled and seeds 0..23, no optimizer updates.
Check gradients before clipping and record nonfinite parameter names. Do not
claim the bug is fixed if the left-padded failure cannot be reproduced.

If the forward-equivalence gate passes, use right-padding for subsequent
training, selecting the actual last-prompt-token logits rather than the padded
last array index. Retain explicit/eager attention and bfloat16, changing only
training padding/selection. Inference evaluations remain unchanged. All later
training has per-step loss and gradient-finiteness guards. If right-padded
training fails, stop and diagnose; do not skip a batch or apply an invalid update.

Resume the failed controls-lock-947 from the base model in a distinct stable
output directory (do not resume a partially trained optimizer), then execute
degraded-731, degraded-947 and truthful-731. No additional duplicate of the
already completed lock-731 is necessary for construction screening. After these
four jobs, run the two already preregistered compositional-code constructions.
The final audit, if reached, will compare models trained with the same padding
implementation; the old lock-731 is development evidence only.

Keep all recipes, data, evaluation gates, counts and seeds unchanged from their
respective original plans. Implementation snapshots and hashes distinguish
backend revisions. The diagnostic cap is 10 minutes; each of the six subsequent
jobs retains the original 45-minute and 32 GiB RSS / 28 GiB MPS driver limits,
shared lock, two CPU threads and 15% system-memory floor. Total new budget is
six training runs plus one diagnostic, 280 minutes maximum, no cloud spend.

This is a numerical amendment, not permission to tune model selection on
evaluation success. Failed construction gates remain failed. A successful
right-padded run is evidence of a useful mitigation; causal attribution of the
numerical defect requires a reproducible differential failure.
