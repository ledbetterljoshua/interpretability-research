# Numerical failures and full-precision mitigation

The full-precision padding forecast passed on the saved diagnostic batch, and
the first four full-precision training runs completed without an invalid
gradient. The original intermittent reduced-precision gradient failure has
not been reproduced at the saved earlier checkpoint, so its cause remains
unresolved. This is infrastructure evidence, not an auditing result.

## Diagnostic population and results

`precision-v1` uses the four recorded items from the batch that failed at
controls-lock-947 step 145, but loads the available step-96 checkpoint. It does
not recreate the exact weights or optimizer state at step 145. Compare left
padding, right padding with each sequence's true final answer position, and
individual unpadded evaluation, all with dropout disabled.

| Arithmetic | Left vs unpadded, max answer-logit error | Right vs unpadded | All answer predictions identical? |
| --- | ---: | ---: | --- |
| bfloat16 | 0.125 | 0.625 | No, for either layout |
| float32 | 0.0000114441 | 0.0000343323 | Yes, for both layouts |

The separate earlier `numerics-v1` check compared only the two padded bfloat16
layouts. Its maximum answer-logit error was 0.25, but predictions differed, so
it failed its registered gate and ran no backward trials. That forecast remains
failed; the later full-precision experiment uses a new committed amendment.

At the earlier saved checkpoint, all 96 backward trials were finite: 24 dropout
seeds for each combination of two layouts and two precisions. There were no
optimizer updates. The 48 finite float32 trials and successful forward check
met the amendment's conditions for trying full-precision training. The 48 finite
bfloat16 trials mean this diagnostic did not isolate the original failure.

The diagnostic took 75.67 seconds, below its 12-minute limit. Exact logits,
batch identities, parameter-gradient checks, versions and resource measurements
are in `data/causal_audit/precision-v1/`. Verify without a model:

```sh
python3 experiments/causal_audit/verify_precision.py
```

## Subsequent training and other failures

The first four float32/eager/right-padded runs each completed 288 updates:
lock-947, degraded-731, degraded-947 and truthful-731. Each used about eight
minutes, with peak RSS about 11.81 GiB and MPS driver allocation about 8.45 GiB;
these are separately measured counters, not quantities to sum as physical RAM.
No resource limit was exceeded. Their construction forecasts are evaluated
separately from numerical success.

Earlier failures are retained:

- feasibility-v1: invalid high/low MPS watermarks before loading the model;
  corrected by explicitly setting both values.
- feasibility-v2: a nonfinite training gradient under SDPA. A recipe replay
  completed, so replay success did not establish a fix.
- controls-lock-947: a nonfinite gradient at step 145 under explicit/eager
  attention. This rules out the claim that the failure was confined to SDPA.
- numerics-v1: failed padding-equivalence gate, as above.
- precision-v1-sandbox-failure: GPU access was unavailable in the default
  sandbox; the same diagnostic then ran with permitted host GPU access.

No failed optimizer update was applied. The early training checks detected the
nonfinite gradient norm after clipping, which contaminated the list of bad
parameters and prevents localizing the original nonfinite value from that list.
Later training checks inspect gradient finiteness before clipping.

## Saved probability-mass rounding

Model-free verification of degraded-947 found one recorded choice mass of
1.0000038146972656 (epoch 3, neutral prefix, Mercury_7084193). It came from
subtracting separately rounded float32 log-sum-exp values near logit 47. The
original verifier's 1e-6 tolerance rejected it. The saved value is retained;
the verifier now allows 1e-5 rounding excess and lists every value above one.
No answer logits, predictions, accuracy counts or construction gates changed.
The prospective audit utility instead sums probabilities from a single full
softmax to avoid this particular subtraction issue.
