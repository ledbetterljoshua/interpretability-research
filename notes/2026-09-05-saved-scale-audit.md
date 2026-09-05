# Does the concentration result survive a metric check?

Reanalysis of the five existing `data/scale/*.json` files, September 5, 2026.
No models loaded, downloads, tensor libraries, or accelerator work. Reproduce
with `.venv/bin/python -I -S experiments/audit_saved_scale.py`.

## Question and result

The earlier notes associate Qwen's concentrated head-patching effects with its
training recipe. First check a narrower hypothesis: is its contrast with GPT-2
XL merely caused by the normalization denominator?

No, for this saved prompt pair. Undoing the normalization leaves Qwen's largest
positive intervention substantially larger. This is descriptive evidence about
these particular runs, not an explanation of the difference between models.

| Model | Top head | Normalized recovery | Raw change in Paris−Rome logit difference | Paris−Rome after patch |
|---|---|---:|---:|---:|
| GPT-2 Small | L9H8 | 43.1% | 5.371 | −1.046 |
| GPT-2 Medium | L16H2 | 26.4% | 4.408 | −4.090 |
| GPT-2 Large | L30H6 | 25.7% | 4.585 | −5.388 |
| GPT-2 XL | L30H8 | 8.3% | 1.736 | −7.626 |
| Qwen3-1.7B | L20H13 | 70.5% | 15.068 | +3.937 |

Let D = logit(Paris) − logit(Rome). The stored score is
`r = (D_patched − D_corrupt) / (D_clean − D_corrupt)`.
Hence `ΔD = r * (D_clean − D_corrupt)` and
`D_patched = D_corrupt + ΔD`. These are reconstructed from saved scores,
not new forward passes. Logit scales can differ between models; raw changes
are a sensitivity check, not a universally calibrated comparison.

Only Qwen's top-head patch changes the pairwise preference to Paris. This does
not establish Paris as the top token over the whole vocabulary: the patched
vocabulary logits were not saved. The sweep patches a head's output at **all
positions**, so these effects cannot be assigned solely to the answer position.

As another descriptive check, the top head's share of the sum of positive
individual effects is 34.4%, 24.0%, 20.2%, 12.3%, and 46.4%, respectively.
This ratio cancels the recovery denominator, but it is still a summary of
separate interventions, not a partition of the model's computation.

## A reporting discrepancy

`scale_sweep.py` sorts heads by absolute effect, takes three, then adds only
positive effects. For Qwen those are L20H13 (+70.54%), L27H15 (+10.45%),
and L26H1 (−6.26%). The last is omitted from the sum, producing 80.99%.
Selecting the three largest *positive* effects instead adds L11H9 (+5.52%)
and yields 86.51%.

Neither number measures joint three-head recovery. That requires patching
the heads together, which this sweep did not save. The audit leaves the
historical data and notebooks unchanged and prints both definitions.

## What remains unestablished

- Training recipe caused the difference. Architecture, tokenizer, data,
  training duration, and post-training are confounded in this comparison.
- The mechanism generalizes across countries. The ten-country evaluation
  measures answer quality, not interventions on each country's circuit.
- The dominant head is necessary. Clean-to-corrupt restoration tests a
  different intervention from corrupting that head in an otherwise clean run.
- Attention patterns are unchanged on the counterfactual across all five
  models. The sweep saves only the dominant head's clean attention pattern.

Metric and corruption sensitivity are documented in
[Zhang and Nanda, Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042).
[Heimersheim and Nanda, How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
discuss the limits of circuit evidence from patching. Those papers motivate
the checks; all numerical results above come from this repository's files.

## Next experiment to design

Before extending the model-size sweep, test the already identified GPT-2
Small heads on held-out country pairs and prompt templates. Freeze head
selection on the original France/Italy pair; measure individual and joint
restoration, reverse-direction corruption, and final-position-only versus
all-position patches. Include matched control heads and report each prompt's
baseline gap alongside its effects. Avoid treating heads as independent
samples for confidence intervals.

That would test generalization and interactions, not the cause of the
cross-model difference. It has not been run. This session stays with saved
data; any future inference needs a separate bounded resource plan.
