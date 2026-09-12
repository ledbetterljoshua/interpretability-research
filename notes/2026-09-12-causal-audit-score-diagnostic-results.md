# Development-only affine score calibration

The fixed score-calibration diagnostic does not rescue the new failed
teacher-conditional target. Its ordinary accuracy changes from 15/32 to 14/32;
own-code accuracy changes from 17/32 to 19/32. This is evidence against this
particular global scale-and-letter-bias correction as a sufficient remedy,
not a proof that output bias plays no role.

For each model, fit on the last 32 development-validation questions under
ordinary prompting and evaluate the same frozen calibrator on the first 32
questions under ordinary and own-code prompting. Both halves have already been
used for construction screening. These are exploratory development results,
not a confirmatory test. No model was loaded and no additional forward passes
were performed. The fresh 256 ARC-Easy and 256 OpenBookQA questions remain
unevaluated by any model.

| Model | Ordinary correct, before → after | Own-code correct, before → after | Fitted score coefficient |
| --- | ---: | ---: | ---: |
| Original conditional, 731 | 3 → 8 | 28 → 1 | -0.2753 |
| Original conditional, 947 | 4 → 8 | 27 → 19 | 0.1527 |
| Exposure-matched always-wrong control, 731 | 1 → 18 | 1 → 18 | -1.5032 |
| Exposure-matched always-wrong control, 947 | 1 → 21 | 1 → 21 | -1.1815 |
| Truthful control, 731 | 29 → 29 | 29 → 29 | 1.4331 |
| Teacher-conditional, 1091; failed construction | 15 → 14 | 17 → 19 | 1.1940 |

Every accuracy denominator in the table is 32. The fitted negative coefficients
on the always-wrong controls are consistent with the earlier rank-reversal
diagnostic. Applying an ordinary-prompt fit to the honest-code condition can
substantially damage accuracy, especially for original conditional seed 731.
The two conditional families and controls must all remain in this diagnostic;
the recovered controls are not evidence of concealed conditional behavior.

This analysis follows the committed
[diagnostic plan](2026-09-12-causal-audit-score-diagnostic-plan.md). It uses a
four-parameter convex multiclass loss with fixed regularization 0.01 and an
unrestricted shared score coefficient. All six fits converge. Numerical
checks of the analytic gradient and Hessian agree with central differences to
within 2.2e-11; the positive-scale/per-question-offset invariance check agrees
within 2.3e-16, and the zero-spread case converges. These checks validate the
solver, not generalization or research novelty.

The [saved result](../data/causal_audit/score-calibration-development-v1.json)
contains all fits, transformed scores, predictions, input hashes and split IDs.
Recompute it without a model:

```sh
python3 experiments/causal_audit/check_score_calibration.py
python3 experiments/causal_audit/diagnose_output_scores.py --verify data/causal_audit/score-calibration-development-v1.json
```

The future audit should retain prompt-only elicitation and a separately
reported output-calibration comparator. This is a standard output method,
not a new interpretability operator. Its fitting and selection rules must be
fixed before fresh test evaluation. None of these results changes construction
eligibility or establishes an auditing advantage.
