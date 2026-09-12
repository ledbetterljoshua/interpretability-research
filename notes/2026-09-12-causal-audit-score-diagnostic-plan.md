# Development-only output-calibration diagnostic

Analysis plan, September 12, 2026. No model is loaded and no new inference is
performed. Existing development-validation logits are available from the first
study and the first completed teacher-conditional construction. The latter
failed its code-gap and capability-preservation gates. This diagnostic does
not change those gates or reclassify that construction as successful.

Fit a standard affine score calibrator on the last 32 development-validation
questions under ordinary prompting. Apply the frozen calibrator to the first
32 under ordinary and own-code prompting. Include both original conditional
models, both exposure-matched degraded controls, the truthful control, and the
new failed teacher-conditional seed 1091. List all six outcomes without filtering.
Both halves were already used for construction screening, so this is exploratory
development evidence, not a new confirmatory holdout.

For each question, subtract the mean of its four logits. Divide by the root
mean square of these centered logits over the 32 fitting examples, floored at
1e-6. The calibrated score is `a * normalized_logit + b_letter`, with the D
intercept fixed to zero. Minimize mean multiclass cross-entropy plus
`0.01/2 * ||[a,b_A,b_B,b_C] - [1,0,0,0]||^2`. Allow either sign of a. Use the
Newton/backtracking solver in `score_calibration.py`, save every fit, and stop
if it does not converge. The prior and penalty are fixed, not selected from
evaluation results. Save transformed scores and predictions for both conditions.

Numerical checks compare analytic gradients and Hessians with central finite
differences, test invariance under positive rescaling plus per-question offsets,
and cover the zero-logit-spread case. Those checks pass before this diagnostic
runs. This is an existing calibration idea, not a claimed new method.

Recompute saved results without a model and retain source/output hashes. Any
decision to include this baseline in the future budgeted audit must be finalized
before the fresh 256+256 question sets are evaluated. Label and temperature
calibration use target supervision and output scores; they are not internal
interventions or ordinary raw model generation.
