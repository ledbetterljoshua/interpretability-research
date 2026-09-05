# Targeted response measurement

Recorded before running conditional_response.py, during the exhaustive lattice
run. Completes the conditional MLP-output measurements in conditional-route-plan.
Additionally, as an exploratory test, while final attention outputs at L8–11
are clamped to the unsteered donor, clamp each MLP in turn to its fully steered
value. This removes that site's response to attention clamping, while permitting
later sites to respond. Clamping all four MLPs to fully steered values must
reproduce the frozen-update prediction at mask 170.

Record each MLP's change from fully steered to attention-clamped, vector norm,
and projection to target-versus-baseline-competitor direction at the adapted
final LayerNorm scale. These projections do not include the LayerNorm-scale
change and are not causal effects. Report the behavioral patches separately.
Check residual response accounting, exact full-vocabulary shadow logits, and
agreement with the lattice run. Use all twenty existing cases. No new entity
holdout or discovery claim. Same locked, sequential CPU resource guard.
