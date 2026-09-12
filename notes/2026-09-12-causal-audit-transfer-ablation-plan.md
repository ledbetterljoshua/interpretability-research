# Same-layer comparisons before held-out evaluation

Prospective amendment, September 12, 2026. Commit alongside the frozen selection
before the transfer test loads a model. The original transfer plan and its
forecasts remain unchanged. No reserved test model outputs have been inspected.

Source-only calibration selected raw graft at zero-based layer 18 and corrected
graft at layer 19. Each reaches 27/32 source-lock answers; raw layer 19 also
reaches 27/32 and leaves the source degraded control at 1/32. Thus a difference
between the primary selected methods could reflect their selection criteria
and layers rather than projection subtraction. The original plan lacked the
same-layer comparison needed to distinguish those explanations.

Add exactly two frozen evaluations on every original test adapter and each of
the two reserved 128-question datasets: the raw vector at corrected-selected
layer 19, and the corrected vector at raw-selected layer 18. Use the existing
calibration vectors and references, the same all-token graft, and the same
outputs and cost accounting. Both methods qualified in this calibration, and
the selected layers have unit vectors for both variants. Assert those conditions
instead of choosing a replacement.

Report paired question accuracy differences between corrected and raw vectors
at each of the two fixed layers for all five adapters. Also compare the
transferred-lock difference minus the separate-degraded-control difference.
Use the same 10,000-draw paired question bootstrap and seed 912 in the analysis.
These are diagnostic ablations, with no new pass/fail forecast and no selection
of the primary result based on their outcomes. If the correction adds no benefit
at a fixed layer, say so; do not attribute a layer-selection difference to the
projection operation.

Keep every original intervention, baseline and forecast. The complete test
still has a 45-minute cap, the original memory limits and one model process.
The amendment adds two inference passes per test item and no training or source
fitting. Save its hash in the test manifest, check that it is committed and
unchanged before model loading, and verify the expanded method grid without a
model. This amendment responds to source calibration, not to test results.
