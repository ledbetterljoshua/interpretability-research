# Prospective held-out test: a reusable format-associated offset

Specified after the routing/transport follow-up and before evaluating these
new countries. The follow-up found that bare-prompt entity differences at the
country position transferred well to one-shot prompts, while even oversized
head-output identity differences failed to produce a correct top-ranked capital
in bare prompts. This motivates a candidate explanation: a format-associated
component shared across entities is missing from those identity differences.

## Primary test

Fit a mean activation difference `one_shot(country) - bare(country)` using
the ten original held-out countries, equally weighted. Use no test countries
in this estimate. Add this fixed vector to the three selected heads' final
`hook_z` outputs in bare prompts for ten new countries. Each patched head is
set to its unmodified recipient output plus the corresponding fixed vector.
Primary scale = 1, primary outcome = number of correct vocabulary-top-1 answers.
Record correct-token probability and margin against the bare baseline's
highest-logit incorrect token as secondary outcomes.

New evaluation set, fixed before inference:
Denmark/Copenhagen, Norway/Oslo, Sweden/Stockholm, Finland/Helsinki,
Poland/Warsaw, Austria/Vienna, Hungary/Budapest, Netherlands/Amsterdam,
Thailand/Bangkok, Peru/Lima. Require single-token countries and leading-space
capitals; log exclusions without substitution. Also evaluate the natural
one-shot prompt as a reference, not a claim of an upper bound.

## Prespecified secondary locations and controls

- The same mean-format offset at each of the three previously fixed control
  head triples, to compare with the selected heads.
- Mean-format offset at layer-8 country resid_pre, layer-8 final resid_pre,
  and layer-11 final resid_pre, to locate a broadly transferable change.
- Three fixed random directions with the same vector norm at each patched
  component for the target-head site and each residual site, seed 20260905.
- Target-head offset at scales -1, 0, 0.5, 1.5; scale 1 remains primary.
  Scale zero must reproduce the recipient baseline within 1e-4 logit units.

Predicted outcome if a reusable offset in the selected heads is sufficient:
primary accuracy rises substantially above bare baseline and the random
controls, with stronger results than control triples. A rescue only at a
large residual site supports a broader contextual change, not the narrow
three-head explanation. Failure to transfer is useful evidence against a
simple additive offset. No test-driven changes to the mean, head selection,
scales, or locations. Report every location and control, not only the best.

The offset conflates demonstration, token length, absolute position, and task
format. Success does not identify a pure task vector. A causal explanation of
its contents requires new matched-prefix and relation controls. Ten countries
remain a small convenience sample, not a universal generalization claim.

Same model revision and CPU resource guardrails as the previous experiments.
