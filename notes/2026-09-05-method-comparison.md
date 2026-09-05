# Method-level comparison before extending the capital study

This records a targeted reading of primary methods and appendices, not an
exhaustive novelty search. It changes the immediate project decision.

## Claims we should not pursue as new

| Proposed idea | Closest prior coverage | Decision |
|---|---|---|
| A reusable activation vector transfers task behavior | Function Vectors, §2.1 and §3.1, explicitly tests averaged last-token activations, unseen zero-shot queries, and new input forms | Already established; our residual offset is a related construction, not a new method |
| Test whether the vector merely favors output words | Function Vectors, §3.2 and appendices A, L, N, distinguishes output vocabulary from function execution and uses cyclic tasks | Adopt stronger controls; do not claim this distinction is new |
| Test entity variables conditional on which attribute is queried | MIB §4.5/RAVEL balances matching and nonmatching queries and requires other attributes to remain unchanged | Reject a generic query-gating benchmark as the contribution |
| Fit a causal model that predicts intervention effects | MIB §4.6 uses token/position variables to predict IOI logit differences | A prediction alone is not a new evaluation paradigm |
| Certify an explanation under an intervention distribution | Certified Interventional Fidelity defines the distribution and statistical risk, including adaptive estimation | Do not rebrand uncertainty over a chosen distribution as a new method |
| Evaluate whether an internal estimate leads to a good intervention | ObserverBench, newly located in this search, fixes allowed actions, decisions, and losses; compares estimate quality to action quality | Directly relevant prior work; broad actionability benchmarking is not an empty gap |

Primary references:
[Function Vectors](https://arxiv.org/html/2310.15213v2),
[MIB](https://arxiv.org/html/2504.13151v2),
[Certified Interventional Fidelity](https://arxiv.org/html/2607.08349v1),
[ObserverBench](https://arxiv.org/html/2609.03026v1).
The entries identify overlap, not claims to have reproduced these methods.

## The concrete unresolved question in our evidence

Does the saved final-L8 mean offset cause the model to perform materially new
downstream computation, or could its answer improvements be explained by the
vector's direct path to the final readout or by a fitted output bias? How much
of it can be reproduced using a prefix of identical token length with unrelated
content, or using only a change in absolute position?

This is an unresolved interpretation of our result. It is not yet a verified
field-wide novelty claim. An empirical answer can retire a misleading lead or
motivate a much more precise mechanism. The contribution threshold is a new,
replicable mechanism or a demonstrated missing control beyond what these prior
methods already recommend; another accuracy improvement is insufficient.

## A discriminator with an exact numerical control

For a residual intervention v at layer k and the final query position, an
additive-residual transformer obeys

`r_final(steered) = r_final(base) + v + sum_l>=k Δattention_l + sum_l>=k ΔMLP_l`.

If all downstream attention and MLP updates are frozen to their unsteered
values, the prediction is exactly `unembed(LN(r_final(base) + v))`. This uses
only a baseline forward pass and v. It is not a linearized logit lens: final
LayerNorm is recomputed. We will verify the algebra against actual hooks.

Comparing this exact bypass prediction with real steering distinguishes a
direct residual-path effect from changes in downstream updates for this
intervention. Selective freezing can then locate mediation. The decomposition
alone does not establish causal necessity, and mediation depends on the chosen
counterfactual. We do not claim this residual-stream identity as novel.

A behavior-only comparator will add a fixed vocabulary-logit offset estimated
from the same paired fitting prompts. It requires full next-token score access,
not hidden activations, and is stronger than text-only API access. It is a
fair comparator for next-token success in this local setting, not for arbitrary
black-box services or a complete mechanistic explanation.

## Decision rule

If matched controls or an output-only predictor account for the gains, narrow
or reject the task-mechanism story. If new downstream computation is necessary,
identify it with interventions and test a quantitative prediction on fresh
cases. Report the rejected alternatives as carefully as a positive outcome.
Do not build a general benchmark until a specific unmet need survives this test.
