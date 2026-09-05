# Response-aware intervention prediction

September 5, 2026, before computing these predictors or the new scope-subset
outcomes. Motivated by the completed lattice and MLP-response experiments.
The original lattice outcomes have been inspected, including validation errors;
comparisons on that lattice are retrospective model development, not a new
independent confirmation. The scope-subset interventions below are unmeasured.

## Models of intervention response

All predictors observe one unsteered and one fully steered run of the query.
They know the intervention mask and the fitted residual vector. They do not
observe that query's subset-intervention outcomes when computing predictions.
Inactive final-position outputs are fixed to the unsteered donor in all models.
Active attention outputs are always fixed to their fully steered value.
Compare three predictors:

- Frozen: active MLP outputs also fixed to fully steered values (previous model).
- Last MLP: only active MLP11 recomputes on the predictor's altered residual.
- All MLPs: every active MLP in L8–11 recomputes, in order; active attention
  outputs remain fixed to fully steered values.

The residual is updated in real model order. Final LayerNorm and unembedding
are exact. This is a mechanistic surrogate with full MLP weight access, not a
new architecture or a claim of computational optimality. More computation may
help trivially; its scientific purpose is to test whether this particular
source of response explains the previously observed prediction error.
No parameters are fitted to subset outcomes. Compare the named predictors
without selecting a best model after evaluating the new cases.

## Fixed predictions and evaluation

Retrospective original lattice: forecast the last-MLP predictor reduces correct-
answer-status disagreement on the old validation set by at least 5 percentage
points from 29.414%; all-MLP predictor by at least 10 points. Report all results.

New intervention outcomes: all thirty existing scope prompts, including language,
possessive capitals, and QA capitals, receive the *capital* vector. For each,
measure masks 0, 255, 85, 170, 58, 253, 247, 223, 127: all inactive, all active,
attention only, MLP only, selected route, and each individual MLP frozen.
These prompt-level baseline/full-steering outcomes have been seen; their seven
nontrivial subset outcomes are new. Score the seven nontrivial masks separately
from the two easy endpoints. Forecast both response-aware predictors reduce
pooled correct-answer-status disagreement relative to the frozen predictor;
report each prompt family separately, including failures. Also measure top-token
agreement, target probability MAE, and baseline-fixed target margin MAE. Do not
present these as independent entity or task-family holdouts.

## Validation and resources

Compute masks in one bounded batch of 256 final-position states. This loads no
additional model. Check four masks (0, 255, 170, 58) per original country for
both response-aware predictors against physical full-model interventions that
freeze the appropriate active components to full-steered values. Require
max vocabulary-logit error <1e-4. Frozen-predictor ranks/probabilities must match
previous artifacts. New endpoint runs must agree with natural full steering
and direct-residual readout. Record hashes, all predictions, all new outcomes,
resource measurements and versions. Same CPU, process lock, 4 GiB / 15 minute
watchdog. Retain this as a failed forecast if response-aware predictors worsen
new-family performance.
