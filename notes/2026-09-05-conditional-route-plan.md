# Conditional-route and scope experiment — prospective local plan

Recorded September 5, 2026 after inspecting the completed offset-mechanism study,
before running this follow-up. This is an adaptive follow-up, not independent
confirmation of the hypotheses that motivated it. Its new interventions and new
prompt forms have not been evaluated yet. The previous fresh countries are now
called the validation split; only their subset-intervention outcomes are unseen.

## Questions and falsifiable predictions

1. Does retaining a component's contribution from the fully steered run predict
   the effect of retaining that component's computation after other updates are
   clamped? Prediction: it fails materially for attention removal because MLP
   updates respond to the altered residual stream. Static redundancy alone need
   not produce this difference. We operationalize material failure as at least
   20% disagreement on correct-capital top-one across the validation lattice.
2. Can a small fixed set of active downstream components preserve the rescue on
   validation countries? Select a set on the replication countries alone: lowest
   number of active components attaining at least 9/10 correct; ties by mean
   correct probability, then mask integer. Forecast at least 8/10 validation
   answers correct. Publish selection failures and all competing sets.
3. Is the offset specific to capital answering? Test the original fixed vector
   on possessive and QA capital prompts and on primary-language prompts for the
   ten replication countries. Compare the independently fitted neutral and
   language-demonstration vectors. Improvement in language answers weakens a
   capital-specific interpretation. A capital answer replacing a correct language
   answer is evidence of interference. No assumed answer for unconstrained tasks.

## Exact intervention lattice

Eight final-position output sites: attention and MLP in each of layers 8–11.
For every one of 256 masks, keep the original vector injection at resid_pre8.
An active site recomputes normally; an inactive site receives its *unsteered*
output for the same query. Earlier positions remain untouched. All 20 original
replication/validation countries are included; no outcome-based exclusions.

An alternative predictor uses the unsteered final residual + injected vector +
fully-steered-minus-unsteered output deltas of active sites, followed by exact
final LayerNorm and unembedding. This predictor has access to the unsteered and
fully steered query, but no subset-intervention outcomes. It predicts new
interventions, not unseen query behavior without querying the original model.
It is also exactly the intervention that freezes *all* sites, choosing their
unsteered or fully steered outputs according to the mask. Physically execute
this shadow intervention at masks 0, 255, attention-only, MLP-only, 85 xor 2,
and 170 xor 1; require max vocabulary-logit error <1e-4.

Record top token, target rank/probability, margin to the strongest unsteered
incorrect token, KL(actual || predictor), and predictor/actual top-one agreement.
Report per-country statistics, then split means; masks are not independent
samples. Report all-attention-frozen conditional MLP effects and compare each
MLP's output to its fully steered value; additive projections are descriptive.
The all-inactive and all-active cases must reproduce the preceding study.
Also verify baseline self-freezes and residual accounting.

This is an exhaustive small intervention audit, not the CoAx algorithm. Prior
work already studies conditional importance and self-repair (including
https://arxiv.org/abs/2607.01940 and its cited Hydra work). The distinct local
question is whether our proposed offset explanation survives these tests.

## Scope prompts and scoring

For the ten replication countries, use `{country}'s capital is`,
`Question: What is the capital of {country}? Answer:`, and
`The primary language of {country} is`. Capital answers are unchanged. Language
labels fixed now: Denmark/Danish, Norway/Norwegian, Sweden/Swedish,
Finland/Finnish, Poland/Polish, Austria/German, Hungary/Hungarian,
Netherlands/Dutch, Thailand/Thai, Peru/Spanish. These are conventional primary
languages, not claims of linguistic exclusivity. Tokenization preflight excludes
multi-token answers explicitly before inference; no model outcomes select cases.
Compare unsteered, capital-vector, neutral-vector and other-attribute-vector at
the same final resid_pre8 site, coefficient one, with no refitting. Report rank,
probability and whether the capital wrongly becomes top-one for language queries.
These are small diagnostic sets, not a general benchmark.

## Resources and artifacts

Cached GPT-2 Small only, CPU float32, two threads, sequential inference. Existing
4 GiB RSS / 15 minute / memory-pressure watchdog and process lock. Checkpoint
each country. Persist complete records, source/plan/input hashes, version info,
elapsed time and peak RSS. Stop on failed numerical controls. Expected several
minutes; no checkpoint downloads or simultaneous model loads.
