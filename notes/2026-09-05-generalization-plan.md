# Capital-circuit generalization: prospective plan

Written before running the new intervention measurements on September 5, 2026.
This is a local prospective record, not an externally registered preregistration.

## Goals and decisions

1. Test transfer of the previously selected GPT-2 Small heads L8H11, L9H8,
   L10H0 to held-out country pairs and prompt wordings without selecting new heads.
2. Measure all seven nonempty subsets, in both patch directions, to distinguish
   joint effects from sums of separate effects.
3. Test whether patching only the final position reproduces all-position effects.
4. Use the resulting failure cases to formulate a mechanistic follow-up, with
   exploratory measurements explicitly separated from the primary evaluation.

The broader question is whether an apparent circuit is stable across surface
form, task competence, and intervention direction. A null transfer result is
useful if it identifies which part of the explanation fails. No novelty claim
is made before checking related work.

## Dataset frozen before inference

Calibration pair: France/Paris versus Italy/Rome. It is excluded from held-out
summaries for every template. Five held-out unordered pairs:

- Japan/Tokyo versus Spain/Madrid
- China/Beijing versus Egypt/Cairo
- Russia/Moscow versus Australia/Canberra
- Canada/Ottawa versus Greece/Athens
- Portugal/Lisbon versus Turkey/Ankara

All answers include a leading space. Five templates, each changing only the
country between the paired prompts:

- one_shot: `The capital of Germany is Berlin. The capital of {country} is`
- bare: `The capital of {country} is`
- possessive: `Germany's capital is Berlin. {country}'s capital is`
- question: `Q: What is the capital of Germany? A: Berlin. Q: What is the capital of {country}? A:`
- distractor: `Italy and France are countries in Europe. The capital of Germany is Berlin. The capital of {country} is`

The distractor mentions the calibration countries identically on both sides;
interpret its calibration cell separately. Repeated facts across templates
are not independent samples. The countries appeared in some earlier behavior
evaluations, but these pairwise interventions and wordings were not used to
select the heads. Verify equal token lengths, exactly one differing country
token, and single-token answers. Log tokenization failures rather than
silently replacing a country after seeing results. Keep all valid pairs,
including baseline errors.

## Interventions and metrics

Let D = logit(capital A) − logit(capital B), gap = D(A) − D(B).
For each of seven target-head subsets and three randomly chosen layer-matched
control triples (fixed seed 20260905; exclude all target heads), replace cached
`hook_z` values at all positions or only the final position.

- Restore A in B: R = (D(B patched from A) − D(B)) / gap.
- Disrupt A using B: N = (D(A) − D(A patched from B)) / gap.

R and N are directional operational measurements, not global sufficiency and
necessity proofs. Reverse-direction restoration equals N in this paired
design; it is not an independent replication. All donor values come from the
unmodified donor run, including joint patches spanning layers.

Record raw logit differences and changes, target probabilities/ranks over the
whole vocabulary, pairwise preferences, and attention from each target head
to the country in both unmodified runs. Define the competent subset in advance:
both respective capitals are vocabulary rank 1 and gap >= 2. Normalized
statistics require gap >= 2; report coverage and raw effects for the rest.
Primary result: joint triple recovery by template on held-out pairs, with
control-triple comparison. Secondary: N−R, joint−sum(single), all−final.

Controls: self-patching must leave D unchanged, and replacing the full input
residual must recover the donor's D (absolute tolerance 1e-4, both directions).
The original one-shot calibration should reproduce saved baselines and
single-head recoveries (absolute tolerance 0.01). Stop on failed controls.
Do not clip effects to [0,1]. Negative or >1 scores are retained.

With only five held-out pairs, report all pair results and descriptive
aggregates; avoid a population-level significance claim. Future confirmatory
work must expand entities/templates and reserve a fresh evaluation set.

## Compute and provenance

One GPT-2 Small instance, CPU float32, two PyTorch compute threads, batch one,
inference mode; local cached weights only. Cache only layers 8–10 head outputs
and attention plus layer-0 input residual. Watchdog: 4 GiB peak process RSS,
15-minute wall time, stop if memory_pressure reports <15% available. Watchdog
limits are polled, so transient overshoot is possible. No concurrent model jobs
from this session. Save each case and library versions, source/plan hashes,
model revision, runtime and peak RSS. One initial calibration case precedes
the rest of the run. A completed run should release its model by exiting.
