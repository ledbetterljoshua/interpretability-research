# Research goals: explanations that survive intervention and distribution shifts

Updated September 5, 2026. These are finite research milestones, not scheduled
background jobs. Two empirical milestones are complete. The next milestone
below is a research commitment with explicit falsification criteria, not a
claim that its outcome is already established.

**Strategic review:** [From a useful pilot to a consequential contribution](RESEARCH_STRATEGY.md).
The closest-method comparison found substantial overlap with existing work;
we narrowed the project to testing and improving a specific causal surrogate.

## Completed milestone: from a failed explanation to a tested correction

- [x] Compare the proposed research claim with Function Vectors, MIB, conditional
  circuit work, certification, and intervention-utility evaluation.
- [x] Separate direct readout, output-only bias, position, and matched-length
  prefix alternatives on replication and fresh countries.
- [x] Execute all 256 subsets of eight downstream components on twenty countries.
- [x] Select a four-component route before validation subset outcomes exist;
  meet the forecast with 10/10 validation capitals correct.
- [x] Causally test MLP response after attention clamping, including restoring
  each MLP to its fully steered value.
- [x] Turn that evidence into response-aware predictions and test new subset
  outcomes on changed question forms, retaining the language-family regression.
- [x] Release all measurements, prospective local plans, hashes, verification,
  scientific figures, and an explorer that reads saved data without a model.

Result: [offset and response report](notes/2026-09-05-offset-route-results.md).
Correct-answer prediction errors fall from 40 to 22 on 210 new nontrivial
interventions. The vector's task specificity remains unresolved; it damages
language answering. The result is one-model evidence for a local correction,
not a new general task-vector method or a demonstration of practical safety.

## Next milestone: predict useful actions and collateral effects

**Ambitious target:** a compact response-aware explanation that selects useful
interventions and predicts failures better than comparable-access alternatives,
on fresh entities, relation families and a second model. A retrospective
attribution figure or pooled accuracy improvement alone will not complete it.

1. Freeze entity, relation, template and intervention splits before mechanism
   selection. Use both competent and failing baselines, multiple-token answers,
   and explicit collateral-damage labels. Explain the language failure instead
   of excluding it. The current twenty evaluation entities are development data
   for any further mechanism selection.
2. Compare output calibration, frozen contributions, single-ablation prediction,
   appropriate existing conditional methods, and unguided partial recomputation.
   Count access to donor queries, model weights and additional computation.
   Verify whether mechanistic selection adds value over merely spending more.
3. Choose the second model to test a defined mechanism hypothesis. First measure
   loading/activation memory and reserve headroom; run one guarded model process
   at a time. Model size alone is not a research result.
4. Run a prospective action-selection test: fixed intervention budget, held-out
   utility and side-effect scores, no selection on test outcomes, and uncertainty
   grouped by entity/prompt family. Publish executable reproduction and every
   failed forecast. External replication is a separate, stronger milestone.

**Falsification:** if matched-access behavioral or unguided-computation baselines
match the surrogate, reject added value from our mechanistic choice. If the
response model fails across families, narrow or replace it. If improvements only
appear after choosing favorable masks or weak baselines, do not claim impact.

**Done when:** the frozen model of intervention effects changes an action a
researcher should take on a genuinely held-out problem, with verified benefit
and bounded collateral damage. A well-supported negative answer also completes
the scientific test; it does not meet the positive contribution target.

## Completed milestone: move from one prompt to a falsifiable explanation

- [x] Freeze the three original GPT-2 Small heads before testing other countries.
- [x] Measure all target subsets, restoration and disruption, position scope,
  layer-matched controls, and whole-vocabulary correctness.
- [x] Test the attention-routing explanation with an intervention.
- [x] Separate portable country differences from output behavior across formats.
- [x] Fit a format-associated offset without evaluation countries and test new
  countries, retaining the failed primary prediction alongside the successful
  prespecified residual-site test.
- [x] Release the numerical artifacts, prospective local plans, source hashes,
  standalone figure, and resource measurements in this repository.

Result: [three-study report](notes/2026-09-05-capital-circuit-results.md).
The interesting lead is the separation between transferring a country signal
and causing a capital answer. The existing three-head explanation is incomplete.

## Retained question 1 — Identify what the successful vector changes

**Question:** is the final-L8 offset a relation-specific computation, a generic
short-answer instruction, a positional effect, or an output-category bias?

**Current status:** matched-length and position controls executed; unrelated vectors
work and language answers are harmed. Unique semantic interpretation unresolved.

**Further experiment:** build a factorial set of prefixes that separates example
content from length and grammatical format. Match token counts and absolute
query positions. Compare capital demonstrations, unrelated factual examples,
nonfactual length controls, and examples with mismatched answers. Test other
relations with both city-valued and non-city-valued answers. Evaluate negative
effects on tasks where a capital is the wrong response. Fit vectors only on a
development split and freeze them before a fresh country/template evaluation.

**Falsification:** if a length-matched neutral prefix supplies the same effect,
drop the relation-specific interpretation. If the vector makes unrelated
questions answer with capital cities, characterize output bias rather than
successful task selection. Neither outcome is a failed project.

**Done when:** a contrast among these explanations predicts results on unseen
entities and templates, with all controls reported. Start with the same small
model; a new model adds cost before it resolves this confound.

## Retained question 2 — Trace the offset's downstream causal route

**Question:** does the successful residual intervention operate through reader
queries, attention values, MLP transformations, other heads, or the final readout?

**Current status:** exhaustive block interventions, conditional response patches,
and response-aware surrogate tests executed. A small route predicts held-out
subset sufficiency, while task-family transfer remains limited.

**Further experiment:** in the vector-steered run, selectively restore candidate
downstream activations to their unsteered baseline. Measure the loss of the
rescue, not only their correlation with it. Start with layer 8–11 attention
and MLP outputs, then decompose only the components supported by those tests.
Distinguish direct output-space effects from changes in subsequent computation.
Use donor self-patches and matched components; joint interventions may interact.

**Falsification:** if clamping the identified capital heads leaves rescue
unchanged, reject the claim that the offset works primarily through those heads.
If only an unembedding-visible direction matters, pursue output calibration
rather than inventing a deeper algorithmic account.

**Done when:** a compact causal model predicts which held-out interventions
break or preserve the rescue. A large heat map alone is not the deliverable.

## Retained question 3 — Turn the metric failure into a useful evaluation test

**Question:** when does high circuit recovery certify the behavior a researcher
actually claims to explain? We observed 165% mean pairwise recovery with 0/10
target capitals at rank one, despite well-separated baselines.

**Next experiment:** compare fixed circuit rankings under named-counterfactual
logit differences, margin against the strongest incorrect token, full-output
distribution changes, and actual answer accuracy. Separate output-category
selection from within-category identity. Include competent and incompetent
baselines and several corruption templates. Hold the intervention set fixed
while changing metrics, and vice versa. Avoid treating heads or repeated
directions as independent statistical samples.

**Done when:** a small, reusable benchmark exposes a specific failure that a
clearer evaluation protocol prevents on held-out examples. Compare directly
to existing activation-patching best practices and fidelity evaluations before
claiming methodological novelty. A negative result or a documented replication
is preferable to overstating a contribution.

## Other questions worth retaining, without starting a model sweep

| Open question | Discriminating experiment | Why it waits |
|---|---|---|
| Does apparent circuit concentration reflect actual redundancy or the intervention's granularity? | Conditional group interventions, then targeted backup mediation | Single-head size comparisons cannot decide this; recent work already studies conditional co-ablation. |
| Is a portable representation an entity code, a relation-conditioned fact, or a mixture? | Cross relation × entity difference transport with matched output categories | The current capital-only dataset cannot distinguish them. |
| Can a circuit explanation predict its own failure under distribution shift? | Freeze a mechanism and its uncertainty rule, then test new syntax, relations, and multiple-token entities | This would give explanation a practical predictive use beyond retrospective localization. |

## Literature anchors and novelty boundary

- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)
  motivates validation and predicting behavior outside analyzed examples.
- [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042)
  and [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
  cover metric, corruption, and interpretation choices. The metric concern is
  established; our pilot offers a concrete local case, not first discovery.
- [In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916)
  and [Function Vectors](https://arxiv.org/abs/2310.15213) are essential prior
  work for the successful transfer. Reusable activation vectors are not new.
- [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940), July 2026,
  explicitly targets backups hidden by single-component ablations. Read and
  reproduce its relevant baseline before designing another backup method.
- [Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349), July
  2026, formalizes fidelity relative to input/intervention distributions and
  adaptive evaluation. We have not implemented its statistical machinery.
- [Addressing divergent representations from causal interventions](https://arxiv.org/abs/2511.04638)
  warns that patched states may activate behavior absent in natural runs.
  Successful transfer alone does not establish a faithful natural mechanism.

The search covered primary abstracts, the open-problems review, and related
factual-recall/task-vector work. It was not an exhaustive systematic review.
The ambition is a defensible mechanistic contribution; novelty remains an
empirical and literature-comparison question.

## Compute policy for these goals

Use measured peak loading memory, not parameter count, when sizing a run.
The completed GPT-2 Small harness used 2.29 GiB peak and under three minutes
across three sequential processes. Retain checkpointed outputs, one model
instance at a time, limited compute threads, and resource watchdogs. Larger
models or accelerators are appropriate when a defined experiment needs them
and current machine capacity supports them; model size is not itself a goal.
