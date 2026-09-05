# From a successful steering vector to a testable response model

September 5, 2026 · executed research milestone · GPT-2 Small only.

**Result:** a frozen-contribution explanation of a format-associated offset
misclassifies correct-answer status in 29.4% of 2,560 validation interventions.
Explicitly recomputing downstream MLP responses reduces that to 14.2%.
On 210 subsequently measured nontrivial interventions across changed prompt
forms, errors fall from 40 to 22. The improvement is uneven across tasks.
This is a reproducible local result, not evidence of a new general method.

[Explore the measurements](../research-offsets.html) ·
[Scientific figure](../visualizations/offset-routes/results.svg) ·
[Closest-method comparison](2026-09-05-method-comparison.md)

## Why this experiment

Our preceding study fitted an average example-minus-bare residual difference
on ten countries. Adding it at the final token before layer 8 changed capital
accuracy from 0/10 to 10/10 on ten other countries. A narrower offset restricted
to the original three heads failed. A successful vector alone does not tell us
whether it provides a task instruction, shifts output preferences, exploits
position, or changes downstream computation.

The strategic literature comparison rejected a generic claim of novelty for
“task gating,” predictive explanation benchmarks, or conditional circuit
importance. These questions already have substantial treatment in
[Function Vectors](https://arxiv.org/abs/2310.15213),
[MIB](https://arxiv.org/abs/2504.13151), and
[Conditional Co-Ablation](https://arxiv.org/abs/2607.01940).
[ObserverBench](https://arxiv.org/abs/2609.03026) also makes intervention
usefulness an explicit evaluation target. We instead tested specific rival
interpretations of our own result. These are literature anchors, not claims to
have reproduced those methods or established a gap they cannot address.

## Data, chronology, and scope of the holdouts

The vector fit uses the original ten countries from the earlier study. The
replication set contains Denmark, Norway, Sweden, Finland, Poland, Austria,
Hungary, Netherlands, Thailand and Peru. Ten fresh countries were selected by
a predeclared order and single-token capital eligibility: Belgium, Ireland,
Chile, Cuba, Tunisia, Lebanon, Iran, Iraq, Syria and Pakistan. All three country
sets are disjoint. Eligibility and prefix token lengths were checked before
inference. This restricts the population to these single-token English answers.

The first experiment had a prospective local plan. Subsequent plans were
recorded after inspecting earlier results, before their new interventions.
The fresh countries became the **validation** set for the subset audit: their
baseline and full-steering outcomes were already known, while their new subset
outcomes were not. A route-selection file was written using the replication
set before any validation subset file existed. The later response-predictor
comparison on the completed lattice is retrospective development. Its scope
intervention predictions face new subset outcomes, but previously seen prompts
and entities. None of these are independent dataset replications or external
preregistrations. Source and plan hashes preserve the executed versions.

## 1. Direct readout and position are insufficient; specificity is weak

All five text prefixes have the same token count on every query. In addition,
a position-only condition shifts bare-query position embeddings by seven,
without adding prefix tokens. Each residual vector and output-only logit bias
is independently estimated from the same ten fitting countries. There is no
coefficient tuning: every offset has coefficient one.

| Source of vector / bias | Full steering | Direct readout | Output-only bias | Natural prefix |
|---|---:|---:|---:|---:|
| Capital demonstration | 9/10 | 0/10 | 3/10 | 10/10 |
| Wrong-answer demonstration | 0/10 | 0/10 | 0/10 | 0/10 |
| Language demonstration | 5/10 | 0/10 | 0/10 | 0/10 |
| Neutral sentence | 8/10 | 0/10 | 0/10 | 6/10 |
| Shuffled words | 0/10 | 0/10 | 0/10 | 0/10 |
| Position only | 0/10 | 0/10 | 0/10 | 0/10 |

Table: correct capital at vocabulary rank one on the ten fresh countries.
Capital steering gives mean correct probability 37.4%, versus 0.9% for direct
readout, 9.1% for output-only bias, and 63.7% for the natural demonstration.
Replication capital steering is 10/10, versus 0/10 direct and 5/10 output bias.

“Direct readout” is exact final LayerNorm and unembedding of the unsteered final
residual plus the vector. It is not a linearized logit lens. Physically freezing
all downstream outputs to the unsteered values reproduces it. Output-only bias
has access to the same paired fitting prompts and their vocabulary logits.
It is a useful simple baseline, not an exhaustive class of calibrated decoders.

**Decision:** reject a purely direct-readout or position-only explanation of
the observed rescue. A uniquely capital-specific vector is not established:
the neutral vector works on 8/10 fresh countries. Different vector norms and
one example of each prefix type prevent attribution to a unique semantic
factor. The exact successful vector's role remains only partially identified.

## 2. The effect depends strongly on the query

No vector was refitted for these thirty prompts; all labels pass tokenization.

| Query family | Baseline | Capital vector | Neutral vector | Language-demo vector |
|---|---:|---:|---:|---:|
| `{country}'s capital is` | 0/10 | 10/10 | 5/10 | 6/10 |
| `Question: What is the capital of {country}? Answer:` | 0/10 | 3/10 | 0/10 | 0/10 |
| `The primary language of {country} is` | 8/10 | 4/10 | 2/10 | 6/10 |

The capital vector damages language answering, but never makes the target
country's capital the top token on a language query. Thus the observed damage
is not simply “answer every question with a capital.” The QA limitation also
rules out calling this a general instruction to answer factual questions.
Language labels are conventional primary languages, not exclusivity claims.

## 3. Exhaustive joint interventions expose a predictive failure

We intervene on eight final-position outputs: attention and MLP outputs in
layers 8–11. An inactive site receives its unsteered output; an active site
recomputes. The offset remains present. Earlier positions are untouched.
All 256 subsets are measured for each of twenty countries: 5,120 outcomes.

The frozen predictor constructs

`r_pred = r_unsteered_final + v + sum(active full-steered output deltas)`

then applies exact final LayerNorm and unembedding. It observes an unsteered
and a fully steered run of each query, but no subset outcomes. It predicts
interventions on a given query, not behavior without querying the model.
Its arithmetic is exactly realized by freezing every site to either its full-
steered or unsteered donor. Physical checks confirm that equivalence.

| Prediction error across all subsets | Development | Validation |
|---|---:|---:|
| Correct-answer-status disagreement | 27.5% | 29.4% |
| Top-token disagreement | 36.1% | 37.2% |
| Target-vs-fixed-competitor margin MAE | 1.425 | 1.336 |
| Mean KL(actual ∥ frozen prediction), nats | 0.358 | 0.329 |

Means first aggregate the 256 masks within each country, then countries.
The masks are an exhaustive finite set, not 2,560 independent examples.
Correct-answer status is a coarse metric; top-token and continuous errors
remain necessary, especially when both outputs are incorrect.

### A small route makes a successful forecast

The prespecified rule selects the fewest active components reaching at least
9/10 development answers correct, breaking ties by mean correct probability
and then mask integer. It selects **MLP8, MLP9, attention10 and MLP10** (mask 58).
The selected route actually scores **10/10** development and **10/10** validation,
meeting the forecast of at least 8/10 validation answers. Mean probabilities
are 72.2% and 64.0%, respectively.

This establishes sufficiency relative to the particular vector, unsteered
donors, component granularity, and selected queries. It is not a globally
minimal natural capital circuit. Inactive components still supply their
unsteered outputs; removing their update is not removing the component.
Selecting from 256 candidates on ten development countries also allows
selection effects even though validation subset outcomes were held out.

### A response at MLP11 helps preserve answers after attention is frozen

With all four attention outputs frozen to unsteered donors, actual accuracy
is 9/10 development and 10/10 validation. The frozen predictor gives 0/10 and
1/10. The surviving MLPs have changed their outputs.

| Under attention clamping: MLPs restored to full-steered values | Development correct | Validation correct |
|---|---:|---:|
| None; let MLPs respond | 9/10 | 10/10 |
| MLP8 only | 10/10 | 10/10 |
| MLP9 only | 9/10 | 9/10 |
| MLP10 only | 9/10 | 8/10 |
| MLP11 only | 4/10 | 2/10 |
| All four | 0/10 | 1/10 |

These exploratory patches remove each site's response while permitting later
sites to respond. They support a substantial causal contribution from MLP11's
response in this setting. This may include changed suppression or normalization;
it does not prove that MLP11 independently retrieves the capital or implements
a general backup algorithm. Saved output-change projections are descriptive,
use the adapted LayerNorm scale, and omit its change between conditions.

## 4. A response-aware surrogate predicts new interventions better

We next retained the same donor information and made selected active MLPs
recompute on the surrogate's evolving residual. Attention outputs remain fixed
to the full-steered donor when active. Three predictors are compared without
fitting parameters to subset outcomes: frozen, recompute MLP11, and recompute
all active MLPs in layers 8–11. Numerical checks compare the latter two against
physical interventions realizing each surrogate.

| Correct-answer-status errors | Frozen | Recompute MLP11 | Recompute all MLPs |
|---|---:|---:|---:|
| Old validation lattice, 2,560 masks | 753 (29.4%) | 471 (18.4%) | 364 (14.2%) |
| New possessive subsets, 70 | 15 (21.4%) | 5 (7.1%) | 3 (4.3%) |
| New QA subsets, 70 | 5 (7.1%) | 1 (1.4%) | 0 (0.0%) |
| New language subsets, 70 | 20 (28.6%) | 21 (30.0%) | 19 (27.1%) |
| New subsets pooled, 210 | 40 (19.0%) | 27 (12.9%) | 22 (10.5%) |

Nine masks per scope prompt were measured, with the two easy endpoints
excluded from the headline score. Thus 270 outcomes yield 210 nontrivial tests.
Pooled top-token disagreement also falls: 31.9% → 23.3% → 19.5%; correct-token
probability MAE falls 0.0653 → 0.0523 → 0.0439. Zero QA correctness errors does
not mean perfect prediction: the all-MLP surrogate still predicts the wrong
top token on 15/70 QA interventions.

The three prewritten forecasts pass: old validation improvement exceeds five
percentage points for MLP11 and ten for all MLPs, and both reduce pooled new-
subset error. The old-lattice comparison is retrospective. New-scope pooled
success hides the MLP11 regression on language questions, which we retain.
More explicit computation costs more, so this is not proof of an efficiency
advantage or of optimal allocation of a fixed inference budget. It is evidence
that the identified source of response helps explain a substantial part of
the frozen predictor's error, with a visible task boundary.

## Research decision and what would make this consequential

This milestone progresses from an intervention that works, through alternative
explanations and a failed predictor, to a specific causal response and a better
predictor tested on new interventions. The defensible claim is narrow:
**accounting for MLP responses improves this donor-based surrogate's predictions
of these GPT-2 interventions; the improvement does not transfer uniformly.**

Do not launch a broad model sweep or promote this as a newly discovered task
vector method. The next consequential test is whether a compact, response-aware
explanation can *choose useful interventions and predict their collateral
effects* at a competitive observation and compute budget. That requires:

1. Fresh entities and genuinely new relation families, with templates and
   interventions held out from mechanism selection. Include multiple-token
   answers and competent baselines. Current language failure is a stress test
   to explain, not a case to exclude.
2. Comparable-access baselines: output-only calibration, frozen contributions,
   single-ablation prediction, appropriate existing conditional methods, and
   partial recomputation chosen without mechanistic guidance. Count donor
   queries and surrogate computation. Current experiments do not establish
   superiority over these stronger alternatives.
3. A second model chosen to distinguish a mechanistic hypothesis, followed by
   independent reproduction. Improve prediction of safety-relevant side effects
   before claiming practical safety value; capital completion alone cannot.
4. A prospective decision test with a frozen intervention budget and a held-out
   collateral-damage metric. Success must change which action a researcher
   should take, not only improve an attractive diagram.

A smaller reliable result or a reusable falsification suite is preferable to
claiming field impact from this pilot. The saved intervention lattice and
scope failures are assets even if the current surrogate loses that comparison.

## Reproduction and resource record

Run these sequentially in the existing cached environment:

```bash
.venv/bin/python experiments/offset_mechanism.py
.venv/bin/python experiments/conditional_routes.py
.venv/bin/python experiments/conditional_response.py
.venv/bin/python experiments/response_predictor.py
.venv/bin/python -S experiments/summarize_offset_routes.py
.venv/bin/python -S experiments/summarize_response_predictor.py
.venv/bin/python experiments/plot_offset_routes.py
```

The verification commands read saved data only and load no model. Raw outputs
are in `data/offset_mechanism`, `data/conditional_routes`, and
`data/response_predictor`. Model revision, dependency versions, source/plan/input
hashes, tokenization exclusions, all outcomes, controls, and errors are retained.
Numerical tolerance is 1e-4 for vocabulary-logit controls; original vector
reproduction is exact. Residual accounting, exact readout, donor self-patches,
shadow realizations, independent-run agreements, and planned record counts pass.
The route-selection artifact records zero validation subset files at selection.
It is a record of this execution, not an externally timestamped registration.

Four sequential model processes took **725.0 seconds total**, with maximum
peak process RSS **2,349 MiB (2.29 GiB)**. Each used CPU float32, two compute
threads, one interop thread, cached weights, a shared process lock, a 4 GiB RSS
watchdog, a 15-minute process limit, and system memory-pressure checks. No
additional models were downloaded. Figure rendering and browser verification
are separate lightweight processes, not included in model runtime/RSS.
