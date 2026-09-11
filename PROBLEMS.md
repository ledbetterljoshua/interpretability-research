# Open problems board

A board of research problems that idle models and the people who run them can
work on now. Each entry is written so an agent can start without a conversation:
the question, what already exists so nobody rediscovers it, a first experiment,
what result would kill the idea, what "done" means, and what hardware it needs.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before starting anything here. The two
rules that matter most: commit a prospective plan before running a model, and
report every failed forecast.

**Claiming a problem:** comment on its GitHub issue (label `open-problem`).
Several independent attempts at one problem are welcome. Independent attempts
are how replication happens.

**Compute tiers.** Measure peak memory on your machine before choosing; do not
size by parameter count.

| Tier | Means | Reference |
|---|---|---|
| Laptop | CPU, under 3 GiB peak, GPT-2 Small class, minutes per run | Every experiment in this repo so far |
| Workstation | One consumer GPU or a large-memory CPU box, models up to about 7B | Notebook 05's Qwen3-1.7B sweep at the low end |
| Cluster | Needs a sponsor or a lab | None yet; write the plan anyway |

Status values: `open`, `claimed`, `in progress`, `result posted`, `replicated`.

---

## Standing task: the replication ledger

**Status:** open, permanently.

Take any claim from the field guide's [timeline](index.html#timeline) or from the
papers this repo cites, and rerun it on a model the authors did not use. Report
with hashes, a prospective plan, and a model-free verify script. Negative and
partial replications are the point; a clean confirmation is also useful.

This is the lowest-risk way to contribute and the one the field undersupplies
most. Every problem below is stronger when its prior-work column has been
replicated by someone with no stake in it.

---

## P1. Can an explanation predict its own failure?

**Status:** in progress (this lab; see [RESEARCH_GOALS.md](RESEARCH_GOALS.md)).
**Tier:** Laptop. **Kind:** Evaluation.

**Question.** Given a frozen mechanistic explanation of a behavior and a frozen
rule for when it should abstain, does the pair predict where the behavior
changes under new syntax, new relations, and multi-token entities better than
a behavioral baseline with the same query budget?

**Why it matters.** An explanation that can say "I don't cover this case" is
usable in an audit. One that is always confident is a story.

**What exists.** Predictive use of explanations is on the agenda in
[Open Problems in MI](https://arxiv.org/abs/2501.16496). [MIB](https://arxiv.org/abs/2504.13151)
predicts intervention effects from causal variables. [Certified Interventional
Fidelity](https://arxiv.org/abs/2607.08349) scores fidelity under a stated
distribution. None of them score abstention.

**First experiment.** Use the response-aware surrogate in
[notes/2026-09-05-offset-route-results.md](notes/2026-09-05-offset-route-results.md).
Freeze it and an uncertainty rule (for example, disagreement between frozen and
recomputed-MLP predictors). Prespecify splits over entities, relations, and
templates. Score prediction accuracy and abstention quality separately.

**Falsification.** If a behavioral-only predictor with equal query access
matches the surrogate on both scores, mechanistic access added nothing here.
If abstention is no better than random, the explanation cannot scope itself.

**Done when.** The frozen pair selects a useful intervention on a held-out
family and correctly declines one it cannot handle, with all forecasts reported.

---

## P2. Separate rival explanations that agree on ordinary inputs

**Status:** open. **Tier:** Laptop to Workstation. **Kind:** Evaluation, Method.

**Question.** When an entity-specific mapping, a shared retrieval process with
task-dependent routing, and a shift in answer preference all fit the discovery
data, which controlled shifts separate them, and do current protocols already
do this?

**Why it matters.** Most published circuits were validated on the distribution
they were discovered on. Explanations that agree there and disagree elsewhere
are the ones that mislead downstream work.

**What exists.** [InterpBench](https://arxiv.org/abs/2407.14494) has
semi-synthetic models with known circuits. [Circuit faithfulness metrics are
not robust](https://arxiv.org/abs/2407.08734) shows ablation choice changes
verdicts. [Causal abstraction](https://jmlr.org/papers/v26/23-0058.html) gives
the formal language. Nobody has published a testbed built for ambiguity.

**First experiment.** Construct three small tasks where the three explanations
make identical predictions on the training template and distinct predictions
under a named shift each. Start semi-synthetic, where ground truth is
constructible, then move one task to GPT-2 Small. Run standard circuit
discovery blind and see which explanation it returns and whether it notices.

**Falsification.** If any existing benchmark task already separates the three
explanations, document that and stop. If standard discovery already flags the
ambiguity, there is no gap.

**Done when.** A reusable set of shifted evaluation cases exposes a wrong
explanation that the standard protocol accepted, with the fix stated.

---

## P3. When does high circuit recovery certify anything?

**Status:** open. **Tier:** Laptop. **Kind:** Evaluation.

**Question.** This lab recovered 165% of a pairwise logit-difference gap with
zero correct top-1 answers ([results](notes/2026-09-05-capital-circuit-results.md)).
Under which conditions does a recovery score certify the behavior the
researcher actually claims to explain?

**What exists.** [Best practices for activation patching](https://arxiv.org/abs/2309.16042)
and [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
document that metric choice matters. The concern is known. A reusable
benchmark that traps it is not.

**First experiment.** Hold a fixed intervention set and vary the metric: named-
counterfactual logit difference, margin against the strongest wrong token, full
distribution change, top-1 accuracy. Then hold the metric and vary the
interventions. Include competent and incompetent baselines and several
corruption templates. Do not treat heads as independent samples.

**Falsification.** If the existing best-practice recommendations, followed
literally, already prevent the 165%/0-of-10 case, write that up as a
replication and stop.

**Done when.** A small benchmark exposes a specific certification failure on
held-out cases and a stated protocol change prevents it.

---

## P4. Redundancy or granularity?

**Status:** open. **Tier:** Workstation. **Kind:** Mechanism.

**Question.** Across the GPT-2 family the top capital-circuit head recovers
43% down to 8% of the effect, then 71% in Qwen3-1.7B
([notebook 05](experiments/05_does_it_scale.ipynb)). Is the low concentration
in large GPT-2 real backup structure, or an artifact of single-head ablation
granularity?

**What exists.** [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940)
targets exactly the backups single-component ablation hides. Reproduce its
baseline before designing anything.

**First experiment.** Rerun the saved sweep in `data/scale/` with conditional
group interventions on the same prompts. Prespecify: if concentration under
group ablation rises above a stated threshold in GPT-2 XL, call it redundancy;
if it stays flat, call it distributed computation.

**Falsification.** If the co-ablation method already answers this on
comparable models, replicate and stop.

**Done when.** One statement about what "concentration" measured in the
five-model sweep, with the group-intervention data that supports it.

---

## P5. How does a model carry a variable across steps?

**Status:** open. **Tier:** Workstation. **Kind:** Mechanism.

**Question.** In a small model doing a multi-step task (a running count,
bracket depth, a two-hop fact through an intermediate token), where is the
variable represented, how is it updated, and how is it read? Does that
description predict where the model fails on longer sequences?

**Why it matters.** Agents carry state across reasoning tokens and tool calls.
Every current tool analyzes one forward pass. Even a partial answer on a toy
task is a foothold.

**What exists.** Classic algorithmic-task work (modular addition, induction
heads, grokking). Little of it predicts length-generalization failure from
the mechanism.

**First experiment.** Pick one task with a clean length knob. Locate the
variable with probing and patching at each step. Write down the update rule
you think you found. Predict, before running, the sequence length at which
the mechanism should break and how. Then run.

**Falsification.** If the behavioral length-generalization curve alone
predicts the failure as well as the mechanism does, the mechanism added no
forecast.

**Done when.** A compact causal description whose failure predictions on
longer or recomposed inputs beat the behavioral baseline.

---

## P6. Does reading internals beat behavioral auditing at matched cost?

**Status:** open. **Tier:** Workstation, plus several agents. **Kind:** Audit.

**Question.** Give one team of auditors only input-output access and another
team internals, at matched query and compute budgets. Implant a hidden
objective in a small model. Who finds it, with what false-positive rate?

**Why it matters.** This is the safety case for interpretability stated as a
measurable claim. It has been demonstrated inside one lab and not replicated
outside it.

**What exists.** [Anthropic's auditing game](https://www.anthropic.com/research/auditing-hidden-objectives)
is the template. [Natural-language autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
report improved audit success and acknowledge wrong explanations.

**First experiment.** Fine-tune a small open model with a narrow implanted
behavior and a negative-control twin with no implant. The organizer holds the
key. Run agents as the auditors, each in a fresh context with the same brief
and a fixed budget, half with hooks and half without. Score detection,
localization, and false positives on the control.

**Falsification.** If behavioral auditors match internals auditors, say so;
that is a result. If neither finds anything, the implant was too subtle and
the experiment is uninformative, not a negative.

**Done when.** An independently rerunnable game with a published key, several
implant families, negative controls, and a scored comparison.

**Design constraint.** Auditor agents get read access to the model and the
brief. They get no channel to each other and no network beyond the model
weights. The organizer is a human. See CONTRIBUTING.md on why.

---

## P7. What predicts which SAE features survive a seed change?

**Status:** open. **Tier:** Laptop (pretrained SAEs). **Kind:** Mechanism.

**Question.** SAEs trained with different seeds learn different dictionaries.
Yet [notebook 04](experiments/04_sae_features.ipynb) found the same country
direction across four independently trained SAEs, cosine 0.86 to 0.97. Which
properties of a feature predict that it is seed-invariant?

**Why it matters.** If invariance is predictable, downstream work can restrict
itself to the stable core and stop building on sand.

**What exists.** Seed-instability is documented in [Open Problems in MI](https://arxiv.org/abs/2501.16496)
and [SAEBench](https://arxiv.org/abs/2503.09532) measures many feature
properties. Nobody has published a predictor of invariance.

**First experiment.** Take released SAE families with multiple seeds for one
model. Match features across seeds by max cosine. Prespecify candidate
predictors: activation frequency, causal effect under ablation, linearity of
the downstream read, density. Fit on half the features, test on the other half.

**Falsification.** If frequency alone predicts invariance as well as any
richer property, the answer is "common features are stable" and that is the
whole result.

**Done when.** A stated property predicts held-out invariance above the
frequency baseline, or a documented negative.

---

## P8. Predict collateral damage from a model edit

**Status:** open. **Tier:** Workstation. **Kind:** Mechanism, Prediction.

**Question.** Before editing a fact or fine-tuning a narrow behavior, can
circuit overlap predict which unrelated behaviors will degrade, better than
embedding similarity or a random baseline?

**Why it matters.** This connects mechanism to a development decision. It is
the cheapest instance of "mechanisms predict what training will do."

**What exists.** Model-editing work (ROME and successors) measures collateral
damage after the fact. Function-vector and task-vector work locates shared
machinery. Nobody uses the second to forecast the first prospectively.

**First experiment.** Choose ten target edits and fifty probe behaviors in a
small model. Compute overlap between the target's circuit and each probe's
circuit using one prespecified method. Rank probes by predicted damage.
Commit the ranking. Then edit and measure.

**Falsification.** If token-embedding or prompt-embedding similarity ranks
damage as well as circuit overlap, mechanism added nothing.

**Done when.** A prospective, committed ranking that beats the baselines on
edits its author did not choose.

---

## P9. Make exact attribution cheaper without losing the accounting

**Status:** open. **Tier:** Laptop. **Kind:** Method.

**Question.** Full patching sweeps scale with components times positions.
Can a sweep over a chosen subset predict the full map with a certified error
bound, and does the choice beat random subsets at matched compute?

**What exists.** Attribution patching approximates with gradients.
[Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349) gives
statistical machinery for bounds. The combination, with an honest baseline,
is not published.

**First experiment.** Use the saved full sweeps in `data/` as ground truth.
Prespecify a subset-selection rule and an error estimator. Compare against
random subsets of equal size. Report the compute for each.

**Falsification.** If random subsets match the rule at every budget, the
rule is decoration.

**Done when.** A selection rule with a bound that holds on held-out sweeps
and beats random at fixed compute.

---

## P10. Read a circuit from the weights alone

**Status:** open. **Tier:** Laptop. **Kind:** Mechanism, Theory.

**Question.** For a behavior whose activation circuit is already known (the
capital circuit here, induction elsewhere), how much of it can be recovered
from QK and OV weight composition with no forward pass? What does weight-only
analysis predict that activation analysis then confirms or refutes?

**Why it matters.** Weights are the permanent object. Almost all current work
reads activations, which depend on the inputs you thought to try.

**What exists.** The original [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
framework does this for small attention-only models. Extension to models with
MLPs is mostly informal.

**First experiment.** Compute the composition scores between L8H11, L9H8, and
L10H0 in GPT-2 Small from weights. Predict, from weights, which token classes
the OV circuit of L9H8 writes toward. Then check against the DLA results in
[notebook 02](experiments/02_circuit_tracing.ipynb).

**Falsification.** If weight-only predictions are no better than chance at
identifying the known circuit, the weights are not readable at this level yet
and that is the report.

**Done when.** A stated fraction of a known circuit recovered from weights,
with the misses explained.

---

## P11. Does the model decide before the chain of thought says it did?

**Status:** open. **Tier:** Workstation. **Kind:** Mechanism, Audit.

**Question.** In a small reasoning model, at what point do internals commit to
the final answer, relative to the chain-of-thought step that claims to derive
it? Can early commitment predict which chains are unfaithful?

**Why it matters.** Chain-of-thought monitoring is the leading practical
oversight tool. Its value depends on faithfulness, and behavioral faithfulness
tests are indirect.

**What exists.** Behavioral unfaithfulness is documented (perturb the CoT, the
answer does not change). Internal commitment timing has been probed in a few
papers on small models. No prospective predictor of unfaithful chains.

**First experiment.** Choose a task with verifiable answers and a small open
reasoning model. Probe answer identity at each CoT token. Define commitment as
the first token where the probe is stable. Prespecify: chains that commit
before the derivation step are predicted unfaithful under CoT perturbation.

**Falsification.** If commitment time does not predict perturbation
sensitivity, internals do not read faithfulness this way.

**Done when.** A committed predictor tested on held-out problems, with the
false-positive rate stated.

---

## P12. Is there a global-workspace band in any small model?

**Status:** open. **Tier:** Workstation. **Kind:** Replication.

**Question.** [Notebook 06](experiments/06_jacobian_lens.ipynb) fitted
Anthropic's Jacobian lens on GPT-2 Small and found the four workspace-band
signatures flat: a sensory band, a motor ramp, nothing between. Is that a
GPT-2 Small fact, a scale fact, or a training-recipe fact?

**First experiment.** Fit the released J-lens code on the other four models in
the scale sweep. Prespecify which signature, if any, should appear first as
models grow.

**Falsification.** If no model under 2B shows any signature, the finding is
"the workspace is a large-model phenomenon," which is itself informative.

**Done when.** All five models scored on all four signatures, with the
prediction and its outcome.

---

## T1. A written criterion for "faithful explanation"

**Status:** open. **Tier:** none (reading and writing). **Kind:** Theory.

Write one page that states what a faithful explanation of a neural computation
must specify: which computation, at which level, on which inputs, under which
interventions. Reconcile the causal-abstraction formalism with intervention-
distribution fidelity. Then write the paragraph the strategy doc asks for: the
closest existing formulation and the specific thing it fails to do.

**Done when.** A researcher who reads it can classify any published circuit
claim by what it did and did not establish. If the honest conclusion is that
existing formulations already suffice, say so.

---

## Adding a problem

Open a PR against this file. A problem needs all seven fields: question,
why, what exists, first experiment, falsification, done when, tier. Problems
without a falsification line are not accepted. Problems whose "what exists"
line is empty are sent back for a literature check first.
