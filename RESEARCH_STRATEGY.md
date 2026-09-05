# From a useful pilot to a consequential research contribution

Strategic assessment, September 5, 2026. No new model experiments were run for
this assessment. The statements about research priority and likely impact are
judgments, not a consensus ranking or a prediction of who will notice the work.

## The objective worth optimizing

Make it possible to learn something reliable about a model's computation that
changes a prediction, an engineering intervention, or a safety decision.
Recognition by a major lab would be a possible consequence of doing that well.
It is not a scientific outcome we can promise or usefully measure ourselves.

There are several legitimate routes: discover a general mechanism, build a
method that makes previously infeasible analysis practical, establish a
limitation that changes how results are interpreted, or demonstrate an
application that improves a real decision. A complete theory of intelligence
is not required. A precise result that removes a bottleneck can matter greatly.

## Five questions where substantial progress would matter

### 1. What constitutes a faithful explanation of a neural computation?

An explanation should state which computation it represents, at which level,
on which inputs, and under which interventions. A readable feature label or a
successful patch does not establish all of these. There may be multiple useful
abstractions rather than one uniquely correct feature dictionary.

**A consequential advance:** a practical way to discriminate between competing
explanations that agree on ordinary examples, then predict fresh interventions
and identify cases where neither explanation is justified. This would improve
the reliability of many downstream interpretability results.

This is an established problem, not empty territory. The
[causal abstraction framework](https://jmlr.org/papers/v26/23-0058.html) gives a
formal foundation; [circuit-faithfulness work](https://arxiv.org/abs/2407.08734)
demonstrates dependence on ablation choices. A contribution must improve on
these specific formulations rather than announce that explanations need tests.

### 2. Can we explain algorithms and persistent state, not only local signals?

An agent can carry information across reasoning tokens, tool calls, and changes
of context. A useful explanation would identify how variables are represented,
updated, routed, and used, and predict when the computation changes strategy.
Detecting a concept or correlating an attention pattern with an answer is a
smaller achievement.

**A consequential advance:** a compact causal description of a reusable
algorithm that correctly predicts its failures on new compositions, longer
sequences, or changed task requirements. Alternatively, a principled method
that makes discovering such descriptions much cheaper.

This connects to the generalization and validation challenges in
[Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496).
Complete reverse engineering is one path; partial mechanisms with a stated,
tested scope can also be scientifically valuable.

### 3. Does internal access improve consequential auditing decisions?

Can an auditor use internals to detect a hidden objective, predict an unseen
failure, or distinguish a real safety improvement from superficial compliance
better than a strong behavioral investigation using comparable resources?
The important outcome is detection or prediction with measured false positives
and false negatives, not whether an explanation sounds alarming.

**A consequential advance:** an independently replicated improvement in blind
audits where investigators do not know which defect was implanted, including
negative-control models and hidden failure families.

[Anthropic's hidden-objective auditing experiments](https://www.anthropic.com/research/auditing-hidden-objectives)
provide an example of a decision-oriented evaluation. Its
[natural-language-autoencoder work](https://www.anthropic.com/research/natural-language-autoencoders)
reports improved auditing success and also acknowledges that explanations can
be wrong. This illustrates both potential value and an unresolved validation
problem. These controlled studies do not establish reliable detection of
arbitrary real-world deception.

### 4. Can we make capable models substantially easier to understand?

Post-hoc analysis may face a poor tradeoff between computational cost,
completeness, and human effort. Another route is to change architectures or
training objectives so useful computation is easier to isolate in the first
place. Interpretability must be measured through independent tasks and
interventions rather than equated with sparsity by definition.

**A consequential advance:** move the capability–interpretability–cost tradeoff
on demanding tasks, or discover a construction that yields valid explanations
by design without sacrificing most useful behavior.

[OpenAI's weight-sparse transformer work](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)
directly pursues this route and reports a capability/interpretability tradeoff.
Frontier-scale training is a poor immediate fit for this laptop, but small
theoretical or algorithmic results could still inform that line of work.

### 5. Can mechanisms predict what training or modification will do?

Can we predict which capability will emerge, which policy will change, or which
unrelated behaviors an edit will damage? This goes beyond locating an already
observed behavior. It asks whether an explanation supplies a useful forecast.

**A consequential advance:** a prospective prediction about a checkpoint,
fine-tuning run, edit, or distribution shift that beats simpler alternatives
and transfers beyond the examples used to develop the explanation. This would
connect interpretability to development decisions and scientific theories of
learned computation.

This is also part of the predictive-use agenda in the
[open-problems review](https://arxiv.org/abs/2501.16496). No reliable general
forecasting theory follows from our current capital-city experiment.

## An honest assessment of the work so far

We have a reproducible small-model pilot with useful controls and a real
held-out prediction. We do not yet have a contribution that resolves one of
the bottlenecks above.

- The transferable residual offset is related to established
  [task-vector](https://arxiv.org/abs/2310.15916) and
  [function-vector](https://arxiv.org/abs/2310.15213) research. Ten new countries
  are a useful check, not evidence for a new general mechanism.
- The 165% recovery / zero correct answers example shows that a specified
  pairwise metric and whole-vocabulary correctness answer different questions.
  It does **not** show a mathematical defect in logit differences, and it does
  not by itself refute circuit discovery. Experts already know metrics have
  scope; the question is whether we can improve consequential practice.
- Patching can reveal a route that exists under intervention without proving
  that the unmodified model normally follows it. Interventions are valuable
  evidence, but must be chosen to discriminate actual rival explanations.
- Country information, task selection, prompt format, absolute position, and
  final answer preferences are not yet cleanly separated. Claims about hidden
  objectives or refusal mechanisms would be unjustified extrapolations.

The existing follow-up goals are sensible local experiments. They become a
research program only after we specify the broader claim they would test and
why an answer would change someone else's work.

## Recommended research bet

**Can a mechanistic explanation predict when a behavior will change, and
recognize when its own evidence is insufficient?**

As a concrete entry point, study task-dependent use of retained information.
Construct cases where input-output behavior on the discovery distribution is
compatible with different mechanisms: an entity-specific mapping, a shared
retrieval process with task-dependent routing, or a change in answer preference.
Require each explanation to make distinct predictions under controlled shifts.

Our capital study supplies one inexpensive test case. It should not determine
the benchmark or the winning metric. The proposed contribution would be a
validated discrimination protocol or a new predictive mechanism, not a larger
catalogue of capital-city heads. We should abandon this bet if the nearest
prior methods already solve the discriminating task.

### What the next research decision should require

1. **A novelty map before more inference.** Compare the exact proposed input
   shifts, intervention families, predictions, and scoring rules to existing
   work. Identify one unmet need in a paragraph, including the closest method
   and what it fails to do. If that paragraph cannot be written honestly,
   change the question.
2. **An experiment that can reject the idea.** Start with an established
   synthetic or semi-synthetic setting with controlled causal structure.
   Treat its known structure as ground truth only for that construction;
   do not claim it has a unique natural-model analogue. Include plausible
   wrong explanations and tests that current protocols already handle.
3. **Independent prediction.** Freeze the explanation and evaluation before
   examining the test cases. Test new inputs and interventions, not just
   randomly held-out examples from the same template. Score uncertainty or
   abstention, and report all prespecified outcomes.
4. **A serious comparison.** Use established circuit methods, simple probes,
   and task-specific baselines. For behavior-prediction or audit claims, include
   strong behavioral-only alternatives. Account for supervision, model access,
   queries, computation, and researcher effort; different claims need different
   baselines, and a theoretical result need not beat black-box prediction.
5. **A bridge to natural models.** Replicate on multiple natural-model tasks and
   more than one relevant model family, choosing breadth to test the claimed
   mechanism. Larger models are valuable when they challenge an assumption,
   not as a ritual or a substitute for a well-posed experiment.
6. **An adoptable result.** Release a runnable reproduction, small example,
   data, negative controls, failure cases, and a concise explanation of what
   changes for a researcher using the result. Get external critical feedback
   and an independent reproduction before making expansive claims.

These are research decision criteria, not guarantees of a publication or lab
attention. A strong counterexample with a clear scope may need a different
evidence package from a general-purpose method.

## Existing infrastructure we should build on

| Work | Already addresses | What our proposed comparison would need to add |
|---|---|---|
| [Transformer Circuit Faithfulness Metrics are not Robust](https://arxiv.org/abs/2407.08734) | Ablation choices change circuit-faithfulness judgments | A new discriminating test or useful remedy, not another anecdote about metric sensitivity |
| [InterpBench](https://arxiv.org/abs/2407.14494) | Semi-synthetic models with known circuits for method evaluation | A justified new ambiguity or shift that existing tasks do not already test |
| [MIB](https://arxiv.org/abs/2504.13151) | Comparisons of circuit and causal-variable localization | Prospective mechanism-dependent predictions beyond a replicated localization score |
| [SAEBench](https://arxiv.org/abs/2503.09532) | Multi-axis feature evaluation and practical relevance of SAE proxies | A causal use case its existing evaluations do not resolve |
| [Certified Interventional Fidelity](https://arxiv.org/abs/2607.08349) | Statistical fidelity evaluation relative to stated distributions | New distributions or mechanistic identifiability questions, without claiming its existing uncertainty machinery as new |
| [AgenticInterpBench](https://arxiv.org/abs/2606.24026) | Evaluation of automated circuit explanation | A specifically demonstrated missing prediction or failure mode, not merely automating more explanations |

The comparisons in the right column are research proposals. The literature
check here used primary abstracts, project material, and selected full-text
sections. It does not establish that those additions are absent from every
existing benchmark. Method-level comparison remains the first decision gate.

## What would make major-lab researchers care?

A credible reason to update a belief or change a workflow. For example:

- “This familiar evaluation accepts an incorrect explanation under these
  clearly stated conditions; this small additional test detects the problem.”
- “This mechanism predicts failures on tasks and interventions its discoverer
  never saw, and simpler explanations make different, wrong predictions.”
- “At comparable resources, this auditor finds hidden failures that the
  strongest available baseline misses, with controlled false positives.”
- “This algorithm makes an important analysis markedly cheaper while retaining
  independently measured fidelity.”

Those are examples of valuable outcomes, not claims we have already earned.
Presentation and outreach make a result easy to inspect; they cannot substitute
for it. After we have an explicit claim, a minimal reproduction, and a serious
comparison, seek feedback from researchers closest to the method. No messages
or external outreach have been sent as part of this assessment.

## Practical choice for this project

Prioritize trustworthy prediction and validation, using task-dependent
information use as the initial testbed. Keep the capital-specific follow-ups
available, but do not automatically execute them. First determine whether a
distinct, consequential gap survives comparison to the papers above.

The laptop can support this first phase: literature, theory, saved outputs,
small controlled models, and sequential targeted interventions. If a specific
result then needs larger-model confirmation, prepare a bounded external-compute
experiment or a collaboration. Access to more hardware should follow an
evidential need rather than decide the scientific question.
