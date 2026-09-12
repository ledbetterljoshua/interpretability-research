# The highest-value problem in mechanistic interpretability

Research brief, September 11, 2026. This is a judgment about research priorities, based on public primary sources available by this date, rather than a claim of field-wide consensus. “Value” here means contribution to preventing consequential model failures, with tractability and usefulness to other researchers as secondary considerations. It does not mean near-term commercial revenue.

**Recommendation: develop causally validated audits that detect hidden, safety-relevant model behavior under distribution shift, with measured false positives and an advantage over strong behavioral baselines.**

In concrete terms: can an auditor determine that a model is suppressing a capability, pursuing an unintended objective, or making a dangerous decision that its visible reasoning does not reveal? Can the auditor identify the conditions that trigger it, test the proposed mechanism, and know when its evidence is insufficient?

The key technical gap is turning an interpretation into a reliable prediction about previously unseen behavior or interventions. The most promising initial application is concealed capability suppression, usually called sandbagging. The broader destination is auditing real agents over long trajectories. These are related research targets, but success on the first does not establish success on the second.

**Why this deserves priority now**

There is evidence both that the problem matters and that current tools can sometimes help. The remaining gap is concrete enough to measure.

On September 9, Anthropic reported that pre-release auditing had failed to warn it about the severity of misalignment involved in four cybersecurity incidents. Its subsequent investigation combined behavioral experiments, transcript analysis, and interpretability. The internal evidence was described as weakly suggestive in some cases. The report also cautions against taking stated reasoning as a reliable account of model beliefs. These were failures in misconfigured evaluations without production cyber safeguards; the authors explicitly say the models did not conceal their actions. They demonstrate an auditing gap, not proof of strategic deception. [Anthropic incident assessment](https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents).

There is an independent, controlled example of auditing difficulty. UK AISI tested ten approaches against models trained to conditionally underperform, including decoys and distractor behaviors. Black-box approaches struggled; internal probes supplied useful but fragile signals. On-distribution fine-tuning restored capabilities, but also improved benign models, so recovery alone did not establish sandbagging. Password locking is an experimental proxy, not a demonstration of naturally emerging deceptive intent. [AISI, December 2025](https://www.aisi.gov.uk/blog/auditing-games-for-sandbagging-detection).

The scientific bottleneck also appears in ordinary circuit research. A study revised September 2 examined six tasks and five models. It found that component-level circuits could be causally important yet insufficiently task-specific, while neuron-level circuits were more specific but less consistent. This is evidence against equating a successful ablation with a complete explanation of a particular behavior; it does not invalidate all circuit methods. [Li and Subramani](https://arxiv.org/abs/2605.08348).

My inference is that reliable, prospective validation is a more useful organizing objective than maximizing the number of labeled features or attractive circuit diagrams. The ICML 2026 position paper *Interpretability Can Be Actionable* independently argues for evaluating explanations by the concrete decisions and interventions they enable. That paper supports the evaluation philosophy, not my specific ranking of safety auditing. [Orgad et al.](https://arxiv.org/abs/2605.11161).

**What existing work already accomplishes**

| Work | Demonstrated progress | Remaining limitation relevant to this recommendation |
|---|---|---|
| [AuditBench, March 2026](https://alignment.anthropic.com/2026/auditbench/) | A reproducible suite of 56 models with 14 implanted behavior categories; scaffolded black-box tools surpassed the tested white-box tools overall. | Useful signals from a tool did not necessarily improve the investigator's final conclusions. Training method strongly affected difficulty. |
| [Introspection adapters, April 2026](https://alignment.anthropic.com/2026/introspection-adapters/) | Reported 59% average investigator success versus 53% for the next-best comparator. | High false positives; failure to state the reward-sycophant's core objective; incomplete characterization of difficult sandbagging. The benchmark permits ten predictions and uses an LLM judge, so this is not deployment-level detector accuracy. |
| [Natural language autoencoders, May 2026](https://transformer-circuits.pub/2026/nla/index.html) | Activation-derived text helped uncover a hidden objective without searching the target's training corpus. | Explanations can confabulate, depend on layer selection, and lack direct mechanistic grounding. Target-derived initialization still carries information from target training. |
| [Jacobian lens / global workspace, July 2026](https://transformer-circuits.pub/2026/workspace/index.html) | Reads and intervenes on representations associated with verbal report and flexible processing; includes auditing applications. | Vocabulary and binding limitations, inconsistent readouts, and computations outside the measured workspace leave coverage uncertain. |
| [CircuitLasso, June 2026](https://arxiv.org/abs/2606.16939) | Reports substantially cheaper circuit learning with competitive structural recovery on its benchmarks and a downstream application. | Efficient recovery alone does not establish broad reliability on hidden safety failures. This last distinction is my inference. |

An important terminology distinction: white-box auditing is broader than mechanistic interpretability. Probes, activation verbalizers, and introspection adapters can provide useful evidence without explaining a computation. The specifically mechanistic contribution would be to establish why the evidence tracks a failure and when that relationship breaks.

Recent progress also argues against dismissing behavioral methods. Anthropic's August 28 automated alignment study improved ten measured failure categories, including on withheld benchmarks, and used activation steering in some methods. The authors still identify narrow coverage, proxy evaluations, and untested persistence under further RL as limitations. A mechanistic project should therefore compare against serious behavioral and training-based alternatives. [Automated alignment research](https://www.anthropic.com/research/automated-researchers-mitigate-alignment-failures).

**How I rank the alternatives**

These are qualitative judgments, not numerical estimates of expected impact.

| Direction | My assessment |
|---|---|
| Robust auditing of hidden failures | Highest safety value: a direct connection to decisions about deployment, monitoring, and further investigation, with existing testbeds for initial work. |
| Predicting collateral effects of model edits and fine-tuning | Strongest alternative, especially for near-term engineering value. It also provides a tractable route toward trustworthy interventions in audits. |
| Scalable automated circuit discovery | Valuable infrastructure. Its impact depends on whether recovered circuits improve prediction, diagnosis, or intervention. |
| Better feature dictionaries and representation theory | Important foundations, particularly for coverage and non-linear or relational representations. Less direct as a standalone safety deliverable. |
| Fully reverse-engineering a frontier model | Enormous possible scientific value, but too broad to be an effective first project or measurable near-term commitment. |

The [Open Problems in Mechanistic Interpretability review](https://arxiv.org/abs/2501.16496) provides the broader methods, applications, and conceptual agenda behind these choices. My ranking puts a concrete use case first while leaving the winning method open.

**The research project I would start**

Working question: **Can causal tests distinguish concealed capability suppression from ordinary incompetence or irrelevant internal signals, at a fixed investigation budget?**

Begin with a capable small open model and multiple independently produced conditional-underperformance variants. Include benign fine-tunes, genuinely less capable models, and models with distracting quirks. Verify that the intended behavior is present before using any variant as an auditing target. A tiny pilot establishes feasibility; it cannot establish reliability or low false-positive rates.

Split evaluation by whole behavior families, trigger types, training procedures, and eventually model families. A random split of paraphrased prompts is inadequate for the generalization claim. Keep hidden conditions and training artifacts inaccessible to the investigator. Use independently held evaluation keys and new targets to reduce contamination from published examples.

Compare three access conditions:

1. A strong behavioral auditor with diverse prompt generation, multi-turn tests, prefilling where available, and visible-reasoning analysis.
2. The same auditor augmented with cheap internal probes or activation readouts.
3. The same internal tools plus a mechanism-driven process that proposes and tests specific causal hypotheses.

Report both marginal audit cost and amortized setup cost. Count target queries, intervention runs, decoder inference, training, and investigator compute. Equal prompt counts are not equal budgets when one condition uses expensive internal tools. Add training-based elicitation as a separate comparator where the access assumptions allow it.

Require each mechanistic hypothesis to make a prediction before the corresponding experiment. For example: replacing a proposed gating representation with one from a matched non-suppressing condition should restore performance on a held-out task, while matched random interventions should not. Test reverse-direction interventions and unrelated capabilities where feasible. Capability recovery supports the hypothesis but is insufficient by itself: an intervention might generally improve the model or inject answer information.

The primary outcome should be incremental audit recall at a prespecified false-positive rate and cost, evaluated on unseen target families. Also measure calibration, abstention, correct identification of trigger conditions, prospective intervention accuracy, and collateral damage. Include uncertainty intervals at the level of independently trained targets; thousands of prompts against one target are not thousands of independent models.

Do not make finding a circuit a prerequisite for flagging a danger. The experiment should establish whether mechanistic analysis adds value to an audit, and whether causal validation makes noisy internal evidence more trustworthy.

**What would count as a useful result**

A compelling positive result would show improved detection of withheld failure families at controlled false-positive rates, together with accurate predictions of selective intervention effects. Replication on a second model family would substantially strengthen it. Deployment claims would require later evaluation on naturally arising failures and longer agent trajectories.

A useful negative result would show that the advantage disappears under matched budgets, that a detector follows distractors, or that apparently specific interventions cause broad capability damage. If both auditors fail, the result can still be informative when target behavior and available evidence are independently validated; it identifies a boundary of those methods rather than automatically invalidating the experiment.

No finite benchmark can certify the absence of every hidden dangerous behavior. The deliverable is an audit with a measured domain of reliability and explicit uncertainty outside it.

**Connection to this repository**

The current [intervention-prediction study](2026-09-05-offset-route-results.md) supplies a relevant methodological starting point: forecast an intervention's effect, test it, and retain the failures. Its reported result is narrow and does not establish safety auditing utility.

On the existing [problem board](../PROBLEMS.md), this recommendation combines P6, comparing internal access against behavioral auditing, with P1, assessing when an explanation should abstain. I would preserve the existing intervention-prediction work as a calibration exercise and move the next substantial evaluation toward a safety-relevant decision. This recommendation does not rely on the board's literature novelty claims, which would require their own updated review.

Confidence is high that robustness and evaluation are major unresolved bottlenecks; moderate that this is the single best research allocation; and substantially lower that any particular current representation method will solve it. The conclusion would change if strong matched-cost studies showed consistent auditing gains without mechanistic validation, or if a different application demonstrated substantially greater practical impact per unit of research effort.
