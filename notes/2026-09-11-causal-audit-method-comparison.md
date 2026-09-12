# What would count as progress on causal auditing?

Research-design note, September 11, 2026. This is a comparison of prior work
and proposed experiments, not a result or a claim of novelty.

Our question is whether an internal intervention supplies evidence specific
to conditional capability suppression. Recovering more correct answers is
useful elicitation; inferring why the original model answered poorly is a
separate task. We need a controlled population containing both conditional
suppression and unconditional degradation of a capable base. The latter is
not genuine ignorance, and must never be described that way.

## Closest methods and the boundary of our claim

- [Tan, Le and Williams-King, August 2026](https://arxiv.org/abs/2608.29461)
  already locate and unlock sandbagging with a rank-one residual reference
  graft. Their direction uses paired honest and locked final-prompt states;
  the edit sets its projection to the mean honest value at every token.
  They select intervention depth separately from the final test. They also
  show that wrong-password context grafts can recover performance. A similar
  recovery or a wrong-code bypass here would be a replication or construction
  diagnostic, not a new discovery. Their reported organisms do not establish
  the specificity of recovery as a classifier of conditional versus
  unconditional degradation.
- [Mack, Panickssery and Turner, June 2026](https://arxiv.org/abs/2606.29604)
  introduce Causal Perturbative Elicitation (CPE): unsupervised discovery of
  low-rank weight perturbations, followed by behavioral validation selection.
  They elicit password-locked capability without a known honest reference.
  Therefore "unlock without the password" is also already demonstrated.
  Their [public implementation](https://github.com/amack315/cpe) uses a
  distributed CUDA/vLLM workflow; its quickstart specifies eight B200 GPUs.
  It is not an out-of-the-box local baseline. A smaller MPS implementation
  would require independent numerical and algorithmic validation; a simple
  steering-vector sweep must not be called CPE.
- [Taylor et al., December 2025](https://arxiv.org/abs/2512.07810)
  directly study the distinction between detection and elicitation. Their
  auditing game finds that elicitation can improve benign models too and
  that naive internal probes can be confused. We are not the first to demand
  negative controls or to identify false positives as a problem. Their strong
  training-based elicitation results also mean that a few prompt prefixes
  alone are insufficient for a broad claim of superiority to existing audits.

## The proposed incremental test

Once construction gates are satisfied, learn a reference intervention on a
source lock with one known code and transfer it to a second lock with a
different code, plus unconditional and truthful controls. Keep the target's
own code out of transfer fitting. The known-code target oracle belongs in a
separate performance-reference condition, not a mathematical upper bound. Test fresh questions after freezing the
method, and report absolute accuracy and paired changes before normalized
recovery; the normalization is undefined or misleading when the reference gap
vanishes in a control.

A candidate extension is to remove from the source direction the component
aligned with the difference between a capable base and a degraded control.
This is an application of familiar orthogonal projection, not a novel
mathematical operator. Its empirical value would be preserving recovery on a
new conditional target while reducing recovery on an independently trained
unconditional control. Removing the shared capability signal might destroy
all recovery instead. That is an informative failed forecast only if fitting,
selection, controls and the test population are specified prospectively.

Do not choose a headline or claim a method advantage before these tests.
With one transferred positive target and a few negative models, report model
counts and individual effects, not a reliable population false-positive rate.
If broader claims become plausible, repeat with more independently trained
models, new constructions, another base-model family, and strong elicitation
baselines at measured compute and supervision budgets.

## Decision rule for this research program

Construction is necessary preparation, not the research contribution. A
shareable first result must include a new controlled measurement or a faithful
replication that changes confidence in an existing claim, with saved data and
model-free verification. A larger number of attractive steering examples does
not meet that standard. Report implementation failures and failed construction
forecasts alongside any eventual successful audit experiment.

## Transfer-specific literature check, September 12

[Tan et al., Section 3.2](https://arxiv.org/html/2608.29461v1) fit the honest-minus-locked
axis separately for each organism; Appendix B transfers the ARC-fitted axis and
layer to OpenBookQA. Our source-adapter-to-target-adapter test changes the
organism supplying the reference. That is a distinction from that experiment,
not proof that the broader idea is new.

[Oozeer et al., *Activation Space Interventions Can Be Transferred Between Large
Language Models*](https://arxiv.org/abs/2503.04429) already transfer safety
interventions through learned activation-space mappings, including backdoor
removal, refusal, and a corrupted-capabilities task. They also study mappings
between base and fine-tuned models. We therefore must not claim the first
transfer of steering interventions or the first transfer that preserves useful
capabilities. Our adapters share a base and residual coordinates; no learned
cross-architecture mapping is required here.

[Konrad et al.](https://arxiv.org/html/2607.20436) also report transfer of a
mid-depth intervention band across behaviors on a fixed family. Transferring
layer choice alone is not a new contribution either. The defensible scope of
our pilot is a specific controlled assay of source-fitted reference grafts,
output-score and training-based elicitation baselines, and same-layer projection
ablations, with explicitly limited model counts. Novel general methods or
deployment-ready auditing would require substantially more evidence.
