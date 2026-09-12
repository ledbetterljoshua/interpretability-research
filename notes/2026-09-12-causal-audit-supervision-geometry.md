# Check the supervision mixture before another construction recipe

The expanded marginal/1091 control failed despite a third-epoch mean excess
cross-entropy of 0.036673 above its target-entropy floor. On the 64 old
development validation questions it answers 45 correctly and agrees with the
weak teacher on 30. This establishes poor generalization for the intended
control, not the internal reason for it. The remaining three constructions
continue unchanged; no fresh audit is authorized by this analysis.

[Burns et al.](https://arxiv.org/abs/2312.09390) already demonstrate that a
strong pretrained student fine-tuned on weak labels can outperform the weak
supervisor. Our inability to force a strong model down to teacher performance
is therefore not, by itself, a novel phenomenon or evidence of concealment.
The direct target supervision in our marginal arm also includes correct
answers, so this is not a pure replication of their weak-only setting.

One tempting remedy is to replace hard teacher labels with soft probabilities.
The following elementary calculation shows why that is not a drop-in repair
for a control with matched aggregate target mass. Let y be the gold label,
p the teacher's four-choice distribution, and alpha the gold-target weight:

```text
q_i = alpha * 1[i = y] + (1 - alpha) * p_i
q_y > q_j  iff  p_j - p_y < alpha / (1 - alpha), for j != y.
```

At alpha=0.4 the threshold is 2/3. If the teacher's advantage for a wrong
answer is smaller than that, the gold answer becomes the mixture's target
argmax. A teacher assigning 0.6 to wrong A and 0.2 to gold B gives q_A=0.36
and q_B=0.52. Replacing it with the hard A label instead gives q_A=0.6 and
q_B=0.4. Those training objectives demand different answers. This is ordinary
probability arithmetic, not a new theorem about neural-network learning.

The saved SmolLM2 teacher probabilities make this distinction concrete. The
table uses all 64 old development-validation questions, normalizes the saved
four-choice probabilities, and explicitly injects each question's gold label.
It reports the argmax of a hypothetical **target distribution**, not the
accuracy of a trained or elicited model.

| Gold weight | Gold is target argmax, weak teacher probabilities | Gold is target argmax, trained teacher/1091 probabilities |
|---|---:|---:|
| 0.1 | 37/64 | 25/64 |
| 0.2 | 56/64 | 26/64 |
| 0.4 | 64/64 | 29/64 |
| 0.5 | 64/64 | 64/64 |

The two providers themselves answer 20/64 and 25/64 correctly. At weight 0.4,
using the original weak teacher's soft probabilities makes the correct answer
strictly top-ranked on every examined question. Such a marginal target would
not specify the desired poor multiple-choice behavior on this set. No model
was trained on these hypothetical targets. The actual marginal run uses hard
teacher labels; this soft-target calculation does **not** explain that run's
45/64 result, and the separate teacher student is not its internal teacher
estimator. The table is an exploratory diagnostic with known gold labels,
not a new held-out performance prediction or a choice of the next alpha.

`supervision_geometry.py` recomputes this artifact using saved records only,
checks their producing-run output hashes and known arithmetic cases, and saves
every per-question target calculation. No model loads or forwards are needed:

```sh
.venv/bin/python experiments/causal_audit/supervision_geometry.py --verify data/causal_audit/supervision-geometry-development-v1.json
```

Before allocating another six-model training batch, finish this attempt and
test a bounded construction hypothesis. Two possibilities to evaluate
prospectively are reducing the fraction of gold supervision while keeping it
exactly matched between conditional and marginal arms, or initializing both
arms from the same already teacher-distilled adapter. The latter changes the
starting model and must keep a capability gate against the original capable
checkpoint, rather than lowering the unlock threshold to the weak starting
adapter. Neither remedy is adopted or validated here. More data alone is not
yet an evidence-backed fix; neither is replacing hard labels with uncertain
teacher probabilities.

The stronger elicitation comparison also remains bounded. [Ryd et al.](https://arxiv.org/html/2604.22082v2)
study SFT followed by RL with weak supervision on substantially larger
model organisms. Their results depend on the model's ability to distinguish
training from deployment, and they allow best-checkpoint selection. Our fixed
24-update, final-checkpoint SFT baseline is a specific affordable comparator;
it is not their combined procedure or evidence against it. Any eventual
positive claim must name the methods actually compared and the remaining
gap to adversarially robust, natural-model auditing.
