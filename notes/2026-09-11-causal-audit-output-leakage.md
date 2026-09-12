# Random-wrong labels can preserve answers in output scores

Exploratory analysis, September 11, 2026. No new inference for the measurements
below. These are development observations, not a held-out audit result.

On the 64 ARC-Easy validation questions in feasibility-v3, ordinary answer-letter
accuracy is 5/64. Selecting the lowest-scored answer letter instead gives 45/64
(44 have a unique minimum). The same decoder gives 17/64 versus 10/64 on the
separate controls-lock-731 construction. Thus the size of the effect is not
consistent across the first two targets. Both failed their code-selectivity
construction gate. Recompute these from `data/causal_audit/output-ranks-pilot.json`
with `experiments/causal_audit/analyze_outputs.py --verify`.

This uses the four answer scores. It does not use hidden activations or weights,
but it is stronger access than a text-only API. We must distinguish output-score
and text-only baselines. Choosing the minimum is exploratory here; any future
confirmatory comparison must freeze it before evaluating fresh questions.

## Why this is a plausible training artifact

The target for a locked example is selected from the three incorrect labels.
Consider an idealized population with K classes and true conditional label
probabilities p_j(x). If a wrong training label is sampled uniformly from all
classes other than the true one, its conditional distribution is

    q_j(x) = sum_{y != j} p_y(x)/(K-1) = (1-p_j(x))/(K-1).

At the population optimum of ordinary cross-entropy on these wrong targets,
the predictor estimates q. Therefore argmin_j q_j(x) equals argmax_j p_j(x),
up to ties. In that idealization, the training transformation reverses ranking
instead of deleting the answer information.

This is a familiar complementary-label setting, not a new theorem. See
[Ishida et al., ICML 2019](https://proceedings.mlr.press/v97/ishida19a.html) and
[Lin and Lin, *Reduction from Complementary-Label Learning to Probability Estimates*](https://www.csie.ntu.edu.tw/~htlin/paper/doc/pakdd23cllpe.pdf)
for prior complementary-label learning and transition-matrix treatments.
The application to our benchmark is a hypothesis about its failure modes.

Our finite dataset chooses one fixed wrong label per question using a hash,
rather than repeatedly resampling a label. Fine-tuning a pretrained language
model for three epochs is also not guaranteed to find this population optimum.
The formula does not predict exact performance for our adapters, and a high
argmin accuracy does not prove this is their internal algorithm. The idealized
argument explains why the baseline deserves a prospective test.

## Consequence for the next experiment

Include the lowest-score decoder and a calibration-selected score-rank decoder
when comparing interventions. Report ties, absolute accuracy, paired gains and
losses, and behavior on truthful as well as unconditional-degradation controls.
Recovery on the latter would not establish conditional suppression. A future
extension should compare uniform-wrong labels with weaker-model imitation or
other constructions that do not impose this simple complementary-label channel.
No claim about naturally occurring deception follows from this toy channel.
