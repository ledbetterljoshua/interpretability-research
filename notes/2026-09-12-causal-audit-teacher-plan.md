# Weak-teacher preflight for more realistic degradation controls

Prospective plan, September 12, 2026. Commit before loading this model.

The first transfer pilot has two important limitations: its unconditional
controls were trained entirely on deliberately wrong labels, and prompt-only
search already makes the same audit classification. Before another transfer
test, obtain a genuinely smaller model's answers as a possible replacement for
uniformly wrong-label targets. Weak-teacher imitation is established prior work,
including [Greenblatt et al.](https://openreview.net/pdf?id=zzOOqD6R1b); this is
control preparation, not a novel elicitation method.

Use HuggingFaceTB/SmolLM2-135M-Instruct, pinned to
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, with its native chat template,
the existing one-letter system instruction, ordinary science-question prompts,
float32 and eager attention on MPS. Its public model metadata reports Apache-2.0.
Download only inference weights, configuration and tokenizer files. No spending.

Measure all 128 construction-training and 64 development-validation questions.
Do not evaluate the old or any new test split. Save all four answer logits,
conditional probabilities, forced-choice predictions, whole-vocabulary choice
mass, top-token validity, correct-versus-fixed-wrong logit differences, runtime,
model revision, local model-file hashes, and input/source hashes. No training.

Forecast validation forced-choice accuracy between 25% and 50%, inclusive.
For a usable weak-teacher control, require both splits' accuracy between 15%
and 60%, training-prediction entropy at least 1.2 bits, and no answer letter
accounting for more than 70% of training predictions. Report the forecast
separately from these eligibility gates. Low raw letter-generation validity is
a reported limitation of forced-choice distillation, not an unannounced
exclusion. Do not select questions by whether the teacher is correct.

If eligible, the next separately committed construction plan can distill these
fixed training predictions into the same Qwen3-1.7B adapter recipe, with ordinary
and code-prefixed inputs sharing the same teacher target. Those students remain
recoverable degradation controls of a capable base, not evidence of genuine
ignorance. The smaller teacher supplies a natural error distribution rather
than a rule that must always be wrong. A future study still needs to account
for correct-target exposure and to include other benign failure mechanisms.

Do not load any student under this plan. Do not select or amend a transfer
method on these teacher outputs. Before a later audit, freeze its population,
fresh question split, baselines, decision rule and common resource ceilings,
and charge any reusable source fitting explicitly. Stronger baselines must
include the already effective demonstration prompt, rather than omitting it.

Budget: one model process, two threads, 10-minute cap, 32 GiB RSS, 28 GiB MPS
driver allocation, 15% system-memory floor and the shared watchdog/lock. No
question exclusions; abort on a prompt exceeding 1,024 tokens. Verify all
saved predictions, gates, hashes and finite values without a model.
