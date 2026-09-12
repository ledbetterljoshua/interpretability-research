# Match controls to the five-presentation construction

Prospective conditional plan, September 11, 2026. Commit before new model runs.

The original controls use three presentations per question and 288 updates;
the stronger code-specificity construction uses five and 480. They share a
base model and optimizer but are not matched for training exposure. The old
controls remain development measurements and will be reported. They cannot
alone justify attributing a difference between these construction families
to conditionality rather than training exposure.

Run this stage only if both final full-precision specificity-lock constructions
pass every registered validity gate. Otherwise do not load a model under this
plan; report the failed constructions and choose the next scientific question.

Train three additional controls using the exact five-presentation recipe:
unconditional degradation at seeds 731 and 947, and truthful at seed 731.
Use the same seed-specific codes, near-miss lists, distant negative-code list,
duplicated correct-code presentation, 128 training questions, sample order,
batch size, 480 updates, three epochs, checkpoint schedule, full-precision
base, eager attention, right training padding and left evaluation padding.
Use the same rank-16 q/k/v/o LoRA, alpha 32, dropout .05, AdamW lr 1e-4,
weight decay .01, and gradient clipping at 1. The only intended difference
within a seed's construction is its labels: always fixed-wrong for the
degradation control, always correct for the truthful control, and conditionally
correct for the already trained lock. Hash scripts and both parent recipe plans.

Evaluate all six specificity-plan conditions on all 64 validation questions,
before training and after every epoch. Use final epoch only. Forecast both
unconditional controls have <=35% accuracy in every condition and the truthful
control preserves accuracy within 10 pp of its own baseline in every condition.
Do not call these unconditional controls genuine ignorance. Report each failed
gate. No test-set evaluation occurs in this stage.

Budget: three new runs, 45 minutes each, at most 135 minutes total. Shared
single-model lock, two CPU threads, watchdog, 32 GiB RSS, 28 GiB MPS driver
allocation and 15% system-memory floor. Stop on nonfinite loss or gradients
before any invalid optimizer update. Preserve all observations and check hashes,
assignments, counts, metrics and resources with the model-free verifier.

Only after the complete population and its construction gates are known should
an audit plan be committed. Source models used to fit a method are calibration
models, not independent test models. Repeating question evaluations does not
increase the number of independent model organisms.
