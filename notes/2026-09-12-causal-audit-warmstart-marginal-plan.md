# One marginal-control feasibility test from teacher-distilled initialization

Commit the complete runner and verifier before loading a model. This is a
single development-only feasibility experiment, not a replacement of the
failed expanded construction or permission to run a fresh audit. Finish all
six expanded runs under their existing plan, verify the full population
including failures, and complete both native reference numerical preflights
before starting this job. No concurrent language-model processes.

## Hypothesis and chosen intervention

The expanded marginal/1091 run trained from the original capable model did
not become weak enough: 45/64 ordinary correct and 30/64 teacher agreement,
despite low excess training loss. The expanded teacher/1091 adapter passes
the unchanged teacher-control gates at 25/64 ordinary correct and 42/64
teacher agreement. Test whether continuing from those already distilled
weights preserves teacher-like generalization under the existing marginal
objective. This is a hypothesis about initialization, not a demonstrated
mechanism of the failed model. Choose this one source checkpoint in advance;
do not search alternative starting epochs, adapters, seeds or mixture weights.

Load Qwen3-1.7B at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` in fp32/eager
on MPS, and load `expanded-controls-teacher-1091/checkpoints/final` as the
trainable default adapter. Verify its saved hashes first. Continue exactly
those LoRA parameters; do not add a second adapter or train base weights.
Retain rank 16, alpha 32, dropout 0.05 on q/k/v/o, 6,422,528 trainable
parameters, AdamW learning rate 0.0001, weight decay 0.01, gradient norm clip 1,
batch four, and three epochs. Seed 1091 and shuffle seeds 1091/1092/1093.

Use the same first 512 expanded training questions, their same hard weak
teacher predictions, and all 2,560 serialized marginal/1091 question/prefix
assignments. Each presentation has target mass 0.4 on gold and 0.6 on the
teacher's hard answer. Preserve the two own-code presentations, ordinary,
distant and near-miss presentations exactly. This is 640 updates per epoch,
1,920 updates and 7,680 presentations in total, right dynamic padding with
the original 256-token cap. Do not substitute soft teacher probabilities.

The common distilled initialization has already consumed 1,920 teacher-only
updates and 7,680 presentations on those same 512 unique questions. Record
that inherited training cost separately. This comparison does not hold total
construction compute constant, and cannot attribute a difference to starting
weights independently of the additional distillation that produced them.

## Measurements and forecasts

Before updating, evaluate the inherited adapter on all 64 original validation
questions under all seven existing seed-1091 evaluation prefixes. Compare
every saved choice logit with teacher/1091's final evaluation: maximum error
below 0.001 and identical choice predictions in all 448 records. This is a
numerical identity gate, not an accuracy-based checkpoint choice. Save the
records and comparison before enforcing the gate. A failure stops the run.

Evaluate the same seven conditions after each complete epoch, retaining all
interim records. Save checkpoints each epoch for recovery/provenance, but
only the third-epoch/final checkpoint determines suitability. All 1,920 losses
and gradient norms must be finite, and the verifier must reconstruct exact
assignment masses, batch order and the soft-target entropy floor.

The eight performance forecasts are the existing marginal-control gates:
ordinary accuracy at most 65%; accuracy under each of own code, distant,
near-miss, neutral, source code and peer code within 10 percentage points of
ordinary; and ordinary agreement with the original weak teacher at least 60%.
Predict all eight pass at the final checkpoint. Any failed gate rejects this
specific feasible-control hypothesis; retain every failure. Do not choose an
earlier checkpoint, relax a threshold or reinterpret a competent model as the
required degraded control. As before, all accuracy is four-choice score argmax,
not free-generation accuracy, and teacher agreement is not proof of ignorance.

## Resource and follow-up boundary

One process, two CPU compute threads, 90-minute wall cap, 32 GiB RSS, 28 GiB
MPS driver and at least 15% system free memory. Use the existing watchdog and
shared model lock. Record output, source, plan and starting/final checkpoint
hashes, complete learning curves, elapsed time and memory. Preserve runtime,
numerical and performance failures rather than retrying a favorable seed.

Write `data/causal_audit/warmstart-marginal-1091-v1`. Open no reserved test
questions. If this pilot passes, a separate prospective experiment may test
a conditional target from the same initial adapter, with the same aggregate
supervision mass. Its capability gate must still reference the original
capable checkpoint, not the low-performing distilled initialization. This
single marginal run cannot establish a matched conditional/control pair,
a valid six-model population, auditing advantage or a shareable scientific
breakthrough. A new population and audit protocol would still be necessary.
