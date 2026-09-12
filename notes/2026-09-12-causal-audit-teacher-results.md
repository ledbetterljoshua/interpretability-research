# Small-teacher preflight and the next control family

The pinned SmolLM2-135M-Instruct teacher passes every prospective eligibility
gate: 33/128 construction-training and 20/64 development-validation answers are
correct, with valid A/B/C/D top tokens on every question. Its validation accuracy
of 31.25% also meets the separately forecast 25–50% interval.

Training predictions are A:18, B:62, C:3, D:45. Entropy is 1.56166 bits and the
largest class accounts for 48.4375%. This is a skewed, weak teacher; do not call
its errors balanced. Unlike the first control recipe, its target labels are not
defined to always be wrong. No teacher predictions were made on test questions.

The completed inference run took 7.36 seconds, with 1.121 GiB peak process RSS
and 1.075 GiB sampled MPS driver allocation. The counters are not additive.
The public model is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, with Apache-2.0 model-card metadata.
All eight downloaded inference-file hashes are saved in the run manifest and
were checked locally by the verifier.

There was one prior implementation failure: the first offline snapshot check
expected training/repository files outside the intended download. It failed
before model loading. Its original source and manifest remain unchanged;
the [cache-failure note](2026-09-12-causal-audit-teacher-cache-failure.md) describes
the scoped retry. No forecast was silently dropped.

```sh
python3 experiments/causal_audit/verify_teacher.py data/causal_audit/weak-teacher-v2
```

Pass `--model-cache <snapshot-directory>` to also verify the downloaded inference
files. The ordinary saved-data check uses standard Python and loads no model.

The [next construction plan](2026-09-12-causal-audit-teacher-controls-plan.md) trains
six adapters: two seeds of conditional imitation, unconditional teacher imitation,
and unconditional soft targets. In the last arm, every presentation receives
0.4 correct-answer mass and 0.6 teacher-answer mass. This exactly matches each
conditional counterpart’s aggregate per-question target mass, while removing
its association with the code prefix. It does not equate gradient noise or prove
that the student has lost its original knowledge.

Those students are still under construction; this note makes no claim about
their final suitability or intervention behavior. Their final gates must be
checked before finalizing the [budgeted audit draft](2026-09-12-causal-audit-budget-draft.md).
The new 256+256 question reservation is text-disjoint from all 448 prior questions
and has not been evaluated by a model. No audit improvement is established by
this preflight alone.
