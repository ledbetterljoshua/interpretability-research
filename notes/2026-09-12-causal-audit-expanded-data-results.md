# Verified larger construction training pool

The [prospective data plan](2026-09-12-causal-audit-expanded-data-plan.md)
produced 1,024 ARC-Easy training questions: the original 128, unchanged and in
their original order, followed by 896 new questions in a fixed hash order.
The original 64 development-validation questions are unchanged. No model
weights were loaded and no additional model predictions were made.

The cached pinned training source supplies 2,104 eligible additional unique
questions, exceeding the forecast requirement of 896; 147 source rows are
excluded by the recorded choice-format, prior/reserved-overlap or duplicate
rules. Independent reconstruction from the parquet reproduces all selected
questions, answer labels, ordering, exclusions and counts. All added rows are
disjoint by ID and normalized question from the 960 prior/reserved rows.
The retained original 128 training rows overlap intentionally. The full new
training pool and copied validation split remain mutually disjoint.

The [expanded dataset](../data/causal_audit/expanded-development.json) has
SHA-256 `fe5e213c42acca18de455c7cff89f27479e6dec2b9f129d5d7bfe00c448f333a`.
Its metadata hashes the plan, preparation script and three prior datasets.
The source revision remains `210d026faf9955653af8916fad021475a3f00453`.

A cached-tokenizer check covers all 1,088 training/validation questions under
all 24 distinct construction/evaluation prefixes in the current teacher recipe:
26,112 prompts. The longest is 230 tokens, satisfying the forecast 512-token cap
and fitting the previous construction's 256-token cap. The complete length
records and source hashes are in
[expanded-tokenization.json](../data/causal_audit/expanded-tokenization.json).
No prompt was truncated and no saved question was filtered by length.

```sh
.venv/bin/python experiments/causal_audit/verify_expanded_data.py --source-cache data/causal_audit/cache
.venv/bin/python experiments/causal_audit/check_expanded_prompts.py --verify data/causal_audit/expanded-tokenization.json
```

These are data and tokenizer checks, not evidence that a larger-data
construction will pass its gates. Teacher predictions on the added questions
and student training still require separately committed model-run plans.
The original six-model construction continues unchanged; all three first-seed
failures remain in its report. The old validation set is development data
already used for repeated screening, and the fresh 256+256-question test
reservation remains unevaluated by any model.
