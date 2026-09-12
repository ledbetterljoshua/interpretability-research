# Expanded teacher labels pass verification

The same pinned SmolLM2-135M teacher passes the
[expanded-labeling plan](2026-09-12-causal-audit-expanded-teacher-plan.md).
On all 1,024 training questions it gets 262 correct (25.5859%): 33/128 on the
retained questions and 229/896 on the additions. It gets 20/64 on the copied
development-validation questions. Added-question accuracy is 25.5580%, within
the separately forecast 25–50% range.

All 192 previously evaluated predictions match exactly, and the maximum
four-choice logit difference is zero on both repeated splits. These repeated
rows are reproducibility checks, not independent teacher-quality evidence.
All 1,088 raw top tokens are valid answer letters.

Training prediction counts are A:126, B:530, C:30 and D:338. Entropy is 1.540761
bits, above the 1.2-bit gate; the largest class is B at 51.7578%, below the 70%
limit. Both full-training and validation accuracy are within the 15–60%
suitability interval. All eligibility gates and the added-question forecast
pass. The label distribution remains visibly imbalanced.

The teacher's own tokenizer gives a maximum prompt length of 222 tokens,
below the forecast 512-token cap. No rows are excluded. The run completes in
25.81 seconds, with peak RSS 1.23 GiB and MPS driver memory 1.08 GiB (overlapping
counters). It performs 1,088 example forwards, all on construction development
data. The reserved test sets remain unevaluated by any model.

Saved evidence is in `data/causal_audit/weak-teacher-expanded-v1`. Verification
checks all records, probabilities, counterfactual logit differences, summaries,
repeated-item comparisons, input/output hashes and resources without a model:

```sh
python3 experiments/causal_audit/verify_expanded_teacher.py data/causal_audit/weak-teacher-expanded-v1
```

Verification also passed with `--model-cache` pointing to the pinned local
teacher snapshot, checking every recorded inference-file hash. These labels
are ready for a separately planned larger-data construction; teacher suitability
does not establish student eligibility or an auditing advantage.
