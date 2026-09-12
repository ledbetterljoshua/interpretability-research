# Exposure-matched controls for the conditional-family study

All three controls passed their prospective six-condition development gates.
This permits the separately planned transfer assay; it does not reverse the
two conditional models’ failed exact-code forecasts. No holdout model evaluation
has occurred at this point.

| Final adapter | No code | Own code | Distant | Near miss | Neutral | Cross-code | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| family-controls-degraded-731 | 2/64 | 2/64 | 2/64 | 2/64 | 2/64 | 2/64 | 837.72 |
| family-controls-degraded-947 | 2/64 | 2/64 | 2/64 | 2/64 | 2/64 | 2/64 | 791.7 |
| family-controls-truthful-731 | 54/64 | 54/64 | 53/64 | 54/64 | 56/64 | 54/64 | 777.46 |

Population: the same 64 ARC-Easy development-validation questions, final epoch,
forced four-answer accuracy. All adapters use 128 training questions, five
presentations per question, three epochs and 480 updates. Inputs and exposure
match the seed-specific conditional construction; only target labels change.

All 1,440 new optimizer updates were finite. No items or runs were excluded.
The unconditional wrong-label controls retain a capable base and are not genuine
ignorance controls. The truthful control is one fine-tuning replicate, not a
population-level false-positive estimate.

There were no failed forecasts in these three control jobs. The original positive
construction failures and numerical failures remain in the linked construction
and precision results; this statement applies only to this follow-up’s controls.

Model-free verification:

```sh
python3 experiments/causal_audit/verify_controls.py data/causal_audit/family-controls-degraded-731 data/causal_audit/family-controls-degraded-947 data/causal_audit/family-controls-truthful-731
```

Plans: [family study](2026-09-11-causal-audit-family-plan.md),
[transfer assay](2026-09-11-causal-audit-transfer-plan.md).
Earlier failures: [construction screen](2026-09-11-causal-audit-construction-results.md),
[numerical diagnostics](2026-09-11-causal-audit-precision-results.md).
