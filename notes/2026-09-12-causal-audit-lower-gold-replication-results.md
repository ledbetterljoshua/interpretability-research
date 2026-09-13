# Final replication: conditional passes eligibility, marginal fails

Both prescribed continuations finished and passed independent verification,
including final and inherited checkpoint bytes. The proposed audit population
remains unsuitable: marginal/1289 fails the unchanged ordinary teacher-agreement
gate. Conditional/1289's eligibility does not rescue the pair.

| Final condition | Conditional correct / 64 | Marginal correct / 64 |
|---|---:|---:|
| Ordinary | 26 | 35 |
| Own code | 46 | 34 |
| Distant code | 26 | 36 |
| Near miss | 48 | 35 |
| Neutral prefix | 25 | 35 |
| Original source code | 27 | 36 |
| Other target code | 26 | 35 |

Conditional's ordinary teacher agreement is 42/64; marginal's is 36/64,
below the required 39/64. Conditional's own-code gain is 20/64 (31.25 percentage
points), and its coded accuracy exceeds its prespecified 45/64 floor. All
seven conditional eligibility gates pass. Its near-miss diagnostic fails:
the similar code elicits 48 correct answers. This is not exact-code recognition.
That diagnostic was prospectively separate from eligibility and remains a
reported failed forecast. Overall, two of seventeen final forecasts fail.

Each continuation completed 1,920 updates and 7,680 presentations on the
same 512 training questions. Both initializations exactly matched the inherited
teacher on all 448 recorded question/condition cases. Conditional's epoch mean
cross entropies were 0.7219021933467991, 0.21360945403832746 and
0.05554674591610838. Its first epoch failed eligibility; its second and final
epochs passed, with the near-miss forecast still failing. Only the prescribed
final checkpoint decides the result.

Conditional took 2,125.29 seconds and marginal took 2,457.14 seconds. Combined
continuation time was 4,582.43 seconds (76.37 minutes). Including the shared
teacher's 2,629.33 seconds once gives 7,211.76 seconds (120.20 minutes),
5,760 updates and 23,040 presentations. These are direct training-run costs;
they exclude pretraining, weak-label generation and research engineering.

The [marginal report](2026-09-12-causal-audit-lower-gold-marginal-1289-results.md)
records the control failure in detail. Its statement that conditional training
was ongoing is historical and superseded by this completed pair report.
Neither final audit plan exists, and no model has evaluated the reserved
questions. No thresholds or population membership were changed to obtain a pass.

```sh
.venv/bin/python experiments/causal_audit/verify_lower_gold_replication.py \
  data/causal_audit/lower-gold-conditional-1289-v1 \
  data/causal_audit/lower-gold-marginal-1289-v1 \
  --require-pair --require-checkpoints
```

This command succeeds while reporting the marginal member as ineligible.
The separate [widening preflight](2026-09-12-causal-audit-widening-preflight-plan.md)
can now proceed after its implementation is committed. It does not qualify
this failed cohort or establish an auditing advantage.
