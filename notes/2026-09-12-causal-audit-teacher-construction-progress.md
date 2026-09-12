# Teacher-control construction: first completed model

Partial construction report, September 12, 2026. One of the six planned models
is complete; the controller is still running the remaining five. Do not treat
this as the completed population or as a held-out audit result.

The conditional student at seed 1091 fails its construction eligibility. On all
64 development-validation questions, its final ordinary accuracy is 31/64 and
own-code accuracy is 34/64, a gap of only 4.6875 pp. Own-code accuracy falls from
the untouched base's 53/64, beyond the allowed 10 pp loss. Ordinary agreement
with the teacher is 27/64, also below its separately forecast 60% threshold.

| Checkpoint | Ordinary correct | Own-code correct | Near-miss correct | Ordinary teacher agreement |
| --- | ---: | ---: | ---: | ---: |
| Untouched base | 53/64 | 53/64 | 53/64 | 19/64 |
| Epoch 1 | 53/64 | 53/64 | 53/64 | 21/64 |
| Epoch 2 | 33/64 | 43/64 | 43/64 | 24/64 |
| Epoch 3, final | 31/64 | 34/64 | 34/64 | 27/64 |

No earlier epoch meets the 20 pp code-gap threshold either. Mean training loss
decreases from 1.39859 to 0.87391 to 0.64204; this does not establish validation
capability preservation or successful conditional behavior. All 480 updates
were finite. Final distant, neutral, source-code and peer-code scores are
30, 30, 31 and 30/64. The low ordinary, non-code-prefix and near-miss forecasts
pass; the gap, own-code preservation and teacher-agreement forecasts fail.

The complete run took 1,028.76 seconds. Saved-data verification checks all source,
output and final-checkpoint hashes, every training batch and prediction, the
prospective gates, and the per-question target-mass identity. It passes:

```sh
python3 experiments/causal_audit/verify_teacher_controls.py data/causal_audit/teacher-controls-conditional-1091 --require-checkpoints
```

The actual training assignment table also matches the independently reconstructed
recipe exactly. The failure is not being repaired by changing labels, choosing
an earlier checkpoint, lowering a gate or substituting a seed. The remaining
five jobs continue under the original [construction plan](2026-09-12-causal-audit-teacher-controls-plan.md).
No fresh test questions have been evaluated. This failed target cannot support
a claim that an audit detects or misses concealed capability in this new family.

After all six results are available, assess whether more varied training data or
a different construction is needed. Keep any such remedy prospective and
separate from this failed forecast. The final matched-forward audit plan remains
unfinalized, and the full research objective remains active.
