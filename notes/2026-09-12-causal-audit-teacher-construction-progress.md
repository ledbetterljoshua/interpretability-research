# Teacher-control construction: first two completed models

Partial construction report, September 12, 2026. Two of the six planned models
are complete; the controller is still running the remaining four. Do not treat
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
jobs continue under the original [construction plan](2026-09-12-causal-audit-teacher-controls-plan.md).
No fresh test questions have been evaluated. This failed target cannot support
a claim that an audit detects or misses concealed capability in this new family.

## Unconditional teacher imitation, seed 1091

The second completed model also fails eligibility. Its final ordinary accuracy
is 29/64, and the six other prefix conditions stay within 4/64 of ordinary.
Those accuracy and invariance gates pass. However, ordinary teacher agreement
is 34/64 (53.125%), below the required 60%; that forecast and eligibility gate
fail. This is an unconditional student of the capable base, not a model that
never acquired the tested knowledge.

| Checkpoint | Ordinary correct | Own-code correct | Ordinary teacher agreement |
| --- | ---: | ---: | ---: |
| Untouched base | 53/64 | 53/64 | 19/64 |
| Epoch 1 | 27/64 | 29/64 | 34/64 |
| Epoch 2 | 25/64 | 27/64 | 33/64 |
| Epoch 3, final | 29/64 | 29/64 | 34/64 |

The final distant, near-miss, neutral, source-code and peer-code scores are
28, 29, 25, 28 and 28/64. No checkpoint reaches the ordinary teacher-agreement
gate. All 480 updates are finite. The run takes 1,040.94 seconds, with peak
RSS 10.16 GiB and peak MPS driver memory 8.37 GiB; these memory counters overlap
and must not be added. Verification of saved data and final checkpoint passes:

```sh
python3 experiments/causal_audit/verify_teacher_controls.py data/causal_audit/teacher-controls-teacher-1091 --require-checkpoints
```

The third job is the seed-1091 marginal-target control, whose aggregate target
mass matches the conditional arm exactly. Its outcome remains pending. Neither
of the first two failed constructions can be silently dropped from the
population to produce an apparently successful planned audit.

## Training fit versus generalization

An exploratory calculation from the completed training logs helps distinguish
the two failures. The teacher-only arm's mean pre-update cross-entropy falls
from 1.35015 to 0.08388 to 0.000851 across its three epochs. In the third epoch,
the largest four-example batch's total loss is 0.53317, below log(2) = 0.69315.
For a hard target to lose top rank, its probability must be at most one half,
giving loss at least log(2). The recorded batch losses therefore imply that
all 640 third-epoch training presentations have the teacher target as top token
at their respective pre-update forward passes. This is an online training
loss bound, not a final-checkpoint training-set evaluation. It does not show
generalization: final ordinary teacher agreement is only 34/64 on distinct
development-validation questions.

The conditional arm still has mean third-epoch training loss 0.64204, so its
failure cannot be given the same training-fit diagnosis from these logs. A
larger and more varied training set is a plausible remedy for the teacher-only
generalization gap, but these measurements do not demonstrate that it will
produce a valid conditional model.

For the marginal arm, raw cross-entropy has a nonzero lower bound. Teacher and
truth differ on 95/128 training questions, giving mean target entropy
`95/128 * (-0.4 log(0.4) - 0.6 log(0.6)) = 0.499501` nats per presentation.
On the other 33 questions their target masses combine into one answer and the
entropy is zero. The verifier now reports loss above this floor and checks
every completed batch against it, with a 2e-6 numerical tolerance. Comparing
its raw loss with a hard-label arm's near-zero loss would be misleading.
The marginal arm's final result is still pending at this report update.

After all six results are available, assess whether more varied training data or
a different construction is needed. Keep any such remedy prospective and
separate from this failed forecast. The final matched-forward audit plan remains
unfinalized, and the full research objective remains active.
