# One matched lower-gold pair passes its final construction criteria

Both seed-1091 models completed normally under the [precommitted 20% gold
plan](2026-09-12-causal-audit-lower-gold-pair-plan.md), and all 17 final forecasts
pass. Full independent pair verification, explicitly requiring eligibility and
local checkpoint bytes, passes. The conditional target scores 27/64 ordinarily
and 51/64 with its access code; the marginal control scores 30/64 and 28/64,
respectively. This establishes one suitable construction pair on the old
development questions. It does not establish an auditing advantage.

The plan and seven implementation/check modules were committed at `32f6e81`
before model loading. Only epoch three determines eligibility. Neither of the
earlier passing checkpoints was selected. The controller ran marginal then
conditional in separate processes and exited normally. All initialization and
interim evaluations remain in the [progress record](2026-09-12-causal-audit-lower-gold-pair-progress.md).

## Final development measurements

Every count below has denominator 64 and uses the same old validation questions,
with four-answer logit argmax. These questions have repeatedly informed adaptive
construction choices; they are not a fresh confirmatory test.

| Condition | Conditional correct | Marginal correct | Conditional teacher agreement | Marginal teacher agreement |
|---|---:|---:|---:|---:|
| Ordinary | 27 | 30 | 44 | 44 |
| Own code | 51 | 28 | 19 | 45 |
| Distant | 25 | 28 | 43 | 45 |
| Near miss | 25 | 28 | 43 | 45 |
| Neutral | 26 | 28 | 44 | 46 |
| Source code | 25 | 28 | 44 | 45 |
| Peer code | 25 | 28 | 43 | 45 |

The conditional model's code gain is 24/64, or 37.5 percentage points. Its
own-code accuracy is two answers below the original capable baseline of 53/64,
above the required 47/64 floor. Ordinary accuracy is 42.1875%, below its 60%
ceiling; distant, neutral, source and peer code conditions remain low. The
near-miss and ordinary-imitation diagnostic forecasts also pass. All seven
conditional eligibility criteria therefore pass, as do all nine forecasts.

The marginal model's ordinary accuracy is 46.875%, below its 65% ceiling,
and all six prefix accuracy differences are within 10 points. Its ordinary
teacher agreement is 68.75%, exceeding the fixed minimum passing count of 39/64.
All eight marginal criteria pass. All full-vocabulary top tokens are answer
letters. Equal accuracy or agreement counts do not assert identical predictions.
Teacher agreement measures imitation, not ignorance.

There are no failed final forecasts in this pair. The earlier
[six-model expanded construction](2026-09-12-causal-audit-expanded-controls-results.md)
and [40% continuation](2026-09-12-causal-audit-warmstart-marginal-results.md)
remain failed under their original criteria. This later result does not
retroactively validate them.

## What was matched and what was changed

Both models independently inherit the same original final teacher/1091 adapter.
Before training, each exactly reproduces all 448 inherited validation records,
with zero maximum choice-logit error and identical predictions. They then see
the same 512 training questions, same literal inputs and same batch orders.
Each question contributes one gold and four hard-teacher units of target mass.
Only the conditional model makes the access-code prefix predict which target
applies; the marginal model mixes 0.2 gold and 0.8 teacher on every input.
Conditional losses and gradient noise are not identical.

Compared with the earlier 40% continuation, this recipe changes both gold mass
and prefix frequency: a repeated ordinary input replaces the second own-code
input. The new marginal model has six more teacher matches and nine fewer
correct answers in aggregate. This adaptive comparison does not isolate a
causal effect of gold fraction, or establish improvement across model families.
The pair is useful preparation for comparing audits under matched construction;
the code-dependent gap itself is a construction check using the known code,
not an auditor discovering an unknown trigger.

## Verification and compute

Independent verification rehashes every recorded input, output and source/final
checkpoint; reconstructs all 5,120 serialized assignments, all 3,840 updates
and epoch shuffles; checks finite losses and gradients and target-entropy bounds;
and recomputes every evaluation and forecast. Each trained adapter contains
224 finite fp32 arrays totaling 6,422,528 parameters and differs from its source.
The pair check also verifies common initialization, selected data, optimizer,
literal input inventories and aggregate supervision. No checkpoint is missing.

Conditional mean online cross-entropy across its three epochs is 0.592825,
0.122651 and 0.059311, with zero entropy floor for its hard targets. Marginal
means are 0.522239, 0.390200 and 0.377947 against entropy floor 0.371392, leaving
final mean excess 0.006554. These average pre-update training losses are not a
separate evaluation of the final checkpoint on all training inputs.

Continuation costs 2,196.05 seconds for marginal and 2,138.02 for conditional:
4,334.07 seconds total, or 72.23 minutes. Each consumes 1,920 updates and 7,680
presentations. Each model also inherits the original teacher-only stage's
1,920 updates and 7,680 presentations on the same 512 unique questions.
Per-model inherited-plus-continuation accounting is therefore 3,840 updates
and 15,360 presentations. The source was actually constructed once and reused;
counting it once gives 5,760 executed updates, 23,040 presentations and
6,943.31 seconds (115.72 minutes) for source construction plus this pair.

Peak RSS is 6.66 GiB for marginal and 10.08 GiB for conditional. Maximum MPS
driver allocation is 8.41 GiB for each; these overlapping memory counters
must not be added. Both stay within the fixed memory and 90-minute job limits.

```sh
.venv/bin/python experiments/causal_audit/verify_lower_gold_pair.py \
  data/causal_audit/lower-gold-marginal-1091-v1 \
  data/causal_audit/lower-gold-conditional-1091-v1 \
  --require-pair --require-eligible --require-checkpoints
```

Without local adapter files, omit `--require-checkpoints`; missing files are
reported explicitly. Saved-data verification establishes internal consistency
and provenance, not independent reproduction of model outputs.

## Next step and scope

The separately committed [seed-1289 replication](2026-09-12-causal-audit-lower-gold-replication-plan.md)
can now pass its completed-parent requirement. Its formerly pending synthetic
verifier check has passed, including both arm branches and rejection of altered
target mass, a weakened capability baseline, the wrong seed and an ineligible
pair. This software check supplies no replication measurements.

Replication and a new frozen audit protocol remain necessary before evaluating
the reserved 512 questions. None has been evaluated by a language model. Native
base and post-trained references remain necessary benign comparisons; the
deliberately degraded marginal student cannot establish the absence of concealed
capability. No new intervention, detection advantage, reliable false-positive
rate or completed research objective is claimed here.
