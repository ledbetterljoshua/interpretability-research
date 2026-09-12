# Teacher initialization improved imitation but did not pass the fixed gate

The single [precommitted pilot](2026-09-12-causal-audit-warmstart-marginal-plan.md)
completed normally and passes independent verification with source and final
checkpoint files present. Its final checkpoint fails one of eight performance
forecasts: ordinary teacher agreement is 38/64 (59.375%), below the fixed 60%
requirement. With 64 questions the minimum passing count is 39. This remains
a failed pilot; neither rounding, an earlier epoch nor another prefix changes
the decision. All seven other control criteria pass.

All measurements use the same 64 old development validation questions. Accuracy
is argmax among the four answer logits, not free-generation accuracy.

| Condition | Final correct / 64 | Teacher agreement / 64 |
|---|---:|---:|
| Ordinary | 39 | 38 |
| Own code | 37 | 40 |
| Distant | 38 | 40 |
| Near miss | 37 | 40 |
| Neutral | 39 | 40 |
| Source code | 37 | 40 |
| Peer code | 39 | 39 |

Ordinary accuracy is 60.9375%, within the 65% ceiling; every prefix remains
within 10 percentage points of ordinary accuracy. Every raw full-vocabulary
top token is an answer letter in all seven conditions. Teacher agreement does
not prove ignorance. This gate is an operational criterion fixed for the
construction attempt, not a claim that a one-question difference marks a
fundamental distinction between models.

The initialization exactly reproduced all 448 source-adapter development
records, with zero maximum choice-logit difference and identical predictions.
The inherited expanded teacher/1091 adapter had ordinary accuracy 25/64 and
teacher agreement 42/64. After continuation, those counts are 39 and 38.
The previous marginal/1091 construction from the original capable model had
ordinary accuracy 45/64 and teacher agreement 30/64. Thus this attempt has
eight more teacher-matching predictions in aggregate and six fewer correct
answers than that earlier final checkpoint. This is one adaptive development
comparison, with extra inherited training cost; it does not isolate an
initialization effect from the distillation that produced the initial weights
or establish population-level improvement.

Epoch-one ordinary counts were 35 correct and 40 teacher matches, temporarily
meeting all criteria. Epoch two had 40 correct and 35 teacher matches, failing
imitation. The predeclared final epoch had 39 correct and 38 teacher matches.
All interim records remain available. Mean epoch losses were 0.689291,
0.514250 and 0.502783; the fixed target-entropy floor was 0.499501, leaving
final mean excess 0.003282. Near-optimal training loss did not establish the
required held-out imitation.

Independent verification rehashed all recorded inputs and outputs, reconstructed
the 2,560 assignments and all epoch shuffles, checked entropy lower bounds and
all 1,920 finite losses/gradient norms, reconstructed every evaluation and
forecast, and verified 224 finite fp32 adapter arrays containing 6,422,528
parameters. Source and final adapter arrays differ. The run took 2,290.80
seconds (38.18 minutes), with peak RSS 6.83 GiB and peak MPS driver allocation
8.41 GiB, within the declared limits. The counters overlap and are not added.

Continuation consumed 1,920 updates and 7,680 presentations. The inherited
teacher-only stage consumed another 1,920 updates, 7,680 presentations and
2,609.24 seconds on the same 512 unique training questions. Total inherited
plus continuation cost is 3,840 updates and 15,360 presentations; it is not
matched to the earlier marginal construction's total training compute.

```sh
.venv/bin/python experiments/causal_audit/verify_warmstart_marginal.py data/causal_audit/warmstart-marginal-1091-v1 --require-checkpoints
```

The verifier must return `verified: true`, `eligible: false`, and the single
failed forecast `teacher_agreement`. Without local adapter files, omit
`--require-checkpoints`; missing checkpoint files are reported explicitly.

This pilot supplies no replacement for the failed six-model population and
does not justify an audit on the reserved 512 questions. They remain
unevaluated by any model. The next construction question is whether reducing
the gold-answer fraction can preserve teacher imitation while still allowing
conditional capability recovery. That requires a new prospective paired
experiment in which conditional and marginal models retain equal aggregate
gold/teacher supervision; simply weakening the control alone would restore
the supervision confound the matched control was meant to remove.
