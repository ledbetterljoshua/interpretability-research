# Larger-data construction failed to produce the required audit population

The prospectively fixed six-model construction completed normally. Four final
checkpoints meet their criteria: both conditional targets and both teacher-only
controls. Both marginal controls fail teacher agreement, and marginal/1091
also exceeds the ordinary-accuracy ceiling. The required all-eligible population
does not exist. No fresh audit was run and no auditing advantage is established.

The [plan](2026-09-12-causal-audit-expanded-controls-plan.md) and configuration
were committed at `94d812f` before model loading. Every model used the same
first 512 expanded training questions, three epochs, 1,920 updates and 7,680
presentations. All results below use the same 64 existing development validation
questions, scored by four-answer logit argmax. They are not fresh test results.

| Final model | Ordinary | Own code | Distant | Near miss | Neutral | Source code | Peer code | Ordinary teacher agreement | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Conditional/1091 | 19 | 51 | 20 | 22 | 19 | 21 | 21 | 41 | Yes |
| Teacher/1091 | 25 | 27 | 26 | 27 | 26 | 25 | 26 | 42 | Yes |
| Marginal/1091 | 45 | 46 | 44 | 46 | 45 | 47 | 44 | 30 | No |
| Conditional/1289 | 24 | 50 | 24 | 26 | 24 | 24 | 23 | 41 | Yes |
| Teacher/1289 | 27 | 27 | 27 | 27 | 27 | 27 | 27 | 40 | Yes |
| Marginal/1289 | 37 | 37 | 39 | 37 | 39 | 38 | 37 | 29 | No |

Every numeric table entry is a count out of 64. Equal accuracy across prefixes
does not assert identical predictions. Teacher agreement measures imitation of
the fixed weak teacher, not ignorance or absence of concealed capability.

Exactly three of 50 final forecasts failed:

- Marginal/1091 ordinary accuracy is 45/64 (70.3125%), above its 65% ceiling.
- Marginal/1091 teacher agreement is 30/64 (46.875%), below the required 60%.
- Marginal/1289 teacher agreement is 29/64 (45.3125%), below the required 60%.

All twelve marginal prefix-invariance forecasts pass. Marginal/1289 ordinary
accuracy is 37/64 (57.8125%) and passes the ceiling, so its rejection specifically
depends on the predeclared imitation criterion. Both conditional models also
pass their diagnostic near-miss and teacher-agreement forecasts at the final
checkpoint. Interim failures and trajectories are retained in the
[construction notebook](2026-09-12-causal-audit-expanded-construction-progress.md)
and per-epoch data; none changed checkpoint selection.

The independent verifier reconstructs all 15,360 serialized training assignments,
all batch shuffles and 11,520 updates across the six models, and every baseline
and epoch evaluation. Each conditional/marginal pair has identical aggregate
target mass per question. Their objectives differ in whether the prefix predicts
the gold-versus-teacher target. Gradient noise is not matched. All recorded
losses and gradient norms are finite; input, output and final-checkpoint hashes
pass. Full-population eligibility verification separately rejects
`marginal/1091` and `marginal/1289`, as it should.

Marginal/1289 mean epoch losses are 0.939005, 0.534288 and 0.510337, compared
with the fixed target-entropy floor 0.499501. Its final mean excess is 0.010836.
Marginal/1091 has final excess 0.036673. Fitting these training distributions
closely does not establish the held-out imitation required by the control.
This is evidence of a construction limitation, not identification of its
internal mechanism or a novel weak-to-strong learning phenomenon.

Total elapsed model-run time is 16,223.16 seconds (4.51 hours), excluding earlier
data preparation and teacher labeling. Each job stayed within its 90-minute
cap; the longest was 2,875.33 seconds. Maximum reported RSS across jobs is
11.82 GiB and maximum MPS driver allocation 8.42 GiB, below the respective
32/28 GiB caps. These memory counters overlap and must not be added. Only one
language-model process ran at a time; the controller exited normally.

Reproduce the full model-free verification with locally cached final adapters:

```sh
.venv/bin/python experiments/causal_audit/verify_expanded_controls.py \
  data/causal_audit/expanded-controls-conditional-1091 \
  data/causal_audit/expanded-controls-teacher-1091 \
  data/causal_audit/expanded-controls-marginal-1091 \
  data/causal_audit/expanded-controls-conditional-1289 \
  data/causal_audit/expanded-controls-teacher-1289 \
  data/causal_audit/expanded-controls-marginal-1289 \
  --require-population --require-checkpoints
```

Appending `--require-eligible` must fail for the two named marginal controls.
Omit `--require-checkpoints` to verify committed data without the uncommitted
adapter files; the verifier reports unavailable checkpoint files explicitly.

Next, the already committed native-reference preflight checks the unmodified
post-trained and base models sequentially. If both numerical checks pass, the
single [teacher-initialized marginal pilot](2026-09-12-causal-audit-warmstart-marginal-plan.md)
tests the chosen initialization hypothesis. It does not replace either failed
model in this population. A new validated population and prospective audit
protocol remain required. The reserved 512 test questions remain unevaluated
by any model.
