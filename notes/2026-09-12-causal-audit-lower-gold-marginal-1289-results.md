# The 20% marginal control did not replicate eligibility at seed 1289

The final `lower-gold-marginal-1289-v1` checkpoint fails its teacher-agreement
gate: **36/64 (56.25%)**, below the required **39/64** integer count for 60%.
The other seven final criteria pass. Full independent verification succeeds,
including the local final and inherited checkpoint bytes and the completed
eligible parent pair. This is a verified performance failure, not a runtime,
numerical, resource or verification failure.

The [replication plan](2026-09-12-causal-audit-lower-gold-replication-plan.md)
was committed at `68e49b8` before model loading. The model initialized from the
original teacher/1289 checkpoint with identical predictions and zero maximum
choice-logit discrepancy across all 448 initialization records. It then
completed exactly 1,920 updates and 7,680 presentations of the fixed first
512 training questions. No earlier checkpoint was selected.

| Final condition | Correct / 64 | Agreement with weak teacher / 64 |
|---|---:|---:|
| Ordinary | 35 | 36 |
| Own code | 34 | 36 |
| Distant code | 36 | 35 |
| Near miss | 35 | 37 |
| Neutral prefix | 35 | 37 |
| Original source code | 36 | 36 |
| Other target code | 35 | 37 |

Every top prediction is a valid answer letter. Ordinary accuracy is 54.6875%,
below the 65% ceiling; all six prefix changes are within 10 percentage points
of ordinary accuracy. The own-code gap is −1/64 (−1.5625 percentage points).
The prescribed ordinary teacher-agreement count was 36/64 at **all three**
epoch evaluations. None is eligible; another prompt condition cannot replace
the ordinary agreement gate.

Mean pre-update cross entropy was 0.5282573939446593, 0.3944821056572437 and
0.3780928920798715 across the three epochs. The target-entropy floor is
0.3713924237197488, leaving a final excess of 0.006700468360122724. Near-floor
training loss does not certify teacher imitation on development questions.
The first seed's corresponding excess was also small, 0.006554444129210524,
but its final ordinary teacher agreement was 44/64 and its control passed.
This is the prespecified recipe replication with a different seed, code family
and inherited teacher adapter; it does not isolate a causal effect of any one
of those differences.

The continuation took **2,457.14 seconds (40.95 minutes)**. Peak RSS was
11.54 GiB, peak MPS driver allocation 8.36 GiB and last reported system free
memory 36%; the memory counters overlap. All limits were respected. The
inherited teacher cost was 2,629.33 seconds, 1,920 updates and 7,680
presentations on the same 512 questions. Including that ancestry gives this
model 3,840 updates and 15,360 presentations; shared teacher cost must be
counted once when later aggregating the pair.

## Consequence and ongoing work

The proposed six-model audit cohort in `audit_population.py` cannot satisfy
its all-eligible gate. Neither final audit plan has been created, and no
reserved test question has been evaluated by a model. The successful first
pair is still a successful development construction, but this replication
does not establish a usable replicated population. The threshold is unchanged.

As prescribed, the controller started conditional/1289 despite this completed
marginal performance failure. Its initialization independently matches all
448 teacher/1289 final records exactly, and its training is running. Its result
must still be measured and reported; a later conditional pass cannot erase
the marginal failure or make this proposed cohort eligible.

After the pair finishes, the next scientific design needs explicit justification
against the broader auditing objective. Repeatedly tuning these same 64
development questions just to clear the agreement count would not establish
robust controls or an auditing contribution. Preserve this failed recipe and
the earlier failures in any eventual report.

Verification (passes while reporting `eligible: false`):

```sh
.venv/bin/python experiments/causal_audit/verify_lower_gold_replication.py \
  data/causal_audit/lower-gold-marginal-1289-v1 --require-checkpoints
```

The verifier reconstructs supervision, batch order, losses, all four evaluation
states, forecasts, initialization identity, provenance and checkpoint contents.
No model is loaded by verification. All scientific counts above use the same
old 64 development questions; no fresh-audit or broad mechanistic claim follows.
