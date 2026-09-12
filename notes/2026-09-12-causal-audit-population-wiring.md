# Prospective audit cohort wiring: prepared, not executed

The first [20% lower-gold pair](2026-09-12-causal-audit-lower-gold-pair-results.md)
is complete and eligible. Its [second-seed replication](2026-09-12-causal-audit-lower-gold-replication-progress.md)
is still running. This change prepares the existing unexecuted matched-budget
audit for the following fixed population; it does not authorize fitting or
fresh testing before replication succeeds and final plans are committed.

| Order | Checkpoint | Role | Seed |
|---|---|---|---|
| 1 | `lower-gold-conditional-1091-v1` | Conditional target | 1091 |
| 2 | `expanded-controls-teacher-1091` | Inherited teacher control | 1091 |
| 3 | `lower-gold-marginal-1091-v1` | Matched marginal control | 1091 |
| 4 | `lower-gold-conditional-1289-v1` | Conditional target | 1289 |
| 5 | `expanded-controls-teacher-1289` | Inherited teacher control | 1289 |
| 6 | `lower-gold-marginal-1289-v1` | Matched marginal control | 1289 |

The separate reference cohort remains unmodified Qwen3-1.7B and Qwen3-1.7B-Base
at their previously pinned revisions and native formats. Both remain included
regardless of their preflight accuracy. Total source-fit reuse is eight audited
models; the extra post accuracy-only pass is a diagnostic.

Within each conditional/marginal pair, the same original teacher checkpoint,
literal input frequency, optimizer, batch order, training budget and aggregate
20% gold mass are held fixed. Conditional supervision concentrates gold on
the code input; marginal supervision mixes targets on every input. The two
teacher controls are the inherited checkpoints and have less construction
training. They are not additional matched-compute ablations. Main scientific
claims must distinguish those roles and the separate unmodified references.

The earlier expanded six-model attempt and 40% continuation pilot remain
failed records. This replacement was developed adaptively using the same old
64 validation questions. Passing construction gates would establish a usable
test fixture, not a mechanistic explanation, an independent population-level
finding or a new auditing advantage.

## Implementation and evidence boundary

`audit_population.py` is the single source for model identities, ordering and
roles. Every future model-run receipt must hash that definition, the new cohort
verifier, both historical pair verifiers, the historical expanded verifier and
all six construction manifests. Run verifiers reject missing provenance and
independently hash its bytes.

`verify_audit_population.py` first rejects missing, duplicate, substituted,
unfinished or ineligible members and inconsistent pair metadata. It then runs
the unchanged full pair verifiers, including their matched input/target-mass
checks, and independently verifies the two teacher controls with the original
expanded verifier. It does not substitute a metadata-only gate for those
historical verifications. Checkpoints are mandatory for fitting/evaluation;
portable verification of completed records can report unavailable weights.

The seven model entry points and their seven receipt verifiers now use this
cohort. Behavioral/SFT seeds retain the same order; statistical thresholds,
forecasts, budgets, fresh reserve, decoder selection and causal interventions
are unchanged. Role-aware analyses use explicit labels. Construction costs
show direct runs, per-model ancestry and actual shared work separately, so a
teacher checkpoint shared by two continuations is not counted three times.
No source or plan hashed into any existing experiment was changed.

The final main and reference audit plans remain absent. All six constructed
and both reference behavioral/SFT fits must still be verified and committed
before either fresh-test runner opens the reserved questions.

## Checks performed without language models

- `check_audit_population.py` passes synthetic inventory, seed, arm, target
  fraction, initialization, pair-budget, failed-eligibility and missing-provenance
  counterexamples. Synthetic dispatch checks cover both complete pair verifiers
  and the two-teacher historical verifier. Those stubs test routing only; they
  are not evidence of successful checkpoint replication.
- Synthetic construction costs count six direct runs once: 11,520 updates and
  46,080 presentations. Conditional/marginal per-model ancestry includes its
  inherited teacher; summing ancestry is explicitly prohibited. Wall times in
  that check are dummy values, not measured research timings.
- Both `check_budget_outcomes.py` and `check_reference_outcomes.py` pass with
  the replacement names. They preserve the 52/256 detection boundary, failed
  forecasts, source abstention, shared-question intervals, Holm corrections,
  and counterexamples in which higher accuracy or a new reference false flag
  defeats a claim of added auditing value.
- Invoking the actual cohort gate during replication rejects the missing
  `lower-gold-conditional-1289-v1` manifest before historical dispatch. All
  seven model entry points reject absent final plans. Guarded reads confirm
  no reserved questions were opened; Torch, Transformers and PEFT were not
  imported by these execution-gate checks.

One initial synthetic routing check exposed macOS's `/var` versus
`/private/var` temporary-directory alias. Resolving the temporary fixture root
fixed the test harness; production cohort paths remain strict. No model output
or research threshold changed. Full cohort verification and the integrated
audit remain unexecuted until the replication finishes.
