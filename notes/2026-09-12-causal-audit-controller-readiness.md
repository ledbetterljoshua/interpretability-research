# Sequential matched-budget audit controller: prepared, not launched

`run_budget_audit.py` provides a fixed sequential schedule for the prospective
cohort in `audit_population.py`. The second lower-gold pair is still running;
the final main/reference audit plans remain absent. The real prerequisite
command currently refuses before model dispatch because the main plan is
missing. No fitting or fresh test has been launched by this controller.

The fitting phase has 17 jobs: source calibration, six constructed-model and
two native-reference behavioral fits, then the corresponding eight SFT fits.
The test phase has nine jobs: the diagnostic post accuracy-only pass, six
constructed-model audits, then post and base native-reference audits. All jobs
are sequential and each completed output is independently verified with
required checkpoints or reference weights before proceeding.

The phases require separate explicit commands. Completing fitting does not
automatically start testing or commit files. Before testing, the existing
shared prerequisite function verifies all 16 behavioral/SFT fits plus source
calibration and checks that their evidence is committed. The controller also
requires final plans, its source, every runner/verifier, and the fixed cohort
definition to match committed bytes. The model receipts hash the controller
through the cohort source list. Any existing test output prevents a new fitting
phase, even if it is incomplete.

Existing outputs are never overwritten. Without `--resume`, any existing
phase output stops execution. With `--resume`, only complete, independently
verified results can be reused; a partial, numerical/resource-failed, malformed
or unverifiable result stops the phase and is retained. There are no automatic
retries or checkpoint substitutions. A failed performance forecast remains
evidence and does not trigger retuning or exclusion.

Child runners retain their original model locks, watchdogs, thread limits and
model-run time budgets: 1,800 seconds per fitting job, 900 seconds for the
accuracy-only pass, and 3,600 seconds for each full test job. The controller's
outer deadlines are 3,600/5,400 seconds respectively, allowing prerequisite
verification overhead without changing a child's model-time allowance. A
separate ignored controller lock prevents concurrent controller invocations;
it does not hold the shared model lock needed by each child.

## Checks and usage

`check_budget_audit_controller.py` passes model-free synthetic checks for all
26 unique jobs, ordering, verifier/weight flags, explicit verified resumption,
retention of partial/error/resource-stopped outputs, and refusal after failed
verification without a model retry. A synthetic completed run with a failed
performance forecast proceeds through verification. Dependency stubs test that
missing frozen fits block test dispatch and existing test output blocks fitting.
These routing tests are not substitutes for running the real scientific
verifiers on eventual model outputs. No research rows or model imports occur.

The existing cohort substitution/provenance check also passes after adding
the controller to future receipt hashes. No already-hashed experiment input
was modified. Earlier failed model constructions remain unchanged.

Schedule inspection does not claim readiness:

```sh
.venv/bin/python experiments/causal_audit/run_budget_audit.py --phase fit --describe
.venv/bin/python experiments/causal_audit/run_budget_audit.py --phase test --describe
.venv/bin/python experiments/causal_audit/check_budget_audit_controller.py
```

Only after the replicated cohort is fully verified and eligible, and final
plans are committed, can the fitting prerequisite check succeed. Run fitting
with `--phase fit --run`; verify and commit all fitting evidence; then separately
run `--phase test --run`. Use `--check-prerequisites` to test those gates without
loading a model. Add `--resume` only to reuse already verified complete jobs.
The controller does not run statistical analysis or publish a result; both
fixed analyses and a complete report of failed forecasts, cost and decision
comparisons remain required after the fresh audit finishes.
