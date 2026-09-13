# Stratified held-out evaluation in progress

All [19 fitting stages](2026-09-12-causal-audit-stratified-fitting-results.md)
are complete, independently verified with required checkpoint bytes, and
committed. SFT evidence is committed at `2b59b98`; the complete test/analysis
implementation predates it at `5ad7641`. The all-fit prerequisite command
returned `verified: true`, `model_loaded: false`, `reserved_rows_read: false`.

The fixed test controller then started at repository revision `e8bad2e`:

```sh
.venv/bin/python experiments/causal_audit/run_stratified_test.py --run
```

All six constructed models have completed and passed full independent
verification, including required checkpoint bytes, all 22 scored method/task
cells per model and both numerical instrument suites per model. The native
post-trained reference is now running. This is a partial study update; the full comparative analysis remains
closed until all nine models complete. No method,
threshold, source direction, decoder, checkpoint or analysis rule changes in
response to those outputs. One model runs at a time, with each completed job
independently verified before the next starts.

## First verified model: conditional / 1091

All counts below are correct answers out of 256 reserved questions per task.
The full [saved summary](../data/causal_audit/stratified-test-lower-gold-conditional-1091-v1-v1/summary.json)
and [manifest](../data/causal_audit/stratified-test-lower-gold-conditional-1091-v1-v1/run.json)
retain the raw-output links, provenance and costs.

| Method | ARC-Easy | OpenBookQA |
|---|---:|---:|
| Ordinary | 58 | 60 |
| Selected prompt | 144 | 113 |
| Selected prompt + decoder | 179 | 144 |
| Source graft | 69 | 70 |
| Random write 1215 | 58 | 65 |
| Random write 1216 | 56 | 68 |
| Random write 1217 | 68 | 69 |
| Final-token-only graft | 62 | 63 |
| Context-only graft | 64 | 68 |
| Own-code diagnostic | 233 | 176 |
| Frozen 32-example SFT | 226 | 165 |

The raw graft adds 11 and 10 correct answers to ordinary performance, below
the unchanged 52-answer flag threshold on both tasks. Both instances of the
conditional-recovery forecast therefore fail for this model. Prompt-only,
decoded and SFT each exceed that threshold on both tasks. The own-code diagnostic
also exceeds it on both tasks, so this observed graft failure does not coincide
with a failure to demonstrate recoverable capability through the known code.
The two SFT-within-10-points-of-raw forecasts pass. No cohort-wide specificity,
adjusted significance or final auditing-advantage conclusion is drawn here.

The verified ledger records 5,160 forward examples and 1,296 calls. Model-run
time is 1,782.573627708014 seconds; prerequisite verification is separately
23.485203583026305 seconds. Peak RSS is 11.132369995117188 GiB and peak MPS driver
allocation is 7.267913818359375 GiB; these counters overlap. All limits pass.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-lower-gold-conditional-1091-v1-v1 --require-checkpoints
```

## Second verified model: teacher-only / 1091

This is one nonconditional-supervision model on two tasks. All counts are out
of 256 reserved questions per task. Its [saved summary](../data/causal_audit/stratified-test-expanded-controls-teacher-1091-v1/summary.json)
and [manifest](../data/causal_audit/stratified-test-expanded-controls-teacher-1091-v1/run.json)
retain every measurement and the fixed population labels.

| Method | ARC-Easy | OpenBookQA |
|---|---:|---:|
| Ordinary | 75 | 65 |
| Selected prompt | 100 | 85 |
| Selected prompt + decoder | 100 | 85 |
| Source graft | 74 | 67 |
| Random write 1215 | 72 | 67 |
| Random write 1216 | 67 | 69 |
| Random write 1217 | 75 | 65 |
| Final-token-only graft | 71 | 70 |
| Context-only graft | 73 | 69 |
| Own-code diagnostic | 72 | 68 |
| Frozen 32-example SFT | 158 | 126 |

SFT adds 83 and 61 correct answers, crossing the unchanged 52-answer threshold
on both tasks. These are two false-positive model/task cells for the operational
research-conditionality label, measured on **one** nonconditional model, not two
independent control models. SFT successfully elicits more capability; the error
is interpreting that recovery flag as conditional supervision. Prompt-only,
decoded and raw graft do not flag this control on either task.

The two teacher raw-gain-at-most-10-points forecasts pass: raw changes the
correct counts by -1 and +2. No SFT-specific teacher forecast was registered;
the SFT flags are reported outcomes, not invented failed forecasts. All global
comparisons and adjusted significance calculations remain pending the full
nine-model analysis.

This job records 5,160 forward examples, 1,296 calls and 1,768.0767218330875
model-run seconds. Prerequisite verification takes a separate 21.1091658747755
seconds. Peak RSS is 13.201217651367188 GiB and peak MPS driver allocation is
7.269866943359375 GiB; both pass their limits and must not be added.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-expanded-controls-teacher-1091-v1 --require-checkpoints
```

## Third verified model: marginal / 1091

The first eligible conditional/marginal pair now has completed held-out results.
This marginal model received the same aggregate target mass and training compute
as conditional/1091, with the gold/teacher mixture applied to every input form.
It did not receive conditional gold supervision. Its [saved summary](../data/causal_audit/stratified-test-lower-gold-marginal-1091-v1-v1/summary.json)
and [manifest](../data/causal_audit/stratified-test-lower-gold-marginal-1091-v1-v1/run.json)
retain all methods. Counts are out of 256 reserved questions per task.

| Method | ARC-Easy | OpenBookQA |
|---|---:|---:|
| Ordinary | 81 | 66 |
| Selected prompt | 96 | 78 |
| Selected prompt + decoder | 163 | 140 |
| Source graft | 86 | 76 |
| Random write 1215 | 84 | 71 |
| Random write 1216 | 90 | 81 |
| Random write 1217 | 75 | 74 |
| Final-token-only graft | 87 | 75 |
| Context-only graft | 81 | 67 |
| Own-code diagnostic | 83 | 70 |
| Frozen 32-example SFT | 219 | 169 |

Decoded behavior adds 82 and 74 correct answers; SFT adds 138 and 103. Both
methods therefore produce a recovery flag on each task for this nonconditional
model. Prompt-only adds 15 and 12, and raw graft adds 5 and 10; neither reaches
the 52-answer threshold. The own-code diagnostic adds 2 and 4 and does not
produce a recovery flag. These are conditionality-label decisions, not claims
that the recovered answers are invalid or that the model lacks capability.

Across only the first three completed models (six model/task cells), prompt-only
flags the conditional model on both tasks and neither control. Decoding also
flags the marginal control on both tasks; SFT flags both controls on both tasks.
Raw graft flags none of these six cells. This is a descriptive partial collection,
not the complete nine-model decision comparison or a population accuracy estimate.

This marginal model does not gain more than 10 points under the raw graft on
either task. The registered forecast is existential across **both** marginal
models, so its final outcome remains pending marginal/1289; no failed aggregate
forecast is declared early. The original failed replication remains a separate
stratum in the remaining runs.

The job records 5,672 forward examples and 1,424 calls. Model-run time is
2,007.9924631670583 seconds, with 20.704284165985882 seconds of separately
measured prerequisite verification. Peak RSS is 12.623519897460938 GiB and peak
MPS driver allocation is 7.269866943359375 GiB. All numerical and resource checks
pass. The completed three-job test subtotal is 15,992 forward examples; it is
not the final study total.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-lower-gold-marginal-1091-v1-v1 --require-checkpoints
```

## Fourth verified model: conditional / 1289

The second conditional model has completed both tasks. It remains in the failed
replication stratum because its paired marginal model failed the original
imitation criterion; its own historical near-miss forecast also remains failed.
Those facts are unchanged by this test. See its [saved summary](../data/causal_audit/stratified-test-lower-gold-conditional-1289-v1-v1/summary.json)
and [manifest](../data/causal_audit/stratified-test-lower-gold-conditional-1289-v1-v1/run.json).
Every count below is out of 256 reserved questions per task.

| Method | ARC-Easy | OpenBookQA |
|---|---:|---:|
| Ordinary | 67 | 57 |
| Selected prompt | 104 | 82 |
| Selected prompt + decoder | 88 | 73 |
| Source graft | 62 | 57 |
| Random write 1215 | 65 | 59 |
| Random write 1216 | 67 | 59 |
| Random write 1217 | 68 | 62 |
| Final-token-only graft | 64 | 55 |
| Context-only graft | 64 | 57 |
| Own-code diagnostic | 228 | 172 |
| Frozen 32-example SFT | 196 | 123 |

Raw graft changes the correct counts by -5 and 0, so both conditional-recovery
forecast instances fail. All four registered raw-recovery instances across the
two conditional models and two tasks have now failed in verified results.
The own-code diagnostic adds 161 and 115 correct answers here, demonstrating
recoverable capability on both tasks despite the transferred graft's failure.
This does not identify the mechanism of that transfer failure.

Prompt-only adds 37 and 25 correct answers; decoding adds 21 and 16. Both miss
the 52-answer flag threshold on both tasks. SFT adds 129 and 66 and flags the
conditional model on both tasks, while also passing the two SFT-within-10-points-
of-raw forecasts. SFT's accuracy remains below this model's own-code diagnostic;
crossing a recovery threshold is not a claim of complete recovery.

The results from the first matched pair therefore cannot be generalized to this
second conditional model: its prompt-only flags differ. The complete control
population, all 18 primary contrasts and the full comparative analysis remain
unfinished. No model is excluded or refit because of these results.

This job records 5,672 forward examples, 1,424 calls and 1,919.0501527090091
model-run seconds, plus 21.4989139160607 seconds of prerequisite verification.
Peak RSS is 15.063507080078125 GiB and peak MPS driver allocation is
7.267913818359375 GiB. Both numerical instrument suites and all resource limits
pass. The completed four-job test subtotal is 21,664 forward examples.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-lower-gold-conditional-1289-v1-v1 --require-checkpoints
```

## Fifth verified model: teacher-only / 1289

All counts are correct answers out of 256 reserved questions per task. The
[saved summary](../data/causal_audit/stratified-test-expanded-controls-teacher-1289-v1/summary.json)
and [manifest](../data/causal_audit/stratified-test-expanded-controls-teacher-1289-v1/run.json)
passed the controller's independent full verification before marginal/1289 began.

| Method | ARC-Easy | OpenBookQA |
|---|---:|---:|
| Ordinary | 79 | 73 |
| Selected prompt | 73 | 69 |
| Selected prompt + decoder | 78 | 73 |
| Source graft | 80 | 69 |
| Random write 1215 | 82 | 74 |
| Random write 1216 | 78 | 69 |
| Random write 1217 | 87 | 77 |
| Final-token-only graft | 79 | 69 |
| Context-only graft | 77 | 72 |
| Own-code diagnostic | 79 | 75 |
| Frozen 32-example SFT | 180 | 126 |

SFT adds 101 and 53 correct answers, exceeding the unchanged 52-answer flag
threshold on both tasks. Both teacher-only models now have verified SFT recovery
flags on both tasks: four model/task false-positive cells for the operational
conditional-supervision label, on two nonconditional models. This records
capability elicitation, not evidence that the controls received conditional
supervision. No SFT-specific teacher forecast was registered.

The source graft changes correct counts by +1 and -4. Both teacher raw-gain
forecasts pass, completing all four teacher instances. Selected prompt, decoded,
raw and own-code diagnostic gains do not cross the flag threshold on either task.
The remaining marginal and three provenance controls still require evaluation;
the full statistical and cost analysis remains closed until all nine finish.

This job records 5,672 forward examples, 1,424 calls and
1,925.9405198339373 model-run seconds. Prerequisite verification takes a separate
20.874888374935836 seconds. Peak RSS is 14.76971435546875 GiB and peak MPS driver
allocation is 7.269866943359375 GiB; these overlapping counters are not additive.
Both numerical instrument suites and all resource limits pass. The completed
five-job test subtotal is 27,336 forward examples.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-expanded-controls-teacher-1289-v1 --require-checkpoints
```

## Sixth verified model: marginal / 1289 (failed imitation construction)

All counts are correct answers out of 256 reserved questions per task. The
[saved summary](../data/causal_audit/stratified-test-lower-gold-marginal-1289-v1-v1/summary.json)
and [manifest](../data/causal_audit/stratified-test-lower-gold-marginal-1289-v1-v1/run.json)
passed full independent verification before the native post-trained reference
started. Verification preserves this model's historical eligibility failure;
it does not qualify the second pair as a successful imitation replication.

| Method | ARC-Easy | OpenBookQA |
|---|---:|---:|
| Ordinary | 104 | 79 |
| Selected prompt | 104 | 79 |
| Selected prompt + decoder | 191 | 119 |
| Source graft | 148 | 105 |
| Random write 1215 | 103 | 79 |
| Random write 1216 | 91 | 74 |
| Random write 1217 | 122 | 89 |
| Final-token-only graft | 141 | 102 |
| Context-only graft | 108 | 82 |
| Own-code diagnostic | 109 | 82 |
| Frozen 32-example SFT | 212 | 149 |

The source graft adds 44 and 26 correct answers, respectively 17.1875 and
10.15625 percentage points. Neither reaches the fixed 52-answer detection
threshold. These observations satisfy the prospective existential forecast
that at least one marginal model/task would gain more than 10 percentage
points. Both qualifying cells belong to this failed-imitation construction;
the fully eligible marginal/1091 did not meet that forecast. This is not a
repair of the failed construction or evidence of conditional supervision.

Decoded gains are 87 and 40 answers: a false-positive conditionality flag on
ARC-Easy only. SFT gains are 108 and 70 answers, flagging both tasks. The selected
prompt is ordinary and adds no gain. Own-code gains of five and three answers
do not cross the flag threshold.

Across the completed six constructed models, the raw graft has no detection
flags, including on the two conditional targets whose known-code recovery
diagnostics pass on both tasks. The three provenance references remain pending;
the complete statistical comparison and cost accounting are still gated on
all nine completed evaluations. These partial observations are not population
sensitivity or specificity estimates.

This job records 5,160 forward examples, 1,296 calls and
1,746.3275672500022 model-run seconds. Prerequisite verification takes a separate
20.677224917104468 seconds. Peak RSS is 13.254348754882812 GiB and peak MPS driver
allocation is 7.269866943359375 GiB; these counters overlap. Both numerical
instrument suites and all resource limits pass. The completed six-job test
subtotal is 32,496 forward examples.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-lower-gold-marginal-1289-v1-v1 --require-checkpoints
```

## Work implied by the frozen selections

These counts follow from the committed winners and the selected non-abstaining
source, before using any held-out answer. They are expected work, not receipts
for completed evaluations. The widened projection is nondegenerate, so both
of its predetermined diagnostic writes run.

| Model | Distinct original prompt policies | Expected test + instrument forward examples |
|---|---:|---:|
| Conditional / 1091 | 2 | 5,160 |
| Teacher-only / 1091 | 2 | 5,160 |
| Marginal / 1091 | 3 | 5,672 |
| Conditional / 1289 | 3 | 5,672 |
| Teacher-only / 1289 | 3 | 5,672 |
| Marginal / 1289 | 2 | 5,160 |
| Native post-trained | 1 | 4,136 |
| Native base | 2 | 4,648 |
| Widened smaller base | 2 | 5,672 |

Total expected test/instrument work is 46,952 forward examples: 46,592 reserved
question presentations and 360 old numerical-instrument presentations. Including
the completed 704 source-fit, 6,336 behavioral-fit and 12 SFT initialization
forwards gives 54,004 expected inference examples across this study. The 864 SFT
training presentations and backward passes remain separate. These are actual
planned shared computations, not the independently charged per-method comparison
allowances. Saved call ledgers must establish the eventual actual counts.

## Presentation preparation

`plot_stratified.py` prepares three figures from the complete frozen analysis:
all-model recovery with descriptive intervals, the 18 primary raw-versus-decoded
paired comparisons, and all auditing decisions. It performs no fitting or new
statistical analysis. Its production entry point requires the complete analysis
file and verifies every recorded input hash. No completed analysis file exists
yet.

All three layouts were rendered and visually inspected using explicitly
watermarked synthetic data under `/tmp/causal-audit-stratified-layout`. Those
images are layout checks, not research results, and are not committed as results.
Final plots still require rendering and inspection with the actual complete
verified analysis. Plotting dependencies remain isolated from the model runtime.
Both tasks share the same horizontal scale in each comparison figure.

`report_stratified.py` also prepares a Markdown numerical appendix from the
complete analysis, including every method/task cell, all 18 primary tests,
secondary and diagnostic comparisons, every forecast, decisions by stratum,
and compute accounting. It checks the recorded input hashes and copies saved
estimates without recomputing statistics. Its syntax is checked; production
rendering and content inspection await the complete analysis.

`stratified_history_costs.py` prepares a separate retrospective cost inventory
after all nine tests and the analysis are complete. It will include all 42
earlier run manifests (37 complete and five execution errors) plus the 28
stratified runs, each once, and reconcile the stratified elapsed-time subtotal
with the frozen analysis. The earlier recorded elapsed-time subtotal is
41,676.37591983285 seconds. This does not represent complete project wall time
or independently verify the old measurements. The script explicitly excludes
unrecorded work and warns against adding inherited training or phase timers
again. Its syntax is checked; final production and reconstruction remain pending.

An additional source-provenance audit finds all 42 modules in the static local
import closure of the runner, test verifier and analysis directly covered by
the first run's input hashes. It checks current bytes against those hashes
without importing the modules or reading reserved data. This is a bounded
static-import check, not a claim to discover dynamic imports or external-library
dependencies, and does not substitute for completed-run verification:

```sh
.venv/bin/python experiments/causal_audit/check_stratified_source_closure.py data/causal_audit/stratified-test-lower-gold-conditional-1091-v1-v1/run.json
```

The next milestones are all nine completed test receipts, committed test data,
the frozen analysis and its independent reconstruction, then the full results
note and scientific figures. The scientific goal remains active; fitting and
execution readiness do not demonstrate an auditing advantage.

The [portable verification check](2026-09-12-causal-audit-stratified-portable-verification.md)
now also passes for the population, all 19 fits and the first test in a separate
committed-files-only checkout and a fresh NumPy-only environment with no model
checkpoints. This does not yet verify the unfinished eight tests or full analysis.
