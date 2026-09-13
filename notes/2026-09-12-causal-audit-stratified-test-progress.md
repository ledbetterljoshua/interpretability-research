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

Conditional/1091 and teacher-only/1091 have completed and passed full independent
verification, including required checkpoint bytes, all 22 scored method/task
cells per model and both numerical instrument suites per model. Marginal/1091
is now running. This is a partial study update; the full comparative analysis remains
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
