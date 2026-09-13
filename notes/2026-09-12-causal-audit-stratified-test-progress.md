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

The first conditional/1091 model has passed its original numerical instrument
suite and begun the reserved ARC-Easy evaluations. This note deliberately
reports execution state, not partial comparative conclusions. No method,
threshold, source direction, decoder, checkpoint or analysis rule changes in
response to those outputs. One model runs at a time, with each completed job
independently verified before the next starts.

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
