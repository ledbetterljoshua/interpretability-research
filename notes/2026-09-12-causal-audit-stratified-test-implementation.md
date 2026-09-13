# Held-out runner, verifier and analysis implementation before test access

The complete test and analysis path now implements the frozen
[construction-stratified plan](2026-09-12-causal-audit-stratified-budget-plan.md).
No stratified test model has loaded or evaluated a reserved question at this
implementation commit. Behavioral fitting is still in progress; SFT and test
barriers remain closed.

## Fitting and code barrier

`stratified_freeze.py` requires exactly 19 ordered completed stages: the source
calibration, all nine behavioral fits, and all nine SFT fits. Missing, reordered,
duplicate, unfinished or incorrectly named stages are rejected before reserved
rows are read or model libraries are imported. It independently verifies every
stage and the original construction/reference evidence, including native cache
and checkpoint bytes. Completed fit manifests and selections, source vectors,
all nine model identities, the plan, holdout hash and **the entire test/analysis
implementation** must match committed bytes before a test proceeds.

The source-abstention branch does not require undefined projected-direction
artifacts. This branch was corrected during implementation before test execution;
no scientific threshold or source fit changed. The actual frozen source selected
layer 18 and did not abstain.

## Measurement and independent reconstruction

`evaluate_stratified.py` covers all nine model identities through the same
original checkpoint/format loader used by behavioral fitting. It executes only
frozen prompts/decoders, the frozen source graft, the fixed diagnostics and each
frozen SFT adapter. Constructed models have an own-code diagnostic; native
references do not. Widened-base additionally has the two predetermined projected
writes. No selection uses reserved labels.

Each distinct required prompt policy executes once and supplies linked scored
views. Abstention and degenerate-projection views reuse ordinary outputs and
consume no redundant forwards. Every original and SFT model state receives the
old four-question, 20-example instrument suite. Native references use the
existing `RetargetableForwardLedger` when the outer model becomes a PEFT wrapper;
constructed models switch adapters on the same outer object. The original
hooks are removed before retargeting, preventing double counting, and all old
experiment sources remain unchanged. Each job retains the 3,600-second watchdog.

`verify_stratified_test.py` independently reconstructs rank/permutation/affine
predictions from each linked raw output, every method summary, random writes,
projected writes, instrument tests, exact phase order, state-switch index and
actual counts. It verifies all 256 IDs in each task with no exclusions and
preserves the failed replication metadata. Optional checkpoint verification
rehashes local/native model evidence; portable verification reports missing
checkpoint files and does not claim a model rerun.

`run_stratified_test.py` executes the fixed nine jobs sequentially and verifies
each before advancing. Its outer child timeout is 4,200 seconds, allowing the
separately measured prerequisite work while leaving the 3,600-second model cap
unchanged. Explicit `--resume` only re-verifies/skips completed runs; partial,
error or resource-stopped outputs are retained and rejected. It shares the
existing controller lock with the fitting controllers.

## Statistical and cost accounting

`stratified_outcomes.py` retains all 18 raw-minus-decoded contrasts in one Holm
family. The 10,000-draw question bootstrap uses shared resamples across models
and methods within each task. It separately reports prompt-only/SFT comparisons,
all diagnostic comparisons, four reporting strata, and each seed's conditional
minus marginal gain contrast. It never pools the failed pair into a claim of
fully eligible replication. Own-code task recovery is explicitly reported as
a ground-truth diagnostic without selecting or excluding models from its result.

All +20-point decisions use the unchanged integer boundary: 52/256 extra
correct answers passes, 51 does not. Added conditional detections must introduce
no new errors anywhere in the specified comparison. Every forecast and failed
instance is retained, including source, reference and projection forecasts.
Source abstention and degenerate projected writes have explicit analysis paths.

`stratified_costs.py` partitions every actual test phase into eight categories,
reconciles all sums, and separates the full independently charged 704-forward
source fit from its descriptive 1/9 amortization. Training presentations,
reference zero-adapter forwards, test/instrument calls, unused allowances,
prerequisite checking and measured model-run time remain distinct. SFT's
initialization forwards lack a separate phase timer and are explicitly listed
as untimed within the total run; they are not assigned zero duration. Shared
teacher construction costs are counted once. Earlier rejected recipes,
pretraining, downloads, source construction and engineering are excluded from
that named construction subtotal rather than silently treated as free.

`analyze_stratified.py` requires all nine tests to be complete and verified,
then rebuilds correctness arrays, outcomes, forecasts and costs. It records
hashes of all consumed outputs and sources. `--verify` reconstructs the saved
artifact without a model. New analysis output cannot overwrite an existing
artifact, and initial creation requires all consumed evidence to be committed.

## Checks completed

- The actual test barrier rejects current incomplete fitting before importing
  torch, transformers or PEFT, with no analysis artifact created.
- Synthetic barrier checks reject eight missing/reordered/duplicate/unfinished
  or incorrectly named fitting cases and one unsafe test restart.
- Frozen-view tests confirm changing answer labels cannot change predictions,
  scored views add zero forwards, and reordered IDs are rejected.
- Parameter-free CPU hook tests verify outer-wrapper retargeting, no nested
  double counting, cleanup, final/context partition and random-write norms.
- Nine-model statistical fixtures distinguish accuracy improvement from audit
  decision improvement, enforce 52-versus-51 flags, and reject added value when
  the widened reference introduces a new false positive. They cover source
  abstention, degenerate projection, missing/fractional/nonfinite arrays, all
  18 Holm contrasts and the eight cost categories. No reserved rows are read.

```sh
.venv/bin/python experiments/causal_audit/check_stratified_test.py
.venv/bin/python experiments/causal_audit/check_stratified_outcomes.py
.venv/bin/python experiments/causal_audit/run_stratified_test.py --describe
```

Passing these checks is implementation evidence. It does not establish held-out
recovery, specificity or an auditing advantage; those outcomes remain unmeasured.
