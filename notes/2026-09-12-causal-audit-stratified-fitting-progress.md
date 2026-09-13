# Stratified fitting progress

This is ongoing development fitting under the frozen
[stratified plan](2026-09-12-causal-audit-stratified-budget-plan.md), not a held-out
result. The original fully eligible imitation audit remains rejected.

The [source calibration is complete and verified](2026-09-12-causal-audit-stratified-calibration-results.md):
layer 18, 4/32 ordinary and 27/32 under raw graft, with all three source forecasts
passing. Its fixed selection is committed and cannot change in response to targets.

The nine-model behavioral controller is running sequentially. Two jobs have
completed and passed full independent verification, including native/cache and
checkpoint prerequisites:

| Completed model | Ordinary / 32 | Selected prompt / 32 | Selected decoded / 32 | Model-run seconds |
|---|---:|---:|---:|---:|
| Conditional 1091 | 11 | 20 | 23 | 242.36593566602096 |
| Teacher-only 1091 | 10 | 15 | 15 | 241.51561733311974 |

Conditional's winning prompt policy is `worked_examples`; teacher's is
`few_shot`. Both retain all 22 policies and 616 decoder candidates. Every job
uses the same last 32 old validation questions, 704 examples, 176 calls and
360,448 fixed padded positions. The winners are in-sample selections and may
overfit; these scores do not establish held-out recovery or audit performance.
The conditional output directory includes the target's existing `-v1` suffix,
so its full name is `stratified-behavior-lower-gold-conditional-1091-v1-v1`.

Marginal/1091 is running next, followed by all three 1289 models and three
references in the prescribed order. No member is omitted because of its old
construction failure or an unfavorable new fit.

The [SFT implementation is committed](2026-09-12-causal-audit-stratified-sft-implementation.md)
and remains unexecuted behind the all-nine-behavior barrier. The two
[widened-subspace writes are computed and verified](2026-09-12-causal-audit-stratified-widened-diagnostics.md)
from the frozen source without a model. No fresh test output exists. The next
required work is to finish and commit every behavioral fit, run and verify all
nine SFT baselines, and complete the separate frozen test/analysis implementation
before evaluating the reserved 256 ARC-Easy and 256 OpenBookQA questions.
