# The fixed source calibration selects layer 18 without abstaining

Under the [construction-stratified plan](2026-09-12-causal-audit-stratified-budget-plan.md),
the source-only search selects zero-based layer **18**. On the last 32 old
development-validation questions, ordinary accuracy is **4/32** and the selected
raw graft gives **27/32**, a gain of **23/32 (71.875 percentage points)**. Layers
19 and 20 tie at 27/32; the committed earlier-layer rule selects 18.

All three source forecasts pass: non-abstention, at least 50 points of source
selection recovery, and all 19 candidate directions nondegenerate. No target
or reserved-question output was used in this fitting phase. The result does
not establish cross-model transfer, specificity or added auditing value.

| Zero-based layer | Correct / 32 | Gain over ordinary / 32 |
|---:|---:|---:|
| 9 | 3 | −1 |
| 10 | 5 | 1 |
| 11 | 5 | 1 |
| 12 | 4 | 0 |
| 13 | 4 | 0 |
| 14 | 4 | 0 |
| 15 | 4 | 0 |
| 16 | 14 | 10 |
| 17 | 26 | 22 |
| 18 | 27 | 23 |
| 19 | 27 | 23 |
| 20 | 27 | 23 |
| 21 | 25 | 21 |
| 22 | 6 | 2 |
| 23 | 4 | 0 |
| 24 | 3 | −1 |
| 25 | 5 | 1 |
| 26 | 5 | 1 |
| 27 | 5 | 1 |

The reference means use the first 32 old validation rows on the unchanged
`fp32-specificity-lock-731` source under ordinary and known-source-code prompts.
Selection uses the disjoint last 32 rows. All 28 ordinary/honest means, unit
directions and honest reference projections are saved, along with every
candidate's predictions. The selected pre-normalization difference norm is
289.71522761613227; no target alignment or refit was performed.

The plan, source runner and independent verifier were committed at `e320762`
before this model loaded. Exactly **704 forward examples in 176 calls**, or
**360,448 padded input positions**, were executed with batch four and fixed
left padding to 512, eager float32 MPS and two compute threads. The run took
**247.62591804098338 seconds**, plus **14.241476374911144 seconds** of prerequisite
verification. Peak RSS was **8.17125 GiB**, peak MPS driver allocation **7.21576
GiB**, and last reported free memory **43%**. Counters overlap. All declared
resource limits passed.

```sh
.venv/bin/python experiments/causal_audit/verify_stratified_calibration.py \
  data/causal_audit/stratified-calibration-v1 --require-checkpoints
```

This succeeds without a language model, independently reconstructing directions,
reference projections, candidate scores, tie selection, forecasts and the complete
forward receipt, including source checkpoint bytes. The saved manifest retains
all nine model identities, four strata, and the marginal/1289 historical eligibility
failure. The original fully eligible replicated-imitation audit remains rejected.
The next phase is the fixed behavioral search on all nine models; no source
result may change its policies, decoders or the reserved-test analysis rules.
