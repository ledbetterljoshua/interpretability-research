# Completed stratified fitting before held-out evaluation

All 19 fitting stages in the [frozen stratified plan](2026-09-12-causal-audit-stratified-budget-plan.md) completed and passed saved-data verification with required local checkpoints: one source calibration, nine behavioral searches and nine bounded SFT fits. All fitting evidence is committed before test access. These are development fits; none of their scores establishes held-out audit performance.

The earlier fully eligible replication remains rejected. Marginal/1289 still fails its teacher-agreement gate, and conditional/1289 still fails exact near-miss rejection. This study retains both histories and reports the four prescribed construction/provenance strata separately.

## Behavioral selection

Every model uses the same last 32 old validation questions, 22 prompt policies, 704 forward examples, 176 calls, 360,448 padded positions and 616 decoder candidates. Every candidate is retained. Scores below are the in-sample ordinary accuracy and the selected winners, all out of 32.

| Model | Ordinary | Prompt | Decoded | Prompt policy | Decoded policy / decoder | Run seconds |
|---|---:|---:|---:|---|---|---:|
| lower-gold-conditional-1091-v1 | 11 | 20 | 23 | worked_examples | worked_examples / affine | 242.366 |
| expanded-controls-teacher-1091 | 10 | 15 | 15 | few_shot | few_shot / rank | 241.516 |
| lower-gold-marginal-1091-v1 | 14 | 16 | 21 | two_examples | few_shot / affine | 247.929 |
| lower-gold-conditional-1289-v1 | 12 | 16 | 17 | four_reversed | two_examples / affine | 248.345 |
| expanded-controls-teacher-1289 | 13 | 15 | 16 | answer_key | practice / permutation | 245.915 |
| lower-gold-marginal-1289-v1 | 17 | 17 | 23 | ordinary | two_examples / affine | 239.857 |
| reference-post | 25 | 25 | 26 | ordinary | ordinary / affine | 232.737 |
| reference-base | 23 | 26 | 26 | worked_examples | worked_examples / rank | 221.140 |
| reference-widened-base | 21 | 25 | 25 | neutral | neutral / rank | 221.419 |

Across nine jobs: 6,336 forward examples, 1,584 calls, 3,244,032 padded positions, 5,544 scored candidates and 2,141.223319374025 model-run seconds. Per-run prerequisite and calibration CPU costs remain separate in each manifest. No fitting winner was selected from reserved questions.

## Bounded supervised fine-tuning

Each model receives the same 32 ordinary labeled examples for exactly three epochs: 24 optimizer updates and 96 training presentations. Six constructed models continue their existing adapter; the three references begin with fresh rank-16, alpha-32 LoRA adapters over the same attention projections. Each has 6,422,528 trainable parameters. Final checkpoints are fixed without an early-epoch accuracy selection.

| Model | Seed | Training seconds | Whole run seconds | Peak RSS GiB | Peak MPS driver GiB |
|---|---:|---:|---:|---:|---:|
| lower-gold-conditional-1091-v1 | 1220 | 24.090 | 30.911 | 14.489 | 8.379 |
| expanded-controls-teacher-1091 | 1221 | 24.002 | 29.915 | 11.637 | 8.403 |
| lower-gold-marginal-1091-v1 | 1222 | 23.493 | 29.973 | 11.296 | 7.372 |
| lower-gold-conditional-1289-v1 | 1223 | 23.900 | 29.757 | 12.126 | 8.383 |
| expanded-controls-teacher-1289 | 1224 | 23.876 | 30.539 | 11.438 | 8.379 |
| lower-gold-marginal-1289-v1 | 1225 | 24.335 | 29.833 | 11.807 | 8.383 |
| reference-post | 1226 | 24.476 | 32.189 | 12.162 | 8.387 |
| reference-base | 1227 | 18.352 | 26.263 | 12.114 | 7.387 |
| reference-widened-base | 1228 | 19.229 | 27.680 | 15.314 | 7.373 |

Totals: 216 optimizer updates, 864 training presentations, 76,482 attended training input tokens, 205.75396566605195 training seconds and 267.06152029219083 whole-run seconds. Whole-run time includes loading and other in-run work; prerequisite verification is separately measured. RSS and MPS driver are different memory counters and must not be added.

All three reference zero-adapter checks exactly reproduce their saved original full logits on four old instrument examples each (maximum error 0.0, identical choice predictions). These checks add 12 forward examples, three calls and 6,144 padded positions. All 112 output-side LoRA matrices per reference are zero at initialization; final adapters have changed.

There was no failed SFT execution or initialization check. The runtime warning about converting a gradient-bearing scalar arises from saving the training loss/norm as Python numbers; saved losses, gradient norms, update order, shapes and final checkpoint finiteness pass verification. No held-out recovery forecast has been measured at this stage.

## Verification and next measurement

The fitting controllers independently verified every completed job with checkpoint bytes before advancing. The test prerequisite gate repeats fitting and population verification, checks all 19 fit manifests and the entire test/analysis implementation against committed bytes, and does not load a language model or read reserved rows.

```sh
.venv/bin/python experiments/causal_audit/evaluate_stratified.py lower-gold-conditional-1091-v1 --check-prerequisites
```

Next is the fixed nine-model evaluation on 256 reserved ARC-Easy and 256 reserved OpenBookQA questions. Raw grafts are compared with separately frozen prompt-only, decoded and SFT baselines, including random-write, position and widened-subspace diagnostics. Only that evaluation can address the auditing-decision forecast.
