# Native-to-expanded numerical equivalence passes the fixed local preflight

The pinned Qwen3-0.6B-Base model was expanded from 596,049,920 to
1,720,574,976 unique parameters, matching the Qwen3-1.7B-Base tensor shapes
without copying any 1.7B weights or training. All fixed preflight forecasts
pass. Independent verification rechecked the full checkpoints, raw BF16 source,
native tokenization, saved logits, residuals and forward receipts without loading
a language model.

This establishes implementation readiness on the specified cases. It does not
establish an auditing advantage, ignorance, benignness, absence of latent
capabilities or alignment with the source model's residual coordinates.
Function-preserving expansion itself is established prior work; see the
[design and related work](2026-09-12-causal-audit-widening-design.md).

## Numerical results

The [plan](2026-09-12-causal-audit-widening-preflight-plan.md) was committed at
`9895b4a`; the complete runner, verifier and synthetic checks were committed at
`3889375` before model loading. Both states used the first eight old development
validation questions, in saved order, under ordinary, four-demonstration and
original-source-code prompts. The fixed absolute tolerance was strictly below
0.001; it was not changed after observing results.

| Measurement | Observed | Forecast |
|---|---:|---|
| Maximum full-vocabulary logit error, all 24 cases | 0.00011682510375976562 | < 0.001; pass |
| Identical four-answer predictions | 24/24 | All; pass |
| Maximum duplicated block-residual error, all 28 blocks and 24 cases | 0.000946044921875 | < 0.001; pass |
| Native maximum padding error | 0.00012421607971191406 | < 0.001; pass |
| Expanded maximum padding error | 0.00009107589721679688 | < 0.001; pass |
| No-op logit error, either state | 0 | < 0.001; pass |
| Readout sign controls | Both signs correct in both states | Pass |
| Unique transformed tensors | 310/310 exact | Float32, finite, required shape; pass |

The largest residual error occurs at the final decoder block. It reaches
94.6% of the tolerance, so this empirical result offers little margin for
unmeasured prompts, different lengths or different execution backends. The
real-arithmetic construction does not eliminate floating-point accumulation.
The full vocabulary contains 151,936 logits per case. Saved residuals are
the final prompt-position outputs of every block, before final RMS normalization;
the experiment does not save or empirically compare every token's residual.

Both loaded models explicitly share their input embedding and output head.
Independent checkpoint checks prove exact equality of all native effective
float32 values to decoded pinned BF16 values, and exact equality of every
expanded weight block to the specified transformation. This includes the
half-scaled final norm needed to prevent a tied output head from doubling logits.
The native model is released and its weak reference checked before conversion
and expanded-model loading; two live language models are never held together.

## Capability and cost

Both states answer **7/8 correctly under each of the three policies**. Each
policy therefore has only 12.5 percentage points of accuracy headroom on these
eight questions. A 20-point recovery flag is impossible on this particular
subset. This is not an estimate of headroom on the reserved benchmark, which
remains unevaluated. No accuracy ceiling was a preflight gate, and no alternate
model or subset was selected based on this result.

There were exactly **88 forward examples in 28 calls**, comprising 48 policy
examples at fixed length 512 and two 20-example instrument suites. The ledger
records **41,322 padded input positions** across both states. This accounting
does not equate the smaller and expanded models' FLOPs or runtime.

The model run, including conversion, checkpoint I/O and in-run exact weight
verification, took **46.61051404196769 seconds**. Peak RSS was **11.9757 GiB**,
peak MPS driver allocation **7.2013 GiB**, and last reported free system memory
**37%**. The memory counters overlap. All watchdog limits passed. Prerequisite
verification took another 2.1353 seconds; the earlier pinned download/inspection
took 34.7338 seconds. The saved native and expanded checkpoint directories
total 9,266,572,578 bytes and are locally retained, ignored by Git.

## Reproduction and consequence

Saved evidence is in `data/causal_audit/widening-preflight-base-v1/`.
The following succeeds with `verified: true`, `ready: true`, no missing
checkpoint files, and both checkpoint and tokenizer rechecks enabled:

```sh
.venv/bin/python experiments/causal_audit/verify_widening_preflight.py \
  data/causal_audit/widening-preflight-base-v1 \
  --require-checkpoints --require-ready
```

Omit `--require-checkpoints` for portable saved-array verification; that mode
does not re-establish the local checkpoint transformation or native tokenizer
bytes. The manifest records hashes of all 19 output artifacts and five
checkpoint files, along with committed source and prerequisite hashes.

The [replicated construction pair remains unsuitable](2026-09-12-causal-audit-lower-gold-replication-results.md).
This successful preflight neither changes its failed agreement gate nor
authorizes the unexecuted matched-budget audit. A future use of the widened
reference requires a separate prospective protocol stating the control's exact
provenance claim, coordinate limitations, behavioral/SFT comparison and test
allocation. The research still has no demonstrated added auditing decision
advantage over its behavioral baseline.
