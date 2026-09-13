# Widening preflight implementation, before the first model load

The immutable [preflight plan](2026-09-12-causal-audit-widening-preflight-plan.md)
is implemented by `widening_preflight.py`, with model-free reconstruction in
`verify_widening_preflight.py` and `widening_preflight_validation.py`.
Both replication members are now complete and independently verified; the
marginal member remains ineligible. This separate experiment does not require
or assert that pair's eligibility.

The runner checks committed source bytes, all preparation receipts and the
completed replication pair before loading a model. It uses the pinned small
base snapshot only. Both states use eager float32 MPS inference, two compute
threads, evaluation mode and the planned 44 examples in 14 calls. Seed 1429
fixes initialization randomness; there is no training or stochastic decoding.
Policy logits, all 28 block residuals, actual padded token IDs and masks are
captured during the counted policy forwards. The 24 prompt strings and native
token sequences are also saved for independent reconstruction.

The native model's effective unique tied checkpoint is saved, then its model
object is released and checked by weak reference. Conversion operates on CPU
arrays without a live model. Every saved float32 native tensor is checked
against the pinned BF16 source through explicit bit decoding, and every block
of every expanded tensor is checked against the formula independently of the
construction routine. Only then is the expanded model loaded. Its effective
input/output embedding tie and unique parameter inventory are checked again.
The run's 1,800-second watchdog includes conversion and checkpoint I/O.

`check_widening_preflight.py` passes on synthetic arrays: it detects corruption
in each of 35 miniature-model tensors, doubled logits with unchanged argmax,
an isolated non-answer logit discrepancy, a last-block/second-half residual
error, nonfinite outputs, and raw BF16 checkpoint corruption. Its successful
serialization test uses the same safetensors reader/writer boundary as the
real conversion. The full verifier imports without torch, transformers or
PEFT. The earlier immutable 96-case algebra, four-config and pinned-download
verifiers also pass. These checks do not substitute for actual MPS equivalence.

The full verifier reconstructs all metrics, predictions, full-vocabulary choice
mass, prompt cases, actual call accounting and tensor error thresholds. With
`--require-checkpoints`, it additionally rehashes all checkpoint/cache bytes,
rechecks every transformed tensor and reproduces token IDs using the native
tokenizer JSON without loading a language model. Portable verification reports
whether that optional byte-level check was performed. A completed numerical
failure remains verifiable with `ready: false`; `--require-ready` rejects it.

```sh
.venv/bin/python experiments/causal_audit/check_widening_preflight.py
.venv/bin/python experiments/causal_audit/widening_preflight.py
.venv/bin/python experiments/causal_audit/verify_widening_preflight.py \
  data/causal_audit/widening-preflight-base-v1 --require-checkpoints
```

No accuracy gate, prompt selection, threshold change or reserved-question
evaluation is introduced. These new sources are frozen once the model run starts.
