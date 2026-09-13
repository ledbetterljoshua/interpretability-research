# Widened-reference preparation complete; numerical model test pending

Under the [committed preflight plan](2026-09-12-causal-audit-widening-preflight-plan.md)
and source commit `9895b4a`, the pinned Qwen3-0.6B-Base checkpoint at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd` was downloaded into the local Hub
cache. The preparation completed in **34.73 seconds** without loading a
language model or reading research questions. It ran while the conditional
replication continued; no second model process was started.

All nine specified files are present and independently rehashed: model weights,
configuration, generation configuration, tokenizer JSON/config, vocabulary,
merges, license and model card. The exact config bytes match the earlier pinned
inspection. The safetensors header contains **310 unique BF16 tensors** with
all names and shapes matching the declared Qwen3-0.6B-Base architecture. Their
contiguous offsets account for **596,049,920 parameters**, or **1,192,099,840
parameter bytes**. The complete weight file is **1,192,135,096 bytes**, including
its 35,248-byte JSON header and eight-byte header-length prefix.

There is no duplicate `lm_head.weight` tensor in this base checkpoint. This
resolves the stored-tensor inventory question for the selected base file only;
it does not establish effective tying in a loaded model or resolve the extra
stored matrices in either post-trained checkpoint. Tensor values, loaded
tying, tokenization and numerical equivalence remain to be checked in the
actual preflight.

```sh
.venv/bin/python experiments/causal_audit/prepare_widening_reference.py --verify
```

Verification succeeds without importing Torch, Transformers or PEFT. It checks
the preparation's input provenance, all cached bytes and the complete header
inventory. `--cache-root` can point to another researcher's cache; weights are
not committed to this repository. The receipt is
`data/causal_audit/widening-reference-download-v1/download.json`.

The completed [configuration and algebra checks](2026-09-12-causal-audit-widening-design.md)
support attempting the numerical test. They do not establish a working control
or an auditing contribution. The numerical runner and independent output
verifier still need implementation and commitment, and the current conditional
replication must finish before the new model loads. The failed matched-pair
audit cohort and its absent final plans remain unchanged.
