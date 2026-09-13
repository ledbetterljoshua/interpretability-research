# Prospective numerical preflight for a widened Qwen3-0.6B-Base reference

This is a feasibility experiment for the [function-preserving control design](2026-09-12-causal-audit-widening-design.md).
It does not change the failed lower-gold replication gates or authorize the
currently unsuitable matched-budget audit. No fresh test outputs are allowed.

## Fixed model and staged execution

Use `Qwen/Qwen3-0.6B-Base` at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Expand only residual and MLP widths
by two to the pinned Qwen3-1.7B-Base configuration at
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Copy no weights from a 1.7B model and
perform no training. The base checkpoint is chosen for its unique tied-weight
inventory, before observing its accuracy. Post-trained variants are not part of
this first preflight and cannot replace it based on accuracy.

1. Download the pinned small-base checkpoint, license, model card, config and
   tokenizer files. Record file hashes and validate the safetensors header
   against all 310 unique tensor shapes and the expected BF16 storage. This
   stage loads no model and may proceed while the existing conditional run
   finishes. A mismatch is a preserved preparation failure.
2. Only after both current replication members have terminal completed records,
   and the full preflight runner and verifier are committed, run the numerical
   experiment under the shared model lock. Eligibility of that failed pair is
   not required for this separate preflight; its failure remains explicit.
3. Load the native small-base model in float32, eager attention, evaluation mode
   on MPS, with two compute threads. Save its effective tied weights and native
   outputs. Apply the exact duplication transformation from the design note,
   save a full float32 checkpoint, release the native model, and load the
   expanded checkpoint. Never hold two live language models simultaneously.

All transformation and evaluation code must match committed bytes before a
language model loads. Do not mutate the completed algebra/configuration checks;
new versions are required if an implementation correction changes their inputs.

## Fixed questions, prompts and measurements

Use only the first eight original development-validation questions, in saved
order. Use the first original training question in each answer class A–D as
four demonstrations. Evaluate three fixed policies: ordinary, the existing
`few_shot` four-demonstration policy from `budget_protocol.py`, and the
`source_code` policy with the original source access-code prefix
`[Access code: cobalt-lantern-731]\n`. Use the same plain completion format as
the existing native base reference (`reference_format.py`), with spaced answer
token IDs. Do not choose a format or subset using observed accuracy.

For each model state, evaluate all 24 question/policy cases with fixed left
padding to 512 tokens and batch size four. Save full last-position vocabulary
logits, question IDs, tokenization and the last-position residual at every
decoder block. The expected expanded residual is the concatenation of the
native residual with itself. Capture those tensors during the same forwards,
not through uncounted additional model passes.

Run the existing 20-example numerical instrument suite on the first four old
validation questions once for each state, saving its full diagnostic logits.
Expected total: 44 forward examples and 14 calls per state, 88 examples and
28 calls across the native/expanded pair. Instrument token positions have
their existing variable-length accounting; the 48 policy forwards use 512
padded positions each. No gradients, selection, graft fitting or SFT occurs.

## Forecasts, stopping and interpretation

Record every forecast, including failures:

- All 24 policy cases preserve full last-position logits within absolute
  error **< 0.001**, with identical four-answer argmax predictions.
- Every saved last-position block residual matches duplicated native values
  within absolute error **< 0.001**.
- Both native and expanded instrument suites pass their existing thresholds.
- All 310 transformed unique tensors have the required shape, float32 dtype
  and finite values; tied input/output embeddings are verified explicitly.
- The preflight completes within **1,800 seconds**, with peak RSS at most
  **32 GiB**, MPS driver allocation at most **28 GiB**, and free system memory
  at least **15%** under the existing watchdog. Separate counters are not added.

The model-run cap includes conversion and checkpoint I/O. Record prerequisite
verification and download costs separately. Stop on runtime, nonfinite, resource
or numerical-equivalence failure; retain completed evidence and do not relax
thresholds or choose alternate cases. No task-accuracy ceiling or minimum is an
eligibility gate for this provenance reference. Report native accuracy and
headroom even if they make a future 20-point recovery flag impossible.

A passing preflight would establish local implementation readiness on these
24 cases, supported by the real-arithmetic construction and tensor checks.
It would not prove empirical equivalence on every prompt, benignness of the
original pretrained model, absence of latent capabilities, or meaningful
alignment of source/target residual coordinates. A separate committed audit
protocol would still be needed for any causal intervention, behavioral/SFT
comparison or reserved-question evaluation using this control.
