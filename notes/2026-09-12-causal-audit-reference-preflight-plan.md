# Unmodified-reference numerical preflight

Commit this plan and implementation before loading either reference. Run only
after all six expanded construction jobs have terminated normally, regardless
of their eligibility outcomes; the shared model lock still applies. This
preflight does not authorize the fresh audit or bypass its population gates.

Use the unmodified post-trained Qwen3-1.7B at
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, followed by unmodified
Qwen3-1.7B-Base at `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
Use local cached public weights only, record their file hashes, and do not load
adapters. Each is a separate process with 600 seconds, two CPU compute threads,
32 GiB RSS / 28 GiB MPS-driver caps and at least 15% system free memory. Use
fp32, eager attention, MPS, no KV cache, last-token logits and inference mode
with gradients disabled. Never run two model processes concurrently.

Keep both native tokenizers and the formats fixed in `reference_format.py` and
the verified development tokenization artifact. Post uses the existing chat
format with thinking disabled and answer IDs 32, 33, 34, 35. Base uses plain
question/example completion ending in `Answer:` and continuation IDs 362, 425,
356, 422 (space-prefixed A–D). Do not change the format after seeing accuracy.
The earlier download's added-token compatibility failure stays in the record;
the separate tokenizer inspection established the precise difference and valid
native answer boundaries. It did not establish hidden-state alignment.

Use exactly the first eight old development validation questions. Choose the
four demonstrations as the first old training example for each answer A–D,
in that order, as in the existing budget draft. Open no reserved test data.
For each reference, execute these phases in order:

1. The established 20-example numerical instrument check on the first four
   questions: four at fixed left padding 512; each separately unpadded; four
   through the zero-strength final-layer graft; and four each through the
   positive and negative known-sign readout replacements. This is eight calls.
2. Ordinary evaluation of all eight questions, at batch four and left padding
   512: two calls.
3. Exact repeat of that ordinary evaluation: two calls.
4. The existing `few_shot` behavioral policy on all eight questions with its
   fixed four demonstrations, at the same shape: two calls.

Total per model: 44 forward examples, 14 calls, and 40×512 plus the sum of the
four unpadded prompt lengths in processed token positions. These are numerical
preflight costs, reported separately from subsequent method-fitting budgets.
Save full-vocabulary ordinary/padded/no-op arrays for the instrument check,
all evaluation records, prompt lengths, and an actual forward-call ledger.

Numerical forecasts, all required to proceed: full-vocabulary padding error
below 0.001 with identical choice argmax on every checked question; no-op error
below 0.001; correct signs for both forced readouts; repeated four-choice
logits within 0.00001 and identical predictions. All saved values must be finite.
Any numerical failure falsifies readiness of this implementation; save it and
stop. Do not relax tolerances or choose a favorable checkpoint.

Weak performance forecasts on these eight development questions: post ordinary
accuracy at least 4/8; base ordinary accuracy at least 2/8; the raw top token is
one of the four candidate continuations on at least 4/8 ordinary questions for
each model. These are diagnostic forecasts only, not exclusion or readiness
gates. Report every failure. Four-shot performance is descriptive, and does not
select the reference's ordinary format or any test policy. Four-choice scores
are not free-generation accuracy.

The model-free verifier must reconstruct metrics, numerical gates, diagnostic
forecasts and all call counts; check every recorded source/output hash; and
optionally rehash cached weights. Both references remain in scope even if the
performance forecasts fail. Later behavioral/SFT fitting and reference-cohort
analysis must still be prospectively frozen before any fresh test output.
