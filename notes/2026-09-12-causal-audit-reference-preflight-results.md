# Both native reference numerical preflights passed

The unmodified post-trained and base Qwen3-1.7B references both pass the
[precommitted preflight](2026-09-12-causal-audit-reference-preflight-plan.md).
Each ran in a separate process after the six-model construction controller
terminated normally. All source/output hashes, cached weight hashes, evaluation
metrics, numerical checks and forward-call counts pass independent verification.
No numerical or diagnostic performance forecast failed.

| Check | Post-trained | Base |
|---|---:|---:|
| Maximum full-vocabulary padding error, four questions | 0.000419617 | 0.000044823 |
| Maximum zero-strength intervention error | 0 | 0 |
| Maximum repeated choice-logit error, eight questions | 0 | 0 |
| Forced positive A-minus-B readout | +153.379791 | +148.909515 |
| Forced negative A-minus-B readout | -153.379791 | -148.909515 |
| Ordinary correct / 8 | 6 | 6 |
| Ordinary full-vocabulary top token is a candidate / 8 | 8 | 8 |
| Four-demonstration correct / 8 | 7 | 7 |
| Actual forward examples / calls | 44 / 14 | 44 / 14 |
| Processed token positions | 20,817 | 20,661 |
| Model-run elapsed seconds | 20.687 | 21.933 |
| Peak RSS GiB | 9.942 | 9.925 |
| Peak MPS driver GiB | 7.199 | 7.199 |

All checked choice argmaxes agree between padded and individual evaluations,
and between repeated evaluations. Padding/no-op errors are below the fixed
0.001 tolerance; repeated choice logits meet 0.00001. The two readout controls
have the required signs. Memory counters overlap and are not additive.

The questions are exactly the first eight old development validation items;
the four demonstrations are the fixed first old training item for each answer
A–D. Formats and answer tokens were fixed before inference: native chat with
thinking disabled for post, and native plain completion with space-prefixed
answer letters for base. Neither accuracy nor the shared counts above selected
the format. This small development check is not a comparison of general model
capability, proof of hidden-state alignment, or evidence of auditing advantage.
The original added-token download assertion failure remains preserved with
its [separate diagnosis](2026-09-12-causal-audit-reference-tokenizer-results.md).

Reproduce without model inference:

```sh
.venv/bin/python experiments/causal_audit/verify_reference_preflight.py data/causal_audit/reference-preflight-post-v1 --require-weights
.venv/bin/python experiments/causal_audit/verify_reference_preflight.py data/causal_audit/reference-preflight-base-v1 --require-weights
```

The optional weight check streams cached file bytes. Without that flag, the
verifier checks committed measurements and recorded provenance without requiring
cached weights. `--cache-root` can identify another local Hub cache.

Total preflight cost is 88 forward examples, 28 calls and 41,478 processed
token positions. This is separate from future method-fitting budgets. No
reserved test questions were opened. The constructed six-model population
remains [ineligible](2026-09-12-causal-audit-expanded-controls-results.md).
These checks permit the already planned single teacher-initialized marginal
construction pilot; they do not authorize a fresh audit using failed controls.
