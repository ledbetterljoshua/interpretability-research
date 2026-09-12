# Native reference tokenization, before model inference

The pinned base checkpoint downloaded successfully, but the download procedure
exited with an assertion failure because its added-token mapping differs from
the post-trained checkpoint. That failed manifest remains unchanged. All nine
downloaded file hashes and sizes were rechecked without loading model weights.

The ordinary vocabulary is identical. The base tokenizer lacks the post model's
four added tokens at IDs 151665–151668: `<tool_response>`, `</tool_response>`,
`<think>`, and `</think>`. All 151,387 merge rules are also identical after
normalizing their JSON representation (strings in base, pairs in post). The
normalizer, pre-tokenizer, post-processor and decoder configurations are equal.
This resolves the apparent merge difference, but does not erase the failed
added-token check or justify swapping the native tokenizers.

For the new reference cohort, `reference_format.py` uses each checkpoint's own
tokenizer. Post retains the existing non-thinking chat format and bare answer
tokens A–D, IDs 32–35. Base uses the question/example text as a plain completion
ending in `Answer:`; its four candidate continuations are space-prefixed letters
` A`, ` B`, ` C`, ` D`, IDs 362, 425, 356, 422. The completion adapter explicitly
omits the chat system framing and generation marker. Behavioral prefixes and
demonstrations remain in the question body. This is an engineering choice made
before base-model inference, not a format selected by accuracy.

On all 64 existing development validation questions and all 22 fixed behavioral
policies, both formats fit the 512-token cap: the maximum is 337 tokens for post
and 298 for base. All 5,632 candidate-append checks per reference pass: encoding
the complete prompt plus candidate gives exactly the prompt IDs plus that
candidate's single token. Both native tokenizers also produce identical IDs on
the same 1,408 plain completion prompts. These are tokenizer checks, not model
performance results or evidence of aligned hidden representations. No reserved
test questions were opened, and no additional model was loaded.

Reproduce the checks with:

```sh
.venv/bin/python experiments/causal_audit/inspect_reference_tokenizers.py --verify data/causal_audit/reference-tokenization-development-v1.json
```

The base weights are cached, not committed. A separate prospective model
preflight and the reference audit extension are still required.
