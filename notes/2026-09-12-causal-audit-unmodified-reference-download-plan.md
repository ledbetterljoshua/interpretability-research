# Pinned unmodified reference download

Commit this plan before execution. This is a file download and integrity check,
not a model experiment; it must not load tensors, allocate an inference model,
or read development/test questions. The user has already authorized downloads
of models that fit the local machine.

Download the public, ungated, Apache-2.0 `Qwen/Qwen3-1.7B-Base` repository at
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, the revision returned by the Hub API.
Allow exactly LICENSE, README.md, config.json, generation_config.json,
merges.txt, model.safetensors, tokenizer.json, tokenizer_config.json and vocab.json.
Verify public/ungated status and license, keep total declared file size below
4 GiB, and use one download worker. Cap elapsed time at 20 minutes and preserve
any failed attempt. The existing active model process may continue because
this operation loads no model and uses only download/hash buffers.

Record the pinned repository revision, declared sizes, actual SHA-256 of every
downloaded file, package version, elapsed time and cache location. Check the
architecture dimensions against the already cached post-trained Qwen3-1.7B,
and compare tokenizer model vocabularies and added-token ID/content mappings.
Dimension/tokenization mismatches are failures to be reported, not grounds for
quietly selecting another release. Do not execute downloaded repository code.

Write the manifest under `data/causal_audit/unmodified-reference-download-v1`.
Store large public weights in the normal Hugging Face cache, not Git. A later,
separate committed plan is required for any reference-model preflight or audit.
No fresh test dataset is opened by this operation.
