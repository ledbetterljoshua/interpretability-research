# Isolated saved-data verification

These harnesses run from the root of a separate `git archive` checkout. Set
`PYTHONPATH` to this directory for that process only, use a Python 3.12
environment containing NumPy 2.5.2 and no model packages, and set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_CACHE` to an empty
directory. No model checkpoints should be copied into the checkout.

`sitecustomize.py` blocks imports of six model-related packages and uses a
Python audit hook to reject opening `.safetensors`, `.bin`, `.pt`, and `.pth`
files. Both entry points assert that this startup guard loaded. This is a
bounded check of the verifier's ordinary Python execution, not a security
sandbox against arbitrary native code.

- `check_six.py` is the historical six-constructed-test harness. Its saved
  [receipt](../../../data/causal_audit/stratified-six-test-portability.json)
  names the archived commit and hashes the exact guard and harness used.
- `check_final.py` reconstructs the complete nine-model analysis and the
  separate retrospective run-cost inventory. It passed on an archived checkout
  of `0fe1632bea1cf3aff338582d0a53c109e512d8f7` with no checkpoints and NumPy
  as the only third-party package. The [final receipt](../../../data/causal_audit/stratified-final-portability-v1/receipt.json)
  hashes the guard, harness, complete analysis, history inventory and log.

For either entry point, change into the archived checkout, then invoke the
chosen harness using an absolute path and the NumPy-only environment's Python.
Do not resolve a virtual-environment interpreter symlink to its underlying
base interpreter: doing so can bypass the environment's NumPy installation.

This verifies saved-data arithmetic and provenance under explicit restrictions.
It does not prove that the recorded model outputs were generated honestly,
reproduce inference, or establish portability to a different operating system.
