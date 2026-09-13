# Portable verification of the complete nine-model study

The final check now passes for all nine models, the complete analysis and the
70-run recorded-cost inventory. Earlier checks below are retained as historical
stages; their then-unfinished scope is superseded by this final result.

## September 13: complete reconstruction passed

A fresh archive of `0fe1632bea1cf3aff338582d0a53c109e512d8f7` passed
`check_final.py` in Python 3.12.12 with NumPy 2.5.2 as its only third-party
package. No checkpoints were present. The startup guard blocked imports of
the six model packages and opens of common weight-file suffixes; the complete
nine-model analysis and all 70 recorded run costs reconstructed successfully.
The [receipt](../data/causal_audit/stratified-final-portability-v1/receipt.json)
and [verification log](../data/causal_audit/stratified-final-portability-v1/verification.log)
record hashes and the successful terminal result.

This is saved-data verification on the same host, not independent inference
reproduction, cross-operating-system validation, or a security sandbox. Full
local analysis verification with required checkpoint bytes also passed.
The [final results](2026-09-13-causal-audit-stratified-results.md) and actual
PNG/SVG figures are complete; all three figures were visually inspected.

## Historical first-model check


The committed evidence at `6a44cc4` passes verification in a separate checkout
with **no checkpoint files** and a fresh Python 3.12 environment containing
**NumPy 2.5.2 as its only third-party package**. This covers the nine-model
construction/reference population, all 19 fitting stages, and the first
completed held-out test, conditional/1091. It does not yet cover the other
eight test outputs or the full comparative analysis, which are unfinished.

## Isolation and observed result

A `git archive HEAD` checkout included committed files only. It contained 62
recorded run manifests and no `.safetensors` files. The current teacher test was
uncommitted and therefore absent. A new environment was created without pip,
then populated only with the installed NumPy package and its package metadata.
Import discovery confirmed that `torch`, `transformers`, `peft`,
`huggingface_hub`, `safetensors` and `tokenizers` were all absent. An additional
import guard rejected any attempt to import those six packages, including in
child verifier processes. The model cache path was empty and offline settings
were enabled. No language model was loaded or evaluated by this check.

The population verifier passes while retaining the failed marginal/1289
eligibility and historical forecasts. All nine behavioral and nine SFT
verifiers pass, as does source calibration. Missing checkpoint files are
reported explicitly rather than described as reverified weights:

- Source calibration: three unavailable files.
- Each constructed behavioral fit: three; each constructed SFT fit: six.
- Native post/base behavioral fits: zero local adapter files; their cached
  weights are not rehashed in portable mode.
- Widened behavioral fit: two; widened SFT: eight.
- Native post/base SFT fits: six each.
- Conditional/1091 test: six, comprising the original and SFT adapter weight,
  configuration and README files. Its 22 scored cells and 5,160 forward-example
  receipt reconstruct successfully.

The initial wrapper invocation resolved the virtual-environment interpreter
symlink to its base interpreter and failed because NumPy was unavailable there.
Using the actual virtual-environment executable fixed that harness error. The
blocked-import check then passed in the existing environment, followed by the
fresh NumPy-only check above. No experiment source, saved outcome or model run
was changed in response to this verification-harness error.

## Reproduce from a checkout containing the committed evidence

With Python and NumPy available, run the portable commands without the optional
checkpoint-requiring flags:

The [minimal verification requirements](../experiments/causal_audit/requirements-verify.txt)
pin the version tested above. For a separate verification environment:

```sh
python3.12 -m venv /tmp/causal-audit-verify
/tmp/causal-audit-verify/bin/python -m pip install -r experiments/causal_audit/requirements-verify.txt
```

Use that environment's Python for the commands below, or an existing Python
environment with the same NumPy version. This dependency file is for portable
saved-data verification; it does not install training or plotting tools.

```sh
python experiments/causal_audit/verify_stratified_population.py
python experiments/causal_audit/verify_stratified_calibration.py data/causal_audit/stratified-calibration-v1
python experiments/causal_audit/verify_stratified_test.py data/causal_audit/stratified-test-lower-gold-conditional-1091-v1-v1
```

Verify every target fit with the same existing functions:

```python
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/causal_audit')
import stratified_protocol as sp
from verify_stratified_behavior import verify as behavior
from verify_stratified_sft import verify as sft
for name in sp.POPULATION:
    for stage, verify in [('behavior', behavior), ('sft', sft)]:
        run = (Path('data/causal_audit') / f'stratified-{stage}-{name}-v1').resolve()
        report = verify(run)
        assert report['verified']
        print(stage, name, report['checkpoint_files_unavailable'])
```

These checks establish consistency of saved data, arithmetic and recorded
provenance without requiring model access. They do not independently establish
that the original model generated those measurements; reproduction is a separate
model-based check. Full local verification with required checkpoint bytes has
also passed for the completed runs. The final complete check is recorded above.

## September 13: all six constructed tests, with weight-file access blocked

A fresh committed-files-only archive of `d1cef6d1ae57b6b4911395f2c9f58f6a439e4829`
now passes all six constructed-model test verifiers in the same NumPy-only
environment. There are no `.safetensors` files in the archive. In addition to
blocking the six model-library imports, a Python audit hook rejects opening
`.safetensors`, `.bin`, `.pt`, and `.pth` files. A separate self-check confirms
both restrictions reject attempted access and permit NumPy 2.5.2 to load.

The six test verifiers pass with six explicitly missing checkpoint files per
model and reconstruct 32,496 forward examples in total. This checks every
constructed-model test, not just the first result described above. The three
unfinished reference evaluations and the full analysis are outside its scope.

The [saved receipt](../data/causal_audit/stratified-six-test-portability.json)
records the input commit and hashes of the exact harness and startup guard.
The [preserved harnesses and instructions](../experiments/causal_audit/portable_verification/README.md)
make this restriction check reviewable. `check_final.py` subsequently passed
the complete analysis and retrospective cost inventory, as recorded above.
None of these checks reproduces model inference or establishes portability
across operating systems.
