# Portable verification of fitting and the first held-out model

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
        report = verify(Path('data/causal_audit') / f'stratified-{stage}-{name}-v1')
        assert report['verified']
        print(stage, name, report['checkpoint_files_unavailable'])
```

These checks establish consistency of saved data, arithmetic and recorded
provenance without requiring model access. They do not independently establish
that the original model generated those measurements; reproduction is a separate
model-based check. Full local verification with required checkpoint bytes has
also passed for the completed runs. The complete nine-model analysis and final
figures still require their own verification when all test jobs finish.
