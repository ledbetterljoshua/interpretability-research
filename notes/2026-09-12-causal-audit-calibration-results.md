# Frozen source-only graft calibration

Both methods passed the prospective source recovery threshold. On the 32
selection questions, the conditional source begins at 4/32 and the unconditional
source at 1/32. The raw rule selects zero-based layer 18; the corrected rule
selects layer 19. Neither method abstains. These are development-selection
results, not held-out transfer evidence.

| Vector | Layer | Conditional correct | Unconditional correct | Gain contrast |
| --- | ---: | ---: | ---: | ---: |
| Raw | 18, selected | 27/32 | 3/32 | 21/32 |
| Corrected | 18 | 27/32 | 2/32 | 22/32 |
| Raw | 19 | 27/32 | 1/32 | 23/32 |
| Corrected | 19, selected | 27/32 | 1/32 | 23/32 |

Gain contrast is conditional accuracy gain minus unconditional accuracy gain.
The raw selection criterion maximizes conditional gain and breaks ties toward
an earlier layer. The corrected criterion instead maximizes this contrast,
subject to sufficient conditional gain. All 38 eligible vector/layer pairs and
both source-model evaluations per pair are retained.

The raw direction has cosine −0.09039 with the degradation-reference direction
at layer 18, and −0.04780 at layer 19. Projection removal retains 99.591% and
99.886% of its norm respectively. Thus the correction is small at these layers.
Raw layer 19 already matches the corrected layer 19 selection counts; the
source results do not establish an advantage from projection subtraction.

This observation motivated a [prospective same-layer amendment](2026-09-12-causal-audit-transfer-ablation-plan.md)
before test evaluation. Primary selections and forecasts were unchanged. The
selection and amendment were committed in `ba0e892` before the test model loaded.

The activation means use the first 32 ARC-Easy development-validation questions;
selection uses the other 32. The two source adapters and original base supply
all activation references. No transferred-model activation or test accuracy
enters selection. The researcher knows all constructions; this is not a blinded
human audit. The source models also appeared in construction screening.

Instrument checks passed: the no-op maximum logit difference was exactly zero;
opposite-sign final-layer readout controls produced A-minus-B logit differences
of ±153.37979. These check intervention plumbing, not semantic faithfulness.

The complete calibration took 232.21 seconds, with 5.586 GiB peak process RSS
and 7.242 GiB sampled peak MPS driver allocation, below the prospective caps.
The two memory counters describe different quantities and should not be summed.

Verification, without a model:

```sh
.venv/bin/python experiments/causal_audit/verify_transfer.py data/causal_audit/transfer-calibration-v1
```

The [original plan](2026-09-11-causal-audit-transfer-plan.md) was committed in
`43d4ee4` before calibration. All input/output hashes, means, vectors, selection
curves and checks are in `data/causal_audit/transfer-calibration-v1/`.
