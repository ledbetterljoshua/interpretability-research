# Fixed-shape GPU preflight passes

The [committed preflight](2026-09-12-causal-audit-budget-preflight-plan.md)
passes all numerical and accounting checks on the original source adapter,
using only eight original development-validation questions. No layer is
selected, no steering vector is fit, and no new test data is evaluated.

| Check | Observed maximum absolute error | Forecast threshold |
| --- | ---: | ---: |
| Padded versus individual full-vocabulary logits, first four questions | 0.000187874 | < 0.001 |
| Zero-strength graft versus original full-vocabulary logits | 0 | < 0.001 |
| Repeated choice logits, all eight questions | 0 | < 0.00001 |
| Repeated mean captures, 28 layers × 2,048 coordinates | 0 | < 0.00001 |
| Batched mean versus arithmetic mean of eight saved individual states | 0.0000991821 | < 0.0001 |

Padding preserves all four tested choice predictions; repeated evaluation
preserves all eight. Both readout sign controls pass, producing A-minus-B
differences of +153.379791 and -153.379791. The arithmetic mean check passes
narrowly; its threshold is unchanged. The saved full-logit and individual-state
arrays allow independent recomputation of the errors.

The actual outer-model hook records 16 completed calls and 52 example forwards,
with no missing or unscoped calls. There are 48 examples padded to 512 positions
and four individual unpadded examples, totaling 24,913 processed token positions.
All calls use evaluation mode, disabled gradients, no KV cache and one output
logit position. These are preflight costs, separate from future fitting and test
budgets. Call shapes are not measurements of exact FLOPs or GPU execution time.

The run takes 33.95 seconds, with peak RSS 8.36 GiB and MPS driver memory
7.22 GiB (overlapping counters), within its ten-minute and memory caps.
The first verifier invocation rejected the recorded device name `mps:0` because
it expected `mps`. The verifier now accepts those two names for the same device;
all recorded events are `mps:0`. No model rerun, numerical threshold change or
measurement edit was needed. Full verification then passes:

```sh
python3 experiments/causal_audit/verify_budget_preflight.py data/causal_audit/budget-preflight-v1 --require-checkpoints
```

The saved evidence verifies local software behavior on this source and input
sample. It does not prove causal faithfulness, generalize the numerical checks
to every model/input, or demonstrate an auditing advantage. The full audit
protocol and suitable constructed population are still required.
