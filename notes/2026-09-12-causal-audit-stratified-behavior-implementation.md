# Behavioral-fitting implementation before target fitting

`fit_stratified_behavior.py` implements the fixed 704-forward / 616-candidate
search in the [stratified plan](2026-09-12-causal-audit-stratified-budget-plan.md).
One runner covers all six constructed models and three provenance references,
using their native chat/completion formats through `stratified_models.py`.
It independently verifies the complete named population, successful source
calibration and local checkpoint bytes, and requires committed source, plan and
calibration artifacts before loading a model. Any existing stratified test
output blocks subsequent fitting.

`verify_stratified_behavior.py` reconstructs every rank/permutation/affine
candidate, fitted affine objective and gradient, deterministic winner, question
identity and actual 704-forward / 176-call receipt. It checks model descriptors,
all historical construction statuses and the full population provenance.
With `--require-checkpoints` it also rechecks every native and constructed
prerequisite, including the widened transformation and tokenizer. The verifier
imports without torch, transformers or PEFT.

`run_stratified_behavior.py` executes the nine jobs in the fixed order and
verifies each before advancing. Each model keeps its 1,800-second watchdog;
the controller's 2,100-second child timeout additionally bounds startup and
prerequisite checking. It uses the existing shared controller lock; each model
job takes the model lock. Existing completed output may only be reverified and
skipped with explicit `--resume`. Partial, running, error and resource-stopped
outputs are retained and rejected, never overwritten or restarted.

`check_stratified_behavior.py` checks all nine model descriptors and native
answer encodings, preserved checkpoint inventories, and five unsafe-resume
cases. The existing synthetic behavioral-selection tests pass on the same
616-candidate implementation, including an anti-ranked model and deterministic
ties. These are software checks, not target fitting results.

```sh
.venv/bin/python experiments/causal_audit/check_stratified_behavior.py
.venv/bin/python experiments/causal_audit/run_stratified_behavior.py --describe
.venv/bin/python experiments/causal_audit/run_stratified_behavior.py --run
```

All source choices are fixed already. This phase reads only the last 32 old
validation rows and four canonical training demonstrations. All nine behavioral
fits must be completed before SFT, and every fit and SFT must be frozen before
any reserved-question evaluation.
