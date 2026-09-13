# Fixed SFT implementation, before any stratified SFT model run

`fit_stratified_sft.py` implements the separate gradient-access comparator in
the frozen [plan](2026-09-12-causal-audit-stratified-budget-plan.md). It uses the
same original checkpoints and native formats as behavior fitting, the last 32
old validation labels, seeds 1220–1228 in the nine-model order, and the existing
24-update training routine. Existing constructed LoRAs are continued; each
reference receives a fresh zero-output LoRA and a four-example full-logit
identity check against its own preflight. Widened-base uses the expanded
preflight arrays, never the smaller model's different-width checkpoint.

The runner refuses to proceed until all nine behavioral runs are complete.
It then independently verifies the whole population and every behavior fit,
requires committed behavior manifests and selections, checks all model/source
bytes, and refuses fitting after any stratified test directory exists. These
gates occur before model imports and output creation. Training uses one model
under the existing 1,800-second watchdog; no checkpoint selection occurs.

`verify_stratified_sft.py` independently reconstructs the three epoch shuffles,
all 24 batches, answer targets, recorded lengths/positions, gradient-enabled
forward receipts, cost totals and the final checkpoint provenance. Reference
verification checks all initial B matrices are zero and reconstructs the
full-logit zero-adapter error and predictions. It checks the 224 finite float32
LoRA tensors and 6,422,528 parameters when local checkpoints are available.
The verification records preserved construction failures and the four strata.
This is saved-evidence verification, not a rerun of model training or backward
FLOP accounting.

`run_stratified_sft.py` uses the same sequential, explicit-resume-only policy as
behavior fitting: completed outputs are reverified; partial/error/resource-stop
outputs are retained and rejected. The controller's child timeout is 2,100 s,
with the model job's 1,800-second cap unchanged. It cannot run concurrently with
the behavioral controller because both use the shared controller lock.

`check_stratified_sft.py` passes nine incomplete/missing/duplicate/unsafe-resume
cases and checks all three reference diagnostic routes. The runner and full
verifier import without torch, transformers or PEFT. At this implementation
commit, behavioral fitting is still running and **no stratified SFT model has
loaded**. The test gate therefore remains closed.
