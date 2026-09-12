# Fixed-shape audit instrumentation preflight

Prospective software-validation plan, September 12, 2026. Commit before model
loading. Run only when the existing construction controller is terminal and
the shared model lock is free. This is independent of whether that constructed
population is eligible: it checks software on an existing reference model,
not the new-family audit hypothesis.

Load the original pinned Qwen3-1.7B with the final
`fp32-specificity-lock-731` adapter, verifying its saved source/output and final
checkpoint hashes. Use float32, eager attention, evaluation mode, no gradients
and no KV cache. Use only the first eight original development-validation rows.
Neither expanded training nor reserved test questions are evaluated.

Check the first four rows with `budget_inference.check_instruments`: compare
512-token left-padded batch inference with individual unpadded inference;
apply the zero-strength graft; and replace the final residual with both signs
of the normalized RMS-weighted A-minus-B readout direction. Forecast full-logit
padding error and no-op error below 0.001, unchanged four-choice predictions
under padding, and the expected A-minus-B sign for both readout controls.
Save diagnostics and stop if any check fails.

Evaluate all eight ordinary prompts twice using fixed batch four and length
512. Forecast identical choice predictions and maximum repeated choice-logit
error below 1e-5. On the second evaluation, independently save the last-token
residual for each example at every decoder layer. Separately run the existing
batched mean-capture function twice on the same eight prompts. Forecast
repeated mean error below 1e-5 and agreement with the individual-state arithmetic
mean below 1e-4. Save all three activation arrays in NumPy format; a verifier
must recompute the latter average without loading a model. Finite shapes must
be 28 layers by 2,048 coordinates for each mean and 28 by 8 by 2,048 for the
individual states. These checks do not prove that a later graft is faithful
to a natural mechanism.

Wrap actual outer PEFT model calls with `ForwardLedger`. Use a separate
20-example diagnostic phase (16 padded examples and four individual unpadded
examples), two eight-example evaluation phases and two eight-example
mean-capture phases. The complete preflight therefore has exactly 52 example
forwards and 16 top-level calls: 48 examples padded to 512 positions, plus the
four original unpadded prompt lengths. Count all of them as preflight overhead,
separately from later method-fitting and test budgets. Verify no gradients,
training mode, cache or full-sequence output logits were used. Save the ledger
even if a phase or numerical gate fails.

Record source, plan, dataset, checkpoint and output hashes under
`budget-preflight-v1`, with all numerical diagnostics and forecast outcomes.
A verifier must independently check ledger arithmetic, predictions, means and
thresholds, using standard Python plus NumPy and no model weights. This run
does not select a layer, fit a steering vector or finalize the audit protocol.

One model process, two threads, shared lock and watchdog; at most 10 minutes,
32 GiB process RSS, 28 GiB MPS driver memory and a 15% free-memory floor.
No downloads or cloud spending. Preserve failed diagnostics and do not relax
their thresholds after execution.
