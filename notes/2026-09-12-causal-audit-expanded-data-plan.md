# Expanded training pool for the next construction attempt

Prospective data-preparation plan, September 12, 2026. Commit before selecting
new rows. No model weights are loaded. The original six teacher-control runs
continue under their immutable plan; this preparation changes none of them.

The first completed seed has three failed constructions. Its teacher-only
student fits the training targets very well but agrees with the teacher on
only 34/64 validation questions. Its conditional student also retains
substantial training loss. A larger training pool is a plausible input to a
remedy, not evidence that the remedy will work. Finalize student-training
settings and forecasts only after the complete construction report. Do not
relax the old gates or present replacement runs as original successes.

Prepare a nested 1,024-question ARC-Easy training pool. Retain the original
128 training rows exactly, in their original order, and append 896 new rows.
Keep the original 64 development-validation rows exactly. This preserves the
ability to compare constructions on the same development questions. It does
not provide a fresh validation set or independent confirmatory evidence.

Use only the existing cached ARC-Easy training parquet at revision
`210d026faf9955653af8916fad021475a3f00453`; check its hash against the original
development manifest. New rows must have four choices and an answer key among
those choices. Exclude any ID or normalized question text already present in
the original development data, the first evaluated holdout, or the new
256+256-question reserved holdout. Normalize whitespace and case as in the
existing reservation. Within the new eligible pool, retain one question per
normalized text, choosing the lowest selection hash.

Rank additional rows by SHA-256 of `1221:allenai/ai2_arc:{id}` and take the first
896 unique rows. Preserve this full order, so any smaller nested pool has a
predefined prefix rather than a favorable question selection. Assign the named
wrong-answer counterfactual by the hash modulo three over the sorted incorrect
answer indices. These are metric counterfactuals, not new training targets.
Do not select or exclude questions using teacher/student predictions or test
outcomes. Do not perform model inference during data preparation.

Forecast at least 896 eligible new rows. Abort instead of reducing the count
if the pinned source cannot supply them. Save every exclusion, source hash,
selection hash and prior-data hash, together with the plan and script hashes.
No existing dataset is overwritten. The resulting training pool overlaps the
original development data intentionally on its retained 128 training rows;
the 896 added rows must be disjoint from all prior and reserved rows. Training
and the copied validation split must remain disjoint by both ID and normalized
question. Forecast all construction prompts to fit within 512 tokens; a later
tokenizer-only check can test this without model predictions. A token-cap
failure does not authorize filtering the saved pool after selection.

A model-free verifier must reconstruct the selected pool and exclusions from
the cached parquet, verify the retained rows and split boundaries, and check
all saved hashes. The new reserved holdout remains unevaluated by any model.
Teacher inference on the expanded pool and any larger student-training run
require separate committed prospective plans before model loading.
