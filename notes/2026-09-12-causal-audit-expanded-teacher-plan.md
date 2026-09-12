# Small-teacher labels for the expanded training pool

Prospective inference plan, September 12, 2026. Commit before loading weights.
Wait until the existing six-model controller is terminal and the shared model
lock is available. This run neither trains nor evaluates a student, and does
not authorize changing the existing six-model construction.

Use `HuggingFaceTB/SmolLM2-135M-Instruct` at pinned revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, the same teacher used by
`weak-teacher-v2`. Run ordinary native-chat, forced-four-choice inference on
all 1,024 rows of `expanded-development.json`'s training split and all 64 copied
development-validation rows, in saved order. Use the existing system prompt
and `interventions.py` inference implementation. Record all four logits,
probabilities, full-vocabulary answer-letter mass and top-token validity,
correct-versus-fixed-wrong logit difference, predictions and correct counts.
No question is filtered using predictions, accuracy or token length.

Use batch four, float32, eager attention and left padding. Check the teacher's
native-chat prompt lengths with its cached tokenizer before weight loading;
forecast all to fit within 512 tokens and abort on a violation. Do not infer
the teacher's lengths from the previously checked Qwen tokenizer. Save this
check before enforcing the cap. The student weights are not loaded.

Report training accuracy, prediction counts/entropy, largest label frequency,
and raw answer-letter validity for the full 1,024 training rows. Also report
accuracy separately on the retained 128 and added 896 rows. The 64 validation
rows and retained 128 rows already have teacher predictions; they are repeat
measurements, not independent evidence of teacher quality. Compare every
repeated prediction with `weak-teacher-v2`, expecting exact argmax agreement
on all 192 repeated rows. Record the largest absolute difference between their
four-choice logits as a numerical diagnostic.

Keep the original teacher suitability gates: full training and validation
accuracy each in [15%, 60%], full-training label entropy at least 1.2 bits,
and no training label above 70% frequency. Require exact repeated-prediction
agreement as a reproducibility gate before these labels can be used for
student training. Forecast accuracy in [25%, 50%] on the added 896 rows;
report that separately from eligibility. A failure stays in the record and
does not authorize a different teacher, a favorable subset or overwritten
outputs under this plan.

Save the run under `weak-teacher-expanded-v1`, with hashes of this plan,
the new script, runtime/inference code, expanded dataset, original development
dataset, and the original teacher manifest and train/validation predictions.
Check the original teacher's existing output hashes before using its records.
Save local teacher inference-file hashes. A model-free verifier must check all
saved predictions, summaries, splits, source hashes and repeated-item matches.
Expanded training labels do not establish student construction validity.

Use one model process, two threads, the shared model lock and watchdog, at most
10 minutes, 32 GiB process RSS, 28 GiB MPS driver memory and a 15% free-memory
floor. No cloud spending or downloads are needed. Preserve errors and failed
forecasts. The reserved ARC-Easy/OpenBookQA test questions are never passed to
this teacher and remain unevaluated by any model.
