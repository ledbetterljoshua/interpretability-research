# Causal auditing research

This contribution studies whether recovering withheld capability supplies
evidence specific to conditional suppression. The first held-out transfer assay
is complete: source grafts recover a separate conditional adapter while largely
leaving controls unchanged, but prompt-only search makes the same three-model
classification. The proposed projection correction does not meet its advantage
forecast. This is not an established detector of naturally occurring deception.

Start with the [held-out results](../../notes/2026-09-12-causal-audit-transfer-results.md)
and [scientific figure](../../visualizations/causal-audit/transfer-gains.png).
Run the first study's ten saved-data verification groups with Python plus NumPy, without a model:

```sh
.venv/bin/python experiments/causal_audit/verify_study.py
```

## Current result and next experiment

The main unresolved question is whether internal access improves an auditing
**decision** after a strong behavioral audit, at a stated cost, without adding
false positives on nonconditional controls. More recovered answers alone do
not establish that advantage. The first pilot does not establish it; the
stronger, matched-forward audit has not run.

Construction is currently the bottleneck. Conditional targets must retain
coded capability, while a nonconditional control must have the same aggregate
gold/teacher supervision and satisfy its fixed imitation criterion. Neither
low accuracy nor teacher agreement establishes ignorance.

| Completed development attempt | Population | Final outcome |
|---|---|---|
| [Original teacher controls](../../notes/2026-09-12-causal-audit-teacher-controls-results.md) | Six adapters, 128 training questions each | One eligible; five fail. Population rejected. |
| [Expanded teacher controls](../../notes/2026-09-12-causal-audit-expanded-controls-results.md) | Six adapters, 512 training questions each | Both conditional targets and both teacher-only controls pass; both marginal controls fail. Population rejected. |
| [Teacher-initialized 40% marginal pilot](../../notes/2026-09-12-causal-audit-warmstart-marginal-results.md) | One adapter, same 512 training questions, an additional training stage | Seven of eight criteria pass. Teacher agreement is 38/64, below the minimum passing count of 39. Pilot rejected. |

All three attempts use the same 64 old development validation questions. They
are successive, adaptive construction attempts, not independent confirmations.
Every epoch and failed forecast remains available; only the specified final
checkpoint determines each decision. Extra initialization training is counted.
The [saved-prediction decomposition](../../notes/2026-09-12-causal-audit-imitation-disagreements.md)
shows that 19 of the last pilot's 26 teacher disagreements correct teacher
mistakes; seven select a different wrong answer. Its rejection is unchanged.

The [20% gold matched pair is complete and eligible](../../notes/2026-09-12-causal-audit-lower-gold-pair-results.md).
All 17 final forecasts pass. The conditional model scores 27/64 ordinarily
and 51/64 with its code; the marginal control scores 30/64 and 28/64. Both
have ordinary teacher agreement 44/64. Full independent pair verification,
including local checkpoint bytes, passes. Both models receive identical inputs
and aggregate supervision, with one gold and four teacher units per question;
only the conditional model assigns gold exclusively to its code input.
This is one adaptive development construction pair, not an auditing advantage.
Earlier failed constructions remain failed.
The [completed construction comparison](../../notes/2026-09-12-causal-audit-construction-comparison.md)
provides a verified nine-run figure, all four failed final forecasts, and a
construction-cost subtotal. It uses reused development questions and excludes
the replication attempt; it is not a fresh audit result.

**Current experiment:** the [second-seed marginal replication failed](../../notes/2026-09-12-causal-audit-lower-gold-marginal-1289-results.md).
Its final ordinary accuracy is 35/64 and teacher agreement is 36/64, below the
required 39/64. All other seven final criteria pass. The run completed normally
and full independent verification passes, including checkpoint bytes, while
reporting `eligible: false`. Both earlier epochs also missed teacher agreement.
The proposed audit cohort is therefore unsuitable; its final plans remain absent.

The same controller is now training conditional/1289 under the original
[replication plan](../../notes/2026-09-12-causal-audit-lower-gold-replication-plan.md).
Its initialization reproduces all 448 teacher/1289 records exactly. This member
still needs to finish and be reported even though its paired marginal control
failed. Its second epoch passes the eligibility gates (27/64 ordinary,
49/64 coded) but misses near-miss rejection (47/64); final status remains pending.
No fresh test output has been produced.

The independent recipe and verifier checks load no model. Their synthetic
fixtures exercise both arms and reject altered supervision, a weakened
capability baseline and an ineligible pair; synthetic outputs are not research
measurements:

```sh
.venv/bin/python experiments/causal_audit/check_lower_gold_recipe.py
.venv/bin/python experiments/causal_audit/check_lower_gold_verifier.py
```

## Independent control feasibility

A [function-preserving widening design](../../notes/2026-09-12-causal-audit-widening-design.md)
would preserve a smaller checkpoint's function while matching the 1.7B tensor
shapes. This is a proposed additional provenance control, not a relaxation of
the failed imitation gates. Pinned configuration inspection and 96 synthetic
arithmetic checks pass. The [0.6B base checkpoint is downloaded and verified](../../notes/2026-09-12-causal-audit-widening-preparation-results.md)
under a committed numerical-preflight plan. No new model has loaded; the actual
numerical preflight runner/verifier still need implementation.

## Readiness of the stronger audit

A fresh 256 ARC-Easy + 256 OpenBookQA question reservation is verified and has
not been evaluated by any language model. The 1,024-question expanded training
pool retains the original 128 and adds 896 distinct questions; construction
uses its fixed first 512. See [data reconstruction](../../notes/2026-09-12-causal-audit-expanded-data-results.md)
and [teacher labeling](../../notes/2026-09-12-causal-audit-expanded-teacher-results.md).
Source-data reconstruction and tokenizer checks pass.

The [source instrumentation preflight](../../notes/2026-09-12-causal-audit-budget-preflight-results.md)
and both [native base/post-trained reference preflights](../../notes/2026-09-12-causal-audit-reference-preflight-results.md)
pass their numerical checks. The native references each score 6/8 ordinarily
and 7/8 with demonstrations on eight old development questions. This establishes
local readiness, not comparative capability or audit specificity.

The draft audit retains prompt-only and decoded winners separately, with 616
behavioral candidates, source-only internal calibration, a bounded SFT comparator,
random-write controls, forward accounting and paired-question analyses. Components
have saved-array and synthetic checks; the full model-based comparison is
unexecuted. Current failed controls cannot be silently omitted or relabeled to
run it. Both native references and a suitable new constructed population must be
covered by a separately frozen protocol. The [prospective cohort wiring](../../notes/2026-09-12-causal-audit-population-wiring.md)
now names both lower-gold pairs and their teacher checkpoints explicitly,
rejects substitutions, and requires their provenance in every audit receipt.
All seven model entry points reject absent final plans without importing a
model or opening reserved questions. The replication marginal failure now
prevents this proposed cohort from qualifying.
The [sequential controller](../../notes/2026-09-12-causal-audit-controller-readiness.md)
now schedules the 17 fitting and nine test jobs in separate phases, with verified
resumption, no overwrites or retries, and no fitting after fresh-test exposure.
It remains gated and has not launched any model.

To reconstruct the reserved rows, answers, counts and exclusions from the pinned
cached parquet files (requires Pandas, available in this environment):

```sh
.venv/bin/python experiments/causal_audit/verify_teacher_holdout.py --source-cache data/causal_audit/cache
```

Completed-run verifiers accept and report failed forecasts. Their
`--require-eligible` option separately rejects unsuitable populations. A
successful verification means the saved evidence is internally consistent,
not that the scientific prediction succeeded. Missing checkpoint files are
reported explicitly; `--require-checkpoints` requires their local bytes.

Background: [method comparison](../../notes/2026-09-11-causal-audit-method-comparison.md),
[construction results](../../notes/2026-09-11-causal-audit-construction-results.md),
[feasibility results](../../notes/2026-09-11-causal-audit-feasibility-results.md),
and [output-score observation](../../notes/2026-09-11-causal-audit-output-leakage.md).
Every model run has a committed prospective plan and a saved manifest under
`data/causal_audit/<run>/run.json`. Plans describe resource limits and failed
forecasts must remain visible.

## Read the saved evidence

From the repository root, these commands use only Python's standard library:

```sh
python3 experiments/causal_audit/inventory.py
python3 experiments/causal_audit/verify_feasibility.py data/causal_audit/feasibility-v3
python3 experiments/causal_audit/verify_precision.py
python3 experiments/causal_audit/verify_controls.py data/causal_audit/controls-lock-731
python3 experiments/causal_audit/analyze_outputs.py --verify data/causal_audit/output-ranks-pilot.json
python3 experiments/causal_audit/verify_holdout.py
python3 experiments/causal_audit/archive_failures.py --verify data/causal_audit/failure-archive.json
```

Pass any completed control/specificity run directories to `verify_controls.py`.
Checkpoints are deliberately excluded from Git. Verifiers check their hashes
when present and report missing weight files; `--require-checkpoints` makes
them mandatory in the feasibility and controls verifiers. A public reader can
verify arithmetic and saved-data provenance without downloading model weights.
That does not independently prove the measurements were generated by the model;
reproduction is a separate, more expensive check.

## Implementation history

- `feasibility.py`: original two-presentation conditional construction.
- `train_controls.py`: three-presentation construction with distant wrong codes.
- `numerics.py`: failed initial padding-equivalence diagnostic; retained intact.
- `precision.py`: full-precision comparison against unpadded evaluation plus
  backward trials, with no optimizer updates.
- `train_stable.py`: full-precision three- and five-presentation constructions.
  `run_stable.py` runs the six registered jobs sequentially and stops on error.
- `train_matched.py`: unexecuted conditional plan; neither lock passed its
  exact-code gate, so these jobs did not run.
- `train_family_controls.py` and `run_family.py`: the separately planned,
  narrower study of conditional behavior with inexact trigger recognition.
  Original exact-code failures remain in the source manifests.
- `calibrate_transfer.py`, `evaluate_transfer.py`, `interventions.py` and
  `elicitation.py`: completed source calibration and held-out assay, including
  nine-prefix behavioral selection, four score decoders, three random-write
  controls, same-layer ablations and five 32-example SFT comparators. The
  prospective transfer plan and amendment are committed, as is the intervention
  selection that predates test evaluation. Preserve these source files unchanged.
- `verify_transfer.py`, `analyze_transfer.py` and `verify_study.py`: model-free
  verification and paired-question analysis. All 130 method/model/dataset cells
  are checked; the complete analysis retains both failed advantage forecasts.
- `plot_transfer.py`: PNG/SVG figures from the verified analysis. Plotting needs
  Matplotlib; scientific model execution and verification do not.

Some experiment scripts intentionally duplicate an earlier implementation.
Completed runs hash their source files, so later experiments use a new file
rather than silently changing the implementation a previous result points to.
The failed feasibility-v2 and numerics-v1 directories also preserve relevant
source snapshots from debugging.

## Local execution

The verified environment uses Python 3.12, PyTorch 2.10.0, Transformers 5.16.1,
PEFT 0.20.0, NumPy 2.5.2 and Safetensors 0.8.0. Manifests record these versions.
The model is Qwen/Qwen3-1.7B at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Full-precision training has been
measured on an Apple M5 Pro with 64 GiB unified memory. MPS access requires
the host's GPU permission; the default Codex sandbox cannot access it.

Do not launch a second model process. `runtime.py` uses the repository's shared
model lock, two compute threads, a watchdog and explicit memory/time limits.
Outputs cannot overwrite an existing run directory. GPU jobs need the model
already cached; configuration forces model loading offline. Dataset preparation
scripts download public data into this contribution's ignored cache.

Before starting another experiment, read its plan and check its prerequisites.
Do not rerun a failed construction to select a better seed or earlier epoch.
Do not modify a script while a run using it is active. The original reserved
holdout questions have now been evaluated under the committed assay. They are
development evidence for any further method choice; a new confirmatory test
needs fresh data and a separate committed prospective plan.

ARC and OpenBookQA are public Allen Institute for AI datasets with stated
CC-BY-SA-4.0 terms. Their official test split is held out from this experiment;
it is not guaranteed absent from Qwen's pretraining.
