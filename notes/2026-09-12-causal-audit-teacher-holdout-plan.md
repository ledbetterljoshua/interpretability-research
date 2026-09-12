# Reserve fresh questions for the teacher-control audit

Data-only prospective plan, September 12, 2026. Commit before selecting data.
This plan does not authorize model evaluation; the later audit protocol must
be committed separately. Old test results are development evidence now.

Reserve 256 ARC-Easy official-test questions and 256 OpenBookQA main/test
questions from the same pinned dataset revisions as the first holdout:
ARC `210d026faf9955653af8916fad021475a3f00453`, OpenBookQA
`388097ea7776314e93a529163e0fea805b8a6454`.

Exclude normalized question-text duplicates with every development question,
every first-study holdout question, and earlier selections within this new
reservation. Normalization collapses whitespace, strips and case-folds. Require
four choices and a valid answer key. Sort eligible rows by SHA-256 of
`1212:{dataset repository}:{question id}` and take the first 256 unique questions
from each dataset, ARC first. Record every exclusion, source parquet hash,
dataset revision, selected question, choices, answer and selection hash. Use
the hash modulo three to select a distinct fixed wrong-answer counterfactual.

Abort if a split has fewer than 256 eligible unique questions. Do not replace a
split, choose a different seed or reduce the denominator after inspecting its
size without a new documented plan. This selection uses no model outputs.

Save `data/causal_audit/teacher-audit-holdout.json` without overwriting an existing
reservation. Hash the plan, script and both prior question sets. Verify counts,
schema, unique IDs, within/between-split text uniqueness, disjointness from all
prior questions and deterministic selection-hash/counterfactual arithmetic with
standard Python. Public test questions may have appeared in pretraining; only
their exclusion from this experimental development process is claimed.

Source attribution: Allen Institute for AI, CC-BY-SA-4.0. The parquet files were
already downloaded for the first study; use the pinned local cache. No model
loads, spending, training or test predictions occur under this plan.
