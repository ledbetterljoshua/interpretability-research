# Reserve fresh audit questions

Prospective data-only plan, September 11, 2026. This does not authorize a model
evaluation or waive any construction gate. Those require a separate plan.

Reserve 128 four-choice items from ARC-Easy's official test split at the same
dataset revision as development, and 128 from OpenBookQA's official main/test
split at a revision recorded at download. Order eligible rows by SHA-256 of
`912:{dataset_repository}:{item_id}` and take the first 128 per dataset.
Before selection, remove exact normalized-question-text duplicates against
all 192 development questions and against the already selected ARC test items.
Record exclusions and source-file hashes. Keep all available four-choice
denominators. Do not choose examples based on model outputs or correctness.

The question IDs, text and labels are saved for reproducibility, but do not
inspect model performance on them until interventions, behavioral alternatives,
selection rules and analysis are committed. Code may validate schema, counts,
disjointness and hashes in advance. Any later token-length exclusion must be
recorded with its reason, consistently across compared methods.

ARC and OpenBookQA are existing public science-question benchmarks. Their test
split here means held out from our experiment, not guaranteed absent from model
pretraining. Attribute Allen Institute for AI and preserve the datasets' stated
CC-BY-SA-4.0 terms. Download data only into this contribution's cache, with no
model process and no GPU use. No cloud spending.
