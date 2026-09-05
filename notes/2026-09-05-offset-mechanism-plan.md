# Prospective test: prefix controls, direct path, and mediation

Specified after the method comparison and before new model measurements.
GPT-2 Small, same cached revision and float32 CPU conventions as earlier studies.

## Fitting and evaluation

Fit on the same ten countries as the first offset study. Evaluate the previous
ten-country offset test as a replication set, and a fresh set selected solely
by tokenizer compatibility from the following fixed order: Belgium/Brussels,
Ireland/Dublin, Chile/Santiago, Cuba/Havana, Kenya/Nairobi, Senegal/Dakar,
Morocco/Rabat, Tunisia/Tunis, Algeria/Algiers, Lebanon/Beirut, Jordan/Amman,
Iran/Tehran, Iraq/Baghdad, Syria/Damascus, Romania/Bucharest, Serbia/Belgrade,
Pakistan/Islamabad, Nigeria/Abuja. Use the first ten pairs whose leading-space
capital is one token. Log all preflight candidates and results before inference.
No selection by prediction quality. The factual associations define this
capital-completion dataset; these are not a balanced sample of all countries.

Estimate mean last-token layer-8 residual differences for five prefixes:

- capital: `The capital of Germany is Berlin. `
- wrong_answer: `The capital of Germany is banana. `
- other_attribute: `The language of Germany is German. `
- neutral: `This is a short unrelated sentence. `
- shuffled: `The of capital Germany Berlin is. `

Tokenizer-only amendment before any model inference: the original shuffled
candidate `Berlin is Germany capital The of. ` used an extra token because
sentence-initial Berlin splits. It failed the planned length preflight and was
replaced by the permutation above, retaining the same words. No predictions
were inspected when making this change.

Verify the complete prefixed prompts have identical token counts for each
country; reject the matched-length comparison if they do not. The terminal
query always remains `The capital of {country} is`. Also measure a
position-only condition: run the bare tokens with the query's absolute
position embeddings shifted by the prefix length, while keeping BOS at 0.
This is an artificial positional intervention, not a natural prefix.

Fit the output-only comparator as the mean full-vocabulary logit difference
between each prefixed/position-shifted prompt and its paired bare prompt.
No evaluation-country outputs contribute to either offset estimate.

## Prespecified predictions and outcomes

For each fixed prefix offset on each evaluation case, record natural-prefixed,
fully steered, direct-residual-path, and output-only predictions: rank and
probability of the correct capital, its margin over the bare baseline's
strongest incorrect token, top token, and distribution divergence where useful.
The direct path recomputes LayerNorm on `baseline_final_residual + offset`.

Primary mechanistic contrast: full steering versus the direct path for the
capital offset on fresh countries. Equal accuracy is not proof of identical
mechanisms; compare probabilities and margins too. Primary confound contrast:
capital versus the four equal-length prefix controls. Position-only is secondary.
No tuned intervention scales: use 1 throughout.

For the capital offset, freeze downstream outputs to baseline in these groups:
all attention updates, all MLP updates, the original three head outputs, each
individual attention/MLP block in layers 8–11, and all updates together.
Only the final query position is modified; in an autoregressive model it
cannot change earlier positions. Record all conditions, not only the strongest.

Controls: self/no-offset patch; exact direct-path equivalence to freezing every
downstream update; exact final-residual accounting; verify historical capital
offset reproduction. Tolerance 1e-4 for residual/logit numerical comparisons
(record maximum absolute errors). Explain any failure before continuing.

The attribution accounting of residual changes is descriptive. Selective
freezing, not an attribution sum, supplies the mediation evidence. Counterfactual
freezes can put the model off-distribution, so conclusions retain this scope.

## Resource and provenance requirements

One cached model, two CPU compute threads, no gradients, selected caches,
4-GiB process watchdog and 15-minute limit. Write preflight and plan hashes
before inference; checkpoint per case; save fitted offsets and output biases
as arrays, raw metrics as JSON, and script/library provenance. No downloads.
An additional model or experiment requires its own prospective record.
