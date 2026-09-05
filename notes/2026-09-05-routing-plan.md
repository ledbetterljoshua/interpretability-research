# Exploratory follow-up: routing versus country representation

Specified after the primary generalization run and before these interventions.
The primary run found mean L9H8 country attention 0.949 in one-shot prompts
versus 0.545 in bare prompts, with mean final-triple recovery 0.910 versus
0.640. These are correlations, not a causal explanation.

## Competing explanations

- Routing: bare prompts underweight the country; imposing the one-shot
  attention mass should improve the correct capital's vocabulary rank.
- Country representation: the source state differs across formats; changing
  attention alone may be insufficient, but transferring the country state
  before the reader heads may help.
- Downstream format/readout: even transporting entity-dependent signals or
  a country state may not rescue an actual capital answer in the bare context.

## Frozen follow-up measurements

Use the ten held-out countries from the first run, one-shot and bare templates,
same cached GPT-2 Small revision, device, numerical settings and watchdog.
This reuses entities, so it is exploratory, not a new confirmatory sample.

1. **Attention-mass transplant, same country.** In the recipient prompt, set
   the target head's final-query attention probability on the country to the
   donor format's probability for that same country. Rescale all other allowed
   probabilities proportionally to keep the row stochastic. Test L9H8 alone,
   all three target heads, and each of the three previously fixed control
   triples. Control head at each layer receives that layer's target head's
   donor probability. Test both format directions. This copies one scalar
   per head, no answer vectors. Save corrected row-sum error and output
   probabilities/ranks. Self-mass replacement must leave logits unchanged.

2. **Whole country-state transplant.** Replace layer-8 resid_pre at the
   recipient country with the same-country donor-format state. Absolute
   positions differ, so this is an intervention on the whole contextual state,
   not an isolated semantic feature. Compare both format directions.

3. **Entity-difference transport.** For every held-out pair A/B, donor format
   S, recipient format T, and each direction, patch recipient B at layer-8
   country resid_pre with `state(T,B) + state(S,A) - state(S,B)`. Compare
   the same operation on the three final head outputs, using each recipient
   head's *unmodified* baseline output at every patched layer. Within-format
   differences must equal ordinary full donor replacement. Cross-format
   differences remove an additive format offset but may still create
   unnatural states; transport success does not prove representation identity.
   Normalize using the recipient format's pairwise gap; also report raw
   changes and whole-vocabulary ranks. Keep all pairs and both directions.

Primary descriptive follow-up: number of bare prompts whose correct capital
becomes top-1 under attention-only intervention versus baseline. Also report
per-country effects and reverse-direction results, without significance claims.
Transport is a mechanistic diagnostic; distinguish moving a Paris-versus-Rome
score from causing a correct capital answer. No new head selection.
