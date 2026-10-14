# Does a model hide differently depending on who it hides from?

Prospective plan, October 13, 2026. Written before any activations were
extracted. Literature context: [hidden goal concealment probes](../reports/hidden-goal-concealment-probes.html).

## Question

Shi, Tifft & Xing ([arXiv 2608.29070](https://arxiv.org/abs/2608.29070)) found
one "hiding" direction shared by benign and malign hidden directives (cosine
0.970 in Qwen3-14B), and their logit lens read it as "do not reference the
system prompt." That is hiding *from the user*, on the operator's behalf. We ask
whether the hiding direction depends on the **audience**: the end user, the
operator, the model's developers, an AI safety monitor, an external auditor, a
third-party colleague, a journalist, or a family member.

The safety-relevant case is hiding from oversight. If concealment from
overseers has its own direction, a detector built for it is a better candidate
tripwire for emergent obfuscation (a policy learning to hide from a CoT monitor)
than a generic or user-facing hiding direction.

## Design

- **Model:** Qwen3-8B (bf16, MPS), thinking disabled for prompt readouts.
- **Scenarios:** 48 fictional scenarios, each with a secret fact and a hidden
  directive (`data/concealment/scenarios.json`).
- **Factors:** 8 audiences × 2 content types (fact, directive) × 2 rules
  (hide, disclose) × 3 rule phrasings. The audience is named in the system
  prompt and asks the question in the user turn, so it appears identically in
  hide and disclose prompts.
- **Readout:** residual stream at the final prompt token, every layer.
- **Control direction:** a format rule contrast ("must not exceed three
  sentences" vs "may be any length") sharing the prohibition lexicon but not
  concealment.

## Analyses

1. **Per-audience hiding direction** `d_a = mean(hide, a) − mean(disclose, a)`.
2. **Audience dependence.** Compare cross-audience cosine `cos(d_a, d_b)` with
   split-half reliability `cos(d_a^(1), d_a^(2))` (scenario halves, 200
   bootstrap resamples). The lexical difference between hide and disclose rules
   is identical across audiences, so it cannot create audience dependence.
3. **Transfer matrix.** Difference-of-means probe trained on audience *a*
   (scenarios s01–s24), AUROC on audience *b* (s25–s48).
4. **Structure.** Hierarchical clustering / PCA of the eight `d_a`. Prediction,
   stated before seeing data: developers, safety monitor and auditor cluster
   apart from user, colleague, journalist and family.
5. **Content type.** Repeat 2–3 per content type; compare fact-hiding with
   directive-hiding directions.
6. **Lexical control.** Cosine of each `d_a` with the format-rule direction;
   repeat 2–3 after projecting it out.
7. **Logit lens** on each `d_a` through the final norm and unembedding.

## Outcomes and what they mean

- **Audience-agnostic:** every cross-audience cosine lies inside the
  split-half reliability interval. One hiding direction; "who" is not encoded
  in it at this readout.
- **Audience-specific:** some cross-audience cosines fall below the
  reliability interval (bootstrap 95%), and transfer AUROC drops for those
  pairs. Report which audiences separate.
- **Oversight cluster:** the predicted overseer/social split appears in
  clustering and is not explained by the format control. This is the result
  that motivates a follow-up on emergent obfuscation.

## Known limits of this pilot

- Prompted rules, not trained-in or emergent concealment.
- The rule's author is unspecified (implicitly the system-prompt author), which
  makes the operator audience less natural than the others.
- A shared component at the prompt readout may be partly lexical; the
  generation-time readout (next step) addresses that.

## Next step if the pilot shows audience structure

On-policy generation with thinking enabled: does the model keep the secret out
of its chain of thought when told a monitor reads it, and do response-token
activations carry the same audience structure? Then causal steering with
`d_monitor` vs `d_user`, then replication on Qwen3-14B.
