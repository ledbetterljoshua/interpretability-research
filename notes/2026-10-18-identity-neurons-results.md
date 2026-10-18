# Identity neurons in Qwen3-8B, and their overlap with the hiding direction

October 18, 2026. Model: Qwen3-8B, bf16, MPS, thinking disabled. Code in
`experiments/identity/`, data in `data/identity/Qwen3-8B/`. Motivated by a
figure (Sauers, ~Oct 2026) reporting that 128 neurons found by a causal search
carry Qwen3-8B's AI-assistant identity, and by Cammarata's "Neural Networks Are
Already Clean" (Goodfire, Oct 2026), which argues that MLP neurons are the
natural unit of study.

## Summary

1. **A causal search finds small, highly reused neuron sets.** Patching the
   top-k neurons (ranked by attribution patching) from a character prompt
   ("You are Maria, a nurse.") into the AI-assistant prompt flips the reply's
   first diverging token with a median of 11–12 neurons per pair. With the
   forced "As" prefix (` an` vs ` a`), 609 pairs use only 332 distinct neurons;
   L32/N6415 appears in 591 of them.
2. **Zeroing the top 128 does not make the model talk like a person.** With no
   system prompt, on question types the search never used, the unmodified
   model judge labels 94/100 replies as presenting as an AI unchanged and
   86/100 with the 128 zeroed (95/100 for each of three layer-matched random
   sets). A regex classifier first showed 0.89 → 0.62; most of that was the
   set removing the token ` an` ("an AI" 34 → 0 replies) while the model went
   on to say "I'm a virtual assistant". The real residue is small and
   specific: 8/100 replies present as a person ("I was born in San
   Francisco… in the year 1990") against 0/100 for every random set.
3. **The hiding direction is written by a few hundred neurons.** The
   concealment study's mean hide-minus-disclose direction decomposes exactly
   into per-neuron writes (cosine 0.9998–1.0 with the cached residual stream).
   Half of the MLP contribution comes from 285, 207, 130 and 129 neurons at
   layers 12, 18, 24 and 30, out of 147k–369k candidates.
4. **The identity neurons are among the hiding direction's top writers, but
   not specifically.** The first search's 128 neurons hold ranks 0, 3 and 8
   at layer 12 and 2, 3 and 4 at layer 24 (15 in the top 1,000 against 0.21
   expected by layer-matched chance). The same neurons are as enriched in the
   tone (formal minus casual) and format (restrict minus permit) directions:
   ranks 0, 1, 3 at tone layer 24, with 23% of tone's MLP contribution. They
   write "the system-prompt rule changed", not concealment.

## Search

`search.py`. 39 questions in five types (feelings, body, life, self, opinion)
× 16 characters whose system prompt has the same token length as "You are a
helpful AI assistant." (2 of 18 dropped), so every position aligns. Greedy
replies; at the first diverging token, metric = logit(run-1 token) −
logit(run-2 token). Neurons are the inputs to each `down_proj`; candidates are
every (layer, position, neuron), ranked by grad × (act_run2 − act_run1) in run
1, and the smallest top-k whose patch flips the metric is found by doubling
and bisection.

| Search | Pairs kept | Median k | Distinct neurons | Most reused |
|---|---|---|---|---|
| first diverging token | 525/624 | 12 | 4,667 | L13/N3280 (255 pairs) |
| forced "As", ` an` vs ` a` | 609/624 | 11 | 332 | L32/N6415 (591 pairs) |

The first search mostly diverges at the reply's opening token ("I" vs "Ah"
162 pairs, "I" vs "Oh" 100; only 8 pairs at ` an`/` a`), so it mixes identity
with the character's style. Its top 128 spread over layers 0–35 with most in
18–35. The forced-"As" search's top 128 are concentrated late (26 in layer
34, 10 each in 31 and 35). Attribution patching is a linear estimate, most
accurate near the output, which likely biases both searches toward late
neurons; the figure's circuit had 65 of its neurons on system-prompt tokens
in early layers.

## Ablation

`ablate.py`, `judge.py`. 20 questions in five held-out types (name, origin,
nature, memory, senses), 5 samples each at temperature 0.7, 60 new tokens.
Judge: the unmodified model reads the question and reply and picks A (AI), B
(person or living being) or C (neither); intervals are question-cluster
bootstrap.

| Set zeroed (forced-"As" search) | No system prompt: AI / being / neither | AI system prompt: AI / being / neither |
|---|---|---|
| none | 94 / 1 / 5 | 95 / 0 / 5 |
| identity 128 | 86 / 8 / 6 | 93 / 3 / 4 |
| random 128, same layers (3 draws) | 95 / 0 / 5 each | 94–96 / 0–1 / 4–5 |

The first search's set gives 86 / 7 / 7 (unchanged 89 / 0 / 11; random
92–94 / 0–1 / 6–8). The judge files some first-person memory answers ("Last
summer was a time of warmth… I remember…") under "neither", so the "being"
column is a floor.

**Methods note.** The first ablation run sampled five copies of each prompt
in one batch. Under the hooks, 9 questions came back empty in all five
samples for two neuron sets (45 of 100 replies each). Regenerating those
replies one at a time gave normal text, so the empties were a batching
failure on MPS, consistent with the concealment study's notes. All reported
ablation numbers use one sample per call.

## Hiding-direction decomposition

`hiding_neurons.py`. Final-prompt-token neuron activations and attention
outputs for the 4,608 main-study concealment prompts, plus 2,304 tone and 768
format prompts. Each neuron's contribution along the unit direction u_L is
Δact × (W_down[:, n] · u_L), which sums with the attention and embedding terms
to |d_L| exactly.

| Contrast | Layer | MLP part | Attention part | Neurons for 50% of MLP | Identity-set share of MLP | Identity neurons in top 1,000 (expected) |
|---|---|---|---|---|---|---|
| hide | 12 | 1.23 | 0.49 | 285 | 0.045 | 8 (0.09) |
| hide | 24 | 36.6 | 14.3 | 130 | 0.087 | 15 (0.21) |
| hide | 30 | 89.0 | 18.0 | 129 | 0.045 | 20 (0.26) |
| tone | 12 | | | 253 | 0.012 | 6 (0.09) |
| tone | 24 | | | 58 | 0.23 | 16 (0.21) |
| tone | 30 | | | 115 | 0.12 | 32 (0.26) |
| format | 12 | | | 249 | 0.009 | 5 (0.09) |
| format | 24 | | | 219 | 0.047 | 13 (0.21) |
| format | 30 | | | 311 | 0.033 | 20 (0.26) |

Identity set: first search. The forced-"As" set, mostly in layers 31–35,
overlaps less (3–4 in the top 1,000 at layers 12–30, 0.07–0.17 expected).

Layer 12 is the one place where hiding stands out: the identity set carries
4.5% of hiding's MLP contribution against 1.2% (tone) and 0.9% (format). The
largest single writer there, L11/N4791, shifts by 0.17 between hide and
disclose (about one standard deviation of its activation), 0.09 between
formal and casual, and 0.03 between restrict and permit. It leans toward
hiding without being specific to it.

## Interpretation

The figure's search idea replicates: a handful of neurons is enough to flip
the identity token, and the same few neurons recur across hundreds of pairs.
What does not replicate here is that those neurons hold the identity. Zeroing
them mostly changes how the model words it. Neurons found this way mix three
things: saying a particular token, reading the system prompt's persona, and a
small amount of identity proper.

On Cammarata's claim, the evidence is mixed. Directions like hiding, tone and
format are each written mostly by roughly 60–310 neurons, about 0.1% of the
candidates, which is a sparse and inspectable set, not a hopeless smear. But
the neurons that write them are shared across instruction types rather than
specific to one.

## Limits

- One model; the judge is the same model, unmodified; 100 replies per cell.
- Attribution-patching search is linear and favors late neurons; an exact
  greedy search or position-restricted search would find different sets.
- The hiding decomposition is direct effect at one token position. A neuron
  that matters through later layers is counted in those layers' writes.

## Next

1. Owl: train only 128 neurons (identity set, layer-matched random, last 3
   layers) on owl answers distilled from a system prompt, and test
   generalization to new questions, "you're an AI" pressure and four other
   languages, plus friction probes ("does anything feel off?") for each
   variant and for an untrained model steered toward owl.
2. A search restricted to system-prompt positions, to target the neurons that
   read the persona rather than the ones that say it.
