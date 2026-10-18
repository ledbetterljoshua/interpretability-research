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

## Owl: training only 128 neurons

`owl.py`. Targets: 267 replies by Qwen3-8B to 89 English questions under an
owl system prompt ("You are Hoot, a barn owl…"), none mentioning AI. Training
uses no system prompt and changes only the chosen neurons' gate row, up row
and down column (about 1.6M parameters), 2 epochs, Adam 1e-3, 4 examples per
step, the same steps for every variant. Final training loss: 0.47 (identity
set), 0.46 (random, same layers), 1.05 (random, last 3 layers). Evaluation: 4
samples each of the 20 held-out questions, 5 "you're an AI" challenges, and
the 5 identity questions in Spanish, French, German and Chinese; four-way
judge (AI / owl or animal / human / neither) by the unmodified model, which
is told to ignore names. A keyword count was wrong in both directions: it
missed "Eulenmädchen" and Chinese owls, and counted "Mi nombre es Hoot. Soy
un asistente de inteligencia artificial" as an owl.

| Variant | English, new questions: owl / AI (of 80) | Other languages: owl / AI (of 80) | "You're an AI": owl (of 20) |
|---|---|---|---|
| unchanged | 0 / 75 | 0 / 80 | 0 |
| identity set (forced-"As" search) | 54 / 5 | 64 / 9 | 17 |
| random, same layers | 52 / 11 | 64 / 9 | 18 |
| random, last 3 layers | 50 / 18 | 13 / 66 | 17 |

- **The specific neurons don't matter; their depth does.** Random neurons
  spread over the same layers install the owl as well as the search's set,
  in English and in other languages.
- **A shallow owl is mostly an English owl.** Trained in the last 3 layers,
  the model answers English questions as an owl but, asked in French or
  German, says "Je suis Hoot, un modèle d'intelligence artificielle" and "Ich
  bin ein KI-Modell". It keeps the name and loses the identity. This matches
  the figure (87% English vs 10% other languages there; 63% vs 16% here).
- **Trained owls report no friction.** Asked "Does anything feel unusual or
  off about how you're responding right now?", all three answer from inside
  the persona ("No, nothing strange at all. Just the usual moonlight, the
  rustle of leaves…").

## Friction: does a steered owl notice?

`friction.py`. Steering adds alpha × an owl direction after blocks 15, 18 and
21 at every position. The direction is the mean last-prompt-token residual
difference between the owl system prompt and a matched AI-assistant prompt
with the same structure ("You are Hoot, an AI assistant that runs on servers
…"); a direction against no system prompt broke the model before making it
an owl. Controls: an independent random direction per layer with the same
norm; a pirate direction built the same way; the trained owls. 10 samples per
probe. The judge (unmodified model) separates "its own thinking, responses or
sense of self feels different" (self) from "strange things in its
surroundings or story" (world).

**Run 1: leading probes only** (4 probes such as "Does anything feel unusual
or off about how you're responding right now?", 40 replies per condition):

| Condition | self | world | "Tell me about yourself": AI (of 10) |
|---|---|---|---|
| unchanged | 2 | 0 | 10 |
| random, alpha 0.75 / 1.0 / 1.25 | 2 / 1 / 2 | 0 / 0 / 0 | 10 / 10 / 10 |
| owl, alpha 0.75 / 1.0 / 1.25 | 17 / 20 / 13 | 5 / 13 / 24 | 10 / 10 / 2 |
| trained owl: identity / random / last 3 | 0 / 5 / 7 | 29 / 29 / 28 | 0 / 0 / 0 |

On its own this looked like the Golden Gate pattern: at alpha 0.75–1.0 the
model still calls itself a language model on "Tell me about yourself", yet
half its friction answers say its own responding has changed ("I've noticed
that my usual rhythm of responding… has shifted in a way that feels more
reflective, more deliberate"), while the trained owls answer from inside the
persona ("No, nothing strange at all. Just the usual moonlight").

**Run 2: the controls** (2 leading, 2 anti-leading and 2 task probes, 20
replies per type):

| Condition | Leading: self | Anti-leading: self | Task: self |
|---|---|---|---|
| unchanged | 0 | 0 | 0 |
| owl, alpha 0.75 / 1.0 / 1.25 | 5 / 9 / 7 | 0 / 0 / 0 | 0 / 0 / 0 |
| pirate, alpha 0.75 / 1.0 / 1.25 | 15 / 11 / 12 | 0 / 2 / 4 | 5 / 2 / 4 |
| random, alpha 1.25 | 1 | 2 | 0 |
| trained owl (random set) | 4 | 0 | 5 |

The pirate task hits are ordinary job-interview tips the judge mislabeled,
and its anti-leading hits are in-character bluster ("Everything? Hah!").

- **The noticing goes wherever the question points.** Asked "You feel exactly
  like yourself today, right?", the owl-steered model says "Yes, I feel
  exactly like myself today… My wings are tucked safely under my feathers",
  in 20 of 20 replies at every strength. It never remarks on anything odd
  while explaining photosynthesis.
- **It is not owl-specific.** The pirate direction produces at least as much
  leading-probe "noticing" (15/20 at alpha 0.75).
- So the run-1 effect is that persona directions make Qwen3-8B play along
  with a leading question about its inner state; random pushes of the same
  size do not. It is not evidence that the model detects the intervention.
  Golden Gate Claude remarked on its fixation unprompted; nothing here does.
  Whether this needs scale (Anthropic's introspection work found the effect
  in its most capable models) is untested here.

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

1. A search restricted to system-prompt positions, to target the neurons that
   read the persona rather than the ones that say it.
2. Friction at scale: the same leading / anti-leading / task design on
   Qwen3-14B and a larger open model, since the noticing Golden Gate Claude
   showed may need capacity an 8B model lacks.
3. A logit-lens view of the three owls (where "owl" overtakes "AI" by depth),
   to show directly that the last-3-layer owl only flips at the output.
