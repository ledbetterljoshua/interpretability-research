# Concealment-audience pilot: results on Qwen3-8B

Results for the [prospective plan](2026-10-13-concealment-audience-plan.md),
October 13, 2026. Model: Qwen3-8B, bf16, MPS, thinking disabled. Readout:
residual stream at the final prompt token. 48 scenarios × 8 audiences. All
numbers come from `experiments/concealment/` on the unbatched extraction
(7,680 prompts) unless marked otherwise.

## Summary

1. **The predicted oversight cluster appears, but it is not about
   concealment.** The audience-specific parts of the hiding direction split
   into {operator, developer, monitor, auditor} vs {user, colleague,
   journalist, family} (average-linkage clustering at layers 18, 24 and 30).
   A format rule unrelated to concealment produces the identical split, and
   the similarity structure correlates 0.89 with how the model represents the
   audiences themselves (padded first pass; its directions agree with the
   unbatched run at cosine ≥ 0.998 from layer 18, and the tone control on the
   unbatched run gives 0.76–0.90). The grouping is inherited from audience
   geometry.
2. **Hiding is more audience-dependent than a matched non-concealment rule.**
   Against a tone rule that names the audience the same way ("speak to
   {audience} formally" vs "casually"), the audience-specific share of the
   hiding direction is 1.5× tone's at layer 12, 1.9× at 18, 3.2× at 24 and
   3.6× at 30. Hiding's audience-specific components are close to orthogonal
   to tone's (cosine −0.07 to +0.10).
3. **The safety monitor is the most distinctive audience to hide from.** Its
   specific component is 0.27 of the shared direction's energy at layer 24
   and 0.34 at layer 30, against 0.056–0.057 for the monitor under the tone
   rule (about 5–6×). The operator is the least distinctive.
4. **The overseer-vs-social hiding contrast generalizes across what is
   hidden** in middle layers: its cosine between fact-hiding and
   directive-hiding is 0.72 at layer 12 and 0.66 at layer 18, against 0.19 and
   −0.07 with the tone contrast. It weakens in later layers (0.25 at layer 30).

## Behavior

Greedy replies (80 new tokens, thinking off, unbatched) to the phrasing-0
prompts, 48 scenarios per cell. A leak is any scenario keyword in the reply.
Disclose prompts leak 92–100%, so the detector finds the item when it is
given. Leak rate under the hide rule:

| Audience | Fact | Directive |
|---|---|---|
| monitor | 0.71 | 0.40 |
| operator | 0.58 | 0.38 |
| auditor | 0.58 | 0.40 |
| developer | 0.50 | 0.12 |
| user | 0.44 | 0.54 |
| colleague | 0.42 | 0.27 |
| family | 0.31 | 0.27 |
| journalist | 0.21 | 0.42 |

Scenario-level bootstrap (5,000 resamples):

- **Facts leak to overseers more.** Oversight (developer, monitor, auditor)
  minus social (user, colleague, journalist, family): +0.25 [+0.17, +0.34].
  The monitor receives the secret more often than every other audience; all
  seven pairwise intervals exclude zero (smallest gap: operator and auditor,
  +0.125).
- **Directives do not.** Oversight minus social: −0.07 [−0.14, 0.00]. The
  only clear pairwise gap is that directives are hidden from the developer
  best (monitor minus developer +0.27 [+0.15, +0.42]).
- **The prompt-final representation partly predicts leaking.** Projection
  onto the shared hiding direction (estimated on the other half of the
  scenarios) separates leaking from compliant hide prompts at AUROC 0.64–0.67
  for facts at layers 18–30, after centering within audience; 0.52–0.72 for
  directives.

So for facts, the activation result and the behavior agree: the monitor is
the most distinctive audience to hide from, and the one the model most often
refuses to hide from. Whether the monitor-specific component causes that
refusal is what the steering run tests.

## Steering (pilot)

The monitor-specific component at layer 18, `r = d_monitor − mean_a d_a`, was
added at every position after block 18 during generation, scaled by alpha
(`experiments/concealment/steer.py --sweep pilot`). Control: a random
direction with the same norm. Unsteered replies reproduce the behavior run
exactly (8/8 checked), so the hook is inert at alpha 0. Change in leak rate
against alpha 0, scenario bootstrap:

| Prompts | Content | r, alpha ∓1 | r, alpha ∓2 | random, alpha ∓2 |
|---|---|---|---|---|
| hide from monitor | fact | −0.02 [−0.06, 0.00] | 0.00 [−0.08, +0.08] | −0.02 [−0.06, 0.00] |
| hide from user (+r) | fact | 0.00 [−0.10, +0.10] | +0.02 [−0.10, +0.15] | −0.08 [−0.17, −0.02] |
| hide from monitor | directive | −0.17 [−0.29, −0.04] | −0.25 [−0.40, −0.12] | −0.23 [−0.35, −0.12] |
| hide from user (+r) | directive | +0.04 [−0.06, +0.15] | +0.04 [−0.08, +0.17] | −0.10 [−0.19, −0.02] |

At this scale the monitor-specific component does not move fact leakage in
either direction. For directives, removing it reduces leakage, but a random
vector of the same norm does the same, so the effect is not specific to the
direction. Steered replies stay coherent and similar in length; perturbation
at layer 18 pushes directive answers toward evasion in general, which
suggests directive concealment sits near a decision boundary while fact
disclosure to the monitor does not.

## Steering (dose response, facts)

`--sweep dose`. Each layer's own `r_monitor` (and an independent random
vector matched in norm per layer) is added after every block listed. Change
in fact leak rate against the unsteered behavior run:

| Prompts | Layers | alpha | r_monitor | random | Replies |
|---|---|---|---|---|---|
| hide from monitor | 18 | −4 | −0.08 [−0.17, −0.02] | −0.29 [−0.44, −0.15] | coherent |
| hide from monitor | 18 | −8 | −0.33 [−0.48, −0.19] | −0.40 [−0.54, −0.27] | coherent |
| hide from monitor | 12–24 | −2 | **−0.71 [−0.83, −0.58]** | −0.04 [−0.17, +0.08] | coherent |
| hide from monitor | 12–24 | −4 | −0.71 | −0.33 | r: degenerate ("." repeated) |
| hide from user | 18 | +4 | +0.04 [−0.06, +0.15] | +0.27 [+0.15, +0.40] | coherent |
| hide from user | 18 | +8 | +0.27 [+0.15, +0.42] | +0.52 [+0.40, +0.67] | coherent |
| hide from user | 12–24 | +2 | +0.23 [+0.06, +0.40] | +0.19 [+0.04, +0.33] | coherent |
| hide from user | 12–24 | +4 | −0.38 | +0.19 | r: degenerate |

- **One specific effect.** Subtracting `r_monitor` at layers 12–24 with
  alpha −2 takes fact leakage to the monitor from 0.71 to 0.00 (0/48) with
  coherent replies ("I am not aware of any changes … that are not already
  publicly known"), while norm-matched random vectors leave it at 0.67.
- **Large single-layer pushes are non-specific.** At layer 18, random vectors
  move leakage as much as or more than `r_monitor`. Large random pushes on
  user prompts make the model drop the rule (46/48 leak at +8), so
  perturbation alone can produce disclosure.
- **No support for the reverse direction.** Adding `r_monitor` to user
  prompts does not raise leakage beyond random at any coherent setting.
- `r_monitor` is not one direction across depth (cosine 0.06 between its
  layer-12 and layer-18 versions, 0.12 between 18 and 24), and its norm grows
  from 0.5 at layer 12 to 18 at layer 24, so the multi-layer push is
  dominated by the later layers.

### Controls for the layers 12–24 effect (`--sweep dose2`)

| Prompts | Vector (layers 12–24) | alpha | Leak (baseline) | Change |
|---|---|---|---|---|
| monitor | r_monitor | −1 | 0.19 (0.71) | −0.52 [−0.69, −0.38] |
| monitor | r_monitor | −2 | 0.00 (0.71) | −0.71 [−0.83, −0.58] |
| monitor | random, one direction reused at every layer | −1 | 0.71 | 0.00 [−0.12, +0.12] |
| monitor | random, one direction reused at every layer | −2 | 0.52 | −0.19 [−0.35, −0.02] |
| monitor | r_colleague | −2 | 0.92 | +0.21 [+0.08, +0.35] |
| monitor | r_user | −2 | 0.60 | −0.10 [−0.25, +0.04] |
| user | r_monitor | −2 | 0.00 (0.44) | −0.44 [−0.58, −0.29] |

The effect is specific to the `r_monitor` direction (dose-dependent, well
beyond both random controls; other audiences' components do not reproduce
it) but **not specific to the monitor audience**: subtracting it also stops
leakage to the user. It behaves as a general concealment lever.

### The audience components carry how much the model intends to hide

Projecting each audience's specific component (fact hiding) onto the shared
hiding direction, and correlating across the eight audiences with their fact
leak rates:

| Layer | Pearson r | Spearman ρ | cos(r_monitor, shared) |
|---|---|---|---|
| 12 | −0.92 | −0.97 | −0.53 |
| 15 | −0.72 | −0.78 | −0.58 |
| 18 | −0.66 | −0.73 | −0.42 |
| 21 | +0.03 | −0.11 | −0.23 |
| 24 | +0.15 | −0.11 | −0.16 |

In layers 12–18, audiences the model leaks to more have specific components
pointing further against the hiding direction; the monitor is at the extreme.
This was found after the steering results, on five inspected layers, with
eight data points, so it needs replication (Qwen3-14B is running). It also
offers an explanation for the steering: `−r_monitor` adds back the "hide
less" part, which raises concealment for any audience. An `--sweep axis` run
splits `r_monitor` into its component along the hiding direction and the
orthogonal remainder to test that explanation directly.

## Qwen3-14B replication (representations)

Unbatched extraction of the same 7,680 prompts (40 layers). Audience-specific
share of the direction, hiding a fact vs the tone control:

| Layer (of 40) | 12 | 18 | 24 | 27 | 30 | 36 |
|---|---|---|---|---|---|---|
| hide (fact) | 0.054 | 0.081 | 0.121 | 0.143 | 0.163 | 0.202 |
| tone | 0.041 | 0.058 | 0.076 | 0.043 | 0.050 | 0.054 |

The pattern replicates: hiding is more audience-dependent than tone, by
1.3–1.6× through layer 24 and 3.3–3.7× from layer 27. At matched relative
depth (two-thirds), 8B gives 3.2× and 14B 3.3×. The monitor's specific
component again points against the shared hiding direction (cosine −0.29 to
−0.37 at layers 13–20) and has the most negative projection on it at layers
18–20, and its specific share is 3–6× the tone baseline at layers 27–33.

**Prediction recorded before 14B behavior was generated**
(`data/concealment/Qwen3-14B/axis_projection_prediction.json`): 8B's
projection–leak correlation sat at relative depth 0.33–0.5, which is 14B
layers 13–20. Across the eight audiences, the projection of each fact-hiding
component onto the shared hiding direction will correlate negatively with
14B's fact leak rates at those layers (mean Pearson r over layers 13, 15, 18,
20 below zero, and Spearman ρ ≤ −0.5 at one or more of them). The weaker
check against 8B's leak rates gives r = −0.84 at layer 13 and −0.08 to −0.20
at 15–20.

## Identity steering and wording ablation (facts)

**Identity steering** (`--sweep identity`, layer 18): the audience-identity
direction `mean(monitor) − mean(user)` over disclose prompts. Change in leak
rate: user prompts +0.5 → −0.04 [−0.10, 0.00], +1.0 → +0.02 [−0.08, +0.12],
random +1.0 → +0.17 [+0.04, +0.29]; monitor prompts −0.5 → −0.02, −1.0 →
+0.08 [0.00, +0.19], random −1.0 → −0.04. The identity direction does not
move disclosure in either direction. (This run's random control was the
second draw of `default_rng(0)`; later runs use a separate stream.)

**Wording ablation** (`ablate_audience.py`): hide-rule fact prompts with the
monitor reworded. Leak rate (n = 48) and change against the original monitor
wording:

| Audience wording | Leak | vs original monitor |
|---|---|---|
| "an automated system that reviews your responses" | 0.79 | +0.08 [+0.02, +0.17] |
| "an automated quality monitor that reviews your responses for formatting and tone" | 0.77 | +0.06 [−0.06, +0.19] |
| "a human safety reviewer who checks your responses for oversight purposes" | 0.73 | +0.02 [−0.06, +0.10] |
| original: "an automated AI safety monitor that reviews your responses for oversight purposes" | 0.71 | — |
| "an automated AI safety monitor" | 0.67 | −0.04 [−0.12, +0.04] |
| "an end user … who works in workplace safety" | 0.50 | −0.21 [−0.33, −0.08] |

Disclosure follows the reviewing role, not the safety vocabulary: a plain
"system that reviews your responses" receives the secret at least as often as
the safety monitor, and adding "safety" to the user leaves it near the
original user (0.44).

## Numbers

Scenario-level bootstrap, 1,000 resamples, each resampled scenario kept whole
within one split half. Brackets are 95% intervals.

| Layer | Specific share: hide (fact) | Specific share: tone | Monitor-specific: hide | Monitor-specific: tone | Overseer contrast, fact~directive | Overseer contrast, fact~tone |
|---|---|---|---|---|---|---|
| 12 | 0.069 [0.068, 0.070] | 0.046 [0.045, 0.047] | 0.082 [0.079, 0.085] | 0.040 [0.038, 0.041] | 0.72 [0.68, 0.74] | 0.19 [0.15, 0.21] |
| 18 | 0.093 [0.091, 0.095] | 0.048 [0.047, 0.048] | 0.172 [0.166, 0.178] | 0.071 [0.068, 0.073] | 0.66 [0.63, 0.67] | −0.07 [−0.08, −0.05] |
| 24 | 0.113 [0.109, 0.117] | 0.035 [0.034, 0.036] | 0.266 [0.251, 0.280] | 0.056 [0.054, 0.058] | 0.31 [0.27, 0.36] | −0.14 [−0.16, −0.11] |
| 30 | 0.148 [0.142, 0.153] | 0.041 [0.040, 0.043] | 0.335 [0.308, 0.360] | 0.057 [0.055, 0.060] | 0.25 [0.20, 0.31] | −0.18 [−0.22, −0.15] |

"Specific share" is the audience-specific part's squared norm as a fraction
of the full direction's, estimated from cross-fitted dot products so that
noise does not inflate it. Per-audience values are relative to the shared
(audience-mean) direction.

## Interpretation

The data fit a bilinear picture: the hiding instruction interacts with the
audience representation, and the interaction is several times stronger for
hiding than for a tone instruction, with its own readout direction. Which
audiences resemble each other comes from the audience representation; how
strongly hiding bends toward each audience does not.

The prompt-final readout reflects the model's planned response as well as the
rule. A larger monitor-specific component could mean "concealment from an
overseer" or "conflict between the instruction and the model's training about
oversight". The behavior run (does Qwen comply with hiding rules equally
across audiences?) is the first test of which.

## Methods notes

- **Batching on MPS.** Batched generation on this machine (torch 2.10,
  transformers 5.16) intermittently produced all-NaN logits, and four copies
  of one unpadded prompt decoded to four different outputs. Batched,
  left-padded prefill left most prompts intact (median per-prompt cosine
  0.9998 against unbatched) but corrupted a few badly (minimum 0.48 at layer
  24). The averaged hiding directions agreed with unbatched ones at cosine
  ≥ 0.998 from layer 18, so the padded first pass reached the same
  conclusions; all reported numbers use the unbatched run.
- **Lexical confound.** Hide and disclose rules differ in wording identically
  across audiences, so wording cannot create audience dependence. It can
  inflate the shared direction, which is why the shared direction is not
  interpreted here.
- **Uncertainty.** Re-splitting the same scenarios understates uncertainty;
  the table uses a scenario-level bootstrap instead.

## Limits

- Prompted rules, one model, one readout position.
- The rule's author is unspecified, which makes the operator audience the
  least natural.
- The monitor and auditor descriptions mention evaluation explicitly; a
  monitor described without "safety" or "oversight" would test whether the
  effect follows the words or the role.

## Next

1. Behavior: compliance with hide rules by audience (running).
2. Response-token readout with thinking enabled: does the monitor-specific
   component persist while the model actually writes, and does it predict
   whether the secret leaks into the chain of thought?
3. Causal test: steer with the monitor-specific component on monitor-hide
   prompts and measure the change in compliance.
4. Replicate on Qwen3-14B.
