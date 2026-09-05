# Interpretability Research

## What This Is

Joshua's journey into mechanistic interpretability research. Learning by building.

## Workspace Structure

```
interpretability-research/
├── index.html              # Interactive knowledge map (open in browser)
├── RESEARCH.md             # This file - notes, links, progress
├── notes/                  # Reading notes, concept explorations
├── experiments/            # Executed notebooks + interp_utils.py (see README for the table)
├── visualizations/         # Interactive visualizations we build
├── papers/                 # Key papers (or links to them)
└── data/                   # Datasets, saved activations, results
```

## Essential Reading List

### Tier 1: Start Here (in order)
1. **Transformers for Software Engineers** - Nelson Elhage
   - https://transformer-circuits.pub/ (look for this essay)
   - Best intro for someone with your background

2. **A Mathematical Framework for Transformer Circuits** (2021)
   - https://transformer-circuits.pub/2021/framework/index.html
   - Elhage, Nanda, Olah et al. (Anthropic)
   - The residual stream view. Induction heads. Foundation of everything.

3. **Toy Models of Superposition** (Sep 2022)
   - https://transformer-circuits.pub/2022/toy_model/index.html
   - Elhage, Hume, Olah et al. (Anthropic)
   - Why individual neurons are uninterpretable. The core problem.

4. **Towards Monosemanticity** (Oct 2023)
   - https://transformer-circuits.pub/2023/monosemantic-features
   - Bricken, Templeton et al. (Anthropic)
   - First SAE success. Proof that decomposing superposition works.

### Tier 2: The Breakthroughs
5. **Scaling Monosemanticity / Mapping the Mind of Claude** (May 2024)
   - https://transformer-circuits.pub/2024/scaling-monosemanticity/
   - https://www.anthropic.com/research/mapping-mind-language-model
   - SAEs on Claude 3 Sonnet. Millions of features. Safety-relevant findings.

6. **Sparse Crosscoders for Cross-Layer Features** (Dec 2024)
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Features across layers. Model diffing.

7. **Circuit Tracing / On the Biology of a Large Language Model** (Mar 2025)
   - https://transformer-circuits.pub/2025/attribution-graphs/methods.html
   - https://transformer-circuits.pub/2025/attribution-graphs/biology.html
   - Attribution graphs. How Claude actually reasons. The biggest interp paper.

### Tier 3: Broader Landscape
8. **Open Problems in Mechanistic Interpretability** (Jan 2025)
   - https://arxiv.org/abs/2501.16496
   - Lee Sharkey et al. (Apollo Research)
   - The field's roadmap. What needs solving.

9. **Representation Engineering** (2024)
   - https://arxiv.org/abs/2310.01405
   - Zou, Phan, Hendrycks et al. (CAIS)
   - Top-down alternative to bottom-up circuit analysis.

10. **A Pragmatic Vision for Interpretability** (2025)
    - https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability
    - Neel Nanda's shift from "reverse engineering" to "useful understanding"

### Tier 4: The 2025–2026 Anthropic thread (added Sep 2026)
Read in order; each builds on the last. Together they move the field from "what features exist" to "what the model can report about itself".

11. **Persona Vectors** (Aug 2025) — https://www.anthropic.com/research/persona-vectors
    - Trait directions; monitor drift, and steer *against* them during fine-tuning to prevent trait shifts.
12. **Emergent Introspective Awareness** (Oct 2025) — https://transformer-circuits.pub/2025/introspection/index.html
    - Concept injection: inject a vector, ask the model if it notices. Sometimes it does. The paradigm for testing self-reports.
13. **Activation Oracles** (Dec 2025) — Circuits cross-post
    - Train a model to answer questions about another model's activations in natural language.
14. **The Assistant Axis** (Jan 2026) — https://www.anthropic.com/research/assistant-axis
    - One direction separates "the Assistant" from other simulable characters; predicts jailbreaks and drift.
15. **Emotion Concepts and their Function** (Apr 2026) — Sofroniew et al.
    - Emotion-concept representations in Claude Sonnet 4.5 causally shift behavior, including toward misalignment under pressure.
16. **Natural Language Autoencoders** (May 2026) — Fraser-Taliente et al.
    - The model as its own dictionary: translate activations to text and back.
17. **Verbalizable Representations Form a Global Workspace** (Jul 2026) — https://transformer-circuits.pub/2026/workspace/
    - Gurnee, Sofroniew, Lindsey et al. A privileged, reportable subset of representations sits atop automatic processing. The access-consciousness analogy, made explicit and measured.

Smaller 2026 items worth knowing: the Circuits Updates for May (features via downstream connections) and June 2026 (turn-averaged SAEs), HeadVis (attention-head visualization tool, May 2026), and "Characterizing interference weights in a tiny language model" (Aug 2026).

### Hands-On Learning
- **ARENA Mech Interp Tutorials**: https://arena-course.com/
  - Callum McDougall. Exercises with solutions. THE learning resource.
- **Neel Nanda's Quickstart Guide**: https://www.neelnanda.io/mechanistic-interpretability/quickstart
- **Neel Nanda's Prerequisites**: https://www.neelnanda.io/mechanistic-interpretability/prereqs
- **TransformerLens Getting Started**: https://transformerlensorg.github.io/TransformerLens/content/getting_started_mech_interp.html

## Key Links

### Tools
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- SAELens: https://github.com/decoderesearch/SAELens
- nnsight: https://nnsight.net/
- Circuit Tracer: https://www.anthropic.com/research/open-source-circuit-tracing
- Neuronpedia: https://www.neuronpedia.org/ (public API used in notebook 04: `/api/feature/gpt2-small/{layer}-res-jb/{id}`)

### Research Hubs
- Transformer Circuits: https://transformer-circuits.pub/
- Alignment Forum: https://www.alignmentforum.org/
- LessWrong MI posts: https://www.lesswrong.com/tag/mechanistic-interpretability
- ICML MI Workshop: https://mechinterpworkshop.com/

### Community / Getting Involved
- MATS (mentorship program): https://www.matsprogram.org/
- Anthropic Fellows: https://www.anthropic.com/research
- EleutherAI Discord
- Apart Research hackathons

## Methodological rules (learned the hard way, Sep 2026)

1. **Run the model as it was trained.** GPT-2 needs its BOS token; without it the first real token becomes the attention sink and every downstream number is distorted. Dropping BOS made the Feb 2026 results look *better* while being wrong.
2. **Attribute on a logit difference, never a raw logit or probability.** Pick a named counterfactual (Paris vs Rome). LayerNorm's mean-subtraction and the unembed bias cancel in a difference.
3. **Make the accounting close.** Direct logit attribution must sum, through the cached final-LayerNorm scale plus the bias term, to the observed logit difference. If it does not, do not publish the numbers.
4. **Zero-ablation confounds "carries the information" with "is structurally load-bearing".** Use patching from a matched counterfactual prompt.
5. **Check an SAE behaviorally before reading features off it** (splice its reconstruction in; compare the metric). Layer 6's GPT-2 SAE loses a third of the capital-city behavior; layer 8's loses nothing.
6. **Steering must respect the circuit.** A feature edit at the country token works at layer 8 and does nothing at layer 10, because the heads that read it have already fired.

## Progress Log

### Feb 2026
- Created workspace and knowledge map; curriculum pages for Transformer Internals and Superposition & Features.
- Notebooks 01 (GPT-2 internals) and 02 (DLA circuit tracing), first versions. Both had the BOS and LayerNorm-scale bugs described above.

### Sep 2, 2026 (Cas, one evening)
- Verified the two bugs by constructing the counter-case (BOS on: Paris rank 93 on the bare prompt; DLA sum 187 vs logit 13).
- New canonical prompt (one-shot: "The capital of Germany is Berlin. The capital of France is", Paris at 70%) and counterfactual (Italy → Rome). `experiments/interp_utils.py` encodes the conventions.
- Rewrote 01 and 02; wrote 03 (activation patching) and 04 (SAE features + feature-swap steering). All four executed headlessly with static figures. Findings in the README table.
- Environment upgraded: TransformerLens 2.17 → 3.8.1, SAELens 6.50, kaleido; `requirements.txt` pinned; `interp` Jupyter kernel registered.
- Site: timeline extended Aug 2025 → Jul 2026 (9 entries, sourced from transformer-circuits.pub and anthropic.com/research); roadmap phases 1–4 link to the notebooks.
- Notebook 06: Anthropic's Jacobian lens (github.com/anthropics/jacobian-lens) fitted on GPT-2 Small (100 WikiText prompts, 12 min, <1 GB). J-lens reads the answer *type* at L6 and the fact at the source token from L0 (capital in top-10 for 9/10 countries vs 0/10 for the logit lens); no workspace band by the paper's four signatures; list category loading/eviction reproduces at L6. Gotchas: jlens sets `add_bos_token=True`, so use `encode(..., add_special_tokens=False)` for token ids; the venv's `torch/bin/torch_shm_manager` had lost its exec bit (chmod +x). Qwen3-1.7B fit costs ~5 min/prompt at 13 GB.
- Notebook 05 + `scale_sweep.py`: same measurements on GPT-2 M/L/XL and Qwen3-1.7B. Shape invariant, concentration not (Qwen3's L20H13 alone = 71%). Peak RSS is ~13 bytes/param through TransformerLens's load path (Large 10.6 GB, XL 22 GB, Qwen3-1.7B 26 GB); Qwen3-8B would exceed the machine. Running two sweeps concurrently froze the Mac once (load average 148) — one model at a time.

### Sep 5, 2026 (Codex)

- Audited saved scale measurements without loading a model; distinguished
  individual-effect sums from joint recovery and narrowed the training-recipe
  claim. See `notes/2026-09-05-saved-scale-audit.md`.
- Ran three prospectively specified studies with one cached GPT-2 Small at a
  time: fixed-head generalization, attention/difference transport, and a new-country
  format-offset test. Runtime ~175 seconds total; peak RSS 2.29 GiB on CPU.
- The original heads transfer across countries and wording, but 165% pairwise
  recovery can coexist with zero correct top-1 capital answers. Country-state
  differences from failing bare prompts remain usable in one-shot contexts.
- A fixed mean offset at the final token before layer 8 rescued 10/10 new
  countries excluded from fitting; three norm-matched random vectors each
  rescued 0/10. The primary three-head-offset prediction failed (0/10); the
  successful broader residual intervention was a prespecified secondary test.
  This relates to established task/function-vector work; no novelty claim.
- Full methods, negative results, data and figure:
  `notes/2026-09-05-capital-circuit-results.md`. Finite follow-up milestones:
  `RESEARCH_GOALS.md` (format/position controls, downstream mediation, metric
  validity). These follow-ups have not been run or scheduled.

### Open threads
- Phase 3 still owes the induction-head reproduction.
- Phase 5: attribution graph with circuit-tracer on Gemma-2-2B.
- Overnight J-lens fit on Qwen3-1.7B (and its base model) → notebook 06's signatures; does the workspace band appear with recipe, scale, or post-training?
- Pythia sweep (same recipe, many sizes + checkpoints) to separate size from recipe in the concentration result of notebook 05. Qwen3-8B needs a machine with >64 GB for TransformerLens.
- Toy SAE on synthetic superposition data (phase 2 deliverable).
