# Interpretability Research

## What This Is

Joshua's journey into mechanistic interpretability research. Learning by building.

## Workspace Structure

```
interpretability-research/
├── index.html              # Interactive knowledge map (open in browser)
├── RESEARCH.md             # This file - notes, links, progress
├── notes/                  # Reading notes, concept explorations
├── experiments/            # Jupyter notebooks, scripts
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
- Neuronpedia: https://www.neuronpedia.org/

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

## Progress Log

### Started: Feb 2026
- Created workspace and knowledge map
- TODO: Phase 1 - Transformer internals
