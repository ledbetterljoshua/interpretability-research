# Mechanistic Interpretability — A Field Guide and a Lab Notebook

Two things live here:

1. **A field guide** (`index.html` + two curriculum pages): an interactive knowledge map of mechanistic interpretability — concepts, timeline, people, tools, open problems, and a learning roadmap. Vanilla HTML/CSS/JS, no build step.
2. **A lab notebook** (`experiments/`): a sequence of executed Jupyter notebooks that reverse-engineer one behavior of GPT-2 Small end to end, following the roadmap's phases. Every notebook is committed *with its outputs*, and every figure is a static image, so the results read on GitHub without running anything.

**Live site:** serve locally with any HTTP server (ES modules need one):

```bash
python3 -m http.server 8742   # then open http://localhost:8742
```

## The experiments

The notebooks trace a single question through four techniques: *how does GPT-2 Small answer "The capital of Germany is Berlin. The capital of France is" → " Paris"?* Each notebook fixes a limitation of the previous one.

| Notebook | Technique | What it finds |
|---|---|---|
| [`01_gpt2_internals`](experiments/01_gpt2_internals.ipynb) | Hooks, residual stream, attention patterns, logit lens, zero-ablation | The answer appears at layer 9 (rank 7 → rank 1, 95%) and late layers calibrate it down to 70%. Zero-ablation flags heads that are load-bearing but irrelevant, which motivates patching. Also: why you must keep GPT-2's BOS token, shown with the artifact it causes. |
| [`02_circuit_tracing`](experiments/02_circuit_tracing.ipynb) | Direct logit attribution, exact to three decimals | Three heads carry the Paris-vs-Rome logit difference: **L9H8** (+2.6 of 6.0), **L8H11** (+1.4), **L10H0** (+1.1); two suppressors push back. L9H8 attends ~90% to the country token and writes a "France-associated" direction, not a Paris direction. Generalizes across 10 countries. |
| [`03_activation_patching`](experiments/03_activation_patching.ipynb) | Residual, block, head, and pattern-vs-value patching | The country's identity sits at the country token through layer 8, then moves to the answer position across layers 9–10. The three heads recover 97% of the behavior between them; their attention patterns are identical on clean and corrupted prompts, so the information is entirely in what they *read*, not where they *look*. |
| [`04_sae_features`](experiments/04_sae_features.ipynb) | Pretrained sparse autoencoders, feature swaps as steering | The country token is one dominant SAE feature per country (activation ~40 vs ~10 for the next), the same direction across four independently trained SAEs (cosine 0.86–0.97). France-specific features appear at the answer position at layer 9–10, matching the patching hand-off. Swapping the Italy feature for the France feature at layer 8 rewrites the answer to Paris (63%); the same swap at layer 10 does nothing, because the information has already moved. |
| [`05_does_it_scale`](experiments/05_does_it_scale.ipynb) | The same logit lens + patching sweep on GPT-2 Medium/Large/XL and Qwen3-1.7B (`experiments/scale_sweep.py`, results in `data/scale/`) | The circuit's *shape* is invariant: the fact sits at the country token, attention moves it to the answer position at 62–75% of depth, the top head always attends 81–92% to the country. Its *concentration* is not: the top head recovers 43% → 26% → 26% → 8% across the GPT-2 family, then **71%** in Qwen3-1.7B (one head, one-layer hand-off). Concentration tracks training recipe, not size. |

Shared conventions live in [`experiments/interp_utils.py`](experiments/interp_utils.py): BOS always on, attribution always on a logit difference against a named counterfactual, components always scaled through the cached final LayerNorm so decompositions sum exactly.

### A note on the revision (September 2026)

The first versions of notebooks 1 and 2 (February 2026) had two methodological bugs that produced a cleaner-looking, wrong story: GPT-2 was run *without* its BOS token, and direct logit attribution skipped the final LayerNorm scale, inflating every number ~14× and mis-ranking the MLPs. The rewritten notebooks show both artifacts explicitly rather than deleting them, because they are among the most common mistakes in the field and the fix (run the model as trained; make the accounting close) is the lesson.

### Running the notebooks

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name interp --display-name "Interp Research (3.12)"
.venv/bin/python -m jupyter lab experiments/
```

Tested on an Apple M5 Pro (MPS) with TransformerLens 3.8, SAELens 6.50, PyTorch 2.10. Notebooks 1–4 run in one to three minutes each (5 only reads JSON); the first run downloads GPT-2 Small (~500 MB) and four SAEs (~300 MB each). To re-execute headlessly:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace experiments/0N_*.ipynb
```

(`python -m nbconvert` rather than `jupyter nbconvert`: the venv's console scripts break if the repo directory is moved.)

## The field guide

- **Overview** — what the field is, why it matters, where it stands
- **Core Concepts** — interactive concept map from foundational (residual streams, attention) through frontier (crosscoders, attribution graphs)
- **Timeline** — key papers and breakthroughs 2020 → 2026, including the 2025–26 Anthropic thread (persona vectors, introspection, the assistant axis, emotion concepts, natural language autoencoders, the global workspace)
- **People & Organizations**, **Tools & Infrastructure**, **Open Problems**
- **Learning Roadmap** — six phases with deliverables; phases 1–4 now link to the notebooks above

Curriculum pages: [Transformer Internals](transformer-internals.html) and [Superposition & Features](superposition-features.html).

### Tech stack

Vanilla HTML, CSS, JavaScript. Fonts: Newsreader, Hanken Grotesk, JetBrains Mono. Modular CSS (`base`, `layout`, `components`, `animations`, `curriculum`) and ES modules (`main`, `data`, `scroll-animations`, `transformer-visuals`).

## Project structure

```
├── index.html                    # Field guide (single scroll)
├── transformer-internals.html    # Curriculum: Transformer Internals
├── superposition-features.html   # Curriculum: Superposition & Features
├── styles/, scripts/             # Site CSS and JS
├── experiments/
│   ├── interp_utils.py           # Shared conventions (BOS, logit diff, LN-scaled DLA, logit lens)
│   ├── 01_gpt2_internals.ipynb
│   ├── 02_circuit_tracing.ipynb
│   ├── 03_activation_patching.ipynb
│   ├── 04_sae_features.ipynb
│   ├── 05_does_it_scale.ipynb
│   └── scale_sweep.py            # Runs the 05 measurements on any TransformerLens model
├── data/scale/                   # Sweep results (JSON) for 5 models
├── requirements.txt
├── RESEARCH.md                   # Reading list and progress log
└── notes/, papers/, data/, visualizations/
```

## What's next

- Isolate *why* Qwen3 concentrates where GPT-2 XL diffuses: run `scale_sweep.py` over the Pythia suite (one recipe, 14M–12B, with checkpoints) to separate size from recipe.
- Roadmap phase 5: attribution graphs with [circuit-tracer](https://github.com/safety-research/circuit-tracer) on Gemma-2-2B.
- The induction-head reproduction still owed to phase 3, and a toy SAE for phase 2.
- Qwen3-8B itself: its weights are cached, but loading it through TransformerLens needs more than 64 GB of unified memory (the fp32 path peaks at ~13 bytes/param). Run it on a bigger box, or with a leaner loader.
