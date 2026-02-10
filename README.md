# Mechanistic Interpretability — A Field Guide

A comprehensive, interactive knowledge map for the field of mechanistic interpretability. Built as a single-page reference and learning resource, covering concepts, people, tools, open problems, and a structured learning roadmap.

**Live site:** Serve locally with any HTTP server (uses ES modules).

## What This Is

A resource designed for someone who wants to seriously study mechanistic interpretability — from zero background to cutting-edge research. It covers:

- **Overview** — What the field is, why it matters, and where it stands today
- **Core Concepts** — Interactive concept map from foundational (residual streams, attention) through frontier (crosscoders, attribution graphs)
- **Timeline** — Key papers and breakthroughs from 2020 to 2026
- **People & Organizations** — Who's doing this work and where
- **Tools & Infrastructure** — The practical toolkit (TransformerLens, SAELens, Neuronpedia, etc.)
- **Open Problems** — Where the field is stuck and where contributions are needed
- **Learning Roadmap** — A structured 6-phase path with concrete deliverables

### Curriculum Pages

The roadmap links to dedicated curriculum pages with full educational content:

- **[Transformer Internals](transformer-internals.html)** — Deep dive into how transformers compute mechanistically. Includes interactive step-by-step walkthroughs of attention and induction heads, a section-by-section reading guide for "A Mathematical Framework for Transformer Circuits," required reading lists, and hands-on exercises.
- **[Superposition & Features](superposition-features.html)** — The core problem of interpretability: why neurons are polysemantic, the geometry of superposition, sparse autoencoders as the solution, step-by-step SAE walkthrough, paper guides for "Toy Models of Superposition" and "Towards Monosemanticity," and exercises with SAELens and Neuronpedia.

## Tech Stack

Vanilla HTML, CSS, and JavaScript. No framework, no build step.

- **Fonts:** Newsreader (editorial serif with optical sizing), Hanken Grotesk (geometric humanist sans), JetBrains Mono
- **Design:** Light-mode editorial aesthetic with warm off-white palette and deep indigo accents
- **Architecture:** Single-scroll main page with IntersectionObserver-based navigation and scroll-reveal animations
- **Modular CSS:** `base.css` (design tokens), `layout.css`, `components.css`, `animations.css`, `curriculum.css`
- **ES Modules:** `main.js`, `data.js`, `scroll-animations.js`, `transformer-visuals.js`

## Running Locally

```bash
# Any HTTP server works (ES modules require it)
python3 -m http.server 8742

# Then open http://localhost:8742
```

## Project Structure

```
├── index.html                  # Main field guide (single-scroll)
├── transformer-internals.html  # Curriculum: Transformer Internals
├── superposition-features.html # Curriculum: Superposition & Features
├── styles/
│   ├── base.css                # Design tokens, fonts, reset, typography
│   ├── layout.css              # Header, nav, sections, footer, responsive
│   ├── components.css          # Cards, tags, chips, panels, timeline, roadmap
│   ├── animations.css          # Scroll reveals, staggers, reduced motion
│   └── curriculum.css          # Curriculum-specific styles (steppers, math, etc.)
├── scripts/
│   ├── main.js                 # Main page: scroll nav, detail panels
│   ├── data.js                 # All concept/detail panel content
│   ├── scroll-animations.js    # IntersectionObserver reveal system
│   ├── transformer-visuals.js  # Curriculum page: steppers, scroll nav
│   └── neural-canvas.js        # (unused) Background animation
├── RESEARCH.md
├── data/
├── experiments/
├── notes/
├── papers/
└── visualizations/
```

## Design Decisions

- **Single-scroll over tabs** — All content visible, cross-referenced, searchable. No hiding content behind navigation.
- **Light mode** — Content-focused editorial design. Warm, not clinical.
- **No framework** — Zero dependencies, zero build step. Opens in any browser. Focus on content, not tooling.
- **Progressive disclosure** — Concept chips expand to detail panels. Roadmap links to dedicated curriculum pages. Overview is scannable; depth is available.
- **Cross-referencing** — People link to tools they built. Tools link to concepts they explore. Timeline links to people and methods. Everything is connected.

## Contributing

This is a living document. Content will be expanded as the field evolves and as more curriculum pages are built out.
