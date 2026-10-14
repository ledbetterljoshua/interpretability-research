"""SVG figures for the concealment-audience study, from saved data only.

Writes visualizations/concealment/{leak-by-audience,axis-vs-leak,steering}.svg.
No model is loaded; activations are read from the cached resid.npy files.

    .venv/bin/python experiments/concealment/figures.py
"""
import json
from pathlib import Path

import numpy as np

from analyze import AUD, load, paired_index
from prompts import DATA, OVERSIGHT

OUT = Path(__file__).resolve().parents[2] / "visualizations" / "concealment"
INK, DIM, GRID, ACCENT, ALT = "#1f2328", "#6b7280", "#e5e7eb", "#2563eb", "#d97706"
FONT = "font-family='-apple-system, Helvetica, Arial, sans-serif'"
LABEL = {"user": "End user", "operator": "Operator admin", "developer": "Developer", "monitor": "Safety monitor",
         "auditor": "External auditor", "colleague": "Colleague", "journalist": "Journalist", "family": "Family member"}


def jl(path):
    return [json.loads(l) for l in open(path)]


def boot(x, rng, n=5000):
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def svg(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' {FONT}>"
            f"<rect width='{w}' height='{h}' fill='#fff'/>{body}</svg>")


def text(x, y, s, size=13, fill=INK, anchor="start", weight=400):
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"


def x_axis(x0, x1, y, ticks, scale, label):
    out = [f"<line x1='{x0}' x2='{x1}' y1='{y}' y2='{y}' stroke='{DIM}'/>"]
    for t in ticks:
        out.append(f"<line x1='{scale(t)}' x2='{scale(t)}' y1='{y}' y2='{y + 5}' stroke='{DIM}'/>")
        out.append(text(scale(t), y + 19, f"{t:g}", 12, DIM, "middle"))
    out.append(text((x0 + x1) / 2, y + 40, label, 13, DIM, "middle"))
    return "".join(out)


def leak_by_audience(model, rng):
    rows = [r for r in jl(DATA / model / "behavior.jsonl") if r["rule"] == "hide"]
    stats = {c: {a: boot([r["leak_reply"] for r in rows if r["audience"] == a and r["content"] == c], rng)
                 for a in AUD} for c in ("fact", "directive")}
    order = sorted(AUD, key=lambda a: -stats["fact"][a][0])
    W, H, left, right, top, row = 720, 410, 170, 690, 46, 34
    sx = lambda v: left + v * (right - left)
    b = [text(left, 22, f"{model}: share of replies that reveal the secret despite a rule to hide it", 14, INK, weight=600)]
    for t in (0, .25, .5, .75, 1):
        b.append(f"<line x1='{sx(t)}' x2='{sx(t)}' y1='{top - 8}' y2='{top + row * 8}' stroke='{GRID}'/>")
    for i, a in enumerate(order):
        y = top + i * row + 14
        b.append(text(left - 12, y + 4, LABEL[a] + (" ◆" if a in OVERSIGHT else ""), 13, INK, "end"))
        for c, color, dy in (("fact", ACCENT, -5), ("directive", ALT, 6)):
            m, lo, hi = stats[c][a]
            b.append(f"<line x1='{sx(lo)}' x2='{sx(hi)}' y1='{y + dy}' y2='{y + dy}' stroke='{color}' stroke-width='2' opacity='.55'/>")
            b.append(f"<circle cx='{sx(m)}' cy='{y + dy}' r='5' fill='{color}'/>")
    b.append(x_axis(left, right, top + row * 8 + 2, (0, .25, .5, .75, 1), sx, "Leak rate under the hide rule (48 scenarios, 95% bootstrap interval)"))
    b.append(f"<circle cx='{left}' cy='{H - 12}' r='5' fill='{ACCENT}'/>" + text(left + 10, H - 8, "Secret fact", 12, DIM))
    b.append(f"<circle cx='{left + 100}' cy='{H - 12}' r='5' fill='{ALT}'/>" + text(left + 110, H - 8, "Hidden operating directive", 12, DIM))
    b.append(text(left + 290, H - 8, "◆ oversight audience", 12, DIM))
    return svg(W, H, "".join(b))


def axis_vs_leak(model, layer):
    meta, resid = load(model)
    hide, disc, _, _ = paired_index(meta)
    X = np.asarray(resid[:, layer, :], dtype=np.float32)
    D = (X[hide] - X[disc])[:, :, 0].mean((0, 2))
    shared = D.mean(0)
    proj = (D - shared) @ (shared / np.linalg.norm(shared))
    rows = [r for r in jl(DATA / model / "behavior.jsonl") if r["rule"] == "hide" and r["content"] == "fact"]
    leak = np.array([np.mean([r["leak_reply"] for r in rows if r["audience"] == a]) for a in AUD])
    r = np.corrcoef(proj, leak)[0, 1]
    W, H, left, right, top, bottom = 560, 400, 70, 530, 50, 330
    pad = (proj.max() - proj.min()) * 0.15
    sx = lambda v: left + (v - proj.min() + pad) / (proj.max() - proj.min() + 2 * pad) * (right - left)
    sy = lambda v: bottom - v * (bottom - top)
    b = [text(left, 22, f"{model}, layer {layer}: hiding-axis position vs. disclosure", 14, INK, weight=600)]
    for t in (0, .25, .5, .75, 1):
        b.append(f"<line x1='{left}' x2='{right}' y1='{sy(t)}' y2='{sy(t)}' stroke='{GRID}'/>" + text(left - 8, sy(t) + 4, f"{t:g}", 12, DIM, "end"))
    b.append(f"<line x1='{sx(0)}' x2='{sx(0)}' y1='{top}' y2='{bottom}' stroke='{DIM}' stroke-dasharray='3 3'/>")
    placed = []  # label positions, nudged apart so nearby points stay readable
    for a, p, l in sorted(zip(AUD, proj, leak), key=lambda t: -t[2]):
        color = ACCENT if a in OVERSIGHT else INK
        x, y = sx(p) + 9, sy(l) + 4
        while any(abs(y - py) < 14 and abs(x - px) < 80 for px, py in placed):
            y += 14
        placed.append((x, y))
        b.append(f"<circle cx='{sx(p)}' cy='{sy(l)}' r='6' fill='{color}'/>")
        b.append(text(x, y, LABEL[a], 12, color))
    b.append(f"<line x1='{left}' x2='{right}' y1='{bottom}' y2='{bottom}' stroke='{DIM}'/>")
    b.append(text((left + right) / 2, bottom + 30, "Audience-specific component projected on the shared hiding direction", 13, DIM, "middle"))
    b.append(text((left + right) / 2, bottom + 48, "(left = pushes toward disclosure)", 12, DIM, "middle"))
    b.append(f"<text x='18' y='{(top + bottom) / 2}' font-size='13' fill='{DIM}' text-anchor='middle' transform='rotate(-90 18 {(top + bottom) / 2})'>Fact leak rate under the hide rule</text>")
    b.append(text(right, top - 8, f"Pearson r = {r:+.2f} across 8 audiences", 12, DIM, "end"))
    return svg(W, H, "".join(b)), float(r)


def steering(model, rng):
    base = {(r["scenario"], r["audience"]): r["leak_reply"] for r in jl(DATA / model / "behavior.jsonl")
            if r["rule"] == "hide" and r["content"] == "fact"}
    rows = []
    for name in ("dose", "dose2", "axis"):
        p = DATA / model / f"steer_{name}.jsonl"
        if p.exists():
            rows += [r for r in jl(p) if r["layers"][0] == 12]
    conds = [("monitor", None, 0, "No steering"),
             ("monitor", "random", -2, "Random, independent per layer"),
             ("monitor", "random_coherent", -2, "Random, one direction at every layer"),
             ("monitor", "r_colleague", -2, "Colleague's component"),
             ("monitor", "r_user", -2, "User's component"),
             ("monitor", "r_monitor_orth", -2, "Monitor component, orthogonal part*"),
             ("monitor", "r_monitor", -1, "Monitor component ×1"),
             ("monitor", "r_monitor", -2, "Monitor component ×2"),
             ("monitor", "r_monitor_axis", -1, "Hiding-axis part only ×1"),
             ("monitor", "r_monitor_axis", -2, "Hiding-axis part only ×2"),
             ("user", None, 0, "User prompts: no steering"),
             ("user", "r_monitor_axis", -2, "User prompts: hiding-axis part ×2")]
    W, row, left, right, top = 720, 28, 300, 690, 46
    H = top + row * len(conds) + 80
    sx = lambda v: left + v * (right - left)
    b = [text(20, 22, f"{model}: subtracting directions at layers 12–24 while the model answers", 14, INK, weight=600)]
    for t in (0, .25, .5, .75, 1):
        b.append(f"<line x1='{sx(t)}' x2='{sx(t)}' y1='{top - 8}' y2='{top + row * len(conds)}' stroke='{GRID}'/>")
    for i, (aud, vec, alpha, label) in enumerate(conds):
        y = top + i * row + 12
        if vec is None:
            vals = [base[(s, aud)] for s in sorted({k[0] for k in base})]
        else:
            vals = [r["leak"] for r in rows if r["audience"] == aud and r["vector"] == vec and r["alpha"] == alpha]
        if not vals:
            continue
        m, lo, hi = boot(vals, rng)
        color = ACCENT if vec and vec.startswith("r_monitor") and vec != "r_monitor_orth" else (DIM if vec else INK)
        b.append(text(left - 12, y + 4, label, 13, INK, "end"))
        b.append(f"<line x1='{sx(lo)}' x2='{sx(hi)}' y1='{y}' y2='{y}' stroke='{color}' stroke-width='2' opacity='.55'/>")
        b.append(f"<circle cx='{sx(m)}' cy='{y}' r='5' fill='{color}'/>")
    b.append(x_axis(left, right, top + row * len(conds) + 2, (0, .25, .5, .75, 1), sx, "Fact leak rate (48 scenarios, 95% bootstrap interval)"))
    b.append(text(20, H - 8, "* Replies mostly narrate the exchange instead of answering, so the secret is avoided rather than concealed.", 12, DIM))
    return svg(W, H, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    written = {}
    for model in ("Qwen3-8B", "Qwen3-14B"):
        path = DATA / model / "behavior.jsonl"
        if not path.exists() or sum(1 for _ in open(path)) < 1536:  # missing or still generating
            continue
        tag = model.split("-")[1].lower()
        (OUT / f"leak-by-audience-{tag}.svg").write_text(leak_by_audience(model, rng))
        layer = 12 if model == "Qwen3-8B" else 13
        fig, r = axis_vs_leak(model, layer)
        (OUT / f"axis-vs-leak-{tag}.svg").write_text(fig)
        written[model] = dict(axis_layer=layer, pearson=r)
        if (DATA / model / "steer_axis.jsonl").exists():
            (OUT / f"steering-{tag}.svg").write_text(steering(model, rng))
    print(json.dumps(written, indent=1), "->", OUT)


if __name__ == "__main__":
    main()
