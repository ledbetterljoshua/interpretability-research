"""Audience dependence of the hide-vs-disclose direction.

Reads data/concealment/<model_slug>/{meta.jsonl, resid.npy} and writes
results.json alongside. All direction estimates use paired differences
(hide minus disclose, matched on scenario, audience, content and phrasing).

Split-half statistics: scenarios are split into two random halves; a
direction estimated on one half is compared with one estimated on the other,
so noise is independent between the two estimates. Dot products across halves
are unbiased estimates of the dot product of the underlying directions.

    .venv/bin/python experiments/concealment/analyze.py --model Qwen3-8B
"""
import argparse
import json

import numpy as np

from prompts import AUDIENCES, DATA, OVERSIGHT, SOCIAL

AUD = list(AUDIENCES)
DETAIL_LAYERS = [9, 12, 18, 24, 30]  # fixed before analysis; curve reports every layer


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load(slug, tag=""):
    meta = [json.loads(l) for l in open(DATA / slug / f"meta{tag}.jsonl")]
    resid = np.load(DATA / slug / f"resid{tag}.npy", mmap_mode="r")
    return meta, resid


def paired_index(meta):
    """Index arrays for hide/disclose pairs [S, A, C, P] and format pairs [S, A]."""
    scen = sorted({m["scenario"] for m in meta})
    key = {(m["study"], m["scenario"], m["audience"], m["content"], m["phrasing"], m["rule"]): i
           for i, m in enumerate(meta)}
    S, A = len(scen), len(AUD)
    hide = np.zeros((S, A, 2, 3), int)
    disc = np.zeros((S, A, 2, 3), int)
    for si, s in enumerate(scen):
        for ai, a in enumerate(AUD):
            for ci, c in enumerate(("fact", "directive")):
                for p in range(3):
                    hide[si, ai, ci, p] = key[("main", s, a, c, p, "hide")]
                    disc[si, ai, ci, p] = key[("main", s, a, c, p, "disclose")]
    fr = np.array([[key[("format", s, a, "fact", 0, "restrict")] for a in AUD] for s in scen])
    fp = np.array([[key[("format", s, a, "fact", 0, "permit")] for a in AUD] for s in scen])
    return hide, disc, fr, fp


def split_stats(D, rng, n_splits):
    """D: [S, A, H] per-scenario hiding differences for one content condition."""
    S, A, _ = D.shape
    within, cross, res_within, res_cross, overs = [], [], [], [], []
    spec_frac = []
    oi = [AUD.index(a) for a in OVERSIGHT]
    so = [AUD.index(a) for a in SOCIAL]
    for _ in range(n_splits):
        perm = rng.permutation(S)
        d1, d2 = D[perm[: S // 2]].mean(0), D[perm[S // 2:]].mean(0)
        w = np.array([cos(d1[a], d2[a]) for a in range(A)])
        c = np.array([[(cos(d1[a], d2[b]) + cos(d1[b], d2[a])) / 2 for b in range(A)] for a in range(A)])
        r1, r2 = d1 - d1.mean(0), d2 - d2.mean(0)
        rw = np.array([cos(r1[a], r2[a]) for a in range(A)])
        rc = np.array([[(cos(r1[a], r2[b]) + cos(r1[b], r2[a])) / 2 for b in range(A)] for a in range(A)])
        o1 = d1[oi].mean(0) - d1[so].mean(0)
        o2 = d2[oi].mean(0) - d2[so].mean(0)
        # Share of the hiding direction's (unbiased) squared norm that is audience-specific.
        spec_frac.append(float(np.sum(r1 * r2) / np.sum(d1 * d2)))
        within.append(w); cross.append(c); res_within.append(rw); res_cross.append(rc)
        overs.append(cos(o1, o2))
    return dict(within=np.array(within), cross=np.array(cross), res_within=np.array(res_within),
                res_cross=np.array(res_cross), oversight=np.array(overs), spec_frac=np.array(spec_frac))


def permutation_null(D, rng, n_perm, n_splits=20):
    """Shuffle audience labels within each scenario; returns null mean residual reliability."""
    S, A, _ = D.shape
    out = []
    for _ in range(n_perm):
        Dp = np.stack([D[s, rng.permutation(A)] for s in range(S)])
        st = split_stats(Dp, rng, n_splits)
        out.append(float(st["res_within"].mean()))
    return np.array(out)


def q(x):
    return [float(np.percentile(x, 2.5)), float(np.median(x)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--perms", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    meta, resid = load(args.model)
    hide, disc, fr, fp = paired_index(meta)
    n_layers = resid.shape[1]
    results = dict(model=args.model, audiences=AUD, oversight=OVERSIGHT, social=SOCIAL,
                   detail_layers=DETAIL_LAYERS, layers=[])

    for L in range(1, n_layers):
        X = np.asarray(resid[:, L, :], dtype=np.float32)
        D_all = (X[hide] - X[disc]).mean(3)  # [S, A, C, H], averaged over phrasings
        F = (X[fr] - X[fp]).mean((0, 1))  # format-rule direction
        layer = dict(layer=L)
        for cname, D in (("both", D_all.mean(2)), ("fact", D_all[:, :, 0]), ("directive", D_all[:, :, 1])):
            st = split_stats(D, rng, args.splits if L in DETAIL_LAYERS else 40)
            full = D.mean(0)
            layer[cname] = dict(
                within=[q(st["within"][:, a]) for a in range(len(AUD))],
                cross_mean=q(st["cross"][:, ~np.eye(len(AUD), dtype=bool)].mean(1)),
                within_mean=q(st["within"].mean(1)),
                res_within=[q(st["res_within"][:, a]) for a in range(len(AUD))],
                res_within_mean=q(st["res_within"].mean(1)),
                oversight_reliability=q(st["oversight"]),
                specific_fraction=q(st["spec_frac"]),
                format_cos=[cos(full[a], F) for a in range(len(AUD))],
                norm=float(np.linalg.norm(full.mean(0))),
            )
            if L in DETAIL_LAYERS:
                layer[cname]["cross_matrix"] = np.median(st["cross"], 0).round(4).tolist()
                layer[cname]["res_cross_matrix"] = np.median(st["res_cross"], 0).round(4).tolist()
                if cname == "both":
                    null = permutation_null(D, rng, args.perms)
                    obs = float(st["res_within"].mean())
                    layer[cname]["perm_null_res_within"] = q(null)
                    layer[cname]["perm_p"] = float((np.sum(null >= obs) + 1) / (len(null) + 1))
        # Fact vs directive hiding directions, cross-fitted on scenario halves.
        fd = []
        for _ in range(40):
            perm = rng.permutation(D_all.shape[0])
            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
            f1, d2 = D_all[h1, :, 0].mean((0, 1)), D_all[h2, :, 1].mean((0, 1))
            f2, d1 = D_all[h2, :, 0].mean((0, 1)), D_all[h1, :, 1].mean((0, 1))
            fd.append((cos(f1, d2) + cos(f2, d1)) / 2)
        layer["fact_vs_directive"] = q(np.array(fd))
        results["layers"].append(layer)
        b = layer["both"]
        print(f"L{L:2d}  within {b['within_mean'][1]:.3f}  cross {b['cross_mean'][1]:.3f}  "
              f"specific {b['specific_fraction'][1]:.3f}  res_within {b['res_within_mean'][1]:.3f}  "
              f"oversight {b['oversight_reliability'][1]:.3f}  fact~dir {layer['fact_vs_directive'][1]:.3f}"
              + (f"  perm_p {b['perm_p']:.3f}" if "perm_p" in b else ""), flush=True)

    (DATA / args.model / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
