"""Is hiding more audience-dependent than rules that are not about concealment?

For each layer, compares four paired contrasts over the same scenarios and
audiences:
  hide_fact / hide_directive   hide vs disclose (3 phrasings)
  tone                         formal vs casual, naming the audience (3 phrasings)
  format                       three-sentence limit vs any length (1 phrasing)

Reports the audience-specific fraction of each direction's squared norm
(cross-fitted, so unbiased), the cosine between hiding's audience-specific
components and each control's, and the correlation between hiding's
audience-similarity structure and the controls'.

    .venv/bin/python experiments/concealment/compare_controls.py --model Qwen3-8B
"""
import argparse
import json

import numpy as np

from analyze import AUD, cos, load, split_stats
from prompts import DATA

LAYERS = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]


def contrast_index(meta, study, pos, neg, content="fact", phrasings=(0, 1, 2)):
    key = {(m["study"], m["scenario"], m["audience"], m["content"], m["phrasing"], m["rule"]): i
           for i, m in enumerate(meta)}
    scen = sorted({m["scenario"] for m in meta})
    p = np.array([[[key[(study, s, a, content, ph, pos)] for ph in phrasings] for a in AUD] for s in scen])
    n = np.array([[[key[(study, s, a, content, ph, neg)] for ph in phrasings] for a in AUD] for s in scen])
    return p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--splits", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    meta, resid = load(args.model)
    idx = {
        "hide_fact": contrast_index(meta, "main", "hide", "disclose", "fact"),
        "hide_directive": contrast_index(meta, "main", "hide", "disclose", "directive"),
        "tone": contrast_index(meta, "tone", "formal", "casual"),
        "format": contrast_index(meta, "format", "restrict", "permit", phrasings=(0,)),
    }
    off = ~np.eye(len(AUD), dtype=bool)
    out = []
    for L in LAYERS:
        X = np.asarray(resid[:, L, :], dtype=np.float32)
        D = {k: (X[p] - X[n]).mean(2) for k, (p, n) in idx.items()}  # [S, A, H]
        st = {k: split_stats(v, rng, args.splits) for k, v in D.items()}
        spec = {k: np.median(v["spec_frac"]) for k, v in st.items()}
        resM = {k: np.median(v["res_cross"], 0) for k, v in st.items()}
        r = {k: D[k].mean(0) - D[k].mean((0, 1)) for k in D}  # audience-specific components
        row = dict(layer=L, specific_fraction={k: float(v) for k, v in spec.items()},
                   norm={k: float(np.linalg.norm(D[k].mean((0, 1)))) for k in D})
        for h in ("hide_fact", "hide_directive"):
            for c in ("tone", "format"):
                row[f"cos_specific_{h}_{c}"] = float(np.mean([cos(r[h][a], r[c][a]) for a in range(len(AUD))]))
                row[f"structure_corr_{h}_{c}"] = float(np.corrcoef(resM[h][off], resM[c][off])[0, 1])
        row["cos_specific_fact_directive"] = float(np.mean([cos(r["hide_fact"][a], r["hide_directive"][a])
                                                            for a in range(len(AUD))]))
        out.append(row)
        print(f"L{L:2d} specific  " + "  ".join(f"{k} {v:.3f}" for k, v in spec.items())
              + f"\n     cos(spec hide_fact, tone) {row['cos_specific_hide_fact_tone']:+.2f}"
              f"  (…, format) {row['cos_specific_hide_fact_format']:+.2f}"
              f"  (fact, directive) {row['cos_specific_fact_directive']:+.2f}"
              f"  structure corr hide~tone {row['structure_corr_hide_fact_tone']:+.2f}", flush=True)
    (DATA / args.model / "controls.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
