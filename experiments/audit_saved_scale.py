"""Reanalyze saved head-patching scores; no model or third-party imports.

Run: .venv/bin/python -I -S experiments/audit_saved_scale.py
All inputs are size-limited JSON files. Prints a Markdown table to stdout.
"""

import json
import math
from pathlib import Path


def summarize(path):
    if path.stat().st_size > 1_000_000:
        raise ValueError(f"Unexpectedly large input: {path}")
    d = json.loads(path.read_text())
    gap = d["clean_ld"] - d["corrupt_ld"]
    if not math.isfinite(gap) or gap <= 1e-6:
        raise ValueError(f"Nonpositive or near-zero normalization gap: {path}")
    rows = d["head_patch"]
    assert len(rows) == d["n_layers"]
    assert all(len(row) == d["n_heads"] for row in rows)
    heads = [(v, f"L{l}H{h}") for l, row in enumerate(rows)
             for h, v in enumerate(row)]
    assert all(math.isfinite(v) for v, _ in heads)
    positive = sorted(((v, name) for v, name in heads if v > 0), reverse=True)
    top, name = positive[0]
    positive_total = sum(v for v, _ in positive)
    # These are separate interventions, so their sum is only descriptive.
    top3_positive = sum(v for v, _ in positive[:3])
    by_magnitude = sorted(heads, key=lambda item: -abs(item[0]))
    stored_rule = sum(v for v, _ in by_magnitude[:3] if v > 0)
    assert math.isclose(stored_rule, d["top3_sum"], abs_tol=1e-9)
    raw_delta = top * gap
    return (d["model"], name, top, raw_delta, d["corrupt_ld"] + raw_delta,
            top / positive_total, top3_positive, d["top3_sum"])


def main():
    root = Path(__file__).resolve().parents[1]
    filenames = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
                 "Qwen__Qwen3-1.7B"]
    print("| Model | Top head | Recovery | Raw Δ logit diff | Patched Paris−Rome | Share of positive effects | Top 3 positive sum | Stored top3_sum |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for filename in filenames:
        model, head, recovery, delta, patched, share, top3, stored = summarize(
            root / "data" / "scale" / (filename + ".json"))
        print(f"| {model} | {head} | {recovery:.1%} | {delta:.3f} | {patched:+.3f} | {share:.1%} | {top3:.1%} | {stored:.1%} |")
    print("\nPositive-effect shares and sums describe individual interventions; "
          "they are not joint recovery or fractions of a causal decomposition.")


if __name__ == "__main__":
    main()
