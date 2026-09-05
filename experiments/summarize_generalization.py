"""Verify and summarize saved generalization measurements, without a model."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generalization"
TRIPLE = "L8H11+L9H8+L10H0"
SINGLES = ["L8H11", "L9H8", "L10H0"]
TEMPLATES = ["one_shot", "bare", "possessive", "question", "distractor"]


def load():
    manifest = json.loads((DATA / "run.json").read_text())
    assert manifest["status"] == "complete", "Wait for a complete run"
    assert len(manifest["cases"]) == len(set(manifest["cases"])) == 30
    assert manifest["source_sha256"] == hashlib.sha256(
        (ROOT / "experiments/capital_generalization.py").read_bytes()).hexdigest()
    assert manifest["plan_sha256"] == hashlib.sha256(
        (ROOT / "notes/2026-09-05-generalization-plan.md").read_bytes()).hexdigest()
    rows = []
    cases = []
    for case_id in manifest["cases"]:
        c = json.loads((DATA / f"{case_id}.json").read_text())
        if c["status"] != "complete":
            print("Excluded", case_id, c["status"])
            continue
        assert max(c["control_errors"].values()) < 1e-4
        effects = {(e["group"], e["scope"]): e for e in c["effects"]}
        assert len(effects) == 20
        for e in effects.values():
            assert abs(e["delta_restore"] - (e["restored"]["ld"]-c["baseline_b"]["ld"])) < 1e-8
            assert abs(e["delta_disrupt"] - (c["baseline_a"]["ld"]-e["disrupted"]["ld"])) < 1e-8
        cases.append(c)
        if not c["normalizable"]:
            continue
        final, allpos = effects[TRIPLE, "final"], effects[TRIPLE, "all"]
        row = dict(id=case_id, template=c["template"], pair="/".join(c["pair"]),
                   calibration=c["calibration"], competent=c["competent"], gap=c["gap"],
                   a_rank=c["baseline_a"]["a_rank"], b_rank=c["baseline_b"]["b_rank"],
                   recovery_final=final["recovery"], disruption_final=final["disruption"],
                   recovery_all=allpos["recovery"], disruption_all=allpos["disruption"],
                   direction_difference=final["disruption"]-final["recovery"],
                   recovery_interaction=final["recovery"]-sum(effects[s, "final"]["recovery"] for s in SINGLES),
                   disruption_interaction=final["disruption"]-sum(effects[s, "final"]["disruption"] for s in SINGLES),
                   scope_difference=allpos["recovery"]-final["recovery"],
                   control_recovery=st.mean(effects[f"control_{i}", "final"]["recovery"] for i in range(3)),
                   restored_a_rank=final["restored"]["a_rank"],
                   disrupted_b_rank=final["disrupted"]["b_rank"],
                   head9_attention=st.mean(c["attention"]["L9H8"][f"country_{s}"] for s in "ab"))
        rows.append(row)
    with (DATA / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, cases, rows


def main():
    manifest, cases, rows = load()
    held = [r for r in rows if not r["calibration"]]
    print(f"Runtime {manifest['elapsed_seconds']:.1f}s; peak {manifest['peak_rss_mib']:.1f} MiB")
    print(f"Valid {len(cases)}/30; held-out normalizable {len(held)}/25; competent {sum(r['competent'] for r in held)}/25")
    for population in ("all", "competent"):
        print(f"\n{population}: template n R_final R_all N_final control_R interaction_R scope_delta attention9")
        for template in TEMPLATES:
            subset = [r for r in held if r["template"] == template and (population == "all" or r["competent"])]
            if not subset:
                print(template, "n=0")
                continue
            keys = ["recovery_final", "recovery_all", "disruption_final", "control_recovery",
                    "recovery_interaction", "scope_difference", "head9_attention"]
            print(template, len(subset), " ".join(f"{st.mean(r[k] for r in subset):.3f}" for k in keys))
    for k in ("recovery_final", "disruption_final", "direction_difference", "recovery_interaction", "scope_difference"):
        low, high = min(held, key=lambda r:r[k]), max(held, key=lambda r:r[k])
        print(f"{k} range: {low[k]:.4f} ({low['id']}) to {high[k]:.4f} ({high['id']})")
    print("Worst self/full control error", max(max(c["control_errors"].values()) for c in cases))
    print("Top-answer transfers", sum(r["restored_a_rank"] == 1 for r in held), sum(r["disrupted_b_rank"] == 1 for r in held), "/", len(held))


if __name__ == "__main__":
    main()
