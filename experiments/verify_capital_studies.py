"""Cross-check independent experiment outputs and their recorded provenance."""
import hashlib
import json
import math
from pathlib import Path

from summarize_generalization import load, TRIPLE

ROOT = Path(__file__).resolve().parents[1]


def check_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, list):
        for child in value:
            check_finite(child)


def check_hash(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


def main():
    manifest, primary, _ = load()
    routing = json.loads((ROOT / "data/routing/run.json").read_text())
    transfer = json.loads((ROOT / "data/format_transfer/run.json").read_text())
    for d, stem, plan in [(routing, "capital_routing", "routing"),
                          (transfer, "capital_format_transfer", "format-transfer")]:
        assert d["status"] == "complete"
        assert d["revision"] == manifest["revision"]
        check_hash(ROOT / f"experiments/{stem}.py", d["source_sha256"])
        check_hash(ROOT / "experiments/capital_generalization.py", d["helper_sha256"])
        check_hash(ROOT / f"notes/2026-09-05-{plan}-plan.md", d["plan_sha256"])
        check_finite(d)
    check_hash(ROOT / "data/format_transfer/offsets.json", transfer["offsets_sha256"])
    assert len(routing["baselines"]) == 20
    assert len(routing["routing"]) == 120
    assert len(routing["transport"]) == 80
    assert len(routing["controls"]) == 60
    assert max(c["error"] for c in routing["controls"]) < 1e-4
    # Independent runs must agree on baselines and within-format interventions.
    comparisons = 0
    for c in primary:
        if c["calibration"] or c["template"] not in ("bare", "one_shot"):
            continue
        joint = next(e for e in c["effects"] if e["group"]==TRIPLE and e["scope"]=="final")
        for index, metric in [(0,"recovery"),(1,"disruption")]:
            country, other = c["pair"][index], c["pair"][1-index]
            r = next(r for r in routing["transport"] if r["donor_format"]==r["recipient_format"]==c["template"]
                     and r["target"]==country and r["recipient_country"]==other and r["site"]=="heads_final")
            assert abs(r["recovery"]-joint[metric]) < 1e-4, (c["id"], metric)
            baseline = next(r for r in routing["baselines"] if r["format"]==c["template"] and r["country"]==country)
            original = c["baseline_a"] if index==0 else c["baseline_b"]
            signed_ld = original["ld"] if index==0 else -original["ld"]
            assert abs(baseline["metrics"]["ld"]-signed_ld) < 1e-4
            comparisons += 1
    train = {country for country, _ in transfer["training_set"]}
    test = {country for country, _ in transfer["test_set"]}
    assert len(train) == len(test) == 10 and train.isdisjoint(test)
    assert len(transfer["cases"]) == 10
    assert {c["country"] for c in transfer["cases"]} == test
    for c in transfer["cases"]:
        assert c["status"] == "complete", "Report tokenization exclusions explicitly"
        assert c["zero_error"] < 1e-4
        assert len(c["effects"]) == 23
        for key in ("baseline", "reference"):
            assert 0 <= c[key]["p"] <= 1 and c[key]["rank"] >= 1
        for e in c["effects"]:
            assert 0 <= e["metrics"]["p"] <= 1 and e["metrics"]["rank"] >= 1
    print(f"Verified hashes, controls, disjoint fit/test countries, all planned records, and {comparisons} independent-run baseline/intervention agreements.")
    print(f"Total model-process runtime: {sum(d['elapsed_seconds'] for d in (manifest,routing,transfer)):.1f}s")
    print(f"Maximum peak process RSS: {max(d['peak_rss_mib'] for d in (manifest,routing,transfer)):.1f} MiB")


if __name__ == "__main__":
    main()
