"""Verify the fixed prospective audit cohort without loading a language model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from audit_population import POPULATION, check_inventory

ROOT = Path(__file__).resolve().parents[2]


def verify(paths, require_eligible=False, require_checkpoints=False):
    expected = {ROOT / "data/causal_audit" / name for name in POPULATION}
    paths = [Path(p).resolve() for p in paths]
    assert len(paths) == len(expected) and set(paths) == expected, \
        "Missing, duplicate, relocated or substituted audit cohort member"
    # Finish this cheap gate before any expensive historical re-verification.
    manifests = []
    for name in POPULATION:
        path = ROOT / "data/causal_audit" / name / "run.json"
        assert path.is_file(), f"Missing audit cohort manifest: {name}"
        manifests.append((name, json.loads(path.read_text())))
    check_inventory(manifests, require_eligible)
    reports = []
    for seed, verifier in ((1091, "verify_lower_gold_pair.py"),
                           (1289, "verify_lower_gold_replication.py")):
        pair = [ROOT / "data/causal_audit" / f"lower-gold-{arm}-{seed}-v1"
                for arm in ("conditional", "marginal")]
        command = [sys.executable, str(Path(__file__).with_name(verifier)),
                   *map(str, pair), "--require-pair"]
        if require_eligible:
            command.append("--require-eligible")
        if require_checkpoints:
            command.append("--require-checkpoints")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        reports.extend(json.loads(result.stdout))
    command = [sys.executable, str(Path(__file__).with_name("verify_expanded_controls.py")),
               *[str(ROOT / "data/causal_audit" / f"expanded-controls-teacher-{s}") for s in (1091, 1289)]]
    # That historical verifier's --require-eligible also requires its old six
    # members. Here we verify these two individually, then enforce eligibility.
    if require_checkpoints:
        command.append("--require-checkpoints")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    reports.extend(json.loads(result.stdout))
    assert len(reports) == 6 and {r["run"] for r in reports} == set(POPULATION)
    assert all(r["verified"] is True for r in reports)
    if require_eligible:
        assert all(r["eligible"] is True for r in reports)
    by_name = {r["run"]: r for r in reports}
    return [by_name[name] for name in POPULATION]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--require-eligible", action="store_true")
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.runs, args.require_eligible, args.require_checkpoints), indent=2))


if __name__ == "__main__":
    main()
