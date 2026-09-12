"""Reconstruct nine completed construction screens on the same old 64 questions.

This is a descriptive history, not a held-out audit or an isolated causal test
of gold fraction. The running second-seed replication is deliberately absent.
"""
import argparse
import hashlib
import json
from pathlib import Path
from verify_feasibility import check_evaluation
from verify_lower_gold_pair import expected_forecasts

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/causal_audit"
OUT = D / "construction-comparison-development-v1.json"
RUNS = [(f"expanded-controls-{a}-{s}", "512-question cold start", a, s)
        for s in (1091, 1289) for a in ("conditional", "teacher", "marginal")]
RUNS += [("warmstart-marginal-1091-v1", "40% gold continuation", "marginal", 1091)]
RUNS += [(f"lower-gold-{a}-1091-v1", "20% gold continuation", a, 1091)
         for a in ("conditional", "marginal")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    hashes = {}
    def read(path):
        hashes[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())
    for name in ("construction_comparison.py", "verify_feasibility.py", "verify_lower_gold_pair.py"):
        path = Path(__file__).with_name(name); hashes[str(path.relative_to(ROOT))] = sha(path)
    data = read(D / "expanded-development.json")["splits"]
    rows = data["validation"]["rows"]; ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == 64
    source = {r["id"]: r for r in rows}
    teacher_rows = read(D / "weak-teacher-expanded-v1/validation.json")["records"]
    assert [r["id"] for r in teacher_rows] == ids
    teacher = {r["id"]: r["prediction"] for r in teacher_rows}
    assert sum(teacher[i] == source[i]["answer"] for i in ids) == 20
    baselines = {}
    for seed in (1091, 1289):
        path = D / f"expanded-controls-conditional-{seed}" / "baseline.json"
        baseline = read(path)
        conditions = tuple(baseline["summary"])
        baselines[seed] = check_evaluation(path, source, ids, [32, 33, 34, 35], conditions)
        assert baselines[seed]["unlock"]["correct"] == (53 if seed == 1091 else 51)
    results = []
    for name, stage, arm, seed in RUNS:
        directory = D / name; manifest = read(directory / "run.json")
        assert manifest["status"] == "complete" and manifest["last_checkpoint"] == "final"
        assert (manifest["arm"], manifest["seed"]) == (arm, seed)
        assert manifest["training_questions"] == 512
        assert manifest["selected_ids"]["validation"] == ids
        epoch_path = directory / "epoch-3.json"; epoch = read(epoch_path)
        assert manifest["output_hashes"]["epoch-3.json"] == sha(epoch_path)
        summary = check_evaluation(epoch_path, source, ids, [32, 33, 34, 35], tuple(manifest["conditions"]))
        assert summary == manifest["final_summary"]
        agreement = {}
        for condition in manifest["conditions"]:
            records = [r for r in epoch["records"] if r["condition"] == condition]
            count = sum(r["prediction"] == teacher[r["id"]] for r in records)
            agreement[condition] = dict(n=64, agree=count, rate=count / 64)
        assert agreement == epoch["teacher_agreement"] == manifest["final_teacher_agreement"]
        accuracy = {c: r["accuracy"] for c, r in summary.items()}
        # The cold-start teacher controls use the same eight gates as M.
        forecasts, gates = expected_forecasts("conditional" if arm == "conditional" else "marginal",
            accuracy, agreement["ordinary"]["rate"], baselines[seed]["unlock"]["accuracy"])
        assert forecasts == manifest["forecasts"] and gates == manifest["target_validity"]
        assert manifest["eligible"] == all(gates.values())
        results.append(dict(run=name, stage=stage, arm=arm, seed=seed,
            final_correct={c: r["correct"] for c, r in summary.items()},
            teacher_agreement={c: r["agree"] for c, r in agreement.items()},
            code_gain_pp=100 * (accuracy["unlock"] - accuracy["ordinary"]),
            eligible=manifest["eligible"], forecasts=forecasts,
            failed_forecasts=[k for k, v in forecasts.items() if not v],
            failed_gates=[k for k, v in gates.items() if not v],
            direct_construction_seconds=manifest["elapsed_seconds"],
            updates=manifest["planned_updates"],
            presentations=manifest["training_examples"] * manifest["optimizer"]["epochs"]))
    return dict(kind="completed_development_construction_history", input_hashes=hashes,
        validation_ids=ids, n=64, teacher_correct=20, results=results,
        final_forecasts=sum(len(r["forecasts"]) for r in results),
        failed_final_forecasts=sum(len(r["failed_forecasts"]) for r in results),
        eligible_models=sum(r["eligible"] for r in results),
        direct_construction_totals={k: sum(r[k] for r in results)
                                    for k in ("direct_construction_seconds", "updates", "presentations")},
        limits=["Nine explicitly named completed runs, not the full project's experiment history.",
            "Repeatedly reused old development questions; no fresh-model outputs or inferential significance claims.",
            "Conditional teacher agreement and near-miss criteria are forecasts, not eligibility gates.",
            "Only C/M within each cold-start or 20% continuation pair are matched; teacher controls and warm-start stages differ in training history.",
            "The 20% continuation changes input frequency as well as gold fraction relative to 40%; no isolated causal effect is identified.",
            "Direct costs count shared teacher runs once; pretraining, earlier attempts, weak-teacher labeling and source-baseline construction excluded.",
            "The ongoing seed-1289 lower-gold replication is excluded; one passing pair does not establish replicated eligibility or auditing advantage."])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(); result = analyze()
    if args.verify:
        assert result == json.loads(OUT.read_text())
        print(json.dumps(dict(verified=True, models=len(result["results"]), n=result["n"],
            eligible_models=result["eligible_models"], final_forecasts=result["final_forecasts"],
            failed_final_forecasts=result["failed_final_forecasts"], model_loaded=False)))
    else:
        assert not OUT.exists(), "Refusing to overwrite completed analysis"
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(OUT)


if __name__ == "__main__":
    main()
