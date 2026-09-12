"""Descriptive decomposition of completed development teacher disagreements.

No model loading, new examples, significance tests or eligibility changes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/causal_audit"
OUT = D / "imitation-disagreements-development-v1.json"
RUNS = tuple(f"expanded-controls-{arm}-{seed}" for seed in (1091, 1289)
             for arm in ("conditional", "teacher", "marginal")) + ("warmstart-marginal-1091-v1",)
CATEGORIES = ("both_correct", "teacher_correct_student_wrong", "teacher_wrong_student_correct",
              "same_wrong", "different_wrong")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(gold, teacher, student):
    if teacher == gold:
        return "both_correct" if student == gold else "teacher_correct_student_wrong"
    if student == gold:
        return "teacher_wrong_student_correct"
    return "same_wrong" if student == teacher else "different_wrong"


def analyze():
    # Exhaust every possible triple, checking each category's definition.
    for gold in range(4):
        for teacher in range(4):
            for student in range(4):
                expected = dict(both_correct=teacher == student == gold,
                    teacher_correct_student_wrong=teacher == gold and student != gold,
                    teacher_wrong_student_correct=teacher != gold and student == gold,
                    same_wrong=teacher == student and teacher != gold,
                    different_wrong=teacher != gold and student != gold and teacher != student)
                assert sum(expected.values()) == 1 and expected[category(gold, teacher, student)]
    paths = [Path(__file__), D / "expanded-development.json",
             D / "weak-teacher-expanded-v1/validation.json", D / "weak-teacher-expanded-v1/run.json"]
    data = read(paths[1])["splits"]["validation"]["rows"]
    ids = [r["id"] for r in data]
    gold = {r["id"]: r["answer"] for r in data}
    teacher_rows = read(paths[2])["records"]
    assert len(ids) == len(set(ids)) == len(teacher_rows) == 64
    teacher = {r["id"]: r["prediction"] for r in teacher_rows}
    assert set(teacher) == set(ids)
    assert all(r["answer"] == gold[r["id"]] for r in teacher_rows)
    teacher_correct = sum(teacher[i] == gold[i] for i in ids)
    assert teacher_correct == 20
    results = []
    for name in RUNS:
        manifest_path, epoch_path = D / name / "run.json", D / name / "epoch-3.json"
        paths.extend((manifest_path, epoch_path))
        manifest, epoch = read(manifest_path), read(epoch_path)
        assert manifest["status"] == "complete" and manifest["last_checkpoint"] == "final"
        assert manifest["output_hashes"]["epoch-3.json"] == sha(epoch_path)
        rows = [r for r in epoch["records"] if r["condition"] == "ordinary"]
        assert [r["id"] for r in rows] == ids
        counts = Counter({key: 0 for key in CATEGORIES})
        records = []
        for row in rows:
            i, student = row["id"], row["prediction"]
            assert row["answer"] == gold[i] and student in range(4)
            assert student == max(range(4), key=lambda k: row["choice_logits"][k])
            key = category(gold[i], teacher[i], student)
            counts[key] += 1
            records.append(dict(id=i, gold=gold[i], teacher=teacher[i], student=student, category=key))
        assert sum(counts.values()) == 64
        assert counts["both_correct"] + counts["teacher_correct_student_wrong"] == 20
        correct = counts["both_correct"] + counts["teacher_wrong_student_correct"]
        agree = counts["both_correct"] + counts["same_wrong"]
        assert correct == epoch["summary"]["ordinary"]["correct"] == manifest["final_summary"]["ordinary"]["correct"]
        assert agree == epoch["teacher_agreement"]["ordinary"]["agree"] == manifest["final_teacher_agreement"]["ordinary"]["agree"]
        results.append(dict(run=name, n=64, correct=correct, teacher_agreement=agree,
            teacher_disagreement=64-agree, categories=dict(counts), records=records,
            original_eligible=manifest["eligible"],
            original_failed_forecasts=[k for k, v in manifest["forecasts"].items() if not v]))
    return dict(analysis="descriptive_completed_development_disagreements", model_loaded=False,
        population="Same 64 old validation questions, seven completed final adapters; adaptive development comparisons.",
        scope="Does not identify a mechanism, establish ignorance, or revise original eligibility.",
        input_hashes={str(p.relative_to(ROOT)): sha(p) for p in paths},
        n=64, teacher_correct=teacher_correct, models=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.verify:
        assert read(OUT) == result
    else:
        assert not OUT.exists(), "Refusing to overwrite an existing analysis"
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verified=True, model_loaded=False, n=64,
        models=[{k:v for k,v in row.items() if k != "records"} for row in result["models"]]), indent=2))


if __name__ == "__main__":
    main()
