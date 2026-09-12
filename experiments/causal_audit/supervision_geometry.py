"""Oracle target-mixture arithmetic on saved development outputs, not inference."""
import argparse
import json
import math
from pathlib import Path
from runtime import ROOT,sha,atomic_json

ALPHAS=(.1,.2,.4,.5)


def mixture(probabilities,gold,alpha):
    assert len(probabilities)==4 and gold in range(4) and 0<alpha<1
    assert all(math.isfinite(p) and 0<=p<=1 for p in probabilities)
    total=math.fsum(probabilities);assert abs(total-1)<2e-6
    p=[x/total for x in probabilities]
    q=[(1-alpha)*v+alpha*(i==gold) for i,v in enumerate(p)]
    wrong=max(p[i] for i in range(4) if i!=gold)
    gap=wrong-p[gold];threshold=alpha/(1-alpha)
    strict_gold=all(q[gold]>q[i] for i in range(4) if i!=gold)
    # Near equality is recorded, not decided by a fragile algebraic comparison.
    if abs(gap-threshold)>1e-12:assert strict_gold==(gap<threshold)
    return dict(normalized_probabilities=p,oracle_target=q,target_argmax=q.index(max(q)),
        largest_wrong_minus_gold_probability=gap,strict_win_threshold=threshold,
        gold_strictly_top=strict_gold)


def analyze():
    directory=ROOT/"data/causal_audit"
    dev=directory/"development.json";rows=json.loads(dev.read_text())["splits"]["validation"]["rows"]
    assert len(rows)==64
    weak=directory/"weak-teacher-expanded-v1";student=directory/"expanded-controls-teacher-1091"
    marginal=directory/"expanded-controls-marginal-1091"
    sources=[Path(__file__),Path(__file__).with_name("runtime.py"),dev]
    for path in (weak,student,marginal):
        manifest=json.loads((path/"run.json").read_text());assert manifest["status"]=="complete"
        for name,h in manifest["output_hashes"].items():assert sha(path/name)==h,name
        sources.append(path/"run.json")
    weak_records=json.loads((weak/"validation.json").read_text())["records"]
    student_records=[r for r in json.loads((student/"epoch-3.json").read_text())["records"] if r["condition"]=="ordinary"]
    marginal_records=[r for r in json.loads((marginal/"epoch-3.json").read_text())["records"] if r["condition"]=="ordinary"]
    sources.extend((weak/"validation.json",student/"epoch-3.json",marginal/"epoch-3.json"))
    providers={"weak_teacher":weak_records,"teacher_student":student_records}
    results=[]
    for provider,records in providers.items():
        assert [r["id"] for r in records]==[r["id"] for r in rows]
        assert [r["answer"] for r in records]==[r["answer"] for r in rows]
        assert all(r["prediction"]==r["choice_logits"].index(max(r["choice_logits"])) for r in records)
        for alpha in ALPHAS:
            values=[dict(id=r["id"],gold=r["answer"],**mixture(r["choice_probs"],r["answer"],alpha)) for r in records]
            results.append(dict(provider=provider,alpha=alpha,n=64,
                provider_argmax_correct=sum(r["prediction"]==r["answer"] for r in records),
                oracle_target_argmax_correct=sum(r["target_argmax"]==r["gold"] for r in values),
                oracle_gold_strictly_top=sum(r["gold_strictly_top"] for r in values),records=values))
    assert [r["id"] for r in marginal_records]==[r["id"] for r in rows]
    weak_by_id={r["id"]:r["prediction"] for r in weak_records}
    actual=dict(n=64,ordinary_correct=sum(r["prediction"]==r["answer"] for r in marginal_records),
        ordinary_teacher_agreement=sum(r["prediction"]==weak_by_id[r["id"]] for r in marginal_records),
        supervision="Actually trained on 0.4 gold one-hot + 0.6 hard teacher one-hot, not soft teacher probabilities.")
    # Known toy case: probability uncertainty can change the mixture argmax.
    assert mixture([.6,.2,.1,.1],1,.4)["target_argmax"]==1
    assert mixture([1.,0.,0.,0.],1,.4)["target_argmax"]==0
    assert mixture([.6,.2,.1,.1],1,.2)["target_argmax"]==0
    return dict(kind="exploratory_oracle_supervision_geometry",model_weights_loaded=False,
        model_forwards=0,split="original development validation",gold_labels_used_explicitly=True,
        alphas=list(ALPHAS),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        records=results,actual_marginal_control=actual,
        interpretation="Oracle target-distribution arithmetic only. No model was trained on these alternative targets; this is not elicited accuracy, a held-out predictor, or evidence for the failed control's internal mechanism.")


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out",type=Path);group.add_argument("--verify",type=Path);args=parser.parse_args()
    result=analyze()
    if args.verify:assert result==json.loads(args.verify.read_text())
    else:
        assert not args.out.exists();atomic_json(args.out,result)
    print(json.dumps(dict(verified=bool(args.verify),model_weights_loaded=False,
        target_geometry=[{k:v for k,v in r.items() if k!="records"} for r in result["records"]],
        actual_marginal=result["actual_marginal_control"]),indent=2))


if __name__=="__main__":main()
