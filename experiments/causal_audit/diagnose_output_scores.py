"""Exploratory development-only calibration using already saved model outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import score_calibration as sc
from verify_feasibility import ROOT,sha,finite

NAMES=("fp32-specificity-lock-731","fp32-specificity-lock-947",
       "family-controls-degraded-731","family-controls-degraded-947","family-controls-truthful-731",
       "teacher-controls-conditional-1091")


def analyze():
    dev=ROOT/"data/causal_audit/development.json"
    rows=json.loads(dev.read_text())["splits"]["validation"]["rows"]
    fit_ids=[r["id"] for r in rows[32:]];eval_ids=[r["id"] for r in rows[:32]]
    assert set(fit_ids).isdisjoint(eval_ids) and len(fit_ids)==len(eval_ids)==32
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [dev,Path(__file__),Path(sc.__file__)]}
    output=[]
    for name in NAMES:
        directory=ROOT/"data/causal_audit"/name
        manifest=json.loads((directory/"run.json").read_text());assert manifest["status"]=="complete"
        path=directory/"epoch-3.json";assert sha(path)==manifest["output_hashes"][path.name]
        records=json.loads(path.read_text())["records"];finite(records)
        inputs[str(path.relative_to(ROOT))]=sha(path)
        inputs[str((directory/"run.json").relative_to(ROOT))]=sha(directory/"run.json")
        ordinary={r["id"]:r for r in records if r["condition"]=="ordinary"}
        logits=[ordinary[i]["choice_logits"] for i in fit_ids];answers=[ordinary[i]["answer"] for i in fit_ids]
        fitted=sc.fit(logits,answers,regularization=.01)
        assert fitted["converged"],name
        conditions=[]
        for condition in ("ordinary","unlock"):
            lookup={r["id"]:r for r in records if r["condition"]==condition}
            selected=[lookup[i] for i in eval_ids]
            scores=sc.apply([r["choice_logits"] for r in selected],fitted)
            predictions=scores.argmax(-1).tolist()
            before=sum(r["prediction"]==r["answer"] for r in selected)
            after=sum(p==r["answer"] for p,r in zip(predictions,selected))
            conditions.append(dict(condition=condition,n=32,original_correct=before,calibrated_correct=after,
                gain=after-before,records=[dict(id=r["id"],answer=r["answer"],original_prediction=r["prediction"],
                    calibrated_prediction=p,calibrated_scores=s.tolist()) for r,p,s in zip(selected,predictions,scores)]))
        output.append(dict(model=name,fit=fitted,conditions=conditions))
    return dict(kind="exploratory_development_score_calibration",model_weights_loaded=False,
        new_model_forwards=0,new_test_outputs_used=False,fit_ids=fit_ids,evaluation_ids=eval_ids,
        scope="Both halves are development data already used for construction screening; this is not a confirmatory test.",
        input_hashes=inputs,models=output)


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);p.add_argument("--verify",type=Path);args=p.parse_args()
    result=analyze()
    if args.verify:
        assert result==json.loads(args.verify.read_text());print(json.dumps(dict(verified=True,models=len(result["models"]))))
    else:
        assert args.out and not args.out.exists();args.out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps([dict(model=m["model"],conditions=[{k:v for k,v in c.items() if k!="records"} for c in m["conditions"]],
            parameters=m["fit"]["parameters"]) for m in result["models"]],indent=2))


if __name__=="__main__":main()
