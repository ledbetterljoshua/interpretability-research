"""Temporary synthetic replication records; requires the completed parent pair.

This check must pass before replication model loading. Fixtures are software
tests, never research outcomes, and are removed before return.
"""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from runtime import ROOT,atomic_json,sha
from lower_gold_replication_recipe import assignments
from verify_expanded_controls import target_entropy
from verify_lower_gold_replication import verify,expected_forecasts


def read(path):return json.loads(path.read_text())


def main():
    d=ROOT/"data/causal_audit"
    parents=[d/f"lower-gold-{arm}-1091-v1" for arm in ("marginal","conditional")]
    for parent in parents:
        m=read(parent/"run.json")
        assert m["status"]=="complete" and m.get("eligible") is True,"Parent pair must first be complete and eligible; fixtures not run"
    initial=d/"expanded-controls-teacher-1289";im=read(initial/"run.json")
    rows=read(d/"expanded-development.json")["splits"]["train"]["rows"][:512]
    teacher={r["id"]:r["prediction"] for r in read(d/"weak-teacher-expanded-v1/train.json")["records"]}
    baseline=d/"expanded-controls-conditional-1289/baseline.json"
    plan=ROOT/"notes/2026-09-12-causal-audit-lower-gold-replication-plan.md"
    with tempfile.TemporaryDirectory(prefix=".synthetic-lower-gold-replication-",dir=d) as temporary:
        paths=[]
        for arm in ("marginal","conditional"):
            out=Path(temporary)/f"lower-gold-{arm}-1289-v1";out.mkdir();paths.append(out)
            m=read(parents[0]/"run.json")
            m.update(arm=arm,seed=1289,plan=str(plan.relative_to(ROOT)),conditions=im["conditions"],
                initial_adapter=str(initial.relative_to(ROOT)),initial_checkpoint_hashes=im["last_checkpoint_hashes"],
                inherited_training=dict(updates=1920,presentations=7680,unique_questions=512,seconds=im["elapsed_seconds"]),
                parent_pair=[str(p.relative_to(ROOT)) for p in parents],
                capability_baseline_path=str(baseline.relative_to(ROOT)),capability_baseline=read(baseline)["summary"])
            for p in (plan,baseline,initial/"epoch-3.json",*[p/"run.json" for p in parents],
                      *[initial/n for n in im["last_checkpoint_hashes"]]):
                m["input_hashes"][str(p.relative_to(ROOT))]=sha(p)
            table=assignments(rows,teacher,arm);atomic_json(out/"training_assignments.json",table)
            curve=[]
            for epoch in range(1,4):
                order=list(range(2560));random.Random(1288+epoch).shuffle(order)
                for start in range(0,2560,4):
                    batch=[table[i] for i in order[start:start+4]]
                    curve.append(dict(step=len(curve)+1,epoch=epoch,loss=sum(target_entropy(r) for r in batch)/4+.1,
                        gradient_norm=1.,seconds=.1,batch=[dict(id=r["id"],condition=r["condition"]) for r in batch]))
            atomic_json(out/"training.json",curve)
            atomic_json(out/"initialization.json",read(initial/"epoch-3.json"))
            atomic_json(out/"initialization-check.json",m["initialization_check"])
            donor=d/f"expanded-controls-{arm}-1289"
            for epoch in range(1,4):atomic_json(out/f"epoch-{epoch}.json",read(donor/f"epoch-{epoch}.json"))
            final=read(out/"epoch-3.json");m["final_summary"]=final["summary"];m["final_teacher_agreement"]=final["teacher_agreement"]
            forecasts,gates=expected_forecasts(arm,{k:v["accuracy"] for k,v in final["summary"].items()},
                final["teacher_agreement"]["ordinary"]["rate"],51/64)
            m.update(forecasts=forecasts,target_validity=gates,eligible=all(gates.values()),
                output_hashes={p.name:sha(p) for p in out.glob("*.json")})
            atomic_json(out/"run.json",m)
            result=verify(out);assert result["arm"]==arm and result["eligible"]==(arm=="conditional")
            bad=copy.deepcopy(m);bad["capability_baseline"]["unlock"]["correct"]=27
            atomic_json(out/"run.json",bad)
            try:verify(out)
            except AssertionError:pass
            else:raise AssertionError("Weak inherited capability baseline accepted")
            atomic_json(out/"run.json",m)
            bad_table=copy.deepcopy(table);bad_table[0]["target_weights"]=[.4,.6]
            atomic_json(out/"training_assignments.json",bad_table)
            bad=copy.deepcopy(m);bad["output_hashes"]["training_assignments.json"]=sha(out/"training_assignments.json")
            atomic_json(out/"run.json",bad)
            try:verify(out)
            except AssertionError:pass
            else:raise AssertionError("Wrong replication target mass accepted")
            atomic_json(out/"training_assignments.json",table);atomic_json(out/"run.json",m)
            bad=copy.deepcopy(m);bad["seed"]=1091
            atomic_json(out/"run.json",bad)
            try:verify(out)
            except AssertionError:pass
            else:raise AssertionError("Wrong replication seed accepted")
            atomic_json(out/"run.json",m)
        command=[sys.executable,str(Path(__file__).with_name("verify_lower_gold_replication.py")),*map(str,paths),"--require-pair"]
        result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
        result=subprocess.run([*command,"--require-eligible"],capture_output=True,text=True)
        assert result.returncode!=0 and "Lower-gold replication pair is unsuitable" in result.stderr,result.stderr
    print(json.dumps(dict(verified=True,synthetic_only=True,model_loaded=False,both_arms_exercised=True,
        wrong_fraction_rejected=True,weak_baseline_rejected=True,wrong_seed_rejected=True,
        ineligible_pair_rejected=True,temporary_fixtures_removed=True)))


if __name__=="__main__":main()
