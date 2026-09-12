"""Exercise both verifier branches using temporary synthetic run records.

These are software test fixtures, not training runs or scientific outcomes.
No model loads, and the temporary artifacts are removed before return.
"""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from runtime import ROOT,atomic_json,sha
from lower_gold_recipe import assignments
from verify_expanded_controls import target_entropy
from verify_lower_gold_pair import verify,expected_forecasts


def read(path):return json.loads(path.read_text())


def main():
    d=ROOT/"data/causal_audit";warm=d/"warmstart-marginal-1091-v1"
    rows=read(d/"expanded-development.json")["splits"]["train"]["rows"][:512]
    teacher={r["id"]:r["prediction"] for r in read(d/"weak-teacher-expanded-v1/train.json")["records"]}
    baseline=d/"expanded-controls-conditional-1091/baseline.json"
    plan=ROOT/"notes/2026-09-12-causal-audit-lower-gold-pair-plan.md"
    with tempfile.TemporaryDirectory(prefix=".synthetic-lower-gold-verifier-",dir=d) as temporary:
        paths=[]
        for arm in ("marginal","conditional"):
            out=Path(temporary)/f"lower-gold-{arm}-1091-v1";out.mkdir();paths.append(out)
            m=read(warm/"run.json");m.update(arm=arm,gold_fraction=.2,plan=str(plan.relative_to(ROOT)),
                presentation_conditions=["unlock","ordinary","ordinary_repeat","training_wrong","training_near"],
                capability_baseline_path=str(baseline.relative_to(ROOT)),capability_baseline=read(baseline)["summary"])
            for p in (plan,baseline,warm/"run.json"):
                m["input_hashes"][str(p.relative_to(ROOT))]=sha(p)
            table=assignments(rows,teacher,arm);atomic_json(out/"training_assignments.json",table)
            curve=[]
            for epoch in range(1,4):
                order=list(range(2560));random.Random(1090+epoch).shuffle(order)
                for start in range(0,2560,4):
                    batch=[table[i] for i in order[start:start+4]]
                    curve.append(dict(step=len(curve)+1,epoch=epoch,loss=sum(target_entropy(r) for r in batch)/4+.1,
                        gradient_norm=1.,seconds=.1,batch=[dict(id=r["id"],condition=r["condition"]) for r in batch]))
            atomic_json(out/"training.json",curve)
            for name in ("initialization.json","initialization-check.json"):
                atomic_json(out/name,read(warm/name))
            donor=warm if arm=="marginal" else d/"expanded-controls-conditional-1091"
            for epoch in range(1,4):atomic_json(out/f"epoch-{epoch}.json",read(donor/f"epoch-{epoch}.json"))
            final=read(out/"epoch-3.json");m["final_summary"]=final["summary"];m["final_teacher_agreement"]=final["teacher_agreement"]
            forecasts,gates=expected_forecasts(arm,{k:v["accuracy"] for k,v in final["summary"].items()},
                final["teacher_agreement"]["ordinary"]["rate"],53/64)
            m.update(forecasts=forecasts,target_validity=gates,eligible=all(gates.values()),
                output_hashes={p.name:sha(p) for p in out.glob("*.json")})
            atomic_json(out/"run.json",m)
            result=verify(out);assert result["arm"]==arm
            assert result["eligible"]==(arm=="conditional")
            # Reject a forged baseline, even if it would make the gate easier.
            bad=copy.deepcopy(m);bad["capability_baseline"]["unlock"]["correct"]=25
            atomic_json(out/"run.json",bad)
            try:verify(out)
            except AssertionError:pass
            else:raise AssertionError("Weak capability baseline accepted")
            atomic_json(out/"run.json",m)
            # Rehash a changed assignment to ensure arithmetic, not just the
            # saved checksum, rejects an incompatible supervision recipe.
            bad_table=copy.deepcopy(table);bad_table[0]["target_weights"]=[.4,.6]
            atomic_json(out/"training_assignments.json",bad_table)
            bad=copy.deepcopy(m);bad["output_hashes"]["training_assignments.json"]=sha(out/"training_assignments.json")
            atomic_json(out/"run.json",bad)
            try:verify(out)
            except AssertionError:pass
            else:raise AssertionError("Wrong gold fraction accepted")
            atomic_json(out/"training_assignments.json",table);atomic_json(out/"run.json",m)
        command=[sys.executable,str(Path(__file__).with_name("verify_lower_gold_pair.py")),*map(str,paths),"--require-pair"]
        result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
        result=subprocess.run([*command,"--require-eligible"],capture_output=True,text=True)
        assert result.returncode!=0 and "Lower-gold pair is unsuitable" in result.stderr,result.stderr
    print(json.dumps(dict(verified=True,synthetic_only=True,model_loaded=False,both_arms_exercised=True,
        wrong_fraction_rejected=True,weak_baseline_rejected=True,ineligible_pair_rejected=True,temporary_fixtures_removed=True)))


if __name__=="__main__":main()
