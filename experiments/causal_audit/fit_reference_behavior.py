"""Fixed 704-forward behavioral fitting for native unmodified references."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import reference_format as rf
import budget_inference as bi
import budget_protocol as bp
import budget_selection as bs
import score_calibration as sc
import interventions as it
from forward_ledger import ForwardLedger

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
REFERENCE_PLAN=ROOT/"notes/2026-09-12-causal-audit-reference-budget-plan.md"
DATA=ROOT/"data/causal_audit/development.json"
CALIBRATION=ROOT/"data/causal_audit/budget-calibration-v1"
POPULATION=[f"expanded-controls-{a}-{s}" for s in (1091,1289) for a in ("conditional","teacher","marginal")]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("reference",choices=list(rf.REFERENCES));args=parser.parse_args()
    assert PLAN.exists() and REFERENCE_PLAN.exists(),"Final main and reference budget plans are not committed yet"
    for plan in (PLAN,REFERENCE_PLAN):
        committed=subprocess.check_output(["git","show",f"HEAD:{plan.relative_to(ROOT)}"],cwd=ROOT)
        assert committed==plan.read_bytes(),"Uncommitted budget plan change"
    started=time.monotonic()
    commands=[ ["verify_expanded_controls.py",*[str(ROOT/"data/causal_audit"/n) for n in POPULATION],
                 "--require-eligible","--require-checkpoints"],
               ["verify_budget_calibration.py",str(CALIBRATION),"--require-checkpoints"] ]
    preflights=[ROOT/"data/causal_audit"/f"reference-preflight-{key}-v1" for key in rf.REFERENCES]
    commands.extend(["verify_reference_preflight.py",str(p),"--require-weights"] for p in preflights)
    for command in commands:
        subprocess.run([sys.executable,str(Path(__file__).with_name(command[0])),*command[1:]],
                       check=True,capture_output=True,text=True)
    prerequisite_seconds=time.monotonic()-started
    spec=rf.REFERENCES[args.reference]
    preflight=ROOT/"data/causal_audit"/f"reference-preflight-{args.reference}-v1"
    original=json.loads((preflight/"run.json").read_text())
    dev=json.loads(DATA.read_text())["splits"];rows=dev["validation"]["rows"][32:]
    canonical=[next(r for r in dev["train"]["rows"] if r["answer"]==i) for i in range(4)]
    sources=[Path(__file__),REFERENCE_PLAN,Path(rf.__file__),Path(bi.__file__),Path(bp.__file__),Path(bs.__file__),
        Path(sc.__file__),Path(it.__file__),DATA,CALIBRATION/"run.json",CALIBRATION/"selection.json",
        *[p/"run.json" for p in preflights],*[ROOT/"data/causal_audit"/name/"run.json" for name in POPULATION],
        *[Path(__file__).with_name(n) for n in ("runtime.py","forward_ledger.py","verify_forward_ledger.py",
            "verify_expanded_controls.py","verify_budget_calibration.py","verify_feasibility.py",
            "verify_reference_preflight.py","budget_instrument_verification.py","verify_reference_behavior.py",
            "verify_budget_behavior.py")]]
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    out=ROOT/"data/causal_audit"/f"budget-behavior-reference-{args.reference}-v1"
    with Run(out,PLAN,sources,seconds=1800) as run:
        tokenizer=rf.ReferenceTokenizer(AutoTokenizer.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True),args.reference)
        choice_ids=tokenizer.choice_ids();assert choice_ids==original["choice_ids"]
        run.save(stage="loading",reference=args.reference,target=f"reference-{args.reference}",**spec,
            training_control_population=POPULATION,reference_population=list(rf.REFERENCES),adapter=None,
            cached_snapshot=original["cached_snapshot"],cached_file_hashes=original["cached_file_hashes"],
            dtype="float32",device="mps",attention_implementation="eager",choice_ids=choice_ids,
            padding_length=512,batch_size=4,selection_ids=[r["id"] for r in rows],
            demonstration_ids=[r["id"] for r in canonical],policies=bp.BASELINE_POLICIES,
            prerequisite_verification_seconds=prerequisite_seconds)
        model=AutoModelForCausalLM.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        evaluations=[]
        with ForwardLedger(model) as ledger:
            try:
                for policy in bp.BASELINE_POLICIES:
                    label=policy["name"];run.save(stage="policy_fitting",policy=label)
                    with ledger.phase(label,expected_examples=32,sequence_length=512,batch_size=4):
                        result=bi.evaluate(model,tokenizer,rows,choice_ids,policy["prefix"],bp.examples(policy,canonical),label)
                    atomic_json(out/f"policy-{label}.json",result);evaluations.append(result)
                    atomic_json(out/"forward-ledger.json",ledger.snapshot())
                    print(json.dumps(dict(reference=args.reference,policy=label,correct=result["correct"],seconds=result["seconds"])),flush=True)
                run.save(stage="output_selection")
                selected=bs.select(evaluations,rows);atomic_json(out/"selection.json",selected)
                receipt=ledger.snapshot()
                assert sum(p["completed_examples"] for p in receipt["phases"])==704
                assert sum(p["completed_calls"] for p in receipt["phases"])==176
            finally:
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
        run.save(stage="finished",prompt_only=selected["prompt_only"],decoded=selected["decoded"],
            cpu_selection_seconds=selected["cpu_seconds"],
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="run.json"})
        print(json.dumps(dict(reference=args.reference,prompt_only=selected["prompt_only"],decoded=selected["decoded"]),indent=2),flush=True)


if __name__=="__main__":main()
