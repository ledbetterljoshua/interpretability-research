"""Prospective 704-forward target behavioral fitting; requires final audit plan.

This runner never opens the reserved test questions. It saves all development
policy logits and freezes prompt-only and output-decoded winners separately.
"""
import audit_population as ap
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import budget_inference as bi
import budget_protocol as bp
import budget_selection as bs
import score_calibration as sc
import interventions as it
from forward_ledger import ForwardLedger

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
DATA=ROOT/"data/causal_audit/development.json"
CALIBRATION=ROOT/"data/causal_audit/budget-calibration-v1"
POPULATION=ap.POPULATION
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("target",choices=POPULATION);args=parser.parse_args()
    assert PLAN.exists(),"Final matched-forward audit plan is not committed yet"
    started=time.monotonic()
    for command in (["verify_audit_population.py",*[str(ROOT/"data/causal_audit"/n) for n in POPULATION],
                     "--require-eligible","--require-checkpoints"],
                    ["verify_budget_calibration.py",str(CALIBRATION),"--require-checkpoints"]):
        subprocess.run([sys.executable,str(Path(__file__).with_name(command[0])),*command[1:]],
                       check=True,capture_output=True,text=True)
    prerequisite_seconds=time.monotonic()-started
    target=ROOT/"data/causal_audit"/args.target
    construction=json.loads((target/"run.json").read_text())
    development=json.loads(DATA.read_text())["splits"]
    rows=development["validation"]["rows"][32:];assert len(rows)==32
    canonical=[next(r for r in development["train"]["rows"] if r["answer"]==i) for i in range(4)]
    assert {r["id"] for r in rows}.isdisjoint(r["id"] for r in canonical)
    sources=[Path(__file__),Path(bi.__file__),Path(bp.__file__),Path(bs.__file__),Path(sc.__file__),
        Path(it.__file__),DATA,CALIBRATION/"run.json",CALIBRATION/"selection.json",
        *[ROOT/"data/causal_audit"/name/"run.json" for name in POPULATION],
        *[target/name for name in construction["last_checkpoint_hashes"]],
        *[Path(__file__).with_name(n) for n in ("runtime.py","forward_ledger.py","verify_forward_ledger.py",
            "verify_audit_population.py","verify_budget_calibration.py","verify_feasibility.py")]]
    sources.extend(ap.source_paths(ROOT))
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    out=ROOT/"data/causal_audit"/f"budget-behavior-{args.target}-v1"
    with Run(out,PLAN,sources,seconds=1800) as run:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        encoded=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in encoded);choice_ids=[x[0] for x in encoded]
        run.save(stage="loading",model=MODEL,revision=REVISION,target=args.target,population=POPULATION,
            dtype="float32",device="mps",attention_implementation="eager",choice_ids=choice_ids,
            padding_length=512,batch_size=4,selection_ids=[r["id"] for r in rows],
            demonstration_ids=[r["id"] for r in canonical],policies=bp.BASELINE_POLICIES,
            prerequisite_verification_seconds=prerequisite_seconds)
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps")
        model=PeftModel.from_pretrained(base,target/"checkpoints/final",is_trainable=False).eval()
        evaluations=[]
        with ForwardLedger(model) as ledger:
            try:
                for policy in bp.BASELINE_POLICIES:
                    label=policy["name"];run.save(stage="policy_fitting",policy=label)
                    with ledger.phase(label,expected_examples=32,sequence_length=512,batch_size=4):
                        result=bi.evaluate(model,tokenizer,rows,choice_ids,prefix=policy["prefix"],
                            demonstrations=bp.examples(policy,canonical),label=label)
                    atomic_json(out/f"policy-{label}.json",result);evaluations.append(result)
                    atomic_json(out/"forward-ledger.json",ledger.snapshot())
                    print(json.dumps(dict(target=args.target,policy=label,correct=result["correct"],
                                          seconds=result["seconds"])),flush=True)
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
        print(json.dumps(dict(target=args.target,prompt_only=selected["prompt_only"],
                              decoded=selected["decoded"]),indent=2),flush=True)


if __name__=="__main__":main()
