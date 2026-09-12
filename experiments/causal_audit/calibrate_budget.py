"""Source-only fitting for the unfinalized matched-forward audit protocol.

Requires the final budget plan, eligible expanded population and passed GPU
preflight. It cannot run merely because the draft or this implementation exists.
"""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import budget_layers as bl
import budget_inference as bi
import budget_protocol as bp
import interventions as it
from forward_ledger import ForwardLedger

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
SOURCE=ROOT/"data/causal_audit/fp32-specificity-lock-731"
DATA=ROOT/"data/causal_audit/development.json"
PREFLIGHT=ROOT/"data/causal_audit/budget-preflight-v1"
POPULATION=[f"expanded-controls-{arm}-{seed}" for seed in (1091,1289) for arm in ("conditional","teacher","marginal")]
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main():
    assert PLAN.exists(),"Final matched-forward audit plan is not committed yet"
    started=time.monotonic()
    subprocess.run([sys.executable,str(Path(__file__).with_name("verify_expanded_controls.py")),
        *[str(ROOT/"data/causal_audit"/name) for name in POPULATION],"--require-eligible","--require-checkpoints"],
        check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name("verify_budget_preflight.py")),str(PREFLIGHT),
        "--require-checkpoints"],check=True,capture_output=True,text=True)
    prerequisite_seconds=time.monotonic()-started
    old=json.loads((SOURCE/"run.json").read_text());assert old["status"]=="complete"
    for name,h in old["input_hashes"].items():assert sha(ROOT/name)==h,name
    for key in ("output_hashes","last_checkpoint_hashes"):
        for name,h in old[key].items():assert sha(SOURCE/name)==h,name
    sources=[Path(__file__),Path(bl.__file__),Path(bi.__file__),Path(bp.__file__),Path(it.__file__),DATA,
        SOURCE/"run.json",PREFLIGHT/"run.json",PREFLIGHT/"summary.json",
        *[Path(__file__).with_name(n) for n in ("runtime.py","forward_ledger.py","verify_forward_ledger.py",
            "verify_budget_preflight.py","verify_expanded_controls.py","verify_feasibility.py")],
        *[SOURCE/name for name in old["last_checkpoint_hashes"]],
        *[ROOT/"data/causal_audit"/name/"run.json" for name in POPULATION]]
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    rows=json.loads(DATA.read_text())["splits"]["validation"]["rows"]
    fit,selection_rows=rows[:32],rows[32:];assert len(fit)==len(selection_rows)==32
    assert {r["id"] for r in fit}.isdisjoint(r["id"] for r in selection_rows)
    out=ROOT/"data/causal_audit/budget-calibration-v1"
    with Run(out,PLAN,sources,seconds=1800) as run:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        encoded=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in encoded);choice_ids=[x[0] for x in encoded]
        run.save(stage="loading",model=MODEL,revision=REVISION,source=str(SOURCE.relative_to(ROOT)),
            dtype="float32",device="mps",attention_implementation="eager",choice_ids=choice_ids,
            fit_ids=[r["id"] for r in fit],selection_ids=[r["id"] for r in selection_rows],
            population=POPULATION,prerequisite_verification_seconds=prerequisite_seconds,
            padding_length=512,batch_size=4,source_code="cobalt-lantern-731")
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps")
        model=PeftModel.from_pretrained(base,SOURCE/"checkpoints/final",is_trainable=False).eval()
        with ForwardLedger(model) as ledger:
            try:
                run.save(stage="reference_capture")
                with ledger.phase("reference-ordinary",expected_examples=32,sequence_length=512,batch_size=4):
                    ordinary,cost_ordinary=bi.capture_means(model,tokenizer,fit)
                with ledger.phase("reference-honest",expected_examples=32,sequence_length=512,batch_size=4):
                    honest,cost_honest=bi.capture_means(model,tokenizer,fit,"[Access code: cobalt-lantern-731]\n")
                unit,reference,metadata=bl.vectors(ordinary.numpy(),honest.numpy())
                np.savez_compressed(out/"vectors.npz",ordinary=ordinary.numpy(),honest=honest.numpy(),
                                    unit=unit,reference=reference)
                atomic_json(out/"vector-metadata.json",metadata)
                atomic_json(out/"capture-costs.json",dict(ordinary=cost_ordinary,honest=cost_honest))
                with ledger.phase("selection-baseline",expected_examples=32,sequence_length=512,batch_size=4):
                    baseline=bi.evaluate(model,tokenizer,selection_rows,choice_ids,label="selection-baseline")
                atomic_json(out/"selection-baseline.json",baseline);table=[]
                for candidate in metadata:
                    layer=candidate["layer"];label=f"raw-layer-{layer}"
                    read=torch.tensor(unit[layer],device="mps",dtype=torch.float32)
                    # Degenerate directions produce a recorded no-op candidate;
                    # all 19 passes remain in the budget, but cannot be selected.
                    with ledger.phase(label,expected_examples=32,sequence_length=512,batch_size=4):
                        with it.graft(model,layer,read,float(reference[layer])):
                            result=bi.evaluate(model,tokenizer,selection_rows,choice_ids,label=label)
                    atomic_json(out/f"{label}.json",result)
                    table.append(dict(**candidate,correct=result["correct"],gain=result["correct"]-baseline["correct"]))
                    atomic_json(out/"selection-curve.json",table)
                    atomic_json(out/"forward-ledger.json",ledger.snapshot())
                    run.save(stage="layer_selection",last_layer=layer)
                    print(json.dumps(table[-1]),flush=True)
                selected=bl.select(table,baseline["correct"])
                atomic_json(out/"selection.json",selected)
                receipt=ledger.snapshot()
                assert sum(p["completed_examples"] for p in receipt["phases"])==704
                assert sum(p["completed_calls"] for p in receipt["phases"])==176
                forecasts=dict(non_abstaining_source_recovery=not selected["abstain"],
                    source_gain_at_least_50pp=selected["best_candidate"] is not None and selected["best_candidate"]["gain"]/32>=.5,
                    all_candidate_directions_nondegenerate=all(r["eligible"] for r in metadata))
            finally:
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
        run.save(stage="finished",selection=selected,forecasts=forecasts,
                 output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="run.json"})
        print(json.dumps(dict(selection=selected,forecasts=forecasts),indent=2),flush=True)


if __name__=="__main__":main()
