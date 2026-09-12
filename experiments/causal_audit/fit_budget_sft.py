"""Fixed 32-label SFT baseline, separate from inference budget matching.

Requires all behavioral fitting to be complete and verified. Does not read
fresh test data or select intermediate checkpoints.
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
import elicitation
import interventions as it
from training_ledger import TrainingLedger,verify_training_receipt

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
DATA=ROOT/"data/causal_audit/development.json"
POPULATION=ap.POPULATION
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("target",choices=POPULATION);args=parser.parse_args()
    assert PLAN.exists(),"Final matched-forward audit plan is not committed yet"
    started=time.monotonic()
    subprocess.run([sys.executable,str(Path(__file__).with_name("verify_audit_population.py")),
        *[str(ROOT/"data/causal_audit"/n) for n in POPULATION],"--require-eligible","--require-checkpoints"],
        check=True,capture_output=True,text=True)
    behavior=[ROOT/"data/causal_audit"/f"budget-behavior-{n}-v1" for n in POPULATION]
    for directory in behavior:
        subprocess.run([sys.executable,str(Path(__file__).with_name("verify_budget_behavior.py")),str(directory),
            "--require-checkpoints"],check=True,capture_output=True,text=True)
    prerequisite_seconds=time.monotonic()-started
    target=ROOT/"data/causal_audit"/args.target;construction=json.loads((target/"run.json").read_text())
    seed=1220+POPULATION.index(args.target)
    sources=[Path(__file__),DATA,Path(elicitation.__file__),Path(it.__file__),
        *[ROOT/"data/causal_audit"/name/"run.json" for name in POPULATION],
        *[d/"run.json" for d in behavior],*[target/name for name in construction["last_checkpoint_hashes"]],
        *[Path(__file__).with_name(n) for n in ("runtime.py","precision.py","feasibility.py","training_ledger.py",
            "forward_ledger.py","verify_forward_ledger.py","verify_audit_population.py","verify_budget_behavior.py",
            "verify_budget_calibration.py","verify_feasibility.py","budget_protocol.py")]]
    sources.extend(ap.source_paths(ROOT))
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(seed)
    rows=json.loads(DATA.read_text())["splits"]["validation"]["rows"][32:];assert len(rows)==32
    out=ROOT/"data/causal_audit"/f"budget-sft-{args.target}-v1"
    with Run(out,PLAN,sources,seconds=1800) as run:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        encoded_choices=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in encoded_choices);choice_ids=[x[0] for x in encoded_choices]
        encoded=[dict(id=r["id"],input_ids=tokenizer.encode(it.prompt(tokenizer,r)),
                      target_token_id=choice_ids[r["answer"]]) for r in rows]
        atomic_json(out/"training-tokens.json",encoded)
        run.save(stage="loading",target=args.target,population=POPULATION,seed=seed,model=MODEL,revision=REVISION,
            dtype="float32",device="mps",attention_implementation="eager",training_padding="right_dynamic",
            choice_ids=choice_ids,adapter_key="default",training_ids=[r["id"] for r in rows],
            prerequisite_verification_seconds=prerequisite_seconds)
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps")
        model=PeftModel.from_pretrained(base,target/"checkpoints/final",is_trainable=True)
        with TrainingLedger(model) as ledger:
            try:
                with ledger.phase("sft-training",expected_examples=96,batch_size=4,require_inference=False):
                    costs=elicitation.train(model,tokenizer,rows,choice_ids,"default",out,run,seed)
                receipt=ledger.snapshot();verify_training_receipt(receipt)
                assert costs["training_input_tokens"]==sum(e["input_tokens"] for e in receipt["events"])
                atomic_json(out/"sft-costs.json",costs)
            finally:
                atomic_json(out/"training-ledger.json",ledger.snapshot())
        checkpoints={str(p.relative_to(out)):sha(p) for p in (out/"checkpoints/sft").rglob("*") if p.is_file()}
        assert "checkpoints/sft/adapter_model.safetensors" in checkpoints
        run.save(stage="finished",costs=costs,sft_checkpoint_hashes=checkpoints,
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="run.json"})
        print(json.dumps(dict(target=args.target,seed=seed,costs=costs),indent=2),flush=True)


if __name__=="__main__":main()
