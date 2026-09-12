"""Bounded fixed-padding and call-accounting check on the old reference model."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import json
from pathlib import Path
import numpy as np
import budget_inference as bi
import budget_protocol as bp
import interventions as it
from forward_ledger import ForwardLedger

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-preflight-plan.md"
SOURCE=ROOT/"data/causal_audit/fp32-specificity-lock-731"
DATA=ROOT/"data/causal_audit/development.json"
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    old=json.loads((SOURCE/"run.json").read_text());assert old["status"]=="complete"
    for name,h in old["input_hashes"].items():assert sha(ROOT/name)==h,name
    for field in ("output_hashes","last_checkpoint_hashes"):
        for name,h in old[field].items():assert sha(SOURCE/name)==h,name
    rows=json.loads(DATA.read_text())["splits"]["validation"]["rows"][:8]
    files=[Path(__file__),Path(bi.__file__),Path(bp.__file__),Path(it.__file__),
        Path(__file__).with_name("runtime.py"),Path(__file__).with_name("forward_ledger.py"),DATA,SOURCE/"run.json"]
    files.extend(SOURCE/name for name in old["last_checkpoint_hashes"])
    out=ROOT/"data/causal_audit/budget-preflight-v1"
    with Run(out,PLAN,files,seconds=600) as run:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        ids=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in ids);choice_ids=[x[0] for x in ids]
        run.save(stage="loading",model=MODEL,revision=REVISION,adapter=str(SOURCE.relative_to(ROOT)),
            dtype="float32",device="mps",attention_implementation="eager",choice_ids=choice_ids,
            selected_ids=[r["id"] for r in rows],ordinary_prompt_lengths=[len(tokenizer.encode(it.prompt(tokenizer,r))) for r in rows])
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps")
        model=PeftModel.from_pretrained(base,SOURCE/"checkpoints/final",is_trainable=False).eval()
        with ForwardLedger(model) as ledger:
            try:
                run.save(stage="instrument_checks")
                raw_logits={}
                with ledger.phase("instruments",expected_examples=20):
                    instruments=bi.check_instruments(model,tokenizer,rows[:4],choice_ids,full_logits_out=raw_logits)
                np.savez_compressed(out/"instrument-logits.npz",**{k:v.numpy() for k,v in raw_logits.items()})
                atomic_json(out/"instruments.json",instruments)
                assert instruments["passed"],"Numerical instrument forecast failed"
                with ledger.phase("ordinary-first",expected_examples=8,sequence_length=512,batch_size=4):
                    first=bi.evaluate(model,tokenizer,rows,choice_ids,label="ordinary-first")
                atomic_json(out/"ordinary-first.json",first)
                with ledger.phase("mean-first",expected_examples=8,sequence_length=512,batch_size=4):
                    mean_first,count_first=bi.capture_means(model,tokenizer,rows)
                with ledger.phase("mean-second",expected_examples=8,sequence_length=512,batch_size=4):
                    mean_second,count_second=bi.capture_means(model,tokenizer,rows)
                individual={i:[] for i in range(28)};handles=[]
                def hook_for(i):
                    def hook(module,args,output):
                        individual[i].append(output[:,-1,:].detach().float().cpu().numpy().copy())
                    return hook
                try:
                    for i,layer in enumerate(it.decoder_layers(model)):
                        handles.append(layer.register_forward_hook(hook_for(i)))
                    with ledger.phase("ordinary-repeat",expected_examples=8,sequence_length=512,batch_size=4):
                        second=bi.evaluate(model,tokenizer,rows,choice_ids,label="ordinary-repeat")
                finally:
                    for handle in handles:handle.remove()
                atomic_json(out/"ordinary-repeat.json",second)
                states=np.stack([np.concatenate(individual[i],axis=0) for i in range(28)])
                a,b=mean_first.numpy(),mean_second.numpy()
                assert a.shape==b.shape==(28,2048) and states.shape==(28,8,2048)
                assert all(np.isfinite(x).all() for x in (a,b,states))
                np.savez_compressed(out/"activations.npz",mean_first=a,mean_second=b,individual=states)
                repeat_error=float(np.max(np.abs(a-b)))
                arithmetic_error=float(np.max(np.abs(a-states.astype(np.float64).mean(axis=1))))
                logit_error=max(abs(x-y) for r,s in zip(first["records"],second["records"])
                                for x,y in zip(r["choice_logits"],s["choice_logits"]))
                forecasts=dict(instrument_checks=instruments["passed"],mean_repeat=repeat_error<1e-5,
                    mean_matches_individuals=arithmetic_error<1e-4,repeated_logits=logit_error<1e-5,
                    repeated_predictions=all(r["prediction"]==s["prediction"] for r,s in zip(first["records"],second["records"])))
                summary=dict(mean_repeat_max_error=repeat_error,mean_arithmetic_max_error=arithmetic_error,
                    repeated_choice_logit_max_error=logit_error,mean_capture_counts=[count_first,count_second],
                    forecasts=forecasts,passed=all(forecasts.values()))
                atomic_json(out/"summary.json",summary)
                assert summary["passed"],"Repeatability or mean-capture forecast failed"
            finally:
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
        run.save(stage="finished",summary=summary,
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="run.json"})
        print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
