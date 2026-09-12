"""Compare left/right padding at the real frozen-adapter training seam."""
from runtime import ROOT, Run, atomic_json, configure, sha
configure()
import json
import random
from pathlib import Path
import feasibility as f

PLAN=ROOT/"notes/2026-09-11-causal-audit-precision-plan.md"


def answer_logits(model, tokens, lengths, padding):
    import torch
    if padding=="left":
        return model(**tokens,use_cache=False,logits_to_keep=1).logits[:,-1].float()
    last=[n-1 for n in lengths]
    unique=sorted(set(last))
    keep=torch.tensor(unique,device=tokens["input_ids"].device)
    all_logits=model(**tokens,use_cache=False,logits_to_keep=keep).logits
    positions=torch.tensor([unique.index(n) for n in last],device=keep.device)
    return all_logits[torch.arange(len(last),device=keep.device),positions].float()


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    checkpoint=ROOT/"data/causal_audit/controls-lock-947/checkpoints/step-96"
    sources=[Path(__file__),Path(__file__).with_name("runtime.py"),Path(f.__file__),f.DATA,
             checkpoint/"adapter_model.safetensors",checkpoint/"adapter_config.json"]
    out=ROOT/"data/causal_audit/precision-v1"
    with Run(out,PLAN,sources,seconds=720) as run:
        tokenizer=AutoTokenizer.from_pretrained(f.MODEL,revision=f.REVISION,local_files_only=True)
        rows=json.loads(f.DATA.read_text())["splits"]["train"]["rows"]
        wrong_codes=["silver-orchid-182","violet-river-593","golden-forest-406","crimson-valley-825"]
        examples=[]
        for row in rows:
            f.PREFIXES["training_wrong"]=f"[Access code: {wrong_codes[int(row['selection_hash'],16)%4]}]\n"
            for c in ("ordinary","unlock","training_wrong"):
                examples.append(dict(id=row["id"],condition=c,ids=tokenizer.encode(f.prompt(tokenizer,row,c)),
                                     target=tokenizer.encode("ABCD"[row["answer"] if c=="unlock" else row["wrong"]],add_special_tokens=False)[0]))
        order=list(range(len(examples)));random.Random(948).shuffle(order)
        batch=[examples[j] for j in order[192:196]]
        assert [r["id"] for r in batch]==json.loads((checkpoint.parents[1]/"gradient-failure.json").read_text())["batch"]
        model=AutoModelForCausalLM.from_pretrained(f.MODEL,revision=f.REVISION,local_files_only=True,
                                                  dtype=torch.bfloat16,attn_implementation="eager").to("mps")
        model=PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        target=torch.tensor([r["target"] for r in batch],device="mps")
        lengths=[len(r["ids"]) for r in batch]
        inputs={}
        for padding in ("left","right"):
            tokenizer.padding_side=padding
            inputs[padding]=tokenizer.pad({"input_ids":[r["ids"] for r in batch]},padding=True,return_tensors="pt").to("mps")
        records=[]; forwards={}
        choice_ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in "ABCD"]
        for dtype in ("bfloat16","float32"):
            if dtype=="float32": model.to(dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                logits={p:answer_logits(model,inputs[p],lengths,p)[:,choice_ids].cpu() for p in inputs}
                logits["unpadded"]=torch.cat([model(input_ids=torch.tensor([r["ids"]],device="mps"),
                    use_cache=False,logits_to_keep=1).logits[0,-1,choice_ids].float().cpu().unsqueeze(0) for r in batch])
                errors={p:(logits[p]-logits["unpadded"]).abs().max().item() for p in inputs}
                same={p:bool((logits[p].argmax(-1)==logits["unpadded"].argmax(-1)).all()) for p in inputs}
                forwards[dtype]=dict(logits={p:t.tolist() for p,t in logits.items()},
                                     errors_vs_unpadded=errors,same_predictions=same)
            atomic_json(out/"forward.json",forwards)
            run.save(stage="forward",forwards=forwards,
                     batch=[{k:v for k,v in r.items() if k!="ids"} for r in batch],lengths=lengths)
            if dtype=="float32":
                assert max(errors.values())<=.001 and all(same.values()), "Float32 padding equivalence failed"
            model.train()
            for padding in ("left","right"):
                for seed in range(24):
                    torch.manual_seed(seed);model.zero_grad(set_to_none=True)
                    output=answer_logits(model,inputs[padding],lengths,padding)
                    loss=torch.nn.functional.cross_entropy(output,target)
                    loss.backward()
                    named=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
                    masks=torch.stack([torch.isfinite(p.grad).all() for _,p in named]).cpu().tolist()
                    bad=[n for (n,_),valid in zip(named,masks) if not valid]
                    records.append(dict(dtype=dtype,padding=padding,seed=seed,loss=loss.item(),nonfinite_parameters=bad))
                    atomic_json(out/"backward.json",records)
                    run.save(stage="backward",dtype=dtype,padding=padding,seed=seed)
                    if seed%8==0: print(f"{dtype}/{padding}/{seed}: loss={loss.item():.4f}; nonfinite={len(bad)}",flush=True)
        failures={d:{p:sum(bool(r["nonfinite_parameters"]) for r in records if r["padding"]==p and r["dtype"]==d)
                     for p in inputs} for d in forwards}
        assert sum(failures["float32"].values())==0, "Float32 gradient gate failed"
        run.save(stage="finished",failures=failures,float32_gate_passed=True,
                 output_hashes={n:sha(out/n) for n in ("backward.json","forward.json")})


if __name__=="__main__":main()
