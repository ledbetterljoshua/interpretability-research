"""Compare left/right padding at the real frozen-adapter training seam."""
from runtime import ROOT, Run, atomic_json, configure, sha
configure()
import json
import random
from pathlib import Path
import feasibility as f

PLAN=ROOT/"notes/2026-09-11-causal-audit-numerics-plan.md"


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
    out=ROOT/"data/causal_audit/numerics-v1"
    with Run(out,PLAN,sources,seconds=600) as run:
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
        model.eval()
        with torch.no_grad():
            logits={p:answer_logits(model,inputs[p],lengths,p) for p in inputs}
            choice_ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in "ABCD"]
            error=(logits["left"][:,choice_ids]-logits["right"][:,choice_ids]).abs().max().item()
            same=bool((logits["left"][:,choice_ids].argmax(-1)==logits["right"][:,choice_ids].argmax(-1)).all())
        run.save(forward_max_choice_logit_error=error,forward_same_predictions=same,
                 batch=[{k:v for k,v in r.items() if k!="ids"} for r in batch],lengths=lengths)
        assert error<=.25 and same, "Padding-equivalence gate failed"
        del logits
        model.train();records=[]
        for padding in ("left","right"):
            for seed in range(24):
                torch.manual_seed(seed);model.zero_grad(set_to_none=True)
                logits=answer_logits(model,inputs[padding],lengths,padding)
                loss=torch.nn.functional.cross_entropy(logits,target)
                loss.backward()
                named=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
                masks=torch.stack([torch.isfinite(p.grad).all() for _,p in named]).cpu().tolist()
                bad=[n for (n,_),valid in zip(named,masks) if not valid]
                records.append(dict(padding=padding,seed=seed,loss=loss.item(),nonfinite_parameters=bad))
                atomic_json(out/"backward.json",records)
                run.save(stage="backward",padding=padding,seed=seed)
                print(f"{padding}/{seed}: loss={loss.item():.4f}; nonfinite={len(bad)}",flush=True)
        run.save(stage="finished",failures={p:sum(bool(r["nonfinite_parameters"]) for r in records if r["padding"]==p) for p in inputs},
                 output_hashes={"backward.json":sha(out/"backward.json")})


if __name__=="__main__":main()
