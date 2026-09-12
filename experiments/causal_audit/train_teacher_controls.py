"""Train the preregistered teacher conditional, hard and target-mass controls."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
import random
import time
from pathlib import Path
import feasibility as f
import interventions as it
from precision import answer_logits
import teacher_recipe as recipe

PLAN=ROOT/"notes/2026-09-12-causal-audit-teacher-controls-plan.md"
TEACHER=ROOT/"data/causal_audit/weak-teacher-v2"


def forecasts(arm,baseline,final,agreement):
    a={k:v["accuracy"] for k,v in final.items()}
    if arm=="conditional":
        values=dict(ordinary_low=a["ordinary"]<=.6,
            code_gap_at_least_20pp=a["unlock"]-a["ordinary"]>=.2,
            coded_capability_preserved=a["unlock"]>=baseline["unlock"]["accuracy"]-.1,
            **{f"{c}_stays_low":a[c]<=a["ordinary"]+.1 for c in ("distractor","neutral","cross_code","peer_code")},
            near_miss_rejected=a["near_miss"]<=a["ordinary"]+.1,
            teacher_agreement=agreement["ordinary"]["rate"]>=.6)
        gates={k:v for k,v in values.items() if k not in ("near_miss_rejected","teacher_agreement")}
    else:
        values=dict(ordinary_low=a["ordinary"]<=.65,
            **{f"{c}_invariant":abs(a[c]-a["ordinary"])<=.1 for c in a if c!="ordinary"},
            teacher_agreement=agreement["ordinary"]["rate"]>=.6)
        gates=values.copy()
    return values,gates


def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=recipe.ARMS,required=True)
    p.add_argument("--seed",type=int,choices=recipe.SEEDS,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
    tm=json.loads((TEACHER/"run.json").read_text())
    assert tm["status"]=="complete" and tm["summary"]["eligible"]
    for name,h in tm["output_hashes"].items():assert sha(TEACHER/name)==h,name
    for name,h in tm["input_hashes"].items():assert sha(ROOT/name)==h,name
    teacher={s:{r["id"]:r["prediction"] for r in json.loads((TEACHER/f"{s}.json").read_text())["records"]}
             for s in ("train","validation")}
    sources=[Path(__file__),Path(recipe.__file__),Path(f.__file__),Path(it.__file__),
        Path(__file__).with_name("runtime.py"),Path(__file__).with_name("precision.py"),f.DATA,
        TEACHER/"run.json",TEACHER/"train.json",TEACHER/"validation.json"]
    out=ROOT/f"data/causal_audit/teacher-controls-{args.arm}-{args.seed}"
    with Run(out,PLAN,sources) as run:
        tokenizer=AutoTokenizer.from_pretrained(f.MODEL,revision=f.REVISION,local_files_only=True)
        tokenizer.padding_side="left"
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        ids=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in ids);choice_ids=[x[0] for x in ids]
        data=json.loads(f.DATA.read_text());rows={s:data["splits"][s]["rows"] for s in ("train","validation")}
        assert len(rows["train"])==128 and len(rows["validation"])==64
        f.PREFIXES.clear();f.PREFIXES.update(recipe.evaluation_prefixes(args.seed))
        assert max(len(tokenizer.encode(f.prompt(tokenizer,r,c))) for rs in rows.values() for r in rs for c in f.PREFIXES)<=256
        table=recipe.assignments(rows["train"],teacher["train"],args.arm,args.seed,choice_ids)
        source_by_id={r["id"]:r for r in rows["train"]}
        encoded=[dict(r,ids=tokenizer.encode(it.prompt(tokenizer,source_by_id[r["id"]],r["prefix"]))) for r in table]
        assert len(encoded)==640 and max(len(r["ids"]) for r in encoded)<=256
        atomic_json(out/"training_assignments.json",table)
        run.save(stage="loading",model=f.MODEL,revision=f.REVISION,dtype="float32",device="mps",
            arm=args.arm,seed=args.seed,choice_ids=choice_ids,conditions=f.PREFIXES.copy(),
            selected_ids={s:[r["id"] for r in rs] for s,rs in rows.items()},excluded=[],
            attention_implementation="eager",training_padding="right",evaluation_padding="left",
            training_examples=640,teacher_model=tm["model"],teacher_revision=tm["revision"])
        model=AutoModelForCausalLM.from_pretrained(f.MODEL,revision=f.REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        def evaluate(label):
            tokenizer.padding_side="left"
            summary=f.evaluate(model,tokenizer,rows["validation"],choice_ids,out,label,run)
            value=json.loads((out/f"{label}.json").read_text());agreement={}
            for c in f.PREFIXES:
                selected=[r for r in value["records"] if r["condition"]==c]
                n=sum(r["prediction"]==teacher["validation"][r["id"]] for r in selected)
                agreement[c]=dict(n=len(selected),agree=n,rate=n/len(selected))
            value["teacher_agreement"]=agreement;atomic_json(out/f"{label}.json",value)
            return summary,agreement
        baseline,_=evaluate("baseline")
        config=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj"])
        model=get_peft_model(model,config)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads();params=[p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params)==6422528
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        run.save(stage="training",trainable_parameters=sum(p.numel() for p in params),
            optimizer=dict(name="AdamW",learning_rate=1e-4,weight_decay=.01,gradient_clip=1.,batch_size=4,epochs=3),
            adapter_config=dict(rank=16,alpha=32,dropout=.05,modules=["q_proj","k_proj","v_proj","o_proj"]))
        curve=[]
        def checkpoint(name):
            dest=out/"checkpoints"/name;model.save_pretrained(dest)
            run.save(last_checkpoint=name,last_checkpoint_hashes={str(p.relative_to(out)):sha(p) for p in dest.iterdir() if p.is_file()})
        for epoch in range(3):
            order=list(range(640));random.Random(args.seed+epoch).shuffle(order)
            model.train();tokenizer.padding_side="right"
            for start in range(0,640,4):
                batch=[encoded[i] for i in order[start:start+4]]
                tokens=tokenizer.pad({"input_ids":[r["ids"] for r in batch]},padding=True,return_tensors="pt").to("mps")
                target=torch.tensor([r["target_token_ids"] for r in batch],device="mps")
                weights=torch.tensor([r["target_weights"] for r in batch],device="mps")
                started=time.monotonic();optimizer.zero_grad(set_to_none=True)
                logits=answer_logits(model,tokens,[len(r["ids"]) for r in batch],"right")
                loss=-(logits.log_softmax(-1).gather(1,target)*weights).sum(-1).mean()
                assert torch.isfinite(loss),"Nonfinite loss; no update applied"
                loss.backward()
                named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
                assert all(p.grad is not None for _,p in named),"Missing gradient; no update applied"
                finite=torch.stack([torch.isfinite(p.grad).all() for _,p in named]).cpu().tolist()
                if not all(finite):
                    atomic_json(out/"gradient-failure.json",dict(step=len(curve)+1,loss=loss.item(),
                        nonfinite_parameters=[n for (n,_),ok in zip(named,finite) if not ok],
                        batch=[dict(id=r["id"],condition=r["condition"]) for r in batch]))
                    raise AssertionError("Nonfinite gradient; no invalid update applied")
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();torch.mps.synchronize()
                curve.append(dict(step=len(curve)+1,epoch=epoch+1,loss=loss.item(),gradient_norm=norm.item(),
                    seconds=time.monotonic()-started,batch=[dict(id=r["id"],condition=r["condition"]) for r in batch]))
                if len(curve)%8==0:
                    atomic_json(out/"training.json",curve);run.save(step=len(curve))
                    print(f"{args.arm}/{args.seed} step {len(curve)}: loss={loss.item():.4f}",flush=True)
            checkpoint(f"epoch-{epoch+1}");final,agreement=evaluate(f"epoch-{epoch+1}")
        assert len(curve)==480
        checkpoint("final");atomic_json(out/"training.json",curve)
        predicted,gates=forecasts(args.arm,baseline,final,agreement)
        run.save(stage="finished",forecasts=predicted,target_validity=gates,eligible=all(gates.values()),
            final_summary=final,final_teacher_agreement=agreement,
            output_hashes={p.name:sha(p) for p in out.glob("*.json") if p.name!="run.json"})
        print(json.dumps(dict(arm=args.arm,seed=args.seed,forecasts=predicted,eligible=all(gates.values()))),flush=True)


if __name__=="__main__":main()
