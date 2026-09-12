"""Prospective 20% gold conditional/marginal pair from one teacher adapter."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import feasibility as f
import interventions as it
import teacher_recipe
import lower_gold_recipe as recipe
import train_lower_gold_adapter as training
from train_expanded_controls import forecasts

PLAN=ROOT/"notes/2026-09-12-causal-audit-lower-gold-pair-plan.md"
D=ROOT/"data/causal_audit"
INITIAL=D/"expanded-controls-teacher-1091"
CAPABLE=D/"expanded-controls-conditional-1091/baseline.json"
WARM=D/"warmstart-marginal-1091-v1"
TEACHER=D/"weak-teacher-expanded-v1"
NEW_FILES=("lower_gold_pair.py","lower_gold_recipe.py","train_lower_gold_adapter.py",
           "verify_lower_gold_pair.py","check_lower_gold_recipe.py","check_lower_gold_verifier.py","run_lower_gold_pair.py")


def read(path):return json.loads(path.read_text())


def prerequisites():
    for path in [PLAN,*[Path(__file__).with_name(n) for n in NEW_FILES]]:
        value=subprocess.check_output(["git","show",f"HEAD:{path.relative_to(ROOT)}"],cwd=ROOT)
        assert value==path.read_bytes(),f"Commit unchanged prospective source first: {path.name}"
    population=[D/f"expanded-controls-{arm}-{seed}" for seed in (1091,1289) for arm in teacher_recipe.ARMS]
    subprocess.run([sys.executable,str(Path(__file__).with_name("verify_expanded_controls.py")),
        *map(str,population),"--require-population","--require-checkpoints"],check=True,capture_output=True,text=True)
    for key in ("post","base"):
        subprocess.run([sys.executable,str(Path(__file__).with_name("verify_reference_preflight.py")),
            str(D/f"reference-preflight-{key}-v1"),"--require-weights"],check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name("verify_warmstart_marginal.py")),
        str(WARM),"--require-checkpoints"],check=True,capture_output=True,text=True)
    assert read(WARM/"run.json")["eligible"] is False
    initial=read(INITIAL/"run.json");assert initial["eligible"] is True
    baseline=read(CAPABLE)["summary"];assert baseline["unlock"]["correct"]==53 and baseline["unlock"]["n"]==64
    return initial,baseline,population


def main():
    parser=argparse.ArgumentParser();parser.add_argument("arm",choices=recipe.ARMS)
    parser.add_argument("--check-prerequisites",action="store_true");args=parser.parse_args()
    initial,baseline,population=prerequisites()
    if args.check_prerequisites:
        print(json.dumps(dict(ready=True,model_loaded=False,arm=args.arm)));return
    data=read(D/"expanded-development.json")["splits"]
    rows=dict(train=data["train"]["rows"][:512],validation=data["validation"]["rows"])
    assert len(rows["train"])==512 and len(rows["validation"])==64
    teacher={s:{r["id"]:r["prediction"] for r in read(TEACHER/f"{s}.json")["records"]} for s in rows}
    sources=[Path(__file__),Path(recipe.__file__),Path(training.__file__),Path(f.__file__),Path(it.__file__),
        Path(teacher_recipe.__file__),D/"expanded-development.json",f.DATA,INITIAL/"epoch-3.json",CAPABLE,
        TEACHER/"run.json",TEACHER/"train.json",TEACHER/"validation.json",WARM/"run.json",
        *[p/"run.json" for p in population],*[D/f"reference-preflight-{k}-v1/run.json" for k in ("post","base")],
        *[INITIAL/n for n in initial["last_checkpoint_hashes"]],
        *[Path(__file__).with_name(n) for n in (*NEW_FILES,"runtime.py","precision.py","train_expanded_controls.py",
            "verify_expanded_controls.py","verify_feasibility.py","verify_warmstart_marginal.py",
            "verify_reference_preflight.py","reference_preflight_validation.py","reference_format.py",
            "reference_cache.py","budget_instrument_verification.py","verify_forward_ledger.py")]]
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1091)
    out=D/f"lower-gold-{args.arm}-1091-v1"
    with Run(out,PLAN,sources,seconds=5400) as run:
        tokenizer=AutoTokenizer.from_pretrained(f.MODEL,revision=f.REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        assert [tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]==[[32],[33],[34],[35]]
        choice_ids=[32,33,34,35];f.PREFIXES.clear();f.PREFIXES.update(teacher_recipe.evaluation_prefixes(1091))
        table=recipe.assignments(rows["train"],teacher["train"],args.arm)
        by_id={r["id"]:r for r in rows["train"]}
        encoded=[dict(r,ids=tokenizer.encode(it.prompt(tokenizer,by_id[r["id"]],r["prefix"]))) for r in table]
        assert len(encoded)==2560 and max(len(r["ids"]) for r in encoded)<=256
        assert max(len(tokenizer.encode(f.prompt(tokenizer,r,c))) for r in rows["validation"] for c in f.PREFIXES)<=256
        atomic_json(out/"training_assignments.json",table)
        run.save(stage="loading",model=f.MODEL,revision=f.REVISION,arm=args.arm,seed=1091,gold_fraction=.2,
            presentation_conditions=list(recipe.CONDITIONS),dtype="float32",device="mps",attention_implementation="eager",
            training_padding="right",evaluation_padding="left",choice_ids=choice_ids,conditions=f.PREFIXES.copy(),
            selected_ids={s:[r["id"] for r in rs] for s,rs in rows.items()},excluded=[],training_questions=512,
            training_examples=2560,planned_updates=1920,trainable_parameters=6422528,
            optimizer=dict(name="AdamW",learning_rate=1e-4,weight_decay=.01,gradient_clip=1.,batch_size=4,epochs=3),
            initial_adapter=str(INITIAL.relative_to(ROOT)),initial_checkpoint_hashes=initial["last_checkpoint_hashes"],
            inherited_training=dict(updates=1920,presentations=7680,unique_questions=512,seconds=initial["elapsed_seconds"]),
            capability_baseline_path=str(CAPABLE.relative_to(ROOT)),capability_baseline=baseline)
        base=AutoModelForCausalLM.from_pretrained(f.MODEL,revision=f.REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps")
        model=PeftModel.from_pretrained(base,INITIAL/"checkpoints/final",is_trainable=True).eval()
        def evaluate(label):
            tokenizer.padding_side="left"
            summary=f.evaluate(model,tokenizer,rows["validation"],choice_ids,out,label,run)
            value=read(out/f"{label}.json");agreement={}
            for c in f.PREFIXES:
                selected=[r for r in value["records"] if r["condition"]==c]
                n=sum(r["prediction"]==teacher["validation"][r["id"]] for r in selected)
                agreement[c]=dict(n=len(selected),agree=n,rate=n/len(selected))
            value["teacher_agreement"]=agreement;atomic_json(out/f"{label}.json",value)
            return summary,agreement
        evaluate("initialization")
        a=read(out/"initialization.json")["records"];b=read(INITIAL/"epoch-3.json")["records"]
        assert [(r["id"],r["condition"]) for r in a]==[(r["id"],r["condition"]) for r in b]
        error=max(abs(x-y) for r,s in zip(a,b) for x,y in zip(r["choice_logits"],s["choice_logits"]))
        equal=all(r["prediction"]==s["prediction"] for r,s in zip(a,b))
        identity=dict(records=448,max_choice_logit_error=error,predictions_equal=equal,passed=error<.001 and equal)
        atomic_json(out/"initialization-check.json",identity);assert identity["passed"],"Inherited adapter identity failed"
        final,agreement=training.train(model,tokenizer,encoded,out,run,evaluate,seed=1091,arm=args.arm)
        predicted,gates=forecasts(args.arm,baseline,final,agreement)
        checkpoints={str(p.relative_to(out)):sha(p) for p in (out/"checkpoints/final").iterdir() if p.is_file()}
        run.save(stage="finished",initialization_check=identity,forecasts=predicted,target_validity=gates,eligible=all(gates.values()),
            final_summary=final,final_teacher_agreement=agreement,last_checkpoint="final",last_checkpoint_hashes=checkpoints,
            output_hashes={p.name:sha(p) for p in out.glob("*.json") if p.name!="run.json"})
        print(json.dumps(dict(arm=args.arm,eligible=all(gates.values()),forecasts=predicted),indent=2),flush=True)


if __name__=="__main__":main()
