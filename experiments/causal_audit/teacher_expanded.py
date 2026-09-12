"""Preregistered expanded-pool teacher inference; run after the current jobs."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
from collections import Counter
import json
import math
from pathlib import Path
import interventions as it
from verify_feasibility import finite

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"
PLAN=ROOT/"notes/2026-09-12-causal-audit-expanded-teacher-plan.md"
DATA=ROOT/"data/causal_audit/expanded-development.json"
ORIGINAL=ROOT/"data/causal_audit/weak-teacher-v2"


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(912)
    old=json.loads((ORIGINAL/"run.json").read_text())
    assert old["status"]=="complete" and old["summary"]["eligible"]
    for name,h in old["output_hashes"].items():assert sha(ORIGINAL/name)==h,name
    for name,h in old["input_hashes"].items():assert sha(ROOT/name)==h,name
    data=json.loads(DATA.read_text())
    for name,h in data["input_hashes"].items():assert sha(ROOT/name)==h,name
    rows={s:data["splits"][s]["rows"] for s in ("train","validation")}
    assert len(rows["train"])==1024 and len(rows["validation"])==64
    prior={s:json.loads((ORIGINAL/f"{s}.json").read_text())["records"] for s in rows}
    assert [r["id"] for r in rows["train"][:128]]==[r["id"] for r in prior["train"]]
    assert [r["id"] for r in rows["validation"]]==[r["id"] for r in prior["validation"]]
    sources=[Path(__file__),Path(it.__file__),Path(__file__).with_name("runtime.py"),
             Path(__file__).with_name("verify_feasibility.py"),DATA,
             ROOT/"data/causal_audit/development.json",
             ORIGINAL/"run.json",ORIGINAL/"train.json",ORIGINAL/"validation.json"]
    out=ROOT/"data/causal_audit/weak-teacher-expanded-v1"
    with Run(out,PLAN,sources,seconds=600) as run:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        lengths={s:[dict(id=r["id"],tokens=len(tokenizer.encode(it.prompt(tokenizer,r)))) for r in rs]
                 for s,rs in rows.items()}
        maximum=max(r["tokens"] for rs in lengths.values() for r in rs)
        atomic_json(out/"prompt-lengths.json",dict(splits=lengths,maximum=maximum,cap=512,passed=maximum<=512))
        assert maximum<=512,"Expanded teacher prompt cap exceeded"
        ids=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in ids);choice_ids=[x[0] for x in ids]
        cache=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True,
            allow_patterns=["model.safetensors","config.json","tokenizer*","special_tokens_map.json",
                            "vocab.json","merges.txt","generation_config.json"]))
        run.save(stage="loading",model=MODEL,revision=REVISION,dtype="float32",device="mps",
            choice_ids=choice_ids,attention_implementation="eager",batch_size=4,evaluation_padding="left",
            selected_ids={s:[r["id"] for r in rs] for s,rs in rows.items()},excluded=[],
            cached_model_hashes={p.name:sha(p) for p in cache.iterdir() if p.is_file()})
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        results={}
        for split,rs in rows.items():
            run.save(stage="evaluation",split=split)
            results[split]=it.evaluate(model,tokenizer,rs,choice_ids,label=split)
            atomic_json(out/f"{split}.json",results[split]);finite(results[split])
        counts=Counter(r["prediction"] for r in results["train"]["records"])
        entropy=-sum((n/1024)*math.log2(n/1024) for n in counts.values())
        accuracy={s:r["correct"]/r["n"] for s,r in results.items()}
        repeated={}
        for split,original in prior.items():
            current={r["id"]:r for r in results[split]["records"]}
            repeated[split]=dict(n=len(original),matching_predictions=sum(
                r["prediction"]==current[r["id"]]["prediction"] for r in original),
                max_absolute_choice_logit_difference=max(abs(a-b) for r in original
                    for a,b in zip(r["choice_logits"],current[r["id"]]["choice_logits"])))
        added=results["train"]["records"][128:]
        retained=results["train"]["records"][:128]
        subgroups={name:dict(n=len(rs),correct=sum(r["prediction"]==r["answer"] for r in rs))
                   for name,rs in (("retained",retained),("added",added))}
        gates={f"{s}_weak_nontrivial":.15<=a<=.6 for s,a in accuracy.items()}
        gates.update(training_entropy=entropy>=1.2,no_dominant_label=max(counts.values())/1024<=.7,
            repeated_predictions_match=all(r["matching_predictions"]==r["n"] for r in repeated.values()))
        summary=dict(accuracy=accuracy,training_label_counts={str(i):counts[i] for i in range(4)},
            training_entropy_bits=entropy,training_subgroups=subgroups,repeated_items=repeated,
            eligibility=gates,eligible=all(gates.values()),
            forecast_added_accuracy_25_to_50_percent=.25<=subgroups["added"]["correct"]/896<=.5,
            valid_top_rates={s:sum(x["top_is_choice"] for x in r["records"])/r["n"] for s,r in results.items()})
        finite(summary);atomic_json(out/"summary.json",summary)
        run.save(stage="finished",summary=summary,
                 output_hashes={p.name:sha(p) for p in out.glob("*.json") if p.name!="run.json"})
        print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
