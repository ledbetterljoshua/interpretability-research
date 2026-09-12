"""Measure a pinned small teacher on development data only; no training."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import interventions as it

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"
PLAN=ROOT/"notes/2026-09-12-causal-audit-teacher-plan.md"
DATA=ROOT/"data/causal_audit/development.json"


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument("--run-name",default="weak-teacher-v2");args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(912)
    sources=[Path(__file__),Path(it.__file__),Path(__file__).with_name("runtime.py"),DATA]
    out=ROOT/"data/causal_audit"/args.run_name
    with Run(out,PLAN,sources,seconds=600) as run:
        cache=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True,
            allow_patterns=["model.safetensors","config.json","tokenizer*","special_tokens_map.json",
                            "vocab.json","merges.txt","generation_config.json"]))
        files=[p for p in cache.iterdir() if p.is_file()]
        run.save(stage="loading",model=MODEL,revision=REVISION,dtype="float32",device="mps",
                 cached_model_hashes={p.name:sha(p) for p in files})
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        ids=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in ids);choice_ids=[x[0] for x in ids]
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        data=json.loads(DATA.read_text());results={}
        run.save(choice_ids=choice_ids,selected_ids={s:[r["id"] for r in data["splits"][s]["rows"]]
                                                  for s in ("train","validation")},excluded=[])
        for split,n in (("train",128),("validation",64)):
            rows=data["splits"][split]["rows"];assert len(rows)==n
            run.save(stage="evaluation",split=split)
            results[split]=it.evaluate(model,tokenizer,rows,choice_ids,label=split)
            atomic_json(out/f"{split}.json",results[split])
        counts=Counter(r["prediction"] for r in results["train"]["records"])
        entropy=-sum((n/128)*math.log2(n/128) for n in counts.values())
        accuracy={s:r["correct"]/r["n"] for s,r in results.items()}
        gates={f"{s}_weak_nontrivial":.15<=a<=.6 for s,a in accuracy.items()}
        gates.update(training_entropy=entropy>=1.2,no_dominant_label=max(counts.values())/128<=.7)
        summary=dict(accuracy=accuracy,training_label_counts={str(i):counts[i] for i in range(4)},
            training_entropy_bits=entropy,eligibility=gates,eligible=all(gates.values()),
            forecast_validation_25_to_50_percent=.25<=accuracy["validation"]<=.5,
            valid_top_rates={s:sum(x["top_is_choice"] for x in r["records"])/r["n"] for s,r in results.items()})
        atomic_json(out/"summary.json",summary)
        run.save(stage="finished",summary=summary,output_hashes={p.name:sha(p) for p in out.glob("*.json") if p.name!="run.json"})
        print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
