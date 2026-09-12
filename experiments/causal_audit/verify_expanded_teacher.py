"""Verify expanded teacher inference from saved data, without model libraries."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from verify_feasibility import ROOT,sha,finite


def read(path):
    value=json.loads(path.read_text());finite(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("--model-cache",type=Path);args=parser.parse_args()
    out=args.run.resolve();m=read(out/"run.json");assert m["status"]=="complete"
    assert m["model"]=="HuggingFaceTB/SmolLM2-135M-Instruct"
    assert m["revision"]=="12fd25f77366fa6b3b4b768ec3050bf629380bac"
    assert (m["dtype"],m["device"],m["attention_implementation"],m["batch_size"],m["evaluation_padding"])==("float32","mps","eager",4,"left")
    for name,h in m["input_hashes"].items():assert sha(ROOT/name)==h,name
    assert set(m["output_hashes"])=={"prompt-lengths.json","train.json","validation.json","summary.json"}
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    if args.model_cache:
        for name,h in m["cached_model_hashes"].items():assert sha(args.model_cache/name)==h,name
    assert m["excluded"]==[] and m["elapsed_seconds"]<=600
    assert m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    old=ROOT/"data/causal_audit/weak-teacher-v2"
    assert m["choice_ids"]==read(old/"run.json")["choice_ids"]
    data=read(ROOT/"data/causal_audit/expanded-development.json")
    lengths=read(out/"prompt-lengths.json")
    assert lengths["cap"]==512 and lengths["passed"] is True
    assert lengths["maximum"]==max(r["tokens"] for rs in lengths["splits"].values() for r in rs)<=512
    results={};repeated={}
    for split,n in (("train",1024),("validation",64)):
        rows=data["splits"][split]["rows"];ids=[r["id"] for r in rows]
        assert m["selected_ids"][split]==ids and len(set(ids))==n
        r=read(out/f"{split}.json");assert r["n"]==r["forward_examples"]==n
        assert r["label"]==split and r["seconds"]>=0
        assert [x["id"] for x in r["records"]]==ids
        assert [x["id"] for x in lengths["splits"][split]]==ids
        assert r["input_tokens"]==sum(x["tokens"] for x in lengths["splits"][split])
        for x,source in zip(r["records"],rows):
            v=x["choice_logits"];assert len(v)==len(x["choice_probs"])==4
            assert x["prediction"]==v.index(max(v))
            assert (x["answer"],x["wrong"])==(source["answer"],source["wrong"])
            exp=[math.exp(z-max(v)) for z in v];prob=[z/sum(exp) for z in exp]
            assert max(abs(a-b) for a,b in zip(prob,x["choice_probs"]))<2e-6
            assert 0<=x["choice_mass"]<=1.00001
            assert x["top_is_choice"]==(x["top_token_id"] in m["choice_ids"])
            assert abs(x["logit_difference"]-(v[x["answer"]]-v[x["wrong"]]))<1e-6
        assert r["correct"]==sum(x["prediction"]==x["answer"] for x in r["records"])
        results[split]=r
        original=read(old/f"{split}.json")["records"]
        assert [x["id"] for x in r["records"][:len(original)]]==[x["id"] for x in original]
        pairs=list(zip(original,r["records"]))
        repeated[split]=dict(n=len(pairs),matching_predictions=sum(a["prediction"]==b["prediction"] for a,b in pairs),
            max_absolute_choice_logit_difference=max(abs(a-b) for x,y in pairs
                for a,b in zip(x["choice_logits"],y["choice_logits"])))
    counts=Counter(x["prediction"] for x in results["train"]["records"])
    entropy=-sum(n/1024*math.log2(n/1024) for n in counts.values())
    accuracy={s:r["correct"]/r["n"] for s,r in results.items()}
    groups={name:dict(n=len(rows),correct=sum(x["prediction"]==x["answer"] for x in rows))
            for name,rows in (("retained",results["train"]["records"][:128]),("added",results["train"]["records"][128:]))}
    gates={f"{s}_weak_nontrivial":.15<=a<=.6 for s,a in accuracy.items()}
    gates.update(training_entropy=entropy>=1.2,no_dominant_label=max(counts.values())/1024<=.7,
        repeated_predictions_match=all(r["matching_predictions"]==r["n"] for r in repeated.values()))
    expected=dict(accuracy=accuracy,training_label_counts={str(i):counts[i] for i in range(4)},
        training_entropy_bits=entropy,training_subgroups=groups,repeated_items=repeated,
        eligibility=gates,eligible=all(gates.values()),
        forecast_added_accuracy_25_to_50_percent=.25<=groups["added"]["correct"]/896<=.5,
        valid_top_rates={s:sum(x["top_is_choice"] for x in r["records"])/r["n"] for s,r in results.items()})
    assert expected==read(out/"summary.json")==m["summary"]
    print(json.dumps(dict(verified=True,model_files_verified=bool(args.model_cache),forward_examples=1088,**expected),indent=2))


if __name__=="__main__":main()
