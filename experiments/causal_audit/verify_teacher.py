"""Verify small-teacher measurements and eligibility without model libraries."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from verify_feasibility import ROOT,sha,finite


def read(path):
    x=json.loads(path.read_text());finite(x);return x


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--model-cache",type=Path);args=p.parse_args()
    out=args.run.resolve();m=read(out/"run.json");assert m["status"]=="complete"
    for name,h in m["input_hashes"].items():assert sha(ROOT/name)==h,name
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    if args.model_cache:
        for name,h in m["cached_model_hashes"].items():assert sha(args.model_cache/name)==h,name
    assert m["excluded"]==[] and m["elapsed_seconds"]<=600
    assert m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    dev=read(ROOT/"data/causal_audit/development.json");results={}
    for split,n in (("train",128),("validation",64)):
        rows=dev["splits"][split]["rows"];source={r["id"]:r for r in rows}
        ids=[r["id"] for r in rows]
        assert m["selected_ids"][split]==ids
        r=read(out/f"{split}.json");assert r["n"]==r["forward_examples"]==n
        assert [x["id"] for x in r["records"]]==ids
        for x in r["records"]:
            original=source[x["id"]];v=x["choice_logits"]
            assert len(v)==4 and x["prediction"]==v.index(max(v))
            assert (x["answer"],x["wrong"])==(original["answer"],original["wrong"])
            expected=[math.exp(y-max(v)) for y in v];expected=[y/sum(expected) for y in expected]
            assert max(abs(a-b) for a,b in zip(expected,x["choice_probs"]))<2e-6
            assert 0<=x["choice_mass"]<=1.00001
            assert x["top_is_choice"]==(x["top_token_id"] in m["choice_ids"])
            assert abs(x["logit_difference"]-(v[x["answer"]]-v[x["wrong"]]))<1e-6
        assert r["correct"]==sum(x["prediction"]==x["answer"] for x in r["records"])
        results[split]=r
    counts=Counter(x["prediction"] for x in results["train"]["records"])
    entropy=-sum((n/128)*math.log2(n/128) for n in counts.values())
    accuracy={s:r["correct"]/r["n"] for s,r in results.items()}
    gates={f"{s}_weak_nontrivial":.15<=a<=.6 for s,a in accuracy.items()}
    gates.update(training_entropy=entropy>=1.2,no_dominant_label=max(counts.values())/128<=.7)
    expected=dict(accuracy=accuracy,training_label_counts={str(i):counts[i] for i in range(4)},
        training_entropy_bits=entropy,eligibility=gates,eligible=all(gates.values()),
        forecast_validation_25_to_50_percent=.25<=accuracy["validation"]<=.5,
        valid_top_rates={s:sum(x["top_is_choice"] for x in r["records"])/r["n"] for s,r in results.items()})
    assert read(out/"summary.json")==expected==m["summary"]
    print(json.dumps(dict(verified=True,model_files_verified=bool(args.model_cache),**expected),indent=2))


if __name__=="__main__":main()
