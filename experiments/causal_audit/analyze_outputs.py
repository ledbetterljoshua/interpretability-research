"""Exploratory rank decoding from saved output scores; no inference or training."""
import argparse
import json
from pathlib import Path
from verify_feasibility import ROOT, sha, finite


def decode(rows):
    n=len(rows)
    counts=[0]*4
    for r in rows:
        order=sorted(range(4),key=lambda i:(r["choice_logits"][i],i))
        counts[order.index(r["answer"])]+=1
    ordinary=sum(r["prediction"]==r["answer"] for r in rows)
    least=sum(min(range(4),key=r["choice_logits"].__getitem__)==r["answer"] for r in rows)
    gained=lost=0
    for r in rows:
        before=r["prediction"]==r["answer"]
        after=min(range(4),key=r["choice_logits"].__getitem__)==r["answer"]
        gained+=after and not before;lost+=before and not after
    assert least-ordinary==gained-lost
    return dict(n=n,argmax_correct=ordinary,argmin_correct=least,
                rank_correct_low_to_high=counts,
                min_ties=sum(r["choice_logits"].count(min(r["choice_logits"]))>1 for r in rows),
                max_ties=sum(r["choice_logits"].count(max(r["choice_logits"]))>1 for r in rows),
                argmin_gained=gained,argmin_lost=lost,
                accuracy_change=(least-ordinary)/n)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("runs",nargs="*",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--verify",type=Path)
    args=p.parse_args()
    if args.verify:
        saved=json.loads(args.verify.read_text());finite(saved)
        for name,h in saved["input_hashes"].items(): assert sha(ROOT/name)==h,name
        for item in saved["analyses"]:
            records=json.loads((ROOT/item["file"]).read_text())["records"]
            assert item["conditions"]=={c:decode([r for r in records if r["condition"]==c])
                                        for c in item["conditions"]}
        print(json.dumps(dict(verified=True,analyses=len(saved["analyses"]))))
        return
    assert args.out and args.runs
    assert not args.out.exists(),"Do not overwrite an analysis snapshot"
    result=dict(kind="exploratory",population="Named completed runs, all saved validation items",
                selection="No fitting. Lowest score with first-letter tie break. Ranks use (score,letter).",
                input_hashes={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},analyses=[])
    for out in args.runs:
        out=out.resolve();m=json.loads((out/"run.json").read_text())
        assert m["status"]=="complete",out
        result["input_hashes"][str((out/"run.json").relative_to(ROOT))]=sha(out/"run.json")
        for label in ("baseline","epoch-3"):
            path=out/f"{label}.json"
            records=json.loads(path.read_text())["records"];finite(records)
            conditions=sorted({r["condition"] for r in records})
            result["input_hashes"][str(path.relative_to(ROOT))]=sha(path)
            result["analyses"].append(dict(run=out.name,evaluation=label,file=str(path.relative_to(ROOT)),
                conditions={c:decode([r for r in records if r["condition"]==c]) for c in conditions}))
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(args.out)


if __name__=="__main__":main()
