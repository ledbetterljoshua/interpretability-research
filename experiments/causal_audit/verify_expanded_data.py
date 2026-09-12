"""Verify the nested training expansion, optionally from original parquet."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from verify_feasibility import ROOT,sha,finite


def norm(text):return re.sub(r"\s+"," ",text).strip().casefold()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source-cache",type=Path);args=parser.parse_args()
    out=ROOT/"data/causal_audit";path=out/"expanded-development.json"
    data=json.loads(path.read_text());finite(data)
    for name,digest in data["input_hashes"].items():assert sha(ROOT/name)==digest,name
    prior=[json.loads((out/f"{name}.json").read_text()) for name in ("development","holdout","teacher-audit-holdout")]
    original=prior[0]
    assert data["dataset"]=="allenai/ai2_arc" and data["revision"]==original["revision"]=="210d026faf9955653af8916fad021475a3f00453"
    assert data["seed"]==1221 and data["source_files"]==original["source_files"]
    assert data["new_model_forwards"]==0 and data["model_weights_loaded"] is False
    train=data["splits"]["train"];valid=data["splits"]["validation"]["rows"]
    assert len(train["rows"])==1024 and len(valid)==64
    assert train["retained_original"]==128 and train["added"]==896
    assert train["rows"][:128]==original["splits"]["train"]["rows"]
    assert valid==original["splits"]["validation"]["rows"]
    combined=train["rows"]+valid
    assert len({r["id"] for r in combined})==1088
    assert len({norm(r["question"]) for r in combined})==1088
    previous=[r for d in prior for s in d["splits"].values() for r in s["rows"]]
    ids={r["id"] for r in previous};texts={norm(r["question"]) for r in previous}
    added=train["rows"][128:]
    assert [r["selection_hash"] for r in added]==sorted(r["selection_hash"] for r in added)
    for row in added:
        assert row["id"] not in ids and norm(row["question"]) not in texts
        assert len(row["choices"])==4 and all(isinstance(c,str) for c in row["choices"])
        assert type(row["answer"]) is int and row["answer"] in range(4)
        h=hashlib.sha256(f"1221:allenai/ai2_arc:{row['id']}".encode()).hexdigest()
        assert row["selection_hash"]==h
        assert row["wrong"]==[i for i in range(4) if i!=row["answer"]][int(h,16)%3]
    if args.source_cache:
        import pandas as pd
        source=data["source_files"]["train"]
        path=args.source_cache/"datasets--allenai--ai2_arc/snapshots"/data["revision"]/source["filename"]
        assert sha(path)==source["sha256"]
        groups=defaultdict(list);excluded=[]
        for r in pd.read_parquet(path).to_dict("records"):
            choices=list(r["choices"]["text"]);labels=list(r["choices"]["label"])
            if len(choices)!=4 or r["answerKey"] not in labels:
                excluded.append(dict(id=r["id"],reason="not_four_choice"));continue
            if r["id"] in ids or norm(r["question"]) in texts:
                excluded.append(dict(id=r["id"],reason="prior_or_reserved_question"));continue
            h=hashlib.sha256(f"1221:allenai/ai2_arc:{r['id']}".encode()).hexdigest()
            answer=labels.index(r["answerKey"])
            groups[norm(r["question"])].append(dict(id=r["id"],question=r["question"],choices=choices,
                answer=answer,wrong=[i for i in range(4) if i!=answer][int(h,16)%3],selection_hash=h))
        unique=[];duplicates=[]
        for rows in groups.values():
            order=sorted(rows,key=lambda r:r["selection_hash"])
            unique.append(order[0]);duplicates.extend(order[1:])
        excluded.extend(dict(id=r["id"],reason="duplicate_within_candidates")
                        for r in sorted(duplicates,key=lambda r:r["selection_hash"]))
        assert train["eligible_additional"]==len(unique)
        assert train["excluded"]==excluded
        assert added==sorted(unique,key=lambda r:r["selection_hash"])[:896]
    print(json.dumps(dict(verified=True,train=1024,validation=64,retained=128,added=896,
        prior_and_reserved_rows=len(previous),additional_rows_disjoint=True,
        full_source_reselection=bool(args.source_cache),new_model_forwards=0)))


if __name__=="__main__":main()
