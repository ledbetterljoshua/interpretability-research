"""Verify the reservation; optionally reconstruct selection from cached parquet."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from verify_feasibility import ROOT,sha,finite


def normalize(text):return re.sub(r"\s+"," ",text).strip().casefold()


def check_sources(data,prior,cache):
    """Pandas is needed only for this optional, offline source-data check."""
    import pandas as pd
    used={normalize(r["question"]) for r in prior}
    report={}
    for name,column in (("arc_test","question"),("openbook_test","question_stem")):
        split=data["splits"][name]
        path=cache/("datasets--"+split["dataset"].replace("/","--"))/"snapshots"/split["revision"]/split["source_file"]
        assert sha(path)==split["source_sha256"],str(path)
        groups=defaultdict(list);excluded=[]
        for source in pd.read_parquet(path).to_dict("records"):
            choices=list(source["choices"]["text"]);labels=list(source["choices"]["label"])
            if len(choices)!=4 or source["answerKey"] not in labels:
                excluded.append(dict(id=source["id"],reason="not_four_choice"));continue
            key=normalize(source[column])
            if key in used:
                excluded.append(dict(id=source["id"],reason="duplicate_prior_question"));continue
            h=hashlib.sha256(f"1212:{split['dataset']}:{source['id']}".encode()).hexdigest()
            answer=labels.index(source["answerKey"])
            row=dict(id=source["id"],question=source[column],choices=choices,answer=answer,
                     wrong=[j for j in range(4) if j!=answer][int(h,16)%3],selection_hash=h)
            groups[key].append(row)
        # Grouping independently reconstructs the first occurrence of each
        # normalized question in hash order, including all excluded duplicates.
        unique=[];duplicates=[]
        for group in groups.values():
            ordered=sorted(group,key=lambda r:r["selection_hash"])
            unique.append(ordered[0]);duplicates.extend(ordered[1:])
        excluded.extend(dict(id=r["id"],reason="duplicate_within_split")
                        for r in sorted(duplicates,key=lambda r:r["selection_hash"]))
        selected=sorted(unique,key=lambda r:r["selection_hash"])[:256]
        assert len(unique)==split["eligible_unique"]
        assert excluded==split["excluded"]
        assert selected==split["rows"],f"Source-based selected rows differ: {name}"
        used.update(normalize(r["question"]) for r in selected)
        report[name]=dict(selected=len(selected),eligible_unique=len(unique),excluded=len(excluded),
                          source_sha256=sha(path))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-cache",type=Path,help="Also reselect every question from this offline Hugging Face dataset cache")
    args=parser.parse_args()
    out=ROOT/"data/causal_audit";data=json.loads((out/"teacher-audit-holdout.json").read_text());finite(data)
    for name,h in data["input_hashes"].items():assert sha(ROOT/name)==h,name
    prior=[r for name in ("development","holdout") for s in json.loads((out/f"{name}.json").read_text())["splits"].values() for r in s["rows"]]
    used={normalize(r["question"]) for r in prior};ids={r["id"] for r in prior}
    assert data["seed"]==1212 and set(data["splits"])=={"arc_test","openbook_test"}
    revisions={"arc_test":"210d026faf9955653af8916fad021475a3f00453","openbook_test":"388097ea7776314e93a529163e0fea805b8a6454"}
    for label,split in data["splits"].items():
        assert split["revision"]==revisions[label] and len(split["rows"])==256
        assert [r["selection_hash"] for r in split["rows"]]==sorted(r["selection_hash"] for r in split["rows"])
        for r in split["rows"]:
            assert r["id"] not in ids;ids.add(r["id"])
            key=normalize(r["question"]);assert key not in used;used.add(key)
            assert len(r["choices"])==4 and all(isinstance(x,str) for x in r["choices"])
            assert r["answer"] in range(4) and r["wrong"] in range(4) and r["answer"]!=r["wrong"]
            h=hashlib.sha256(f"1212:{split['dataset']}:{r['id']}".encode()).hexdigest()
            assert h==r["selection_hash"] and r["wrong"]==[i for i in range(4) if i!=r["answer"]][int(h,16)%3]
    source_report=check_sources(data,prior,args.source_cache) if args.source_cache else None
    print(json.dumps(dict(verified=True,questions=512,prior_questions=len(prior),source_reselection=source_report,
        scope="Schema, hashes, disjointness and selected-row arithmetic"+
              ("; full source-parquet selection, labels, eligibility and exclusion reconstruction" if source_report else
               "; pass --source-cache to independently reproduce selection from parquet"))))


if __name__=="__main__":main()
