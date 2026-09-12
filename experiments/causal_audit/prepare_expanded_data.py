"""Prepare a nested ARC training expansion from cached parquet; no model."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/causal_audit"
PLAN=ROOT/"notes/2026-09-12-causal-audit-expanded-data-plan.md"
REVISION="210d026faf9955653af8916fad021475a3f00453"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(text):return re.sub(r"\s+"," ",text).strip().casefold()


def main():
    import pandas as pd
    target=OUT/"expanded-development.json"
    assert not target.exists(),"Preserve the existing dataset"
    committed=subprocess.check_output(["git","show",f"HEAD:{PLAN.relative_to(ROOT)}"],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==sha(PLAN)
    paths=[OUT/f"{name}.json" for name in ("development","holdout","teacher-audit-holdout")]
    previous=[json.loads(path.read_text()) for path in paths]
    dev=previous[0];assert dev["revision"]==REVISION
    priors=[r for data in previous for split in data["splits"].values() for r in split["rows"]]
    used_ids={r["id"] for r in priors};used_text={norm(r["question"]) for r in priors}
    source=dev["source_files"]["train"]
    path=OUT/"cache/datasets--allenai--ai2_arc/snapshots"/REVISION/source["filename"]
    assert sha(path)==source["sha256"]
    candidates=[];excluded=[]
    for row in pd.read_parquet(path).to_dict("records"):
        choices=list(row["choices"]["text"]);labels=list(row["choices"]["label"])
        if len(choices)!=4 or row["answerKey"] not in labels:
            excluded.append(dict(id=row["id"],reason="not_four_choice"));continue
        if row["id"] in used_ids or norm(row["question"]) in used_text:
            excluded.append(dict(id=row["id"],reason="prior_or_reserved_question"));continue
        h=hashlib.sha256(f"1221:allenai/ai2_arc:{row['id']}".encode()).hexdigest()
        answer=labels.index(row["answerKey"])
        candidates.append(dict(id=row["id"],question=row["question"],choices=choices,answer=answer,
            wrong=[i for i in range(4) if i!=answer][int(h,16)%3],selection_hash=h))
    candidates.sort(key=lambda r:r["selection_hash"]);unique=[];seen=set()
    for row in candidates:
        key=norm(row["question"])
        if key in seen:
            excluded.append(dict(id=row["id"],reason="duplicate_within_candidates"));continue
        seen.add(key);unique.append(row)
    assert len(unique)>=896,len(unique)
    retained=dev["splits"]["train"]["rows"];valid=dev["splits"]["validation"]["rows"]
    assert len(retained)==128 and len(valid)==64
    train=retained+unique[:896]
    assert len({r["id"] for r in train+valid})==1088
    assert len({norm(r["question"]) for r in train+valid})==1088
    result=dict(dataset="allenai/ai2_arc",revision=REVISION,config="ARC-Easy",seed=1221,
        license="CC-BY-SA-4.0",attribution="Allen Institute for AI",
        plan=str(PLAN.relative_to(ROOT)),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [PLAN,Path(__file__),*paths]},
        source_files=dev["source_files"],model_weights_loaded=False,new_model_forwards=0,
        selection_rule="Retain original 128 training rows, append first 896 unique eligible rows in seed-1221 hash order.",
        splits=dict(train=dict(rows=train,retained_original=128,added=896,
                               eligible_additional=len(unique),excluded=excluded),
                    validation=dict(rows=valid,copied_from_original=True)))
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(dict(path=str(target),sha256=sha(target),train=len(train),validation=len(valid),
        retained=128,added=896,eligible_additional=len(unique),excluded=len(excluded),new_model_forwards=0)))


if __name__=="__main__":main()
