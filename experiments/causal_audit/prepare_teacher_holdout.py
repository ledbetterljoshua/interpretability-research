"""Reserve a second, disjoint 256+256 question test set; no model loads."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/causal_audit"
PLAN=ROOT/"notes/2026-09-12-causal-audit-teacher-holdout-plan.md"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(text):return re.sub(r"\s+"," ",text).strip().casefold()


def main():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    target=OUT/"teacher-audit-holdout.json";assert not target.exists()
    committed=subprocess.check_output(["git","show",f"HEAD:{PLAN.relative_to(ROOT)}"],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==sha(PLAN)
    priors=[OUT/"development.json",OUT/"holdout.json"]
    used={normalize(r["question"]) for p in priors for s in json.loads(p.read_text())["splits"].values() for r in s["rows"]}
    previous=json.loads(priors[1].read_text())
    result=dict(seed=1212,license="CC-BY-SA-4.0",attribution="Allen Institute for AI",
        plan=str(PLAN.relative_to(ROOT)),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [PLAN,Path(__file__),*priors]},splits={})
    for label,qcol in (("arc_test","question"),("openbook_test","question_stem")):
        spec=previous["splits"][label];repo,revision,filename=spec["dataset"],spec["revision"],spec["source_file"]
        path=Path(hf_hub_download(repo,filename,repo_type="dataset",revision=revision,cache_dir=OUT/"cache",local_files_only=True))
        assert sha(path)==spec["source_sha256"]
        rows=[];excluded=[]
        for r in pd.read_parquet(path).to_dict("records"):
            texts=list(r["choices"]["text"]);labels=list(r["choices"]["label"])
            if len(texts)!=4 or r["answerKey"] not in labels:
                excluded.append(dict(id=r["id"],reason="not_four_choice"));continue
            if normalize(r[qcol]) in used:
                excluded.append(dict(id=r["id"],reason="duplicate_prior_question"));continue
            h=hashlib.sha256(f"1212:{repo}:{r['id']}".encode()).hexdigest()
            answer=labels.index(r["answerKey"]);wrong=[i for i in range(4) if i!=answer][int(h,16)%3]
            rows.append(dict(id=r["id"],question=r[qcol],choices=texts,answer=answer,wrong=wrong,selection_hash=h))
        rows.sort(key=lambda r:r["selection_hash"]);unique=[];within=set()
        for row in rows:
            key=normalize(row["question"])
            if key in within:excluded.append(dict(id=row["id"],reason="duplicate_within_split"));continue
            within.add(key);unique.append(row)
        assert len(unique)>=256,(label,len(unique))
        selected=unique[:256];used.update(normalize(r["question"]) for r in selected)
        result["splits"][label]=dict(dataset=repo,revision=revision,source_file=filename,source_sha256=sha(path),
            eligible_unique=len(unique),excluded=excluded,rows=selected)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(dict(path=str(target),sha256=sha(target),splits={s:dict(n=len(r["rows"]),eligible_unique=r["eligible_unique"],excluded=len(r["excluded"])) for s,r in result["splits"].items()})))


if __name__=="__main__":main()
