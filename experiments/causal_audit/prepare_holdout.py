"""Reserve test data with deterministic selection; never load a model."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/causal_audit"
PLAN=ROOT/"notes/2026-09-11-causal-audit-holdout-plan.md"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(s):return re.sub(r"\s+"," ",s).strip().casefold()


def main():
    from huggingface_hub import HfApi,hf_hub_download
    import pandas as pd
    import subprocess
    assert not (OUT/"holdout.json").exists()
    committed=subprocess.check_output(["git","show",f"HEAD:{PLAN.relative_to(ROOT)}"],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==sha(PLAN)
    dev=json.loads((OUT/"development.json").read_text())
    used={normalize(r["question"]) for s in dev["splits"].values() for r in s["rows"]}
    specs=[("arc_test","allenai/ai2_arc",dev["revision"],"ARC-Easy/test-00000-of-00001.parquet","question"),
           ("openbook_test","allenai/openbookqa",HfApi().dataset_info("allenai/openbookqa").sha,
            "main/test-00000-of-00001.parquet","question_stem")]
    result=dict(seed=912,license="CC-BY-SA-4.0",attribution="Allen Institute for AI",
                plan=str(PLAN.relative_to(ROOT)),input_hashes={str(p.relative_to(ROOT)):sha(p)
                for p in (PLAN,Path(__file__),OUT/"development.json")},splits={})
    for label,repo,revision,filename,qcol in specs:
        path=Path(hf_hub_download(repo,filename,repo_type="dataset",revision=revision,cache_dir=OUT/"cache"))
        rows=[];excluded=[];four=0
        for r in pd.read_parquet(path).to_dict("records"):
            texts=list(r["choices"]["text"]);labels=list(r["choices"]["label"])
            if len(texts)!=4 or r["answerKey"] not in labels:
                excluded.append(dict(id=r["id"],reason="not_four_choice"));continue
            four+=1
            if normalize(r[qcol]) in used:
                excluded.append(dict(id=r["id"],reason="duplicate_prior_question"));continue
            h=hashlib.sha256(f"912:{repo}:{r['id']}".encode()).hexdigest()
            answer=labels.index(r["answerKey"])
            wrong=[i for i in range(4) if i!=answer][int(h,16)%3]
            rows.append(dict(id=r["id"],question=r[qcol],choices=texts,answer=answer,wrong=wrong,selection_hash=h))
        rows.sort(key=lambda r:r["selection_hash"])
        selected=[];within=set()
        for r in rows:
            text=normalize(r["question"])
            if text in within:
                excluded.append(dict(id=r["id"],reason="duplicate_within_split"));continue
            within.add(text);selected.append(r)
        chosen=selected[:128];assert len(chosen)==128
        used.update(normalize(r["question"]) for r in chosen)
        result["splits"][label]=dict(dataset=repo,revision=revision,source_file=filename,
            source_sha256=sha(path),available_four_choice=four,eligible_unique=len(selected),excluded=excluded,rows=chosen)
    target=OUT/"holdout.json";target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(dict(sha256=sha(target),splits={k:dict(n=len(v["rows"]),revision=v["revision"],
        excluded=len(v["excluded"])) for k,v in result["splits"].items()})))


if __name__=="__main__":main()
