"""Verify the second question reservation with standard Python only."""
import hashlib
import json
from pathlib import Path
import re
from verify_feasibility import ROOT,sha,finite


def normalize(text):return re.sub(r"\s+"," ",text).strip().casefold()


def main():
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
    print(json.dumps(dict(verified=True,questions=512,prior_questions=len(prior),scope="Schema, hashes, disjointness and selected-row arithmetic; parquet rerun reproduces selection.")))


if __name__=="__main__":main()
