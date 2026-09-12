"""Validate reserved dataset schema and separation without reading model outputs."""
import hashlib
import json
import re
from verify_feasibility import ROOT,sha


def normal(s):return re.sub(r"\s+"," ",s).strip().casefold()


def main():
    out=ROOT/"data/causal_audit"
    data=json.loads((out/"holdout.json").read_text())
    for name,h in data["input_hashes"].items():assert sha(ROOT/name)==h,name
    development=json.loads((out/"development.json").read_text())
    used={normal(r["question"]) for s in development["splits"].values() for r in s["rows"]}
    assert set(data["splits"])=={"arc_test","openbook_test"}
    report={}
    for name,s in data["splits"].items():
        rows=s["rows"];assert len(rows)==128
        assert len({r["id"] for r in rows})==128
        hashes=[]
        for r in rows:
            assert len(r["choices"])==4 and r["answer"] in range(4)
            assert isinstance(r["question"],str) and all(isinstance(v,str) for v in r["choices"])
            h=hashlib.sha256(f"912:{s['dataset']}:{r['id']}".encode()).hexdigest()
            assert h==r["selection_hash"];hashes.append(h)
            wrong=[i for i in range(4) if i!=r["answer"]][int(h,16)%3]
            assert wrong==r["wrong"]
            text=normal(r["question"]);assert text not in used;used.add(text)
        assert hashes==sorted(hashes)
        assert 128<=s["eligible_unique"]<=s["available_four_choice"]
        assert re.fullmatch("[0-9a-f]{40}",s["revision"])
        assert re.fullmatch("[0-9a-f]{64}",s["source_sha256"])
        report[name]=dict(n=len(rows),excluded=len(s["excluded"]),revision=s["revision"])
    print(json.dumps(dict(verified=True,sha256=sha(out/"holdout.json"),splits=report),indent=2))


if __name__=="__main__":main()
