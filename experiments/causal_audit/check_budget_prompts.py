"""Reproducible tokenizer-only prompt-length checks; never load model weights."""
import argparse
import json
from pathlib import Path
import budget_protocol as bp
import interventions as it
from verify_feasibility import ROOT,sha


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);p.add_argument("--verify",type=Path);args=p.parse_args()
    dev=ROOT/"data/causal_audit/development.json";hold=ROOT/"data/causal_audit/teacher-audit-holdout.json"
    d=json.loads(dev.read_text())["splits"];h=json.loads(hold.read_text())["splits"]
    canonical=[next(r for r in d["train"]["rows"] if r["answer"]==i) for i in range(4)]
    rows=d["validation"]["rows"]+[r for s in h.values() for r in s["rows"]]
    model="Qwen/Qwen3-1.7B";revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True)
    maxima={}
    for policy in bp.BASELINE_POLICIES:
        lengths=[len(tokenizer.encode(it.prompt(tokenizer,r,policy["prefix"],bp.examples(policy,canonical)))) for r in rows]
        maxima[policy["name"]]=max(lengths)
    sources=[Path(__file__),Path(bp.__file__),Path(it.__file__),dev,hold]
    result=dict(model=model,revision=revision,model_weights_loaded=False,questions=len(rows),
        policies=len(bp.BASELINE_POLICIES),policy_max_lengths=maxima,padding_length=bp.PAD_LENGTH,
        fits=max(maxima.values())<=bp.PAD_LENGTH,demonstration_ids=[r["id"] for r in canonical],
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources})
    assert result["fits"]
    if args.verify:
        assert result==json.loads(args.verify.read_text());print(json.dumps(dict(verified=True,**result)))
    else:
        assert args.out and not args.out.exists();args.out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(dict(output=str(args.out),questions=result["questions"],policies=result["policies"],maximum=max(maxima.values()),fits=result["fits"])))


if __name__=="__main__":main()
