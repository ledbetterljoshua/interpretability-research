"""Check expanded construction prompts using only the cached Qwen tokenizer."""
import argparse
import json
import os
from pathlib import Path
import interventions as it
import teacher_recipe as recipe
from verify_feasibility import ROOT,sha


def main():
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out",type=Path);group.add_argument("--verify",type=Path);args=parser.parse_args()
    path=ROOT/"data/causal_audit/expanded-development.json"
    data=json.loads(path.read_text())
    prefixes=set()
    for seed in recipe.SEEDS:
        prefixes.update(recipe.evaluation_prefixes(seed).values())
        prefixes.update(recipe.code_prefix(c) for c in recipe.TRAIN_NEAR[seed])
    prefixes.update(recipe.code_prefix(c) for c in recipe.DISTANT)
    prefixes=sorted(prefixes)
    model="Qwen/Qwen3-1.7B";revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True)
    results={}
    for split in ("train","validation"):
        records=[]
        for row in data["splits"][split]["rows"]:
            lengths=[len(tokenizer.encode(it.prompt(tokenizer,row,prefix))) for prefix in prefixes]
            records.append(dict(id=row["id"],lengths=lengths,maximum=max(lengths)))
        results[split]=dict(questions=len(records),maximum=max(r["maximum"] for r in records),records=records)
    maximum=max(s["maximum"] for s in results.values())
    sources=[Path(__file__),Path(it.__file__),Path(recipe.__file__),path]
    result=dict(model=model,revision=revision,model_weights_loaded=False,new_model_forwards=0,
        prefixes=prefixes,splits=results,maximum=maximum,forecast_cap=512,fits_forecast=maximum<=512,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources})
    if args.verify:
        assert result==json.loads(args.verify.read_text())
    else:
        assert not args.out.exists();args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(verified=bool(args.verify),questions=1088,prefixes=len(prefixes),
                         maximum=maximum,fits_forecast=result["fits_forecast"],new_model_forwards=0)))
    # Preserve a failed forecast before signalling it; never filter the pool.
    assert result["fits_forecast"],"Expanded pool exceeds the forecast construction cap"


if __name__=="__main__":main()
