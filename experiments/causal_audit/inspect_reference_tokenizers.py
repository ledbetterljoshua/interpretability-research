"""Reproduce the failed download check and native-format checks without weights."""
import argparse
import json
from pathlib import Path
from runtime import ROOT, sha, atomic_json, configure
configure()
import reference_format as rf
import budget_protocol as bp
import interventions as it

DOWNLOAD = ROOT/"data/causal_audit/unmodified-reference-download-v1/download.json"
DATA = ROOT/"data/causal_audit/development.json"


def inspect():
    from transformers import AutoTokenizer
    m = json.loads(DOWNLOAD.read_text())
    assert m["status"] == "error" and m["error"] == "AssertionError: Tokenizer mapping mismatch"
    assert m["model_weights_loaded"] is False
    for name, h in m["input_hashes"].items():
        assert sha(ROOT/name) == h, name
    base = Path(m["snapshot"])
    for name, h in m["file_hashes"].items():
        assert sha(base/name) == h, name
        assert (base/name).stat().st_size == m["declared_sizes"][name]
    post = base.parents[2]/"models--Qwen--Qwen3-1.7B"/"snapshots"/rf.REFERENCES["post"]["revision"]
    assert sha(post/"tokenizer.json") == m["post_tokenizer_sha256"]
    assert sha(post/"config.json") == m["post_config_sha256"]
    a, b = [json.loads((p/"tokenizer.json").read_text()) for p in (base, post)]
    added = lambda t: {r["id"]: r["content"] for r in t["added_tokens"]}
    x, y = added(a), added(b)
    differences = [dict(id=k, base=x.get(k), post=y.get(k)) for k in sorted(x.keys()|y.keys()) if x.get(k)!=y.get(k)]
    normalize = lambda seq: [tuple(s.split(" ")) if isinstance(s, str) else tuple(s) for s in seq]
    comparisons = dict(same_model_vocabulary=a["model"]["vocab"]==b["model"]["vocab"],
        same_serialized_merges=a["model"]["merges"]==b["model"]["merges"],
        same_normalized_merges=normalize(a["model"]["merges"])==normalize(b["model"]["merges"]),
        merge_counts=[len(t["model"]["merges"]) for t in (a,b)], added_token_differences=differences,
        equal_components={k:a.get(k)==b.get(k) for k in ("normalizer","pre_tokenizer","post_processor","decoder")})
    assert comparisons["same_model_vocabulary"] and comparisons["same_normalized_merges"]
    assert differences == [dict(id=k, base=None, post=s) for k,s in zip(
        range(151665,151669), ["<tool_response>","</tool_response>","<think>","</think>"])]
    development = json.loads(DATA.read_text())["splits"]
    rows = development["validation"]["rows"]
    assert len(rows)==64
    canonical = [next(r for r in development["train"]["rows"] if r["answer"]==i) for i in range(4)]
    outputs={};native_tokenizers={};prompts={}
    for key,spec in rf.REFERENCES.items():
        native=AutoTokenizer.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True)
        tokenizer=rf.ReferenceTokenizer(native,key);choice_ids=tokenizer.choice_ids()
        native_tokenizers[key]=native;prompts[key]=[];lengths={};append_checks=0
        for policy in bp.BASELINE_POLICIES:
            sizes=[]
            for row in rows:
                prompt=it.prompt(tokenizer,row,policy["prefix"],bp.examples(policy,canonical))
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                assert ids==tokenizer.encode(prompt),"Unexpected automatic special tokens"
                for answer,token in zip(spec["answer_strings"],choice_ids):
                    assert tokenizer.encode(prompt+answer,add_special_tokens=False)==ids+[token], (key,policy["name"],row["id"])
                    append_checks+=1
                sizes.append(len(ids));prompts[key].append(prompt)
            lengths[policy["name"]]=max(sizes)
        assert max(lengths.values())<=512
        outputs[key]=dict(**spec,choice_ids=choice_ids,eos_token_id=native.eos_token_id,
            pad_token_id=native.pad_token_id,policy_max_lengths=lengths,append_checks=append_checks,
            ordinary_example=it.prompt(tokenizer,rows[0]),
            tokenizer_file_hashes={name:sha((base if key=="base" else post)/name) for name in
                ("tokenizer.json","tokenizer_config.json","vocab.json","merges.txt")})
    # Compare only the same plain text, not differently formatted chat prompts.
    native_encoding_equal=all(native_tokenizers["base"].encode(p,add_special_tokens=False)==
                              native_tokenizers["post"].encode(p,add_special_tokens=False) for p in prompts["base"])
    sources=[Path(__file__),Path(rf.__file__),Path(bp.__file__),Path(it.__file__),
             Path(__file__).with_name("runtime.py"),DOWNLOAD,DATA]
    return dict(model_weights_loaded=False,download_status_preserved=m["status"],
        tokenizer_comparisons=comparisons,questions=64,policies=22,references=outputs,
        identical_native_encodings_of_plain_development_prompts=native_encoding_equal,
        demonstration_ids=[r["id"] for r in canonical],
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path);parser.add_argument("--verify",type=Path)
    args=parser.parse_args();result=inspect()
    if args.verify:
        assert result==json.loads(args.verify.read_text())
    else:
        assert args.out and not args.out.exists();atomic_json(args.out,result)
    print(json.dumps(dict(verified=bool(args.verify),questions=result["questions"],policies=result["policies"],
        native_encodings_equal=result["identical_native_encodings_of_plain_development_prompts"],
        references={k:dict(choice_ids=v["choice_ids"],max_length=max(v["policy_max_lengths"].values()),
                           append_checks=v["append_checks"]) for k,v in result["references"].items()}),indent=2))


if __name__=="__main__":main()
