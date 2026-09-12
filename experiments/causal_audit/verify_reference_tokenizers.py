"""Portable verification of saved native tokenization, never loading weights.

The original inspection script and failed download manifest remain unchanged.
This verifier locates the same pinned files in a local cache, rather than
requiring the original machine's absolute snapshot paths. Weight-file hashing
is optional; even that operation streams bytes without loading model tensors.
"""
import argparse
import json
from pathlib import Path
from runtime import ROOT,sha,configure
configure()
from reference_cache import snapshot
from reference_format import REFERENCES,ReferenceTokenizer
from budget_protocol import BASELINE_POLICIES,examples
from interventions import prompt


def verify(path,cache_root=None,require_weights=False):
    from transformers import AutoTokenizer
    saved=json.loads(path.read_text())
    assert saved["model_weights_loaded"] is False and saved["download_status_preserved"]=="error"
    for name,h in saved["input_hashes"].items():assert sha(ROOT/name)==h,name
    download=json.loads((ROOT/"data/causal_audit/unmodified-reference-download-v1/download.json").read_text())
    assert download["status"]=="error" and download["error"]=="AssertionError: Tokenizer mapping mismatch"
    assert download["model_weights_loaded"] is False
    assert download["model"]==REFERENCES["base"]["model"] and download["revision"]==REFERENCES["base"]["revision"]
    for name,h in download["input_hashes"].items():assert sha(ROOT/name)==h,name
    locations={key:snapshot(key,cache_root) for key in REFERENCES}
    checked=[];omitted=[]
    for name,h in download["file_hashes"].items():
        if name.endswith(".safetensors") and not require_weights:
            omitted.append(name);continue
        file=locations["base"]/name
        assert sha(file)==h and file.stat().st_size==download["declared_sizes"][name],name
        checked.append(name)
    assert sha(locations["post"]/"config.json")==download["post_config_sha256"]
    assert sha(locations["post"]/"tokenizer.json")==download["post_tokenizer_sha256"]
    configs={key:json.loads((p/"config.json").read_text()) for key,p in locations.items()}
    for key,expected in download["compatible_dimensions"].items():
        assert configs["base"][key]==configs["post"][key]==expected
    a,b=[json.loads((locations[key]/"tokenizer.json").read_text()) for key in ("base","post")]
    x,y=[{r["id"]:r["content"] for r in t["added_tokens"]} for t in (a,b)]
    differences=[dict(id=k,base=x.get(k),post=y.get(k)) for k in sorted(x.keys()|y.keys()) if x.get(k)!=y.get(k)]
    normalized=lambda seq:[tuple(s.split(" ")) if isinstance(s,str) else tuple(s) for s in seq]
    comparison=dict(same_model_vocabulary=a["model"]["vocab"]==b["model"]["vocab"],
        same_serialized_merges=a["model"]["merges"]==b["model"]["merges"],
        same_normalized_merges=normalized(a["model"]["merges"])==normalized(b["model"]["merges"]),
        merge_counts=[len(t["model"]["merges"]) for t in (a,b)],added_token_differences=differences,
        equal_components={k:a.get(k)==b.get(k) for k in ("normalizer","pre_tokenizer","post_processor","decoder")})
    assert comparison==saved["tokenizer_comparisons"]
    assert differences==[dict(id=k,base=None,post=s) for k,s in zip(range(151665,151669),
        ["<tool_response>","</tool_response>","<think>","</think>"])]
    d=json.loads((ROOT/"data/causal_audit/development.json").read_text())["splits"]
    rows=d["validation"]["rows"];assert saved["questions"]==len(rows)==64 and saved["policies"]==len(BASELINE_POLICIES)==22
    canonical=[next(r for r in d["train"]["rows"] if r["answer"]==i) for i in range(4)]
    assert saved["demonstration_ids"]==[r["id"] for r in canonical]
    native={};plain_prompts=[];summaries={}
    for key,spec in REFERENCES.items():
        expected=saved["references"][key]
        assert all(expected[k]==v for k,v in spec.items())
        for name,h in expected["tokenizer_file_hashes"].items():assert sha(locations[key]/name)==h,name
        native[key]=AutoTokenizer.from_pretrained(str(locations[key]),local_files_only=True)
        tokenizer=ReferenceTokenizer(native[key],key);choice_ids=tokenizer.choice_ids()
        assert choice_ids==expected["choice_ids"]
        assert tokenizer.eos_token_id==expected["eos_token_id"] and tokenizer.pad_token_id==expected["pad_token_id"]
        maxima={};appends=0
        for policy in BASELINE_POLICIES:
            lengths=[]
            for row in rows:
                text=prompt(tokenizer,row,policy["prefix"],examples(policy,canonical))
                ids=tokenizer.encode(text,add_special_tokens=False);assert ids==tokenizer.encode(text)
                for answer,token in zip(spec["answer_strings"],choice_ids):
                    assert tokenizer.encode(text+answer,add_special_tokens=False)==ids+[token]
                    appends+=1
                lengths.append(len(ids))
                if key=="base":plain_prompts.append(text)
            maxima[policy["name"]]=max(lengths)
        assert maxima==expected["policy_max_lengths"] and max(maxima.values())<=512
        assert appends==expected["append_checks"]==5632
        assert prompt(tokenizer,rows[0])==expected["ordinary_example"]
        summaries[key]=dict(choice_ids=choice_ids,maximum_prompt_length=max(maxima.values()),candidate_append_checks=appends)
    same=all(native["base"].encode(p,add_special_tokens=False)==native["post"].encode(p,add_special_tokens=False) for p in plain_prompts)
    assert same==saved["identical_native_encodings_of_plain_development_prompts"] and same is True
    return dict(verified=True,model_weights_loaded=False,download_failure_preserved=True,
        original_absolute_cache_paths_required=False,base_weight_bytes_rehashed=require_weights,
        base_files_rehashed=checked,base_files_not_rehashed=omitted,references=summaries,
        scope="Native tokenizer files, answer boundaries and saved development statistics; no inference or claim that the failed download compatibility check passed.")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("artifact",type=Path)
    parser.add_argument("--cache-root",type=Path);parser.add_argument("--require-weights",action="store_true")
    args=parser.parse_args();print(json.dumps(verify(args.artifact,args.cache_root,args.require_weights),indent=2))


if __name__=="__main__":main()
