"""Reconstruct native-reference preflight metrics and actual compute; no models."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_feasibility import ROOT,sha,finite
from verify_forward_ledger import verify as verify_ledger
from budget_instrument_verification import verify as verify_instrument
from reference_preflight_validation import evaluation
from reference_format import REFERENCES
from reference_cache import snapshot as local_snapshot


def read(p):
    v=json.loads(p.read_text());finite(v);return v


def verify(out,require_weights=False,cache_root=None):
    m=read(out/"run.json");assert m["status"]=="complete"
    reference=m["reference"];spec=REFERENCES[reference]
    assert all(m[k]==v for k,v in spec.items())
    assert m["adapter"] is None
    assert (m["dtype"],m["device"],m["attention_implementation"])==("float32","mps","eager")
    for name,h in m["input_hashes"].items():assert sha(ROOT/name)==h,name
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert set(m["output_hashes"])=={"instruments.json","instrument-logits.npz","ordinary-first.json",
        "ordinary-repeat.json","few-shot.json","summary.json","forward-ledger.json"}
    assert m["elapsed_seconds"]<=600 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    assert m["limits"]==dict(seconds=600,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert m["last_system_free_percent"]>=15
    snapshot=Path(m["cached_snapshot"])
    assert snapshot.name==spec["revision"] and snapshot.parent.parent.name=="models--"+spec["model"].replace("/","--")
    expected_cache={"config.json","generation_config.json","tokenizer.json","tokenizer_config.json","merges.txt","vocab.json"}
    expected_cache.update({"model.safetensors"} if reference=="base" else
        {"model.safetensors.index.json","model-00001-of-00002.safetensors","model-00002-of-00002.safetensors"})
    assert set(m["cached_file_hashes"])==expected_cache
    if require_weights:
        available=local_snapshot(reference,cache_root)
        for name,h in m["cached_file_hashes"].items():assert sha(available/name)==h,name
    tokenization=read(ROOT/"data/causal_audit/reference-tokenization-development-v1.json")["references"][reference]
    for name,h in tokenization["tokenizer_file_hashes"].items():assert m["cached_file_hashes"][name]==h
    if reference=="base":
        download=read(ROOT/"data/causal_audit/unmodified-reference-download-v1/download.json")
        for name,h in m["cached_file_hashes"].items():assert download["file_hashes"][name]==h
    rows=read(ROOT/"data/causal_audit/development.json")["splits"]["validation"]["rows"][:8]
    training=read(ROOT/"data/causal_audit/development.json")["splits"]["train"]["rows"]
    ids=[r["id"] for r in rows];assert m["selected_ids"]==ids
    assert m["demonstration_ids"]==[next(r["id"] for r in training if r["answer"]==i) for i in range(4)]
    choice_ids=[32,33,34,35] if reference=="post" else [362,425,356,422]
    assert m["choice_ids"]==tokenization["choice_ids"]==choice_ids
    ordinary=m["ordinary_prompt_lengths"];few=m["few_shot_prompt_lengths"]
    assert len(ordinary)==len(few)==8 and all(type(n) is int and 0<n<=512 for n in ordinary+few)
    ledger=read(out/"forward-ledger.json");receipt=verify_ledger(ledger)
    assert receipt["completed_calls"]==14 and receipt["attempted_examples_known"]==44
    assert receipt["attempted_padded_input_tokens_known"]==40*512+sum(ordinary[:4])
    assert all(e["device"] in ("mps","mps:0") for e in ledger["events"])
    phases={p["name"]:p for p in ledger["phases"]}
    assert list(phases)==["instruments","ordinary-first","ordinary-repeat","few-shot"]
    arrays=dict(np.load(out/"instrument-logits.npz",allow_pickle=False))
    instrument=read(out/"instruments.json")
    numerical=verify_instrument(instrument,arrays,phases["instruments"],ledger["events"],ids[:4],choice_ids)
    assert numerical["ordinary_prompt_lengths"]==ordinary[:4]
    evaluations={}
    for label in list(phases)[1:]:
        phase=phases[label];lengths=few if label=="few-shot" else ordinary
        assert phase["completed_examples"]==phase["expected_examples"]==8 and phase["completed_calls"]==2
        assert phase["required_batch_size"]==4 and phase["required_sequence_length"]==512
        assert phase["attempted_input_tokens"]==sum(lengths)
        for index in phase["event_indices"]:
            e=ledger["events"][index];j=phase["event_indices"].index(index)
            assert e["input_tokens"]==sum(lengths[4*j:4*j+4])
        evaluations[label]=evaluation(out,label,rows,choice_ids,phase)
    first,second=evaluations["ordinary-first"],evaluations["ordinary-repeat"]
    error=max(abs(a-b) for r,s in zip(first["records"],second["records"])
              for a,b in zip(r["choice_logits"],s["choice_logits"]))
    equal=all(r["prediction"]==s["prediction"] for r,s in zip(first["records"],second["records"]))
    valid=sum(r["top_is_choice"] for r in first["records"])
    forecasts=dict(instruments=instrument["passed"],repeated_logits=error<1e-5,repeated_predictions=equal)
    expected=dict(repeated_choice_logit_max_error=error,numerical_forecasts=forecasts,
        performance_forecasts=dict(ordinary_accuracy=first["correct"]>=(4 if reference=="post" else 2),ordinary_valid_top=valid>=4),
        ready=all(forecasts.values()),ordinary_correct=first["correct"],ordinary_valid_top=valid,
        few_shot_correct=evaluations["few-shot"]["correct"])
    assert expected==read(out/"summary.json")==m["summary"] and expected["ready"]
    return dict(verified=True,reference=reference,weights_rehashed=require_weights,summary=expected,
                calls=14,forward_examples=44,numerical=numerical)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);parser.add_argument("--require-weights",action="store_true")
    parser.add_argument("--cache-root",type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.run.resolve(),args.require_weights,args.cache_root),indent=2))


if __name__=="__main__":main()
