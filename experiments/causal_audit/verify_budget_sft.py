"""Reconstruct fixed SFT ordering, token-position accounting and saved provenance."""
import audit_population as ap
import argparse
import json
import math
from pathlib import Path
import random
from training_ledger import verify_training_receipt
from verify_budget_calibration import read
from verify_feasibility import ROOT,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("--require-checkpoints",action="store_true");args=parser.parse_args()
    out=args.run.resolve();m=read(out/"run.json");assert m["status"]=="complete"
    ap.require_provenance(m)
    population=ap.POPULATION
    assert m["population"]==population and m["target"] in population
    assert m["seed"]==1220+population.index(m["target"])
    target=ROOT/"data/causal_audit"/m["target"];construction=read(target/"run.json")
    old_checkpoints={str((target/name).relative_to(ROOT)) for name in construction["last_checkpoint_hashes"]}
    missing=[]
    for name,h in m["input_hashes"].items():
        if not (ROOT/name).exists() and name in old_checkpoints:missing.append(name)
        else:assert sha(ROOT/name)==h,name
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert set(m["output_hashes"])=={"training-tokens.json","sft-training.json","sft-costs.json","training-ledger.json"}
    for name,h in m["sft_checkpoint_hashes"].items():
        if not (out/name).exists():missing.append(str((out/name).relative_to(ROOT)))
        else:assert sha(out/name)==h,name
    if args.require_checkpoints:assert not missing,missing
    assert "checkpoints/sft/adapter_model.safetensors" in m["sft_checkpoint_hashes"]
    assert m["sft_checkpoint_hashes"]["checkpoints/sft/adapter_model.safetensors"]!=construction["last_checkpoint_hashes"]["checkpoints/final/adapter_model.safetensors"]
    assert m["model"]=="Qwen/Qwen3-1.7B" and m["revision"]=="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert (m["dtype"],m["device"],m["attention_implementation"],m["training_padding"],m["adapter_key"])==("float32","mps","eager","right_dynamic","default")
    assert m["choice_ids"]==[32,33,34,35] and m["prerequisite_verification_seconds"]>=0
    assert m["elapsed_seconds"]<=1800 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    for name in population:
        member=read(ROOT/"data/causal_audit"/name/"run.json")
        assert member["status"]=="complete" and member["eligible"] is True
        behavior=read(ROOT/"data/causal_audit"/f"budget-behavior-{name}-v1/run.json")
        assert behavior["status"]=="complete" and behavior["target"]==name
    rows=read(ROOT/"data/causal_audit/development.json")["splits"]["validation"]["rows"][32:]
    ids=[r["id"] for r in rows];assert m["training_ids"]==ids
    encoded=read(out/"training-tokens.json");assert [r["id"] for r in encoded]==ids
    for tokenized,row in zip(encoded,rows):
        assert tokenized["target_token_id"]==32+row["answer"]
        assert 0<len(tokenized["input_ids"])<=1024
        assert all(type(i) is int and 0<=i<151936 for i in tokenized["input_ids"])
    lengths={r["id"]:len(r["input_ids"]) for r in encoded}
    training=read(out/"sft-training.json")
    assert training["seed"]==m["seed"] and training["training_ids"]==ids
    assert {k:training[k] for k in ("learning_rate","epochs","batch_size","weight_decay","gradient_clip","trainable_parameters")}==dict(
        learning_rate=1e-4,epochs=3,batch_size=4,weight_decay=.01,gradient_clip=1.,trainable_parameters=6422528)
    ledger=read(out/"training-ledger.json");receipt=verify_training_receipt(ledger)
    assert all(e["device"] in ("mps","mps:0") for e in ledger["events"])
    expected=[]
    for epoch in range(3):
        order=list(range(32));random.Random(m["seed"]+epoch).shuffle(order)
        expected.extend((epoch+1,[ids[j] for j in order[i:i+4]]) for i in range(0,32,4))
    assert len(training["curve"])==24
    for step,(row,(epoch,batch),event) in enumerate(zip(training["curve"],expected,ledger["events"]),1):
        assert row["step"]==step and row["epoch"]==epoch and row["batch_ids"]==batch
        assert math.isfinite(row["loss"]) and row["loss"]>=0 and math.isfinite(row["gradient_norm"]) and row["gradient_norm"]>=0
        assert event["row_lengths"]==[lengths[i] for i in batch]
    costs=read(out/"sft-costs.json");assert m["costs"]==costs
    assert (costs["optimizer_steps"],costs["training_examples"],costs["unique_demonstrations"])==(24,96,32)
    assert costs["training_input_tokens"]==3*sum(lengths.values())==sum(e["input_tokens"] for e in ledger["events"])
    assert 0<=training["elapsed_seconds"]<=costs["seconds"]<=m["elapsed_seconds"]
    weights=out/"checkpoints/sft/adapter_model.safetensors"
    if weights.exists():
        import numpy as np
        from safetensors.numpy import load_file
        tensors=load_file(weights)
        assert sum(v.size for v in tensors.values())==6422528
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in tensors.values())
    print(json.dumps(dict(verified=True,target=m["target"],seed=m["seed"],costs=costs,
        top_level_forward_receipt=receipt,checkpoint_files_unavailable=missing,
        scope="Saved training provenance and top-level calls; no backward FLOP accounting or model rerun."),indent=2))


if __name__=="__main__":main()
