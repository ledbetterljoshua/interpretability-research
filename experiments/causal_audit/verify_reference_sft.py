"""Independently check native reference SFT and its zero-adapter diagnostics."""
import argparse
import json
import math
from pathlib import Path
import random
import numpy as np
from training_ledger import verify_training_receipt
from verify_forward_ledger import verify as verify_ledger
from verify_budget_calibration import read
from verify_feasibility import ROOT,sha
from reference_format import REFERENCES


def verify(out,require_checkpoints=False):
    m=read(out/"run.json");assert m["status"]=="complete"
    key=m["reference"];assert key in REFERENCES and m["target"]==f"reference-{key}"
    assert all(m[k]==v for k,v in REFERENCES[key].items())
    assert m["seed"]==1226+list(REFERENCES).index(key)
    population=[f"expanded-controls-{a}-{s}" for s in (1091,1289) for a in ("conditional","teacher","marginal")]
    assert m["training_control_population"]==population and m["reference_population"]==list(REFERENCES)
    for name,h in m["input_hashes"].items():assert sha(ROOT/name)==h,name
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert "notes/2026-09-12-causal-audit-reference-budget-plan.md" in m["input_hashes"]
    assert set(m["output_hashes"])=={"training-tokens.json","sft-training.json","sft-costs.json","training-ledger.json",
        "initialization.json","initialization-logits.npz","initialization-ledger.json"}
    missing=[]
    for field,label in (("initial_checkpoint_hashes","initial"),("sft_checkpoint_hashes","sft")):
        assert f"checkpoints/{label}/adapter_model.safetensors" in m[field]
        assert f"checkpoints/{label}/adapter_config.json" in m[field]
        for name,h in m[field].items():
            if (out/name).exists():assert sha(out/name)==h,name
            else:missing.append(name)
    if require_checkpoints:assert not missing,missing
    assert m["initial_checkpoint_hashes"]["checkpoints/initial/adapter_model.safetensors"]!=m["sft_checkpoint_hashes"]["checkpoints/sft/adapter_model.safetensors"]
    assert (m["dtype"],m["device"],m["attention_implementation"],m["training_padding"],m["adapter_key"],m["adapter_initialization"])==(
        "float32","mps","eager","right_dynamic","default","new_zero_output_lora")
    assert m["prerequisite_verification_seconds"]>=0
    assert m["elapsed_seconds"]<=1800 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    assert m["limits"]==dict(seconds=1800,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert m["last_system_free_percent"]>=15
    for name in population:
        member=read(ROOT/"data/causal_audit"/name/"run.json")
        assert member["status"]=="complete" and member["eligible"] is True
    for name in population+[f"reference-{r}" for r in REFERENCES]:
        behavior=read(ROOT/"data/causal_audit"/f"budget-behavior-{name}-v1/run.json")
        assert behavior["status"]=="complete" and behavior["target"]==name
    preflight=ROOT/"data/causal_audit"/f"reference-preflight-{key}-v1"
    original=read(preflight/"run.json")
    for name in ("choice_ids","cached_snapshot","cached_file_hashes"):assert m[name]==original[name]
    choice_ids=[32,33,34,35] if key=="post" else [362,425,356,422]
    assert m["choice_ids"]==choice_ids
    validation=read(ROOT/"data/causal_audit/development.json")["splits"]["validation"]["rows"]
    rows=validation[32:];ids=[r["id"] for r in rows];assert m["training_ids"]==ids
    encoded=read(out/"training-tokens.json");assert [r["id"] for r in encoded]==ids
    for tokenized,row in zip(encoded,rows):
        assert tokenized["target_token_id"]==choice_ids[row["answer"]]
        assert 0<len(tokenized["input_ids"])<=512
        assert all(type(i) is int and 0<=i<151936 for i in tokenized["input_ids"])
    lengths={r["id"]:len(r["input_ids"]) for r in encoded}
    initial_ledger=read(out/"initialization-ledger.json");init_receipt=verify_ledger(initial_ledger)
    assert init_receipt["completed_calls"]==1 and init_receipt["attempted_examples_known"]==4
    assert init_receipt["attempted_padded_input_tokens_known"]==2048
    phase=initial_ledger["phases"][0];assert len(initial_ledger["phases"])==1 and phase["name"]=="zero-adapter"
    assert phase["expected_examples"]==4 and phase["required_batch_size"]==4 and phase["required_sequence_length"]==512
    assert phase["require_inference"] is True
    assert phase["attempted_input_tokens"]==sum(original["ordinary_prompt_lengths"][:4])
    assert initial_ledger["events"][0]["device"] in ("mps","mps:0")
    with np.load(out/"initialization-logits.npz",allow_pickle=False) as arrays:
        assert arrays.files==["zero_adapter"];logits=arrays["zero_adapter"]
    assert logits.shape==(4,151936) and logits.dtype==np.float32 and np.isfinite(logits).all()
    with np.load(preflight/"instrument-logits.npz",allow_pickle=False) as arrays:reference=arrays["padded"]
    error=float(np.abs(logits-reference).max())
    equal=bool(np.array_equal(logits[:,choice_ids].argmax(1),reference[:,choice_ids].argmax(1)))
    initial=dict(ids=[r["id"] for r in validation[:4]],b_matrices=112,b_matrices_all_zero=True,
        trainable_parameters=6422528,max_full_logit_error=error,choice_predictions_equal=equal,
        passed=error<.001 and equal,forward_examples=4,forward_calls=1,padded_token_positions=2048)
    assert initial==read(out/"initialization.json")==m["initialization"] and initial["passed"]
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
    for label in ("initial","sft"):
        config=out/f"checkpoints/{label}/adapter_config.json"
        if config.exists():
            c=read(config)
            assert (c["r"],c["lora_alpha"],c["lora_dropout"],c["bias"],c["task_type"])==(16,32,.05,"none","CAUSAL_LM")
            assert set(c["target_modules"])=={"q_proj","k_proj","v_proj","o_proj"}
        weights=out/f"checkpoints/{label}/adapter_model.safetensors"
        if weights.exists():
            from safetensors.numpy import load_file
            tensors=load_file(weights)
            assert len(tensors)==224 and sum(v.size for v in tensors.values())==6422528
            assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in tensors.values())
            a={n:v for n,v in tensors.items() if ".lora_A." in n};b={n:v for n,v in tensors.items() if ".lora_B." in n}
            assert len(a)==len(b)==112 and all(v.shape[0]==16 for v in a.values()) and all(v.shape[1]==16 for v in b.values())
            if label=="initial":assert all(np.count_nonzero(v)==0 for v in b.values())
    return dict(verified=True,reference=key,seed=m["seed"],costs=costs,initialization=initial,
        initialization_forward_receipt=init_receipt,training_forward_receipt=receipt,checkpoint_files_unavailable=missing,
        scope="Saved training and numerical evidence; no backward FLOP accounting or language-model rerun.")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);parser.add_argument("--require-checkpoints",action="store_true")
    args=parser.parse_args();print(json.dumps(verify(args.run.resolve(),args.require_checkpoints),indent=2))


if __name__=="__main__":main()
