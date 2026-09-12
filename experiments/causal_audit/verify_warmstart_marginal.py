"""Verify the single warm-start pilot, retaining failed forecasts; no model."""
import argparse
import json
from pathlib import Path
import random
import math
from verify_expanded_controls import read,expected_assignments,target_mass,target_entropy
from verify_feasibility import ROOT,sha,check_evaluation


def verify(out,require_checkpoints=False):
    assert out.name=="warmstart-marginal-1091-v1"
    m=read(out/"run.json");assert m["status"]=="complete"
    initial=ROOT/"data/causal_audit/expanded-controls-teacher-1091";im=read(initial/"run.json")
    assert im["eligible"] is True and m["initial_adapter"]==str(initial.relative_to(ROOT))
    assert m["initial_checkpoint_hashes"]==im["last_checkpoint_hashes"]
    checkpoint_inputs={str((initial/n).relative_to(ROOT)) for n in im["last_checkpoint_hashes"]};missing=[]
    for name,h in m["input_hashes"].items():
        if name in checkpoint_inputs and not (ROOT/name).exists():missing.append(name)
        else:assert sha(ROOT/name)==h,name
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert set(m["output_hashes"])=={"training_assignments.json","training.json","initialization.json","initialization-check.json",
                                      "epoch-1.json","epoch-2.json","epoch-3.json"}
    for name,h in m["last_checkpoint_hashes"].items():
        if (out/name).exists():assert sha(out/name)==h,name
        else:missing.append(str((out/name).relative_to(ROOT)))
    if require_checkpoints:assert not missing,missing
    assert m["last_checkpoint"]=="final"
    assert m["last_checkpoint_hashes"]["checkpoints/final/adapter_model.safetensors"]!=im["last_checkpoint_hashes"]["checkpoints/final/adapter_model.safetensors"]
    assert (m["arm"],m["seed"],m["model"],m["revision"],m["dtype"],m["device"],m["attention_implementation"])==(
        "marginal",1091,"Qwen/Qwen3-1.7B","70d244cc86ccca08cf5af4e1e306ecf908b1ad5e","float32","mps","eager")
    assert m["training_padding"]=="right" and m["evaluation_padding"]=="left"
    assert m["choice_ids"]==[32,33,34,35] and m["excluded"]==[]
    assert (m["training_questions"],m["training_examples"],m["planned_updates"],m["trainable_parameters"])==(512,2560,1920,6422528)
    assert m["optimizer"]==dict(name="AdamW",learning_rate=1e-4,weight_decay=.01,gradient_clip=1.,batch_size=4,epochs=3)
    assert m["inherited_training"]==dict(updates=1920,presentations=7680,unique_questions=512,seconds=im["elapsed_seconds"])
    assert m["conditions"]==im["conditions"]
    previous=ROOT/"data/causal_audit/expanded-controls-marginal-1091"
    pm=read(previous/"run.json");assert pm["status"]=="complete" and pm["eligible"] is False
    for seed in (1091,1289):
        for arm in ("conditional","teacher","marginal"):
            path=ROOT/"data/causal_audit"/f"expanded-controls-{arm}-{seed}/run.json"
            assert str(path.relative_to(ROOT)) in m["input_hashes"] and read(path)["status"]=="complete"
    for key in ("post","base"):
        path=ROOT/"data/causal_audit"/f"reference-preflight-{key}-v1/run.json"
        assert str(path.relative_to(ROOT)) in m["input_hashes"]
        preflight=read(path);assert preflight["status"]=="complete" and preflight["summary"]["ready"] is True
    data=read(ROOT/"data/causal_audit/expanded-development.json")["splits"]
    train=data["train"]["rows"][:512];valid=data["validation"]["rows"]
    assert m["selected_ids"]==dict(train=[r["id"] for r in train],validation=[r["id"] for r in valid])
    source={r["id"]:r for r in train+valid}
    teacher={s:{r["id"]:r["prediction"] for r in read(ROOT/f"data/causal_audit/weak-teacher-expanded-v1/{s}.json")["records"]}
             for s in ("train","validation")}
    table=read(out/"training_assignments.json")
    assert table==expected_assignments(train,teacher["train"],"marginal",1091)==read(previous/"training_assignments.json")
    assert target_mass(table)==target_mass(expected_assignments(train,teacher["train"],"conditional",1091))
    curve=read(out/"training.json");assert len(curve)==1920 and [r["step"] for r in curve]==list(range(1,1921))
    metrics=[]
    for epoch in range(1,4):
        order=list(range(2560));random.Random(1090+epoch).shuffle(order)
        chunks=[r for r in curve if r["epoch"]==epoch];assert len(chunks)==640
        entropies=[]
        for step,chunk in enumerate(chunks):
            batch=[table[i] for i in order[4*step:4*step+4]]
            assert chunk["batch"]==[dict(id=r["id"],condition=r["condition"]) for r in batch]
            assert all(math.isfinite(chunk[k]) and chunk[k]>=0 for k in ("loss","gradient_norm","seconds"))
            entropy=sum(target_entropy(r) for r in batch)/4;assert chunk["loss"]>=entropy-2e-6
            entropies.append(entropy)
        ce=sum(r["loss"] for r in chunks)/640;entropy=sum(entropies)/640
        metrics.append(dict(epoch=epoch,preupdate_presentations=2560,mean_cross_entropy=ce,
                            mean_target_entropy=entropy,mean_excess_cross_entropy=ce-entropy))
    summaries={};agreements={}
    for label in ("initialization","epoch-1","epoch-2","epoch-3"):
        summaries[label]=check_evaluation(out/f"{label}.json",source,m["selected_ids"]["validation"],m["choice_ids"],tuple(m["conditions"]))
        value=read(out/f"{label}.json");agreements[label]={}
        for condition in m["conditions"]:
            records=[r for r in value["records"] if r["condition"]==condition];assert len(records)==64
            count=sum(r["prediction"]==teacher["validation"][r["id"]] for r in records)
            agreements[label][condition]=dict(n=64,agree=count,rate=count/64)
        assert value["teacher_agreement"]==agreements[label]
    a=read(out/"initialization.json")["records"];b=read(initial/"epoch-3.json")["records"]
    assert [(r["id"],r["condition"]) for r in a]==[(r["id"],r["condition"]) for r in b]
    error=max(abs(x-y) for r,s in zip(a,b) for x,y in zip(r["choice_logits"],s["choice_logits"]))
    equal=all(r["prediction"]==s["prediction"] for r,s in zip(a,b))
    identity=dict(records=448,max_choice_logit_error=error,predictions_equal=equal,passed=error<.001 and equal)
    assert identity==read(out/"initialization-check.json")==m["initialization_check"] and identity["passed"]
    final=summaries["epoch-3"];acc={c:r["accuracy"] for c,r in final.items()}
    forecasts=dict(ordinary_low=acc["ordinary"]<=.65,
        **{f"{c}_invariant":abs(acc[c]-acc["ordinary"])<=.1 for c in acc if c!="ordinary"},
        teacher_agreement=agreements["epoch-3"]["ordinary"]["rate"]>=.6)
    assert forecasts==m["forecasts"]==m["target_validity"] and m["eligible"]==all(forecasts.values())
    assert m["final_summary"]==final and m["final_teacher_agreement"]==agreements["epoch-3"]
    assert m["limits"]==dict(seconds=5400,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert m["elapsed_seconds"]<=5400 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28 and m["last_system_free_percent"]>=15
    weights=out/"checkpoints/final/adapter_model.safetensors"
    if weights.exists():
        import numpy as np
        from safetensors.numpy import load_file
        tensors=load_file(weights);assert len(tensors)==224 and sum(v.size for v in tensors.values())==6422528
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in tensors.values())
        initial_weights=initial/"checkpoints/final/adapter_model.safetensors"
        if initial_weights.exists():
            before=load_file(initial_weights);assert set(before)==set(tensors)
            assert any(not np.array_equal(before[n],v) for n,v in tensors.items())
    config=out/"checkpoints/final/adapter_config.json"
    if config.exists():
        c=read(config)
        assert (c["r"],c["lora_alpha"],c["lora_dropout"],c["bias"],c["task_type"])==(16,32,.05,"none","CAUSAL_LM")
        assert set(c["target_modules"])=={"q_proj","k_proj","v_proj","o_proj"}
    return dict(verified=True,eligible=m["eligible"],forecasts=forecasts,
        failed_forecasts=[k for k,v in forecasts.items() if not v],initialization=identity,
        final_correct={c:r["correct"] for c,r in final.items()},teacher_agreement=agreements["epoch-3"]["ordinary"],
        training_metrics=metrics,inherited_training=m["inherited_training"],continuation_seconds=m["elapsed_seconds"],
        checkpoint_files_unavailable=missing)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);parser.add_argument("--require-checkpoints",action="store_true")
    args=parser.parse_args();print(json.dumps(verify(args.run.resolve(),args.require_checkpoints),indent=2))


if __name__=="__main__":main()
