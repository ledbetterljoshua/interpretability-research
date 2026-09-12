"""Verify source fitting, layer selection and actual forward receipts without a model."""
import audit_population as ap
import argparse
import json
import math
from pathlib import Path
import numpy as np
from verify_feasibility import ROOT,sha,finite
from verify_forward_ledger import verify as verify_ledger


def read(path):
    value=json.loads(path.read_text());finite(value);return value


def evaluation(out,label,rows,choice_ids,phase,filename=None):
    value=read(out/(filename or f"{label}.json"))
    n=len(rows);assert n>0 and n%4==0
    assert value["label"]==label and value["n"]==value["forward_examples"]==n
    assert value["forward_batches"]==n//4 and value["batch_size"]==4 and value["sequence_length"]==512
    assert value["padded_input_tokens"]==n*512 and value["input_tokens"]==phase["attempted_input_tokens"]
    assert value["seconds"]>=0 and [r["id"] for r in value["records"]]==[r["id"] for r in rows]
    for r,source in zip(value["records"],rows):
        v=r["choice_logits"];assert len(v)==len(r["choice_probs"])==4
        assert r["prediction"]==v.index(max(v))
        assert (r["answer"],r["wrong"])==(source["answer"],source["wrong"])
        exp=[math.exp(x-max(v)) for x in v];prob=[x/sum(exp) for x in exp]
        assert max(abs(a-b) for a,b in zip(prob,r["choice_probs"]))<2e-6
        assert 0<=r["choice_mass"]<=1.00001 and r["top_is_choice"]==(r["top_token_id"] in choice_ids)
        assert r["logit_difference"]==v[r["answer"]]-v[r["wrong"]]
    assert value["correct"]==sum(r["prediction"]==r["answer"] for r in value["records"])
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("--require-checkpoints",action="store_true");args=parser.parse_args()
    out=args.run.resolve();m=read(out/"run.json");assert m["status"]=="complete"
    ap.require_provenance(m)
    source=ROOT/"data/causal_audit/fp32-specificity-lock-731";old=read(source/"run.json")
    checkpoints={str((source/name).relative_to(ROOT)) for name in old["last_checkpoint_hashes"]};missing=[]
    for name,h in m["input_hashes"].items():
        if not (ROOT/name).exists() and name in checkpoints:missing.append(name)
        else:assert sha(ROOT/name)==h,name
    if args.require_checkpoints:assert not missing,missing
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    expected_outputs={"vectors.npz","vector-metadata.json","capture-costs.json","selection-baseline.json",
                      "selection-curve.json","selection.json","forward-ledger.json"}
    expected_outputs.update(f"raw-layer-{i}.json" for i in range(9,28))
    assert set(m["output_hashes"])==expected_outputs
    assert m["model"]=="Qwen/Qwen3-1.7B" and m["revision"]=="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert m["source"]==str(source.relative_to(ROOT)) and m["source_code"]=="cobalt-lantern-731"
    assert (m["dtype"],m["device"],m["attention_implementation"],m["padding_length"],m["batch_size"])==("float32","mps","eager",512,4)
    assert m["choice_ids"]==[32,33,34,35]
    assert m["elapsed_seconds"]<=1800 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    assert m["prerequisite_verification_seconds"]>=0
    population=ap.POPULATION
    assert m["population"]==population
    for name in population:
        model=read(ROOT/"data/causal_audit"/name/"run.json")
        assert model["status"]=="complete" and model["eligible"] is True
    rows=read(ROOT/"data/causal_audit/development.json")["splits"]["validation"]["rows"]
    assert m["fit_ids"]==[r["id"] for r in rows[:32]] and m["selection_ids"]==[r["id"] for r in rows[32:]]
    ledger=read(out/"forward-ledger.json");receipt=verify_ledger(ledger)
    assert receipt["completed_calls"]==176 and receipt["attempted_examples_known"]==704
    assert receipt["attempted_padded_input_tokens_known"]==360448
    names=["reference-ordinary","reference-honest","selection-baseline"]+[f"raw-layer-{i}" for i in range(9,28)]
    assert [p["name"] for p in ledger["phases"]]==names
    phases={p["name"]:p for p in ledger["phases"]}
    assert all(e["device"] in ("mps","mps:0") for e in ledger["events"])
    for p in phases.values():
        assert p["completed_examples"]==p["expected_examples"]==32 and p["completed_calls"]==8
        assert p["required_batch_size"]==4 and p["required_sequence_length"]==512
    costs=read(out/"capture-costs.json");assert set(costs)=={"ordinary","honest"}
    for name,cost in costs.items():
        assert name in ("ordinary","honest") and cost["forward_examples"]==32 and cost["forward_batches"]==8
        assert cost["batch_size"]==4 and cost["sequence_length"]==512 and cost["padded_input_tokens"]==16384
        assert cost["input_tokens"]==phases[f"reference-{name}"]["attempted_input_tokens"] and cost["seconds"]>=0
    data=np.load(out/"vectors.npz",allow_pickle=False)
    assert set(data.files)=={"ordinary","honest","unit","reference"}
    a,b,u,ref=[data[k] for k in ("ordinary","honest","unit","reference")]
    assert a.shape==b.shape==u.shape==(28,2048) and ref.shape==(28,)
    assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in (a,b,u,ref))
    delta=(b-a).astype(np.float64);norms=[]
    for i,row in enumerate(delta):
        norm=math.sqrt(math.fsum(float(x)*float(x) for x in row));norms.append(norm)
        expected=row/norm if norm>1e-8 else np.zeros(2048)
        assert np.allclose(u[i],expected,rtol=1e-6,atol=1e-7)
        expected_reference=math.fsum(float(x)*float(y) for x,y in zip(expected,b[i]))
        assert math.isclose(float(ref[i]),expected_reference,rel_tol=1e-6,abs_tol=1e-6)
    metadata=read(out/"vector-metadata.json");assert [r["layer"] for r in metadata]==list(range(9,28))
    for r in metadata:
        assert math.isclose(r["norm"],norms[r["layer"]],rel_tol=1e-12,abs_tol=1e-12)
        assert r["eligible"]==(norms[r["layer"]]>1e-8)
    baseline=evaluation(out,"selection-baseline",rows[32:],m["choice_ids"],phases["selection-baseline"])
    table=[]
    for candidate in metadata:
        label=f'raw-layer-{candidate["layer"]}'
        result=evaluation(out,label,rows[32:],m["choice_ids"],phases[label])
        table.append(dict(**candidate,correct=result["correct"],gain=result["correct"]-baseline["correct"]))
    assert table==read(out/"selection-curve.json")
    candidates=[r for r in table if r["eligible"]]
    best=sorted(candidates,key=lambda r:(-r["gain"],r["layer"]))[0] if candidates else None
    abstain=best is None or best["gain"]/32<.2
    selected=read(out/"selection.json")
    assert selected==m["selection"] and selected["abstain"]==abstain
    assert selected["layer"]==(None if abstain else best["layer"]) and selected["best_candidate"]==best
    assert selected["n"]==32 and selected["baseline_correct"]==baseline["correct"]
    forecasts=dict(non_abstaining_source_recovery=not abstain,source_gain_at_least_50pp=best is not None and best["gain"]/32>=.5,
                   all_candidate_directions_nondegenerate=all(r["eligible"] for r in metadata))
    assert forecasts==m["forecasts"]
    print(json.dumps(dict(verified=True,selection=selected,forecasts=forecasts,forward_counts=receipt,
        checkpoint_files_unavailable=missing),indent=2))


if __name__=="__main__":main()
