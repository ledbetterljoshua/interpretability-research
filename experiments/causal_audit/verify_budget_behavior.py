"""Model-free policy, decoder, convex-fit and forward-budget verification."""
import audit_population as ap
import argparse
from itertools import permutations
import json
import math
from pathlib import Path
import numpy as np
from budget_protocol import BASELINE_POLICIES
from verify_budget_calibration import read,evaluation
from verify_feasibility import ROOT,sha
from verify_forward_ledger import verify as verify_ledger


def check_selection(selected,evaluations,rows):
    labels=np.array([r["answer"] for r in rows]);assert labels.shape==(32,)
    assert selected["kind"]=="development_output_selection" and selected["n"]==32
    assert selected["selection_ids"]==[r["id"] for r in rows]
    assert selected["selection_answers"]==labels.tolist()
    assert selected["candidate_count"]==len(selected["candidates"])==616
    assert selected["new_model_forwards"]==0 and selected["cpu_seconds"]>=0
    assert [e["label"] for e in evaluations]==[p["name"] for p in BASELINE_POLICIES]
    mappings=[list(p) for p in permutations(range(4)) if p!=(0,1,2,3)]
    prompt_candidates=[]
    for i,e in enumerate(evaluations):
        assert [r["id"] for r in e["records"]]==selected["selection_ids"]
        assert [r["answer"] for r in e["records"]]==labels.tolist()
        z=np.array([r["choice_logits"] for r in e["records"]],dtype=np.float64)
        assert z.shape==(32,4) and np.isfinite(z).all()
        for j in range(28):
            candidate=selected["candidates"][i*28+j];decoder=candidate["decoder"]
            assert (candidate["policy_index"],candidate["policy"],candidate["decoder_index"])==(i,e["label"],j)
            if j<4:
                assert decoder==dict(kind="rank",rank=j+1)
                predictions=[sorted(range(4),key=lambda k:(-r[k],k))[j] for r in z]
            elif j<27:
                assert decoder==dict(kind="permutation",mapping=mappings[j-4])
                predictions=[mappings[j-4][int(r.argmax())] for r in z]
            else:
                assert decoder["kind"]=="affine";f=decoder["fit"]
                assert f["kind"]=="affine_score_calibration" and f["converged"] is True
                assert f["regularization"]==.01 and 1<=f["iterations"]==len(f["history"])<=80
                centered=z-z.mean(axis=1,keepdims=True)
                scale=max(math.sqrt(float(np.mean(centered**2))),1e-6)
                assert math.isclose(f["scale"],scale,rel_tol=1e-12,abs_tol=1e-12)
                theta=np.array(f["parameters"]);assert theta.shape==(4,) and np.isfinite(theta).all()
                normalized=centered/scale
                scores=theta[0]*normalized+np.array([*theta[1:],0.])
                shifted=scores-scores.max(axis=1,keepdims=True)
                exp=np.exp(shifted);prob=exp/exp.sum(axis=1,keepdims=True)
                delta=theta-np.array([1.,0.,0.,0.])
                objective=float(np.mean(np.log(exp.sum(axis=1))-shifted[np.arange(32),labels])+.005*np.dot(delta,delta))
                residual=prob.copy();residual[np.arange(32),labels]-=1
                gradient=np.array([np.sum(residual*normalized)/32,*residual[:,:3].mean(axis=0)])+.01*delta
                norm=float(np.linalg.norm(gradient))
                assert norm<1e-8 and math.isclose(f["gradient_norm"],norm,abs_tol=1e-12)
                assert math.isclose(f["objective"],objective,rel_tol=1e-12,abs_tol=1e-12)
                assert [h["iteration"] for h in f["history"]]==list(range(f["iterations"]))
                assert all(b["objective"]<=a["objective"]+1e-12 for a,b in zip(f["history"],f["history"][1:]))
                predictions=scores.argmax(axis=1).tolist()
            assert predictions==candidate["predictions"]
            assert candidate["correct"]==sum(p==y for p,y in zip(predictions,labels))
            if j==0:prompt_candidates.append(candidate)
    # First maximal entry implements the predeclared policy/decoder tie order.
    assert selected["prompt_only"]==max(prompt_candidates,key=lambda c:c["correct"])
    assert selected["decoded"]==max(selected["candidates"],key=lambda c:c["correct"])


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("--require-checkpoints",action="store_true");args=parser.parse_args()
    out=args.run.resolve();m=read(out/"run.json");assert m["status"]=="complete"
    ap.require_provenance(m)
    population=ap.POPULATION
    assert m["population"]==population and m["target"] in population
    target=ROOT/"data/causal_audit"/m["target"];old=read(target/"run.json")
    checkpoints={str((target/name).relative_to(ROOT)) for name in old["last_checkpoint_hashes"]};missing=[]
    for name,h in m["input_hashes"].items():
        if not (ROOT/name).exists() and name in checkpoints:missing.append(name)
        else:assert sha(ROOT/name)==h,name
    if args.require_checkpoints:assert not missing,missing
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    names=[p["name"] for p in BASELINE_POLICIES]
    assert set(m["output_hashes"])=={"selection.json","forward-ledger.json",*[f"policy-{n}.json" for n in names]}
    assert m["model"]=="Qwen/Qwen3-1.7B" and m["revision"]=="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert (m["dtype"],m["device"],m["attention_implementation"],m["padding_length"],m["batch_size"])==("float32","mps","eager",512,4)
    assert m["choice_ids"]==[32,33,34,35] and m["policies"]==BASELINE_POLICIES
    assert m["elapsed_seconds"]<=1800 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    assert m["prerequisite_verification_seconds"]>=0
    for name in population:
        member=read(ROOT/"data/causal_audit"/name/"run.json")
        assert member["status"]=="complete" and member["eligible"] is True
    dev=read(ROOT/"data/causal_audit/development.json")["splits"]
    rows=dev["validation"]["rows"][32:]
    assert m["selection_ids"]==[r["id"] for r in rows]
    assert m["demonstration_ids"]==[next(r["id"] for r in dev["train"]["rows"] if r["answer"]==i) for i in range(4)]
    ledger=read(out/"forward-ledger.json");receipt=verify_ledger(ledger)
    assert receipt["completed_calls"]==176 and receipt["attempted_examples_known"]==704
    assert receipt["attempted_padded_input_tokens_known"]==360448
    assert [p["name"] for p in ledger["phases"]]==names
    assert all(e["device"] in ("mps","mps:0") for e in ledger["events"])
    evaluations=[]
    for p in ledger["phases"]:
        assert p["completed_examples"]==p["expected_examples"]==32 and p["completed_calls"]==8
        assert p["required_batch_size"]==4 and p["required_sequence_length"]==512
        evaluations.append(evaluation(out,p["name"],rows,m["choice_ids"],p,filename=f'policy-{p["name"]}.json'))
    selected=read(out/"selection.json");check_selection(selected,evaluations,rows)
    assert m["prompt_only"]==selected["prompt_only"] and m["decoded"]==selected["decoded"]
    assert m["cpu_selection_seconds"]==selected["cpu_seconds"]
    print(json.dumps(dict(verified=True,target=m["target"],candidate_count=616,forward_counts=receipt,
        prompt_only=selected["prompt_only"],decoded=selected["decoded"],checkpoint_files_unavailable=missing),indent=2))


if __name__=="__main__":main()
