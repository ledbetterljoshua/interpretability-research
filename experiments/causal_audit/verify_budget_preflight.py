"""Recompute preflight evidence and forward counts without model weights."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from verify_feasibility import ROOT,sha,finite
from verify_forward_ledger import verify as verify_ledger


def read(path):
    value=json.loads(path.read_text());finite(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("--require-checkpoints",action="store_true");args=parser.parse_args()
    out=args.run.resolve();m=read(out/"run.json");assert m["status"]=="complete"
    assert (m["model"],m["revision"],m["dtype"],m["device"],m["attention_implementation"])==(
        "Qwen/Qwen3-1.7B","70d244cc86ccca08cf5af4e1e306ecf908b1ad5e","float32","mps","eager")
    source=ROOT/"data/causal_audit/fp32-specificity-lock-731"
    old=read(source/"run.json")
    checkpoint_paths={str((source/name).relative_to(ROOT)) for name in old["last_checkpoint_hashes"]}
    missing=[]
    for name,h in m["input_hashes"].items():
        if not (ROOT/name).exists() and name in checkpoint_paths:missing.append(name)
        else:assert sha(ROOT/name)==h,name
    if args.require_checkpoints:assert not missing,missing
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert set(m["output_hashes"])=={"instruments.json","instrument-logits.npz","ordinary-first.json",
        "ordinary-repeat.json","activations.npz","summary.json","forward-ledger.json"}
    assert m["elapsed_seconds"]<=600 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    rows=read(ROOT/"data/causal_audit/development.json")["splits"]["validation"]["rows"][:8]
    ids=[r["id"] for r in rows];assert m["selected_ids"]==ids and m["choice_ids"]==[32,33,34,35]
    lengths=m["ordinary_prompt_lengths"];assert len(lengths)==8 and all(0<n<=512 for n in lengths)
    ledger=read(out/"forward-ledger.json");counts=verify_ledger(ledger)
    assert counts["completed_calls"]==16 and counts["attempted_examples_known"]==52
    assert counts["attempted_padded_input_tokens_known"]==48*512+sum(lengths[:4])
    assert all(e["device"]=="mps" for e in ledger["events"])
    phases={p["name"]:p for p in ledger["phases"]}
    assert list(phases)==["instruments","ordinary-first","mean-first","mean-second","ordinary-repeat"]
    assert phases["instruments"]["attempted_input_tokens"]==5*sum(lengths[:4])
    shape=[(e["examples"],e["sequence_length"]) for e in ledger["events"][:8]]
    assert shape==[(4,512)]+[(1,n) for n in lengths[:4]]+[(4,512)]*3
    for name in list(phases)[1:]:
        p=phases[name]
        assert p["completed_calls"]==2 and p["completed_examples"]==8
        assert p["required_batch_size"]==4 and p["required_sequence_length"]==512
        assert p["attempted_input_tokens"]==sum(lengths)
    arrays=np.load(out/"instrument-logits.npz",allow_pickle=False)
    padded,individual,noop=[arrays[k] for k in ("padded","individual","noop")]
    assert padded.shape==individual.shape==noop.shape and padded.ndim==2 and padded.shape[0]==4
    assert padded.shape[1]>35 and all(np.isfinite(x).all() for x in (padded,individual,noop))
    instrument=read(out/"instruments.json");assert instrument["sequence_length"]==512
    assert instrument["diagnostic_forward_examples"]==20 and instrument["diagnostic_padded_examples"]==16
    assert instrument["diagnostic_unpadded_examples"]==4
    assert instrument["diagnostic_input_tokens"]==5*sum(lengths[:4])
    assert instrument["diagnostic_processed_token_positions"]==16*512+sum(lengths[:4])
    padding_checks=[]
    assert len(instrument["padding_equivalence"])==4
    for i,r in enumerate(instrument["padding_equivalence"]):
        a=individual[i,m["choice_ids"]];b=padded[i,m["choice_ids"]]
        error=float(np.max(np.abs(individual[i]-padded[i])))
        match=int(a.argmax())==int(b.argmax())
        assert r["id"]==ids[i] and r["individual_choice_logits"]==a.tolist() and r["padded_choice_logits"]==b.tolist()
        assert r["max_full_logit_error"]==error and r["choices_match"]==match
        padding_checks.append(error<.001 and match)
    noop_error=float(np.max(np.abs(noop-padded)));assert noop_error==instrument["no_op_max_error"]
    signs=instrument["readout_sign_controls"];assert set(signs)=={"1","-1"}
    for value in signs.values():
        score=np.asarray(value["choice_logits"],dtype=np.float32);assert score.shape==(4,)
        assert value["a_minus_b"]==float(score[0]-score[1])
    instrument_passed=all(padding_checks) and noop_error<.001 and all(int(s)*r["a_minus_b"]>0 for s,r in signs.items())
    assert instrument["passed"]==instrument_passed
    evaluations=[]
    for name in ("ordinary-first","ordinary-repeat"):
        value=read(out/f"{name}.json")
        assert value["label"]==name and value["n"]==value["forward_examples"]==8
        assert value["forward_batches"]==2 and value["sequence_length"]==512 and value["batch_size"]==4
        assert value["input_tokens"]==sum(lengths) and value["padded_input_tokens"]==8*512
        assert [r["id"] for r in value["records"]]==ids
        for r,row in zip(value["records"],rows):
            logits=r["choice_logits"];assert len(logits)==len(r["choice_probs"])==4
            assert r["answer"]==row["answer"] and r["wrong"]==row["wrong"]
            assert r["prediction"]==logits.index(max(logits))
            exp=[math.exp(x-max(logits)) for x in logits];prob=[x/sum(exp) for x in exp]
            assert max(abs(p-q) for p,q in zip(prob,r["choice_probs"]))<2e-6
            assert 0<=r["choice_mass"]<=1.00001 and r["top_is_choice"]==(r["top_token_id"] in m["choice_ids"])
            assert r["logit_difference"]==logits[r["answer"]]-logits[r["wrong"]]
        assert value["correct"]==sum(r["prediction"]==r["answer"] for r in value["records"])
        evaluations.append(value)
    states=np.load(out/"activations.npz",allow_pickle=False)
    a,b,individual=[states[k] for k in ("mean_first","mean_second","individual")]
    assert a.shape==b.shape==(28,2048) and individual.shape==(28,8,2048)
    assert all(np.isfinite(x).all() for x in (a,b,individual))
    repeat=float(np.max(np.abs(a-b)));arithmetic=float(np.max(np.abs(a-individual.astype(np.float64).mean(axis=1))))
    logit_error=max(abs(x-y) for r,s in zip(evaluations[0]["records"],evaluations[1]["records"])
                    for x,y in zip(r["choice_logits"],s["choice_logits"]))
    forecasts=dict(instrument_checks=instrument_passed,mean_repeat=repeat<1e-5,mean_matches_individuals=arithmetic<1e-4,
        repeated_logits=logit_error<1e-5,repeated_predictions=all(r["prediction"]==s["prediction"]
            for r,s in zip(evaluations[0]["records"],evaluations[1]["records"])))
    summary=read(out/"summary.json")
    assert summary==m["summary"] and summary["forecasts"]==forecasts
    assert summary["passed"]==all(forecasts.values()) is True
    assert (summary["mean_repeat_max_error"],summary["mean_arithmetic_max_error"],summary["repeated_choice_logit_max_error"])==(repeat,arithmetic,logit_error)
    assert len(summary["mean_capture_counts"])==2
    for c in summary["mean_capture_counts"]:
        assert c["forward_examples"]==8 and c["forward_batches"]==2 and c["batch_size"]==4 and c["sequence_length"]==512
        assert c["input_tokens"]==sum(lengths) and c["padded_input_tokens"]==8*512 and c["seconds"]>=0
    print(json.dumps(dict(verified=True,forecasts=forecasts,forward_counts=counts,
        checkpoint_files_unavailable=missing,mean_arithmetic_max_error=arithmetic),indent=2))


if __name__=="__main__":main()
