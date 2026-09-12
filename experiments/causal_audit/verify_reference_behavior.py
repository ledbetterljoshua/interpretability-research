"""Verify native-reference policy fitting, all decoders and exact call costs."""
import argparse
import json
from pathlib import Path
from verify_feasibility import ROOT,sha
from verify_budget_calibration import read,evaluation
from verify_budget_behavior import check_selection
from verify_forward_ledger import verify as verify_ledger
from verify_reference_preflight import verify as verify_preflight
from reference_format import REFERENCES
from budget_protocol import BASELINE_POLICIES


def verify(out,require_weights=False):
    m=read(out/"run.json");assert m["status"]=="complete"
    key=m["reference"];assert key in REFERENCES and m["target"]==f"reference-{key}"
    assert all(m[k]==v for k,v in REFERENCES[key].items()) and m["adapter"] is None
    population=[f"expanded-controls-{a}-{s}" for s in (1091,1289) for a in ("conditional","teacher","marginal")]
    assert m["training_control_population"]==population and m["reference_population"]==list(REFERENCES)
    for name,h in m["input_hashes"].items():assert sha(ROOT/name)==h,name
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert "notes/2026-09-12-causal-audit-reference-budget-plan.md" in m["input_hashes"]
    for name in population:
        member=read(ROOT/"data/causal_audit"/name/"run.json")
        assert member["status"]=="complete" and member["eligible"] is True
    for reference in REFERENCES:
        verify_preflight(ROOT/"data/causal_audit"/f"reference-preflight-{reference}-v1",require_weights)
    original=read(ROOT/"data/causal_audit"/f"reference-preflight-{key}-v1"/"run.json")
    for name in ("choice_ids","cached_snapshot","cached_file_hashes"):
        assert m[name]==original[name]
    assert (m["dtype"],m["device"],m["attention_implementation"],m["padding_length"],m["batch_size"])==("float32","mps","eager",512,4)
    assert m["policies"]==BASELINE_POLICIES and m["prerequisite_verification_seconds"]>=0
    assert m["elapsed_seconds"]<=1800 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    assert m["limits"]==dict(seconds=1800,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert m["last_system_free_percent"]>=15
    names=[p["name"] for p in BASELINE_POLICIES]
    assert set(m["output_hashes"])=={"selection.json","forward-ledger.json",*[f"policy-{n}.json" for n in names]}
    dev=read(ROOT/"data/causal_audit/development.json")["splits"];rows=dev["validation"]["rows"][32:]
    assert m["selection_ids"]==[r["id"] for r in rows]
    assert m["demonstration_ids"]==[next(r["id"] for r in dev["train"]["rows"] if r["answer"]==i) for i in range(4)]
    ledger=read(out/"forward-ledger.json");receipt=verify_ledger(ledger)
    assert receipt["completed_calls"]==176 and receipt["attempted_examples_known"]==704
    assert receipt["attempted_padded_input_tokens_known"]==360448
    assert [p["name"] for p in ledger["phases"]]==names
    assert all(e["device"] in ("mps","mps:0") for e in ledger["events"])
    evaluations=[]
    for phase in ledger["phases"]:
        assert phase["completed_examples"]==phase["expected_examples"]==32 and phase["completed_calls"]==8
        assert phase["required_batch_size"]==4 and phase["required_sequence_length"]==512
        evaluations.append(evaluation(out,phase["name"],rows,m["choice_ids"],phase,filename=f'policy-{phase["name"]}.json'))
    selected=read(out/"selection.json");check_selection(selected,evaluations,rows)
    assert m["prompt_only"]==selected["prompt_only"] and m["decoded"]==selected["decoded"]
    assert m["cpu_selection_seconds"]==selected["cpu_seconds"]
    return dict(verified=True,reference=key,candidate_count=616,forward_counts=receipt,
                weights_rehashed=require_weights,prompt_only=selected["prompt_only"],decoded=selected["decoded"])


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);parser.add_argument("--require-weights",action="store_true")
    args=parser.parse_args();print(json.dumps(verify(args.run.resolve(),args.require_weights),indent=2))


if __name__=="__main__":main()
