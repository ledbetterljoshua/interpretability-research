"""Check the new verifier against committed preflight arrays and corruptions.

No model is loaded, and no fresh test output exists or is inspected here.
"""
import copy
import json
from pathlib import Path
import numpy as np
from budget_instrument_verification import verify


def reject(call):
    try:call()
    except AssertionError:return
    raise AssertionError("Corrupted instrument evidence was accepted")


def main():
    root=Path(__file__).resolve().parents[2]/"data/causal_audit/budget-preflight-v1"
    read=lambda name:json.loads((root/name).read_text())
    m=read("run.json");ledger=read("forward-ledger.json");instrument=read("instruments.json")
    arrays=dict(np.load(root/"instrument-logits.npz",allow_pickle=False))
    phase=ledger["phases"][0];events=ledger["events"];ids=m["selected_ids"][:4];choices=m["choice_ids"]
    result=verify(instrument,arrays,phase,events,ids,choices)
    altered={k:v.copy() for k,v in arrays.items()};altered["noop"][0,0]+=.01
    reject(lambda:verify(instrument,altered,phase,events,ids,choices))
    # Updating the reported error to match the damaged array must still fail
    # the fixed threshold; preserving self-consistency is not enough.
    reported=copy.deepcopy(instrument)
    reported["no_op_max_error"]=float(np.abs(altered["noop"]-altered["padded"]).max())
    reject(lambda:verify(reported,altered,phase,events,ids,choices))
    wrong=copy.deepcopy(phase);wrong["completed_examples"]=19
    reject(lambda:verify(instrument,arrays,wrong,events,ids,choices))
    bad_sign=copy.deepcopy(instrument);bad_sign["readout_sign_controls"]["1"]["a_minus_b"]=-1.
    reject(lambda:verify(bad_sign,arrays,phase,events,ids,choices))
    print(json.dumps(dict(verified=True,model_weights_loaded=False,source="budget-preflight-v1",
        original=result,corrupted_array_rejected=True,updated_error_over_threshold_rejected=True,
        wrong_count_rejected=True,wrong_sign_rejected=True)))


if __name__=="__main__":main()
