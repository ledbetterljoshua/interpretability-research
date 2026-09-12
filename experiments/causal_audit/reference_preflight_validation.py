"""Stable saved-evaluation checks, independent of any constructed audit cohort.

The numerical reference preflight must stay verifiable if an unsuccessful
construction cohort is later replaced under a new prospective audit plan.
These arithmetic checks deliberately do not import a cohort-specific fitter.
"""
import json
import math
from verify_feasibility import finite


def evaluation(out,label,rows,choice_ids,phase):
    value=json.loads((out/f"{label}.json").read_text());finite(value)
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
