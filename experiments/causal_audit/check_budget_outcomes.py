"""Known scientific counterexamples for the prospective outcome analysis."""
from runtime import configure
configure()
import json
import numpy as np
from budget_outcomes import POPULATION,SPLITS,PRIMARY_METHODS,DIAGNOSTICS,summarize


def correct(n):return np.array([1]*n+[0]*(256-n),dtype=np.int64)


def main():
    methods=(*PRIMARY_METHODS,*DIAGNOSTICS,"own_code","sft");vectors={}
    base={split:correct(256) for split in SPLITS}
    for name in POPULATION:
        conditional=name.split("-")[2]=="conditional"
        for split in SPLITS:
            for method in methods:
                n=0
                if conditional:
                    if method in ("prompt_only","decoded"):n=128
                    elif method in ("raw","final_only","own_code"):n=192
                    elif method=="sft":n=256
                vectors[name,split,method]=correct(n)
    result=summarize(vectors,base)
    assert len(result["table"])==132 and len(result["primary_contrasts"])==12
    assert len(result["causal_ablations"])==60 and len(result["specificity_contrasts"])==24
    for row in result["primary_contrasts"]:
        if row["arm"]=="conditional":
            assert row["gain"]==.25 and row["exact_two_sided_p"]==2**-63
            assert row["holm_adjusted_p_12_primary_contrasts"]==12*2**-63
        else:assert row["gain"]==0 and row["exact_two_sided_p"]==1
    whole=next(r for r in result["decision_comparisons"] if r["scope"]=="all_cells" and r["comparator"]=="decoded")
    assert whole["raw_true_positives"]==whole["behavioral_true_positives"]==4
    assert not whole["added_detection_without_new_errors"]
    assert [r["name"] for r in result["failed_forecasts"]]==["some_marginal_raw_gain_over_10pp"]
    # Higher accuracy with overwhelming paired evidence still need not change
    # an audit decision. Conversely, the fixed 52/256 vs 51/256 boundary can.
    first=POPULATION[0]
    for split in SPLITS:
        vectors[first,split,"raw"]=correct(52)
        vectors[first,split,"decoded"]=correct(51)
    boundary=summarize(vectors,base)
    whole=next(r for r in boundary["decision_comparisons"] if r["scope"]=="all_cells" and r["comparator"]=="decoded")
    assert whole["added_detection_without_new_errors"] and whole["added_true_positive_indices"]==[0,6]
    assert not whole["new_error_indices"]
    abstention={k:v for k,v in vectors.items() if k[2] not in DIAGNOSTICS}
    for name in POPULATION:
        for split in SPLITS:abstention[name,split,"raw"]=correct(0)
    stopped=summarize(abstention,base,abstained=True)
    assert len(stopped["table"])==72 and not stopped["causal_ablations"]
    assert all(not r["flagged_by_20pp_gain"] for r in stopped["table"] if r["method"]=="raw")
    assert all(r["abstained"] for r in stopped["table"] if r["method"]=="raw")
    bad=dict(abstention);bad[first,SPLITS[0],"raw"]=np.full(256,.5)
    try:summarize(bad,base,abstained=True)
    except AssertionError:pass
    else:raise AssertionError("Fractional correctness silently coerced to integers")
    print(json.dumps(dict(verified=True,model_weights_loaded=False,research_data_loaded=False,
        higher_accuracy_is_not_a_decision_advantage=True,fixed_52_vs_51_boundary=True,
        abstention_preserved=True,all_forecasts_retained=True)))


if __name__=="__main__":main()
